import re
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, date
from curl_cffi import requests as cffi_requests

st.set_page_config(page_title="SPX GEX Command Center", layout="wide")

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.cboe.com/",
    "Origin": "https://www.cboe.com",
}
OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")
URLS = {
    "SPX": "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json",
    "QQQ": "https://cdn.cboe.com/api/global/delayed_quotes/options/QQQ.json",
    "VIX": "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX.json",
    "VIX1D": "https://cdn.cboe.com/api/global/delayed_quotes/quotes/_VIX1D.json",
}

@st.cache_data(ttl=300)
def fetch_json(url):
    r = cffi_requests.get(url, headers=HEADERS, impersonate="chrome", timeout=30)
    if r.status_code != 200:
        st.error(f"Cboe returned HTTP {r.status_code} for {url}")
        st.stop()
    return r.json()["data"]
    T = np.maximum(T, 10.0 / (252.0 * 390.0))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    gamma = pdf / (S * sigma * sqrt_T)
    vanna = -pdf * d2 / sigma
    charm = pdf * d2 / (2.0 * T)
    return gamma, vanna, charm

def analyze_chain(chain, scope="All Exp"):
    spot = float(chain["current_price"])
    today = date.today()
    strikes, Ts, ivs, ois, vols, signs = [], [], [], [], [], []

    for opt in chain["options"]:
        m = OCC_RE.match(opt["option"])
        if not m:
            continue
        oi = opt.get("open_interest") or 0
        iv = opt.get("iv") or 0
        if oi <= 0 or iv <= 0.01:
            continue
        ymd = m.group(2)
        exp = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        dte = max((exp - today).days, 0)
        if scope == "0DTE" and dte > 0:
            continue
        strike = int(m.group(4)) / 1000.0
        if abs(strike - spot) / spot > 0.20:
            continue

        strikes.append(strike)
        Ts.append(dte / 365.0)
        ivs.append(float(iv))
        ois.append(float(oi))
        vols.append(float(opt.get("volume") or 0))
        signs.append(1.0 if m.group(3) == "C" else -1.0)

    if not strikes:
        return None

    K, T, sigma, OI, VOL, sign = map(np.array, (strikes, Ts, ivs, ois, vols, signs))
    gamma, vanna, charm = bs_greeks(spot, K, T, sigma)

    gex = sign * gamma * OI * 100.0 * (spot**2) * 0.01
    vex = sign * vanna * OI * 100.0 * spot * 0.01
    # Hourly charm in ES-equivalent futures (1 ES = $50/pt)
    charm_es_hr = (sign * charm * OI * 100.0 / (252.0 * 6.5)).sum() * (spot / 50.0) / 100.0

    uniq = np.unique(K)
    net_k, call_k, put_k, vol_oi_k = {}, {}, {}, {}
    for k_val in uniq:
        mask = K == k_val
        c_g = gex[mask & (sign > 0)].sum()
        p_g = gex[mask & (sign < 0)].sum()
        call_k[k_val], put_k[k_val], net_k[k_val] = c_g, p_g, c_g + p_g
        vol_oi_k[k_val] = VOL[mask].sum() / max(OI[mask].sum(), 1.0)

    call_wall = max(call_k, key=call_k.get)
    put_wall = min(put_k, key=put_k.get)
    king = max(net_k, key=lambda x: abs(net_k[x]))

    # True Spot-Shift Zero Gamma (±6%)
    grid = np.linspace(spot * 0.94, spot * 1.06, 51)
    totals = np.array([(sign * bs_greeks(s, K, T, sigma)[0] * OI * 100.0 * (s**2) * 0.01).sum() for s in grid])
    cross = np.where(np.diff(np.sign(totals)) != 0)[0]
    zero_gamma = None
    if len(cross) > 0:
        i = cross[0]
        zero_gamma = grid[i] - totals[i] * (grid[i + 1] - grid[i]) / (totals[i + 1] - totals[i])

    return {
        "spot": spot, "net_gex": gex.sum() / 1e9, "net_vex": vex.sum() / 1e9,
        "charm_es_hr": charm_es_hr, "call_wall": (call_wall, call_k[call_wall] / 1e9),
        "put_wall": (put_wall, put_k[put_wall] / 1e9), "king": (king, net_k[king] / 1e9),
        "zero_gamma": zero_gamma, "net_k": net_k, "vol_oi_k": vol_oi_k,
    }

# --- UI ---
st.title("SPX Multi-Factor GEX Command Center")
col_a, col_b = st.columns([2, 1])
scope = col_a.radio("Expiration Scope", ["All Exp", "0DTE"], horizontal=True)
if col_b.button("Refresh Data Now"):
    st.cache_data.clear()

with st.spinner("Fetching Cboe chains & volatility..."):
    spx = analyze_chain(fetch_json(URLS["SPX"]), scope=scope)
    qqq = analyze_chain(fetch_json(URLS["QQQ"]), scope="All Exp")
    vix = fetch_json(URLS["VIX"])
    vix1d = fetch_json(URLS["VIX1D"])

if not spx:
    st.warning("No contracts found for 0DTE (market may be closed or weekend). Switch to All Exp.")
    st.stop()

vix_px, vix_chg = float(vix.get("current_price", 0)), float(vix.get("price_change", 0))
v1d_px, v1d_chg = float(vix1d.get("current_price", 0)), float(vix1d.get("price_change", 0))
v_ratio = v1d_px / vix_px if vix_px > 0 else 1.0

# --- CONFLUENCE SCORE ---
score, notes = 0, []
if spx["net_vex"] > 0 and vix_chg < -0.2:
    score += 2
    notes.append("VIX bleeding lower: Positive VEX is forcing dealer buying.")
elif spx["net_vex"] > 0 and vix_chg > 0.2:
    score -= 2
    notes.append("VIX curling up: Positive VEX flips to dealer selling.")
else:
    notes.append("VIX flat: No strong Vanna tailwind yet.")

if v_ratio < 0.95:
    score += 1
    notes.append(f"VIX1D/VIX ({v_ratio:.2f}) < 0.95: Same-day fear is draining.")
elif v_ratio > 1.05:
    score -= 1
    notes.append(f"VIX1D/VIX ({v_ratio:.2f}) > 1.05: Traders paying up for 0DTE protection.")

if spx["charm_es_hr"] > 0:
    score += 1
    notes.append(f"Charm positive (+{spx['charm_es_hr']:.0f} ES/hr): Time decay supports upside drift.")
else:
    score -= 1
    notes.append(f"Charm negative ({spx['charm_es_hr']:.0f} ES/hr): Time decay leans bearish.")

if qqq and qqq["spot"] >= qqq["king"][0]:
    score += 1
    notes.append(f"QQQ ({qqq['spot']:.2f}) holding above its King ({qqq['king'][0]:.0f}): Tech confirming.")
elif qqq:
    score -= 1
    notes.append(f"QQQ ({qqq['spot']:.2f}) below its King ({qqq['king'][0]:.0f}): Tech lagging.")

# --- METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("SPX Spot", f"{spx['spot']:.2f}", f"Zero Γ: {spx['zero_gamma']:.1f}" if spx["zero_gamma"] else "Zero Γ: N/A")
m2.metric("Net GEX", f"${spx['net_gex']:+.2f}B", "Positive (Chop)" if spx["net_gex"] >= 0 else "Negative (Trend)")
m3.metric("Net VEX", f"${spx['net_vex']:+.2f}B", f"VIX {vix_px:.2f} ({vix_chg:+.2f})")
m4.metric("Confluence Score", f"{score:+d} / 5", "Bullish Push" if score >= 2 else ("Bearish / Reject" if score <= -2 else "Mixed / Wait"))

l1, l2, l3, l4 = st.columns(4)
l1.metric("Put Wall", f"{spx['put_wall'][0]:.0f}", f"${spx['put_wall'][1]:+.2f}B")
l2.metric("King Node", f"{spx['king'][0]:.0f}", f"${spx['king'][1]:+.2f}B")
l3.metric("Call Wall", f"{spx['call_wall'][0]:.0f}", f"${spx['call_wall'][1]:+.2f}B")
l4.metric("VIX1D / VIX", f"{v_ratio:.2f}", f"VIX1D {v1d_px:.2f} ({v1d_chg:+.2f})")

st.subheader("Live Playbook Readout")
for n in notes:
    st.markdown(f"- {n}")

# --- STRIKE CHART (±2.5% around spot) ---
near = sorted([k for k in spx["net_k"] if abs(k - spx["spot"]) / spx["spot"] <= 0.025])
vals = [spx["net_k"][k] / 1e6 for k in near]
colors = ["#00c853" if v >= 0 else "#ff5252" for v in vals]

fig = go.Figure(go.Bar(x=vals, y=[str(int(k)) for k in near], orientation="h", marker_color=colors))
fig.update_layout(
    title="Net GEX by Strike ($M) — ±2.5% Around Spot",
    xaxis_title="Net GEX ($ Millions)", yaxis_title="Strike",
    height=520, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20)
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} | Cboe 15-min delayed feed | Confirm final entries with ES CVD & NYSE $TICK on TradingView.")
