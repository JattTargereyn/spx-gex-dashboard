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

def bs_greeks(S, K, T, sigma, r=0.04):
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

def build_roadmap(spx, score):
    """Finds what is directly above and below spot, and what comes next if it breaks."""
    spot = spx["spot"]
    key_levels = [
        (spx["put_wall"][0], "Put Wall", spx["put_wall"][1]),
        (spx["king"][0], "King Node", spx["king"][1]),
        (spx["call_wall"][0], "Call Wall", spx["call_wall"][1]),
    ]
    if spx["zero_gamma"]:
        key_levels.append((round(spx["zero_gamma"], 1), "Zero Γ Flip", 0.0))

    # Deduplicate close levels and sort
    above = sorted([lvl for lvl in key_levels if lvl[0] > spot + 2], key=lambda x: x[0])
    below = sorted([lvl for lvl in key_levels if lvl[0] < spot - 2], key=lambda x: x[0], reverse=True)

    # Upside verdict
    if above:
        up1 = above[0]
        up_next = f"{above[1][1]} ({above[1][0]:.0f})" if len(above) > 1 else f"Open Air (+{spot*0.01:.0f} pts)"
        voi = spx["vol_oi_k"].get(round(up1[0]), 0)
        if score >= 2 or (up1[2] < 0 and score >= 1) or voi >= 2.0:
            up_status = "LIKELY TO BREAK"
            up_reason = "VIX/Vanna tailwind or heavy intraday flow is pushing into this level."
        elif up1[2] > 0 and score <= 0:
            up_status = "LIKELY TO HOLD (CEILING)"
            up_reason = "Positive dealer gamma here should absorb buying unless VIX drops sharply."
        else:
            up_status = "CONTESTED / WATCH TAPE"
            up_reason = "Wait for a clean 5m ES close above + falling VIX before chasing."
    else:
        up1, up_next, up_status, up_reason = None, "Blue Sky", "ABOVE ALL WALLS", "Spot is above the Call Wall — squeeze territory."

    # Downside verdict
    if below:
        dn1 = below[0]
        dn_next = f"{below[1][1]} ({below[1][0]:.0f})" if len(below) > 1 else f"Open Air (-{spot*0.01:.0f} pts)"
        if score <= -2:
            dn_status = "AT RISK OF BREAKING"
            dn_reason = "Rising VIX + negative dealer flow is pressuring this floor."
        elif score >= 1:
            dn_status = "LIKELY TO HOLD (FLOOR)"
            dn_reason = "Volatility is calm/falling; dips into this level should find buyers."
        else:
            dn_status = "CONTESTED / WATCH TAPE"
            dn_reason = "Watch if VIX spikes on a test; a break accelerates selling."
    else:
        dn1, dn_next, dn_status, dn_reason = None, "Air Pocket", "BELOW ALL WALLS", "Spot is below the Put Wall — high crash/acceleration risk."

    return (up1, up_next, up_status, up_reason), (dn1, dn_next, dn_status, dn_reason)

# --- HEADER ---
st.title("SPX GEX Command Center")
c1, c2 = st.columns([2, 1])
scope = c1.radio("Expiration Scope", ["All Exp", "0DTE"], horizontal=True)
if c2.button("Refresh Live Data"):
    st.cache_data.clear()

with st.spinner("Loading Cboe options chains & volatility..."):
    spx = analyze_chain(fetch_json(URLS["SPX"]), scope=scope)
    qqq = analyze_chain(fetch_json(URLS["QQQ"]), scope="All Exp")
    vix = fetch_json(URLS["VIX"])
    vix1d = fetch_json(URLS["VIX1D"])

if not spx:
    st.warning("No 0DTE contracts active right now (market closed or weekend). Switch to All Exp.")
    st.stop()

vix_px, vix_chg = float(vix.get("current_price", 0)), float(vix.get("price_change", 0))
v1d_px, v1d_chg = float(vix1d.get("current_price", 0)), float(vix1d.get("price_change", 0))
v_ratio = v1d_px / vix_px if vix_px > 0 else 1.0

# --- CONFLUENCE ENGINE ---
score, notes = 0, []
if spx["net_vex"] > 0 and vix_chg < -0.2:
    score += 2
    notes.append("🟢 **VIX Falling:** Positive VEX is forcing dealers to buy futures.")
elif spx["net_vex"] > 0 and vix_chg > 0.2:
    score -= 2
    notes.append("🔴 **VIX Rising:** Positive VEX flips to dealer selling pressure.")
else:
    notes.append("⚪ **VIX Flat:** No strong Vanna push either way yet.")

if v_ratio < 0.95:
    score += 1
    notes.append(f"🟢 **VIX1D/VIX ({v_ratio:.2f}):** Same-day fear is draining (bullish).")
elif v_ratio > 1.05:
    score -= 1
    notes.append(f"🔴 **VIX1D/VIX ({v_ratio:.2f}):** Traders paying up for 0DTE protection (bearish).")

if spx["charm_es_hr"] > 0:
    score += 1
    notes.append(f"🟢 **Charm (+{spx['charm_es_hr']:.0f} ES/hr):** Time decay supports upside drift.")
else:
    score -= 1
    notes.append(f"🔴 **Charm ({spx['charm_es_hr']:.0f} ES/hr):** Time decay leans bearish.")

if qqq and qqq["spot"] >= qqq["king"][0]:
    score += 1
    notes.append(f"🟢 **QQQ ({qqq['spot']:.2f}):** Above its King ({qqq['king'][0]:.0f}) — tech confirming.")
elif qqq:
    score -= 1
    notes.append(f"🔴 **QQQ ({qqq['spot']:.2f}):** Below its King ({qqq['king'][0]:.0f}) — tech lagging.")

# --- TOP VERDICT BANNER ---
regime_txt = "POSITIVE GAMMA (Choppy / Mean-Reverting)" if spx["net_gex"] >= 0 else "NEGATIVE GAMMA (Fast / Trending)"
if score >= 2:
    st.success(f"**BULLISH BIAS ({score:+d}/5) — {regime_txt}** | Tailwinds are active. Favor breakouts above resistance and dip-buys.")
elif score <= -2:
    st.error(f"**BEARISH BIAS ({score:+d}/5) — {regime_txt}** | Selling pressure active. Rallies into walls are likely to reject.")
else:
    st.warning(f"**MIXED / NEUTRAL ({score:+d}/5) — {regime_txt}** | No clear vol tailwind yet. Let price test a key level and confirm with TradingView ES CVD & $TICK.")

# --- WHAT'S NEXT IN LINE (UPSIDE vs DOWNSIDE) ---
(up1, up_next, up_status, up_reason), (dn1, dn_next, dn_status, dn_reason) = build_roadmap(spx, score)

r1, r2 = st.columns(2)
with r1:
    with st.container(border=True):
        st.markdown("### ⬆️ Next Upside Test")
        if up1:
            dist = up1[0] - spx["spot"]
            st.markdown(f"**{up1[1]} at `{up1[0]:.0f}`** (+{dist:.1f} pts away)")
            st.markdown(f"**Verdict:** `{up_status}`")
            st.caption(up_reason)
            st.markdown(f"**If it breaks → Next in line:** `{up_next}`")
        else:
            st.markdown(f"**{up_status}** — {up_reason}")

with r2:
    with st.container(border=True):
        st.markdown("### ⬇️ Next Downside Test")
        if dn1:
            dist = spx["spot"] - dn1[0]
            st.markdown(f"**{dn1[1]} at `{dn1[0]:.0f}`** (-{dist:.1f} pts away)")
            st.markdown(f"**Verdict:** `{dn_status}`")
            st.caption(dn_reason)
            st.markdown(f"**If it breaks → Next in line:** `{dn_next}`")
        else:
            st.markdown(f"**{dn_status}** — {dn_reason}")

# --- KEY METRICS ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("SPX Spot", f"{spx['spot']:.2f}", f"Zero Γ: {spx['zero_gamma']:.1f}" if spx["zero_gamma"] else "Zero Γ: Out of range")
m2.metric("Net GEX", f"${spx['net_gex']:+.2f}B", "Chop" if spx["net_gex"] >= 0 else "Trend")
m3.metric("Net VEX", f"${spx['net_vex']:+.2f}B", f"VIX {vix_px:.2f} ({vix_chg:+.2f})")
m4.metric("VIX1D / VIX", f"{v_ratio:.2f}", f"VIX1D {v1d_px:.2f} ({v1d_chg:+.2f})")

# --- ANNOTATED GAMMA PROFILE CHART ---
near = sorted([k for k in spx["net_k"] if abs(k - spx["spot"]) / spx["spot"] <= 0.025])
vals = [spx["net_k"][k] / 1e6 for k in near]
colors = ["#00e676" if v >= 0 else "#ff5252" for v in vals]

fig = go.Figure(go.Bar(x=vals, y=near, orientation="h", marker_color=colors, name="Net GEX ($M)"))

# Draw Key Horizontal Levels
fig.add_hline(y=spx["spot"], line_dash="dash", line_color="white", line_width=2,
              annotation_text=f"SPOT {spx['spot']:.1f}", annotation_position="top right")
if min(near) <= spx["call_wall"][0] <= max(near):
    fig.add_hline(y=spx["call_wall"][0], line_dash="dot", line_color="#00e5ff",
                  annotation_text=f"CALL WALL {spx['call_wall'][0]:.0f}", annotation_position="top left")
if min(near) <= spx["put_wall"][0] <= max(near):
    fig.add_hline(y=spx["put_wall"][0], line_dash="dot", line_color="#ff4081",
                  annotation_text=f"PUT WALL {spx['put_wall'][0]:.0f}", annotation_position="bottom left")
if min(near) <= spx["king"][0] <= max(near):
    fig.add_hline(y=spx["king"][0], line_dash="solid", line_color="#ffd740", line_width=1.5,
                  annotation_text=f"KING {spx['king'][0]:.0f}", annotation_position="bottom right")
if spx["zero_gamma"] and min(near) <= spx["zero_gamma"] <= max(near):
    fig.add_hline(y=spx["zero_gamma"], line_dash="dashdot", line_color="#ff9100",
                  annotation_text=f"ZERO Γ {spx['zero_gamma']:.1f}", annotation_position="top left")

fig.update_layout(
    title=f"SPX {scope} Gamma Map (±2.5% Around Spot)",
    xaxis_title="Net GEX ($ Millions) — Green = Pin/Support/Ceiling | Red = Volatile/Accelerator",
    yaxis_title="Strike Price",
    height=560, template="plotly_dark", margin=dict(l=15, r=15, t=45, b=20),
    showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

# --- FACTOR BREAKDOWN & TRADINGVIEW COPY BOX ---
with st.expander("Why this score? (Factor Breakdown)", expanded=True):
    for n in notes:
        st.markdown(n)

zg_val = f"{spx['zero_gamma']:.1f}" if spx["zero_gamma"] else "N/A"
st.subheader("Quick-Copy Levels for TradingView")
st.code(
    f"Put Wall: {spx['put_wall'][0]:.0f} | King: {spx['king'][0]:.0f} | Call Wall: {spx['call_wall'][0]:.0f} | Zero Gamma: {zg_val}",
    language="text"
)
st.caption(f"Refreshed: {datetime.now().strftime('%H:%M:%S')} | Cboe 15-min delayed data | Confirm final entries with live ES CVD & NYSE $TICK on TradingView.")
