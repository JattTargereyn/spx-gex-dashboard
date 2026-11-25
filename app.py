import re
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from datetime import datetime, date, timezone
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

@st.cache_data(ttl=180)
def fetch_cboe(url):
    r = cffi_requests.get(url, headers=HEADERS, impersonate="chrome", timeout=30)
    if r.status_code != 200:
        st.error(f"Cboe returned HTTP {r.status_code} for {url}")
        st.stop()
    raw = r.json()
    return raw["data"], raw.get("timestamp", "Unknown")

@st.cache_data(ttl=120)
def fetch_es_data():
    """Fetches 1m ES=F bars for Est. CVD, Absorption, and 9:30-9:45 AM ET Opening Range."""
    try:
        df = yf.Ticker("ES=F").history(period="2d", interval="1m")
        if df.empty or len(df) < 15:
            return None

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        latest_day = df.index[-1].date()
        day_df = df[df.index.date == latest_day].copy()
        if len(day_df) < 15:
            day_df = df.tail(60).copy()

        rng = (day_df["High"] - day_df["Low"]).replace(0, np.nan)
        buy_ratio = ((day_df["Close"] - day_df["Low"]) / rng).fillna(0.5).clip(0, 1)
        day_df["delta"] = day_df["Volume"] * (2.0 * buy_ratio - 1.0)
        day_df["cvd"] = day_df["delta"].cumsum()

        last_15 = day_df.tail(15)
        cvd_15m = float(last_15["delta"].sum())
        vol_15m = float(last_15["Volume"].sum())
        px_chg_15m = float(last_15["Close"].iloc[-1] - last_15["Close"].iloc[0])
        es_last = float(day_df["Close"].iloc[-1])
        es_time = day_df.index[-1].strftime("%H:%M ET")

        delta_pct = abs(cvd_15m) / max(vol_15m, 1.0)
        if delta_pct > 0.22 and abs(px_chg_15m) < 2.5:
            absorb = "BUYER ABSORPTION (Ceiling)" if cvd_15m > 0 else "SELLER ABSORPTION (Floor)"
        elif cvd_15m > 0 and px_chg_15m > 2.5:
            absorb = "CLEAN BUYING PUSH"
        elif cvd_15m < 0 and px_chg_15m < -2.5:
            absorb = "CLEAN SELLING PUSH"
        else:
            absorb = "BALANCED / CHOP"

        or_bars = day_df.between_time("09:30", "09:44")
        if len(or_bars) >= 10:
            or_high = float(or_bars["High"].max())
            or_low = float(or_bars["Low"].min())
            or_mid = (or_high + or_low) / 2.0
            or_ready = True
        else:
            or_high = float(day_df["High"].max())
            or_low = float(day_df["Low"].min())
            or_mid = (or_high + or_low) / 2.0
            or_ready = False

        return {
            "es_last": es_last, "es_time": es_time,
            "cvd_sess": float(day_df["cvd"].iloc[-1]), "cvd_15m": cvd_15m,
            "px_chg_15m": px_chg_15m, "absorb": absorb,
            "or_ready": or_ready, "or_high": or_high, "or_low": or_low, "or_mid": or_mid,
        }
    except Exception:
        return None

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
    dte0_call_vol, dte0_put_vol = 0.0, 0.0
    dte0_strikes = {}

    for opt in chain["options"]:
        m = OCC_RE.match(opt["option"])
        if not m:
            continue
        oi = opt.get("open_interest") or 0
        iv = opt.get("iv") or 0
        vol = float(opt.get("volume") or 0)
        ymd = m.group(2)
        exp = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        dte = max((exp - today).days, 0)
        strike = int(m.group(4)) / 1000.0
        is_call = m.group(3) == "C"

        if dte == 0 and abs(strike - spot) / spot <= 0.05:
            if is_call:
                dte0_call_vol += vol
            else:
                dte0_put_vol += vol
            rec = dte0_strikes.setdefault(strike, {"c_vol": 0.0, "p_vol": 0.0, "oi": 0.0})
            rec["c_vol" if is_call else "p_vol"] += vol
            rec["oi"] += float(oi)

        if oi <= 0 or iv <= 0.01:
            continue
        if scope == "0DTE" and dte > 0:
            continue
        if abs(strike - spot) / spot > 0.20:
            continue

        strikes.append(strike)
        Ts.append(dte / 365.0)
        ivs.append(float(iv))
        ois.append(float(oi))
        vols.append(vol)
        signs.append(1.0 if is_call else -1.0)

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

    grid = np.linspace(spot * 0.94, spot * 1.06, 51)
    totals = np.array([(sign * bs_greeks(s, K, T, sigma)[0] * OI * 100.0 * (s**2) * 0.01).sum() for s in grid])
    cross = np.where(np.diff(np.sign(totals)) != 0)[0]
    zero_gamma = None
    if len(cross) > 0:
        i = cross[0]
        zero_gamma = grid[i] - totals[i] * (grid[i + 1] - grid[i]) / (totals[i + 1] - totals[i])

    hot_0dte = sorted(
        [(k, v["c_vol"] + v["p_vol"], (v["c_vol"] + v["p_vol"]) / max(v["oi"], 1.0),
          "CALLS" if v["c_vol"] >= v["p_vol"] else "PUTS")
         for k, v in dte0_strikes.items()],
        key=lambda x: x[1], reverse=True
    )[:3]

    return {
        "spot": spot, "net_gex": gex.sum() / 1e9, "net_vex": vex.sum() / 1e9,
        "charm_es_hr": charm_es_hr, "call_wall": (call_wall, call_k[call_wall] / 1e9),
        "put_wall": (put_wall, put_k[put_wall] / 1e9), "king": (king, net_k[king] / 1e9),
        "zero_gamma": zero_gamma, "net_k": net_k, "vol_oi_k": vol_oi_k,
        "dte0_cvol": dte0_call_vol, "dte0_pvol": dte0_put_vol, "hot_0dte": hot_0dte,
    }

def format_age(ts_str):
    try:
        dt = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        mins = int((datetime.now(timezone.utc) - dt).total_seconds() // 60)
        if mins < 60:
            return f"{ts_str} UTC ({mins}m old)"
        return f"{ts_str} UTC ({mins // 60}h {mins % 60}m old — Market Closed/Stale)"
    except Exception:
        return f"{ts_str}"

def build_roadmap(spx, score):
    spot = spx["spot"]
    key_levels = [
        (spx["put_wall"][0], "Put Wall", spx["put_wall"][1]),
        (spx["king"][0], "King Node", spx["king"][1]),
        (spx["call_wall"][0], "Call Wall", spx["call_wall"][1]),
    ]
    if spx["zero_gamma"]:
        key_levels.append((round(spx["zero_gamma"], 1), "Zero Γ Flip", 0.0))

    above = sorted([lvl for lvl in key_levels if lvl[0] > spot + 2], key=lambda x: x[0])
    below = sorted([lvl for lvl in key_levels if lvl[0] < spot - 2], key=lambda x: x[0], reverse=True)

    if above:
        up1 = above[0]
        up_next = f"{above[1][1]} ({above[1][0]:.0f})" if len(above) > 1 else f"Open Air (+{spot*0.01:.0f} pts)"
        voi = spx["vol_oi_k"].get(round(up1[0]), 0)
        if score >= 2 or (up1[2] < 0 and score >= 1) or voi >= 2.0:
            up_status = "LIKELY TO BREAK"
            up_reason = "Vanna/0DTE flow or heavy intraday Vol/OI is pressuring this level."
        elif up1[2] > 0 and score <= 0:
            up_status = "LIKELY TO HOLD (CEILING)"
            up_reason = "Positive dealer gamma should absorb buying unless VIX drops."
        else:
            up_status = "CONTESTED / WATCH TAPE"
            up_reason = "Wait for clean 5m ES close above + falling VIX."
    else:
        up1, up_next, up_status, up_reason = None, "Blue Sky", "ABOVE ALL WALLS", "Squeeze territory."

    if below:
        dn1 = below[0]
        dn_next = f"{below[1][1]} ({below[1][0]:.0f})" if len(below) > 1 else f"Open Air (-{spot*0.01:.0f} pts)"
        if score <= -2:
            dn_status = "AT RISK OF BREAKING"
            dn_reason = "Rising VIX + negative dealer flow is pressuring this floor."
        elif score >= 1:
            dn_status = "LIKELY TO HOLD (FLOOR)"
            dn_reason = "Calm/falling vol; dips into this level should find buyers."
        else:
            dn_status = "CONTESTED / WATCH TAPE"
            dn_reason = "Watch if VIX spikes on a test; a break accelerates selling."
    else:
        dn1, dn_next, dn_status, dn_reason = None, "Air Pocket", "BELOW ALL WALLS", "High acceleration risk."

    return (up1, up_next, up_status, up_reason), (dn1, dn_next, dn_status, dn_reason)

# --- HEADER & FRESHNESS ---
st.title("SPX GEX Command Center")
c1, c2 = st.columns([2, 1])
scope = c1.radio("Expiration Scope", ["All Exp", "0DTE"], horizontal=True)
if c2.button("Refresh Live Data"):
    st.cache_data.clear()

with st.spinner("Loading Cboe chains, 0DTE flow, VIX & ES Opening Range..."):
    spx_raw, spx_ts = fetch_cboe(URLS["SPX"])
    qqq_raw, _ = fetch_cboe(URLS["QQQ"])
    vix, _ = fetch_cboe(URLS["VIX"])
    vix1d, _ = fetch_cboe(URLS["VIX1D"])
    spx = analyze_chain(spx_raw, scope=scope)
    qqq = analyze_chain(qqq_raw, scope="All Exp")
    es = fetch_es_data()

st.info(f"**Data Timestamp:** Cboe Chain `{format_age(spx_ts)}` | ES Futures: `{es['es_time'] if es else 'Unavailable'}` | Page Refreshed: `{datetime.now().strftime('%H:%M:%S')}`")

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
    notes.append("⚪ **VIX Flat:** No strong Vanna push yet.")

if v_ratio < 0.95:
    score += 1
    notes.append(f"🟢 **VIX1D/VIX ({v_ratio:.2f}):** Same-day fear is draining.")
elif v_ratio > 1.05:
    score -= 1
    notes.append(f"🔴 **VIX1D/VIX ({v_ratio:.2f}):** Traders paying up for 0DTE protection.")

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
    st.success(f"**BULLISH BIAS ({score:+d}/5) — {regime_txt}** | Tailwinds active. Favor breakouts & dip-buys.")
elif score <= -2:
    st.error(f"**BEARISH BIAS ({score:+d}/5) — {regime_txt}** | Selling pressure active. Rallies into walls risk rejecting.")
else:
    st.warning(f"**MIXED / NEUTRAL ({score:+d}/5) — {regime_txt}** | Wait for a level test + ES CVD/VIX confirmation.")

# --- 9:30-9:45 AM ET OPENING RANGE & BIAS TRACKER ---
if es:
    with st.container(border=True):
        if es["or_ready"]:
            st.markdown("### ⏱️ 9:30–9:45 AM ET Opening Range (ES Futures)")
            if es["es_last"] > es["or_high"]:
                if score >= 1 and vix_chg <= 0:
                    or_badge = "🟢 ABOVE OR HIGH — CONFIRMED BULLISH BREAKOUT"
                    or_tip = "ES broke above the 15m Opening Range with VIX/Vanna support. Look for pullbacks to OR High to target the next upside GEX wall."
                else:
                    or_badge = "⚠️ ABOVE OR HIGH — FAKEOUT / TRAP WARNING"
                    or_tip = "ES poked above OR High, but VIX/Vanna is NOT confirming. Watch for a snap-back inside the range."
            elif es["es_last"] < es["or_low"]:
                if score <= -1 and vix_chg >= 0:
                    or_badge = "🔴 BELOW OR LOW — CONFIRMED BEARISH BREAKDOWN"
                    or_tip = "ES broke below the 15m Opening Range with rising VIX/selling pressure. Look for bounces to OR Low to target the next downside wall."
                else:
                    or_badge = "⚠️ BELOW OR LOW — SWEEP / REVERSAL WARNING"
                    or_tip = "ES broke OR Low, but VIX is not expanding. Watch for a liquidity sweep and reclaim back inside the range."
            else:
                or_badge = "🟡 INSIDE OPENING RANGE — CHOP / WAIT"
                or_tip = "Price is still inside the 9:30–9:45 AM range. Wait for a clean 5m close outside OR High or OR Low."

            o1, o2, o3, o4 = st.columns(4)
            o1.metric("OR High", f"{es['or_high']:.2f}", f"{es['es_last'] - es['or_high']:+.2f} pts")
            o2.metric("OR Mid", f"{es['or_mid']:.2f}")
            o3.metric("OR Low", f"{es['or_low']:.2f}", f"{es['es_last'] - es['or_low']:+.2f} pts")
            o4.metric("ES Last", f"{es['es_last']:.2f}")
            st.markdown(f"**Status:** `{or_badge}`")
            st.caption(or_tip)
        else:
            st.markdown("### 🌙 Pre-Market / Overnight Range (Waiting for 9:45 AM ET)")
            o1, o2, o3 = st.columns(3)
            o1.metric("Overnight High", f"{es['or_high']:.2f}")
            o2.metric("ES Last", f"{es['es_last']:.2f}")
            o3.metric("Overnight Low", f"{es['or_low']:.2f}")
            st.caption("The 9:30–9:45 AM ET Opening Range will lock in automatically at 9:45 AM ET.")

# --- 0DTE FLOW & ES CVD / ABSORPTION PANELS ---
f1, f2 = st.columns(2)
with f1:
    with st.container(border=True):
        st.markdown("### ⚡ 0DTE Flow & Wall Risk")
        tot_0dte = spx["dte0_cvol"] + spx["dte0_pvol"]
        if tot_0dte > 0:
            c_pct = (spx["dte0_cvol"] / tot_0dte) * 100
            st.markdown(f"**0DTE Split:** `{c_pct:.0f}% Calls` ({spx['dte0_cvol']:,.0f}) vs `{100-c_pct:.0f}% Puts` ({spx['dte0_pvol']:,.0f})")
            for k, v, voi, side in spx["hot_0dte"]:
                flag = " ⚠️ **WALL OVERRIDE**" if voi >= 2.0 else ""
                st.markdown(f"- Strike `{k:.0f}` ({side}): `{v:,.0f}` vol | Vol/OI `{voi:.1f}x`{flag}")
        else:
            st.caption("No 0DTE volume recorded yet (pre-market or closed).")

with f2:
    with st.container(border=True):
        st.markdown("### 📊 Est. ES CVD & Absorption (1m Bars)")
        if es:
            st.markdown(f"**ES Last:** `{es['es_last']:.2f}` | **15m Move:** `{es['px_chg_15m']:+.2f} pts`")
            st.markdown(f"**Est. Session CVD:** `{es['cvd_sess']:+,.0f}` | **15m CVD:** `{es['cvd_15m']:+,.0f}`")
            st.markdown(f"**Tape Read:** `{es['absorb']}`")
            st.caption("Estimated from 1m bar close-in-range; confirm exact tick absorption on TradingView.")
        else:
            st.caption("ES=F intraday bars temporarily unavailable.")

# --- WHAT'S NEXT IN LINE ---
(up1, up_next, up_status, up_reason), (dn1, dn_next, dn_status, dn_reason) = build_roadmap(spx, score)
r1, r2 = st.columns(2)
with r1:
    with st.container(border=True):
        st.markdown("### ⬆️ Next Upside Test")
        if up1:
            st.markdown(f"**{up1[1]} at `{up1[0]:.0f}`** (+{up1[0] - spx['spot']:.1f} pts)")
            st.markdown(f"**Verdict:** `{up_status}`")
            st.caption(up_reason)
            st.markdown(f"**Next if it breaks:** `{up_next}`")
        else:
            st.markdown(f"**{up_status}** — {up_reason}")

with r2:
    with st.container(border=True):
        st.markdown("### ⬇️ Next Downside Test")
        if dn1:
            st.markdown(f"**{dn1[1]} at `{dn1[0]:.0f}`** (-{spx['spot'] - dn1[0]:.1f} pts)")
            st.markdown(f"**Verdict:** `{dn_status}`")
            st.caption(dn_reason)
            st.markdown(f"**Next if it breaks:** `{dn_next}`")
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

fig = go.Figure(go.Bar(x=vals, y=near, orientation="h", marker_color=colors))
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
    xaxis_title="Net GEX ($M) — Green = Pin/Ceiling/Support | Red = Volatile/Accelerator",
    yaxis_title="Strike", height=540, template="plotly_dark",
    margin=dict(l=15, r=15, t=45, b=20), showlegend=False
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Why this score? (Factor Breakdown)", expanded=True):
    for n in notes:
        st.markdown(n)

# --- 1-PASTE TRADINGVIEW STRING (WALLS + BASIS + NEXT TARGETS) ---
key_lvls = [spx["put_wall"][0], spx["king"][0], spx["call_wall"][0]]
if spx["zero_gamma"]:
    key_lvls.append(round(spx["zero_gamma"], 1))

above_lvls = sorted([x for x in key_lvls if x > spx["spot"] + 2])
below_lvls = sorted([x for x in key_lvls if x < spx["spot"] - 2], reverse=True)

if len(above_lvls) >= 2:
    up_tgt = above_lvls[1]
elif len(above_lvls) == 1:
    up_tgt = round((above_lvls[0] + spx["spot"] * 0.0075) / 5.0) * 5.0
else:
    up_tgt = round((spx["spot"] * 1.01) / 5.0) * 5.0

if len(below_lvls) >= 2:
    dn_tgt = below_lvls[1]
elif len(below_lvls) == 1:
    dn_tgt = round((below_lvls[0] - spx["spot"] * 0.0075) / 5.0) * 5.0
else:
    dn_tgt = round((spx["spot"] * 0.99) / 5.0) * 5.0

zg_num = f"{spx['zero_gamma']:.1f}" if spx["zero_gamma"] else "0"
basis_val = round(es["es_last"] - spx["spot"], 2) if es else 0.0
tv_string = f"{spx['put_wall'][0]:.0f},{spx['king'][0]:.0f},{spx['call_wall'][0]:.0f},{zg_num},{basis_val:.2f},{up_tgt:.0f},{dn_tgt:.0f}"

st.subheader("1-Paste String for TradingView Indicator")
st.text_input("Tap, Select All, and Copy:", value=tv_string, label_visibility="collapsed")
st.caption(f"Format: Put, King, Call, Zero Γ, Basis (`{basis_val:+.2f}`), Up Target (`{up_tgt:.0f}`), Dn Target (`{dn_tgt:.0f}`)")

