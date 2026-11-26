import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="SPX & ES GEX Dashboard", layout="wide")
NY = ZoneInfo("America/New_York")

CBOE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cboe.com/",
    "Origin": "https://www.cboe.com",
}


# --- 1. DATA FETCHERS ---
@st.cache_data(ttl=60)
def fetch_cboe_chain(ticker: str):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
    r = requests.get(url, headers=CBOE_HEADERS, timeout=20)
    if r.status_code != 200:
        st.error(f"Cboe API returned HTTP {r.status_code} for {ticker}. Try refreshing in a minute.")
        st.stop()
    data = r.json()["data"]
    spot = float(data["current_price"])
    today_yymmdd = datetime.now(NY).strftime("%y%m%d")

    rows = []
    call_0dte_vol, put_0dte_vol = 0, 0
    for opt in data["options"]:
        sym = opt["option"]
        opt_type = sym[-9]
        strike = int(sym[-8:]) / 1000.0
        exp_str = sym[-15:-9]
        oi = float(opt.get("open_interest") or 0)
        vol = float(opt.get("volume") or 0)
        gamma = float(opt.get("gamma") or 0)
        vega = float(opt.get("vega") or 0)
        theta = float(opt.get("theta") or 0)

        if exp_str == today_yymmdd:
            if opt_type == "C":
                call_0dte_vol += vol
            else:
                put_0dte_vol += vol

        if oi <= 0 or gamma == 0:
            continue

        sign = 1.0 if opt_type == "C" else -1.0
        gex = sign * gamma * oi * 100 * (spot ** 2) * 0.01
        vex = sign * vega * oi * 100
        charm = -sign * theta * oi * 100
        rows.append((strike, gex, vex, charm, exp_str == today_yymmdd))

    df = pd.DataFrame(rows, columns=["strike", "gex", "vex", "charm", "is_0dte"])
    df = df[(df["strike"] >= spot * 0.90) & (df["strike"] <= spot * 1.10)]
    agg = df.groupby("strike", as_index=False)[["gex", "vex", "charm"]].sum().sort_values("strike")

    put_wall = float(agg.loc[agg["gex"].idxmin(), "strike"])
    call_wall = float(agg.loc[agg["gex"].idxmax(), "strike"])
    king = float(agg.loc[agg["gex"].abs().idxmax(), "strike"])
    net_gex = float(agg["gex"].sum())
    net_vex = float(agg["vex"].sum())
    net_charm = float(agg["charm"].sum())

    agg["cum_gex"] = agg["gex"].cumsum()
    flips = agg[(agg["cum_gex"].shift(1) * agg["cum_gex"]) <= 0]
    zero_g = float(flips.loc[(flips["strike"] - spot).abs().idxmin(), "strike"]) if not flips.empty else 0.0


    above = agg[(agg["strike"] > spot + 5) & (agg["gex"] > 0)]
    below = agg[(agg["strike"] < spot - 5) & (agg["gex"] < 0)]
    up_tgt = float(above.loc[above["gex"].idxmax(), "strike"]) if not above.empty else call_wall
    dn_tgt = float(below.loc[below["gex"].idxmin(), "strike"]) if not below.empty else put_wall

    return {
        "spot": spot,
        "put_wall": put_wall,
        "king": king,
        "call_wall": call_wall,
        "zero_g": zero_g,
        "net_gex": net_gex,
        "net_vex": net_vex,
        "net_charm": net_charm,
        "up_tgt": up_tgt,
        "dn_tgt": dn_tgt,
        "call_0dte_vol": int(call_0dte_vol),
        "put_0dte_vol": int(put_0dte_vol),
        "df": agg,
    }


@st.cache_data(ttl=60)
def fetch_market_data():
    es_1m = yf.Ticker("ES=F").history(period="1d", interval="1m")
    spx_1m = yf.Ticker("^GSPC").history(period="1d", interval="1m")
    es_5m = yf.Ticker("ES=F").history(period="1d", interval="5m")
    spx_5m = yf.Ticker("^GSPC").history(period="1d", interval="5m")
    vix = yf.Ticker("^VIX").history(period="2d", interval="5m")
    vix1d = yf.Ticker("^VIX1D").history(period="1d", interval="5m")

    basis = 0.0
    if not es_1m.empty and not spx_1m.empty:
        common = es_1m.index.intersection(spx_1m.index)
        if len(common) > 0:
            basis = float(es_1m.loc[common[-1], "Close"] - spx_1m.loc[common[-1], "Close"])
        else:
            basis = float(es_1m["Close"].iloc[-1] - spx_1m["Close"].iloc[-1])

    vix_now = float(vix["Close"].iloc[-1]) if not vix.empty else 0.0
    vix_chg = float(vix["Close"].iloc[-1] - vix["Open"].iloc[0]) if not vix.empty else 0.0
    vix1d_now = float(vix1d["Close"].iloc[-1]) if not vix1d.empty else vix_now

    return es_5m, spx_5m, round(basis, 2), round(vix_now, 2), round(vix_chg, 2), round(vix1d_now, 2)


# --- 2. LOAD DATA & COMPUTE MACRO SCORE ---
now_str = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S ET")
st.title("SPX & ES Live GEX Dashboard")
st.caption(f"Last refreshed: {now_str}")

if st.button("Refresh Live Data"):
    st.cache_data.clear()

spx = fetch_cboe_chain("_SPX")
qqq = fetch_cboe_chain("QQQ")
es_5m, spx_5m, basis_val, vix_now, vix_chg, vix1d_now = fetch_market_data()

# Macro Confluence Score (-5 to +5)
score = 0
if spx["net_vex"] > 0 and vix_chg < -0.20:
    score += 2
elif vix_chg > 0.20:
    score -= 2

fear_ratio = (vix1d_now / vix_now) if vix_now > 0 else 1.0
if fear_ratio < 0.95:
    score += 1
elif fear_ratio > 1.05:
    score -= 1

score += 1 if spx["net_charm"] > 0 else -1
score += 1 if qqq["spot"] >= qqq["king"] else -1

gamma_flag = 1 if spx["net_gex"] >= 0 else -1
gamma_txt = "POSITIVE GAMMA (Chop / Mean-Revert)" if gamma_flag == 1 else "NEGATIVE GAMMA (Fast / Trending)"

if score >= 2:
    st.success(f"BULLISH BIAS ({score:+d}/5) — {gamma_txt}")
elif score <= -2:
    st.error(f"BEARISH BIAS ({score:+d}/5) — {gamma_txt}")
else:
    st.warning(f"NEUTRAL / MIXED ({score:+d}/5) — {gamma_txt}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("SPX Spot", f"{spx['spot']:.2f}", f"Basis {basis_val:+.2f}")
c2.metric("Put Wall (ES)", f"{spx['put_wall'] + basis_val:.1f}", f"SPX {spx['put_wall']:.0f}")
c3.metric("King Node (ES)", f"{spx['king'] + basis_val:.1f}", f"SPX {spx['king']:.0f}")
c4.metric("Call Wall (ES)", f"{spx['call_wall'] + basis_val:.1f}", f"SPX {spx['call_wall']:.0f}")
c5.metric("VIX / VIX1D", f"{vix_now:.2f} ({vix_chg:+.2f})", f"Ratio {fear_ratio:.2f}")

# --- 3. SPX GAMMA EXPOSURE (GEX) PROFILE BARS ---
st.subheader("SPX Net Gamma Exposure (GEX) Profile")
gex_df = spx["df"]
gex_view = gex_df[(gex_df["strike"] >= spx["spot"] - 250) & (gex_df["strike"] <= spx["spot"] + 250)].copy()

bar_colors = ["#00e676" if val >= 0 else "#ff4081" for val in gex_view["gex"]]
gex_fig = go.Figure(data=[go.Bar(
    x=gex_view["gex"] / 1e6,
    y=gex_view["strike"],
    orientation="h",
    marker_color=bar_colors,
    name="Net GEX ($M)"
)])
gex_fig.add_hline(
    y=spx["spot"],
    line_dash="dash",
    line_color="#ffd740",
    annotation_text=f"SPX Spot {spx['spot']:.1f}",
    annotation_position="top right"
)
if spx["zero_g"] > 0:
    gex_fig.add_hline(
        y=spx["zero_g"],
        line_dash="dot",
        line_color="#ff9100",
        annotation_text=f"Zero Γ {spx['zero_g']:.1f}",
        annotation_position="bottom right"
    )
gex_fig.update_layout(
    template="plotly_dark",
    xaxis_title="Net GEX ($ Millions per 1% Move)",
    yaxis_title="SPX Strike",
    height=520,
    margin=dict(l=10, r=10, t=20, b=10),
    showlegend=False
)
st.plotly_chart(gex_fig, use_container_width=True)

# --- 4. 9:30–9:45 AM ET OPENING RANGE (ES FUTURES) ---
st.subheader("9:30–9:45 AM ET Opening Range (ES Futures)")
if not es_5m.empty:
    if es_5m.index.tz is None:
        es_5m.index = es_5m.index.tz_localize("UTC").tz_convert(NY)
    else:
        es_5m.index = es_5m.index.tz_convert(NY)

    es_rth = es_5m.between_time("09:30", "16:15")
    or_bars = es_5m.between_time("09:30", "09:44")

    if not or_bars.empty and not es_rth.empty:
        or_high = float(or_bars["High"].max())
        or_low = float(or_bars["Low"].min())
        or_mid = (or_high + or_low) / 2.0
        es_last = float(es_rth["Close"].iloc[-1])

        o1, o2, o3, o4 = st.columns(4)
        o1.metric("OR High", f"{or_high:.2f}", f"{es_last - or_high:+.2f} pts")
        o2.metric("OR Mid", f"{or_mid:.2f}")
        o3.metric("OR Low", f"{or_low:.2f}", f"{es_last - or_low:+.2f} pts")
        o4.metric("ES Last", f"{es_last:.2f}")

        if es_last > or_high:
            if score >= 0:
                st.success("Status: ABOVE OR HIGH — ALIGNED BREAKOUT")
            elif (es_last - or_high) > 30:
                st.info("Status: EXTENDED COUNTER-MACRO SQUEEZE — Trail stops; do not blindly fade.")
            else:
                st.warning("Status: ABOVE OR HIGH — FAKEOUT / TRAP WARNING (VIX/Vanna not confirming)")
        elif es_last < or_low:
            if score <= 0:
                st.error("Status: BELOW OR LOW — ALIGNED BREAKDOWN")
            elif (or_low - es_last) > 30:
                st.info("Status: EXTENDED COUNTER-MACRO FLUSH — Watch for Put Wall reclaim.")
            else:
                st.warning("Status: BELOW OR LOW — BEAR TRAP WARNING (Macro not confirming)")
        else:
            st.caption("Status: INSIDE OPENING RANGE — Wait for range break.")
    else:
        st.caption("Waiting for 9:30–9:45 AM ET Opening Range bars.")

# --- 5. INTRADAY 5M SPX CANDLES + HISTORICAL GEX WALL SHIFTS ---
st.subheader("Intraday 5m SPX Candles + Historical GEX Wall Shifts")
if not spx_5m.empty:
    if spx_5m.index.tz is None:
        spx_5m.index = spx_5m.index.tz_localize("UTC").tz_convert(NY)
    else:
        spx_5m.index = spx_5m.index.tz_convert(NY)

    fig = go.Figure(data=[go.Candlestick(
        x=spx_5m.index,
        open=spx_5m["Open"],
        high=spx_5m["High"],
        low=spx_5m["Low"],
        close=spx_5m["Close"],
        name="SPX 5m"
    )])

    drew_history = False
    if os.path.exists("gex_history.csv"):
        hist = pd.read_csv("gex_history.csv")
        hist["timestamp_et"] = pd.to_datetime(hist["timestamp_et"]).dt.tz_localize(NY)
        today_str = spx_5m.index[-1].strftime("%Y-%m-%d")
        hist_today = hist[hist["timestamp_et"].dt.strftime("%Y-%m-%d") == today_str]

        if not hist_today.empty:
            drew_history = True
            for col_name, label, color in [
                ("call_wall", "Call Wall", "#00e5ff"),
                ("king", "King Node", "#ffd740"),
                ("put_wall", "Put Wall", "#ff4081"),
                ("zero_gamma", "Zero Γ", "#ff9100"),
            ]:
                vals = hist_today[col_name].replace(0, np.nan)
                fig.add_trace(go.Scatter(
                    x=hist_today["timestamp_et"],
                    y=vals,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2, shape="hv"),
                    marker=dict(size=4)
                ))

    if not drew_history:
        st.caption("Showing current live SPX GEX levels (stepped 15m wall shifts will appear automatically during RTH).")
        for val, label, color in [
            (spx["call_wall"], f"Call Wall {spx['call_wall']:.0f}", "#00e5ff"),
            (spx["king"],      f"King {spx['king']:.0f}",           "#ffd740"),
            (spx["put_wall"],  f"Put Wall {spx['put_wall']:.0f}",   "#ff4081"),
            (spx["zero_g"],    f"Zero Γ {spx['zero_g']:.1f}",       "#ff9100"),
        ]:
            if val > 0:
                fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 6. 0DTE FLOW SPLIT ---
total_0dte = spx["call_0dte_vol"] + spx["put_0dte_vol"]
if total_0dte > 0:
    c_pct = spx["call_0dte_vol"] / total_0dte * 100
    p_pct = spx["put_0dte_vol"] / total_0dte * 100
    st.subheader("0DTE Flow Split")
    st.write(f"**{c_pct:.0f}% Calls** ({spx['call_0dte_vol']:,}) vs **{p_pct:.0f}% Puts** ({spx['put_0dte_vol']:,})")

# --- 7. 1-PASTE STRING FOR TRADINGVIEW ---
tv_string = f"{spx['put_wall']:.0f},{spx['king']:.0f},{spx['call_wall']:.0f},{spx['zero_g']:.1f},{basis_val:.2f},{spx['up_tgt']:.0f},{spx['dn_tgt']:.0f},{score},{gamma_flag}"

st.subheader("1-Paste String for TradingView Live Engine")
st.text_input("Tap, Select All, and Copy:", value=tv_string, label_visibility="collapsed")
st.caption(f"9-Value Format: Put, King, Call, Zero Γ, Basis ({basis_val:+.2f}), UpTgt, DnTgt, MacroScore ({score:+d}), Gamma ({gamma_flag:+d})")
