import os
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="SPX & ES GEX Dashboard", layout="wide")
NY = ZoneInfo("America/New_York")
BASIS_FILE = "last_basis.txt"

CBOE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cboe.com/",
    "Origin": "https://www.cboe.com",
}


def years_to_expiry(exp_date) -> float:
    """Exact time to 4:00 PM ET expiry in years, floored at 15 minutes."""
    exp_dt = datetime.combine(exp_date, dtime(16, 0), tzinfo=NY)
    secs = (exp_dt - datetime.now(NY)).total_seconds()
    return max(secs, 15 * 60) / (365 * 24 * 3600)


def bs_vanna_charm(spot: float, strike: float, t_years: float, iv: float):
    """True Black-Scholes Vanna (dDelta/dVol) and Charm (-dDelta/dT, r=0)."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + 0.5 * (iv ** 2) * t_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t
    pdf_d1 = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vanna = -pdf_d1 * (d2 / iv)
    charm = pdf_d1 * (d2 / (2.0 * t_years))
    return vanna, charm


def compute_gex_metrics(rows, spot):
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["strike", "gex", "vanna", "charm"])
    df = df[(df["strike"] >= spot * 0.90) & (df["strike"] <= spot * 1.10)]
    if df.empty:
        return None
    agg = df.groupby("strike", as_index=False)[["gex", "vanna", "charm"]].sum().sort_values("strike")

    put_wall = float(agg.loc[agg["gex"].idxmin(), "strike"])
    call_wall = float(agg.loc[agg["gex"].idxmax(), "strike"])
    king = float(agg.loc[agg["gex"].abs().idxmax(), "strike"])
    net_gex = float(agg["gex"].sum())
    net_vanna = float(agg["vanna"].sum())
    net_charm = float(agg["charm"].sum())

    agg["cum_gex"] = agg["gex"].cumsum()
    flips = agg[(agg["cum_gex"].shift(1) * agg["cum_gex"]) <= 0]
    zero_g = float(flips.loc[(flips["strike"] - spot).abs().idxmin(), "strike"]) if not flips.empty else 0.0

    above = agg[(agg["strike"] > spot + 5) & (agg["gex"] > 0)]
    below = agg[(agg["strike"] < spot - 5) & (agg["gex"] < 0)]
    up_tgt = float(above.loc[above["gex"].idxmax(), "strike"]) if not above.empty else call_wall
    dn_tgt = float(below.loc[below["gex"].idxmin(), "strike"]) if not below.empty else put_wall

    return {
        "put_wall": put_wall,
        "king": king,
        "call_wall": call_wall,
        "zero_g": zero_g,
        "net_gex": net_gex,
        "net_vanna": net_vanna,
        "net_charm": net_charm,
        "up_tgt": up_tgt,
        "dn_tgt": dn_tgt,
        "df": agg,
    }


def compute_iv_skew(puts_iv, calls_iv, spot):
    if not puts_iv or not calls_iv:
        return None, None, None
    has_deltas = any(d != 0 for _, d, _ in puts_iv) and any(d != 0 for _, d, _ in calls_iv)
    if has_deltas:
        put_iv_25 = min(puts_iv, key=lambda x: abs(x[1] - (-0.25)))[2] * 100.0
        call_iv_25 = min(calls_iv, key=lambda x: abs(x[1] - 0.25))[2] * 100.0
    else:
        put_iv_25 = min(puts_iv, key=lambda x: abs(x[0] - spot * 0.9925))[2] * 100.0
        call_iv_25 = min(calls_iv, key=lambda x: abs(x[0] - spot * 1.0075))[2] * 100.0
    return round(put_iv_25, 2), round(call_iv_25, 2), round(put_iv_25 - call_iv_25, 2)


def log_snapshot(d, every_min: int = 15):
    now = datetime.now(NY)
    if now.weekday() >= 5 or not (570 <= now.hour * 60 + now.minute <= 960):
        return
    path = "gex_history.csv"
    if os.path.exists(path):
        last = pd.read_csv(path, usecols=["timestamp_et"]).tail(1)
        if not last.empty:
            age = (now.replace(tzinfo=None) - pd.to_datetime(last.iloc[0, 0])).total_seconds()
            if age < every_min * 60:
                return
    row = {
        "timestamp_et": now.strftime("%Y-%m-%d %H:%M:%S"),
        "call_wall": d["call_wall"],
        "king": d["king"],
        "put_wall": d["put_wall"],
        "zero_gamma": d["zero_g"],
    }
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


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
    today_date = datetime.now(NY).date()
    today_yymmdd = today_date.strftime("%y%m%d")

    all_rows, wk_rows = [], []
    call_0dte_vol, put_0dte_vol = 0, 0
    calls_0dte_iv, puts_0dte_iv = [], []
    wk_exp_dates = set()
    raw_wk_iv = []

    for opt in data["options"]:
        sym = opt["option"]
        opt_type = sym[-9]
        strike = int(sym[-8:]) / 1000.0
        exp_str = sym[-15:-9]
        try:
            exp_date = datetime.strptime(exp_str, "%y%m%d").date()
            dte = (exp_date - today_date).days
        except ValueError:
            continue

        oi = float(opt.get("open_interest") or 0)
        vol = float(opt.get("volume") or 0)
        gamma = float(opt.get("gamma") or 0)
        iv = float(opt.get("iv") or 0)
        delta = float(opt.get("delta") or 0)

        if exp_str == today_yymmdd:
            if opt_type == "C":
                call_0dte_vol += vol
                if iv > 0 and strike > spot:
                    calls_0dte_iv.append((strike, delta, iv))
            else:
                put_0dte_vol += vol
                if iv > 0 and strike < spot:
                    puts_0dte_iv.append((strike, delta, iv))

        if 1 <= dte <= 7:
            wk_exp_dates.add(exp_date)
            if iv > 0:
                raw_wk_iv.append((exp_date, opt_type, strike, delta, iv))

        if oi <= 0 or gamma == 0:
            continue

        sign = 1.0 if opt_type == "C" else -1.0
        t_years = years_to_expiry(exp_date)
        vanna_unit, charm_unit = bs_vanna_charm(spot, strike, t_years, iv)

        gex = sign * gamma * oi * 100 * (spot ** 2) * 0.01
        vanna_exp = sign * vanna_unit * oi * 100 * spot * 0.01
        charm_exp = sign * charm_unit * oi * 100 * spot / 365.0

        all_rows.append((strike, gex, vanna_exp, charm_exp))
        if 1 <= dte <= 7:
            wk_rows.append((strike, gex, vanna_exp, charm_exp))

    target_wk_exp = None
    if wk_exp_dates:
        fridays = sorted([d for d in wk_exp_dates if d.weekday() == 4])
        target_wk_exp = fridays[0] if fridays else max(wk_exp_dates)

    calls_wk_iv = [(s, d, i) for e, t, s, d, i in raw_wk_iv if e == target_wk_exp and t == "C" and s > spot]
    puts_wk_iv = [(s, d, i) for e, t, s, d, i in raw_wk_iv if e == target_wk_exp and t == "P" and s < spot]

    put_iv_25, call_iv_25, iv_skew = compute_iv_skew(puts_0dte_iv, calls_0dte_iv, spot)
    wk_put_iv_25, wk_call_iv_25, wk_iv_skew = compute_iv_skew(puts_wk_iv, calls_wk_iv, spot)

    intra = compute_gex_metrics(all_rows, spot)
    if intra is None:
        st.error(f"Cboe returned an empty or invalid options chain for {ticker}.")
        st.stop()
    weekly = (compute_gex_metrics(wk_rows, spot) if wk_rows else None) or intra

    return {
        "spot": spot,
        **intra,
        "call_0dte_vol": int(call_0dte_vol),
        "put_0dte_vol": int(put_0dte_vol),
        "put_iv_25": put_iv_25,
        "call_iv_25": call_iv_25,
        "iv_skew": iv_skew,
        "weekly": {
            **weekly,
            "target_exp": target_wk_exp.strftime("%Y-%m-%d") if target_wk_exp else "N/A",
            "put_iv_25": wk_put_iv_25,
            "call_iv_25": wk_call_iv_25,
            "iv_skew": wk_iv_skew,
        },
    }


def _to_ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df.index = df.index.tz_localize("UTC").tz_convert(NY) if df.index.tz is None else df.index.tz_convert(NY)
    return df


@st.cache_data(ttl=60)
def fetch_market_data():
    es_1m = _to_ny(yf.Ticker("ES=F").history(period="1d", interval="1m"))
    spx_1m = _to_ny(yf.Ticker("^GSPC").history(period="1d", interval="1m"))
    es_5m = _to_ny(yf.Ticker("ES=F").history(period="1d", interval="5m"))
    spx_5m = _to_ny(yf.Ticker("^GSPC").history(period="1d", interval="5m"))
    vix = _to_ny(yf.Ticker("^VIX").history(period="2d", interval="5m"))
    vix1d = _to_ny(yf.Ticker("^VIX1D").history(period="1d", interval="5m"))
    vix3m = _to_ny(yf.Ticker("^VIX3M").history(period="2d", interval="5m"))

    # RTH-only matched timestamp basis with last-known RTH fallback
    basis, basis_live, basis_cached = 0.0, False, False
    if not es_1m.empty and not spx_1m.empty:
        common = es_1m.index.intersection(spx_1m.index)
        if len(common) > 0:
            basis = float(es_1m.loc[common[-1], "Close"] - spx_1m.loc[common[-1], "Close"])
            basis_live = True

    if basis_live:
        with open(BASIS_FILE, "w") as f:
            f.write(f"{basis}")
    elif os.path.exists(BASIS_FILE):
        with open(BASIS_FILE) as f:
            basis = float(f.read().strip())
            basis_cached = True

    # Today-only VIX change
    vix_now, vix_chg = 0.0, 0.0
    if not vix.empty:
        vix_now = float(vix["Close"].iloc[-1])
        today_ny = datetime.now(NY).date()
        vix_today = vix[vix.index.date == today_ny]
        if not vix_today.empty:
            vix_chg = float(vix_today["Close"].iloc[-1] - vix_today["Open"].iloc[0])

    vix1d_now = float(vix1d["Close"].iloc[-1]) if not vix1d.empty else vix_now
    vix3m_now = float(vix3m["Close"].iloc[-1]) if not vix3m.empty else vix_now

    return es_5m, spx_5m, round(basis, 2), basis_live, basis_cached, round(vix_now, 2), round(vix_chg, 2), round(vix1d_now, 2), round(vix3m_now, 2)


# --- 2. LOAD DATA ---
now_str = datetime.now(NY).strftime("%Y-%m-%d %H:%M:%S ET")
st.title("SPX & ES Live GEX Dashboard")
st.caption(f"Last refreshed: {now_str} | Note: Cboe OI updates overnight (Morning Map)")

if st.button("Refresh Live Data"):
    st.cache_data.clear()

spx = fetch_cboe_chain("_SPX")
qqq = fetch_cboe_chain("QQQ")
es_5m, spx_5m, basis_val, basis_live, basis_cached, vix_now, vix_chg, vix1d_now, vix3m_now = fetch_market_data()
log_snapshot(spx)

if not basis_live:
    if basis_cached:
        st.info(f"Outside RTH: Using last saved RTH basis ({basis_val:+.2f}) until 9:30 AM ET.")
    else:
        st.warning("Outside RTH: No saved RTH basis yet (set to 0.00). Use Manual Basis Override in TradingView until 9:30 AM ET.")

tab1, tab2 = st.tabs(["Total GEX Morning Map (All Expiries)", "1–7 DTE Weekly Map"])

# ==============================================================================
# TAB 1: TOTAL GEX MORNING MAP (ALL EXPIRIES)
# ==============================================================================
with tab1:
    score = 0
    if spx["net_vanna"] > 0 and vix_chg < -0.20:
        score += 2
    elif vix_chg > 0.20:
        score -= 2

    fear_ratio = (vix1d_now / vix_now) if vix_now > 0 else 1.0
    if fear_ratio < 0.95:
        score += 1
    elif fear_ratio > 1.05:
        score -= 1

    # net_charm < 0 -> dealer delta decaying -> hedge buy-back -> bullish
    score += 1 if spx["net_charm"] < 0 else -1
    score += 1 if qqq["spot"] >= qqq["king"] else -1

    gamma_flag = 1 if spx["net_gex"] >= 0 else -1
    gamma_txt = "POSITIVE GAMMA (Chop / Mean-Revert)" if gamma_flag == 1 else "NEGATIVE GAMMA (Fast / Trending)"

    if score >= 2:
        st.success(f"BULLISH BIAS ({score:+d}/5) — {gamma_txt}")
    elif score <= -2:
        st.error(f"BEARISH BIAS ({score:+d}/5) — {gamma_txt}")
    else:
        st.warning(f"NEUTRAL / MIXED ({score:+d}/5) — {gamma_txt}")

    basis_delta = f"Basis {basis_val:+.2f}" if basis_live else (f"Last RTH {basis_val:+.2f}" if basis_cached else "Basis Stale")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SPX Spot", f"{spx['spot']:.2f}", basis_delta)
    c2.metric("Put Wall (ES)", f"{spx['put_wall'] + basis_val:.1f}", f"SPX {spx['put_wall']:.0f}")
    c3.metric("King Node (ES)", f"{spx['king'] + basis_val:.1f}", f"SPX {spx['king']:.0f}")
    c4.metric("Call Wall (ES)", f"{spx['call_wall'] + basis_val:.1f}", f"SPX {spx['call_wall']:.0f}")
    c5.metric("VIX / VIX1D", f"{vix_now:.2f} ({vix_chg:+.2f})", f"Ratio {fear_ratio:.2f}")

    st.subheader("SPX Net Gamma Exposure (GEX) Profile")
    gex_df = spx["df"]
    gex_view = gex_df[(gex_df["strike"] >= spx["spot"] - 250) & (gex_df["strike"] <= spx["spot"] + 250)].copy()

    bar_colors = ["#00e676" if val >= 0 else "#ff4081" for val in gex_view["gex"]]
    gex_fig = go.Figure(data=[go.Bar(
        x=gex_view["gex"] / 1e6, y=gex_view["strike"], orientation="h",
        marker_color=bar_colors, name="Net GEX ($M)"
    )])
    gex_fig.add_hline(y=spx["spot"], line_dash="dash", line_color="#ffd740",
                      annotation_text=f"SPX Spot {spx['spot']:.1f}", annotation_position="top right")
    if spx["zero_g"] > 0:
        gex_fig.add_hline(y=spx["zero_g"], line_dash="dot", line_color="#ff9100",
                          annotation_text=f"Zero Γ {spx['zero_g']:.1f}", annotation_position="bottom right")
    gex_fig.update_layout(template="plotly_dark", xaxis_title="Net GEX ($ Millions per 1% Move)",
                          yaxis_title="SPX Strike", height=520, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(gex_fig, use_container_width=True)

    st.subheader("9:30–9:45 AM ET Opening Range (ES Futures)")
    if not es_5m.empty:
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

    st.subheader("Intraday 5m SPX Candles + Historical GEX Wall Shifts")
    if not spx_5m.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=spx_5m.index, open=spx_5m["Open"], high=spx_5m["High"], low=spx_5m["Low"], close=spx_5m["Close"], name="SPX 5m"
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
                    ("call_wall", "Call Wall", "#00e5ff"), ("king", "King Node", "#ffd740"),
                    ("put_wall", "Put Wall", "#ff4081"), ("zero_gamma", "Zero Γ", "#ff9100"),
                ]:
                    vals = hist_today[col_name].replace(0, np.nan)
                    fig.add_trace(go.Scatter(x=hist_today["timestamp_et"], y=vals, mode="lines+markers",
                                             name=label, line=dict(color=color, width=2, shape="hv"), marker=dict(size=4)))

        if not drew_history:
            st.caption("Showing current live SPX GEX levels (stepped 15m wall shifts log automatically during RTH).")
            for val, label, color in [
                (spx["call_wall"], f"Call Wall {spx['call_wall']:.0f}", "#00e5ff"),
                (spx["king"],      f"King {spx['king']:.0f}",           "#ffd740"),
                (spx["put_wall"],  f"Put Wall {spx['put_wall']:.0f}",   "#ff4081"),
                (spx["zero_g"],    f"Zero Γ {spx['zero_g']:.1f}",       "#ff9100"),
            ]:
                if val > 0:
                    fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top left")

        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=500,
                          margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("0DTE Flow Split & IV Skew")
    f1, f2 = st.columns(2)
    with f1:
        total_0dte = spx["call_0dte_vol"] + spx["put_0dte_vol"]
        if total_0dte > 0:
            c_pct = spx["call_0dte_vol"] / total_0dte * 100
            p_pct = spx["put_0dte_vol"] / total_0dte * 100
            st.metric("0DTE Volume Split", f"{c_pct:.0f}% C / {p_pct:.0f}% P", f"C: {spx['call_0dte_vol']:,} | P: {spx['put_0dte_vol']:,}")
        else:
            st.metric("0DTE Volume Split", "N/A", "No active 0DTE volume")
    with f2:
        if spx["iv_skew"] is not None:
            skew_val = spx["iv_skew"]
            skew_read = "Steep (Put Bid / Breakdown Risk)" if skew_val >= 4.0 else ("Flat/Inverted (Call Bid / Squeeze Risk)" if skew_val <= 1.5 else "Normal Baseline (Balanced)")
            st.metric("0DTE 25Δ IV Skew (Put − Call)", f"{skew_val:+.2f}%", f"Put IV {spx['put_iv_25']:.1f}% vs Call IV {spx['call_iv_25']:.1f}%")
            st.caption(f"Read: **{skew_read}**")
        else:
            st.metric("0DTE 25Δ IV Skew (Put − Call)", "N/A", "Active during RTH 0DTE session")

    tv_string = f"{spx['put_wall']:.0f},{spx['king']:.0f},{spx['call_wall']:.0f},{spx['zero_g']:.1f},{basis_val:.2f},{spx['up_tgt']:.0f},{spx['dn_tgt']:.0f},{score},{gamma_flag}"
    st.subheader("1-Paste String for TradingView Intraday Engine")
    st.text_input("Tap, Select All, and Copy (Intraday):", value=tv_string, label_visibility="collapsed", key="tv_0dte")
    st.caption(f"9-Value Format: Put, King, Call, Zero Γ, Basis ({basis_val:+.2f}), UpTgt, DnTgt, MacroScore ({score:+d}), Gamma ({gamma_flag:+d})")

# ==============================================================================
# TAB 2: 1–7 DTE WEEKLY MAP (EX-0DTE)
# ==============================================================================
with tab2:
    wk = spx["weekly"]
    qqq_wk = qqq["weekly"]
    term_ratio = (vix_now / vix3m_now) if vix3m_now > 0 else 1.0

    wk_score = 0
    if term_ratio < 0.92:
        wk_score += 2
    elif term_ratio > 1.00:
        wk_score -= 2

    wk_score += 1 if wk["net_vanna"] > 0 else -1
    wk_score += 1 if qqq["spot"] >= qqq_wk["king"] else -1

    wk_gamma_flag = 1 if wk["net_gex"] >= 0 else -1
    wk_gamma_txt = "POSITIVE WEEKLY Γ (Pin / Range)" if wk_gamma_flag == 1 else "NEGATIVE WEEKLY Γ (Multi-Day Trend)"

    if wk_score >= 2:
        st.success(f"WEEKLY BULLISH BIAS ({wk_score:+d}/4) — {wk_gamma_txt}")
    elif wk_score <= -2:
        st.error(f"WEEKLY BEARISH BIAS ({wk_score:+d}/4) — {wk_gamma_txt}")
    else:
        st.warning(f"WEEKLY NEUTRAL / MIXED ({wk_score:+d}/4) — {wk_gamma_txt}")

    w1, w2, w3, w4, w5 = st.columns(5)
    w1.metric("Weekly Put Wall (ES)", f"{wk['put_wall'] + basis_val:.1f}", f"SPX {wk['put_wall']:.0f}")
    w2.metric("Weekly King (ES)", f"{wk['king'] + basis_val:.1f}", f"SPX {wk['king']:.0f}")
    w3.metric("Weekly Call Wall (ES)", f"{wk['call_wall'] + basis_val:.1f}", f"SPX {wk['call_wall']:.0f}")
    w4.metric("VIX / VIX3M", f"{term_ratio:.2f}", "Contango" if term_ratio < 0.92 else ("Backwardation" if term_ratio > 1.0 else "Flat"))
    if wk["iv_skew"] is not None:
        w5.metric(f"Weekly 25Δ Skew ({wk['target_exp']})", f"{wk['iv_skew']:+.2f}%", f"P {wk['put_iv_25']:.1f}% | C {wk['call_iv_25']:.1f}%")
    else:
        w5.metric("Weekly 25Δ Skew", "N/A", wk["target_exp"])

    st.subheader("SPX Weekly Net GEX Profile (1–7 DTE, Ex-0DTE)")
    wk_df = wk["df"]
    wk_view = wk_df[(wk_df["strike"] >= spx["spot"] - 300) & (wk_df["strike"] <= spx["spot"] + 300)].copy()

    wk_colors = ["#00e676" if val >= 0 else "#ff4081" for val in wk_view["gex"]]
    wk_fig = go.Figure(data=[go.Bar(
        x=wk_view["gex"] / 1e6, y=wk_view["strike"], orientation="h",
        marker_color=wk_colors, name="Weekly Net GEX ($M)"
    )])
    wk_fig.add_hline(y=spx["spot"], line_dash="dash", line_color="#ffd740",
                     annotation_text=f"SPX Spot {spx['spot']:.1f}", annotation_position="top right")
    if wk["zero_g"] > 0:
        wk_fig.add_hline(y=wk["zero_g"], line_dash="dot", line_color="#ff9100",
                         annotation_text=f"Weekly Zero Γ {wk['zero_g']:.1f}", annotation_position="bottom right")
    wk_fig.update_layout(template="plotly_dark", xaxis_title="Weekly Net GEX ($ Millions per 1% Move)",
                         yaxis_title="SPX Strike", height=520, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(wk_fig, use_container_width=True)

    wk_tv_string = f"{wk['put_wall']:.0f},{wk['king']:.0f},{wk['call_wall']:.0f},{wk['zero_g']:.1f},{basis_val:.2f},{wk['up_tgt']:.0f},{wk['dn_tgt']:.0f},{wk_score},{wk_gamma_flag}"
    st.subheader("1-Paste String for TradingView Weekly Swing Engine (15m / 1H)")
    st.text_input("Tap, Select All, and Copy (Weekly):", value=wk_tv_string, label_visibility="collapsed", key="tv_weekly")
    st.caption(f"9-Value Format: WkPut, WkKing, WkCall, WkZeroΓ, Basis ({basis_val:+.2f}), UpTgt, DnTgt, WkScore ({wk_score:+d}), WkGamma ({wk_gamma_flag:+d})")
