import os
import csv
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
now_ny = datetime.now(NY)

# Only log during RTH (09:30 - 16:15 ET) so DST changes never log junk
if not (930 <= now_ny.hour * 100 + now_ny.minute <= 1615):
    print(f"Outside RTH ({now_ny.strftime('%H:%M ET')}). Skipping.")
    raise SystemExit(0)

def get_es_spx_basis():
    es = yf.Ticker("ES=F").history(period="1d", interval="1m")
    spx = yf.Ticker("^GSPC").history(period="1d", interval="1m")
    if es.empty or spx.empty:
        return 0.0
    common = es.index.intersection(spx.index)
    if len(common) > 0:
        return round(float(es.loc[common[-1], "Close"] - spx.loc[common[-1], "Close"]), 2)
    return round(float(es["Close"].iloc[-1] - spx["Close"].iloc[-1]), 2)

def fetch_cboe_spx():
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    spot = float(data["current_price"])

    rows = []
    for opt in data["options"]:
        sym = opt["option"]
        # Parse strike & type from OCC symbol (e.g., SPX261125C07000000)
        opt_type = sym[-9]
        strike = int(sym[-8:]) / 1000.0
        oi = float(opt.get("open_interest") or 0)
        gamma = float(opt.get("gamma") or 0)
        if oi <= 0 or gamma == 0:
            continue
        gex = gamma * oi * 100 * (spot ** 2) * 0.01
        if opt_type == "P":
            gex = -gex
        rows.append((strike, gex))

    df = pd.DataFrame(rows, columns=["strike", "gex"]).groupby("strike", as_index=False)["gex"].sum()
    df = df[(df["strike"] >= spot * 0.90) & (df["strike"] <= spot * 1.10)].sort_values("strike")

    put_wall = float(df.loc[df["gex"].idxmin(), "strike"])
    call_wall = float(df.loc[df["gex"].idxmax(), "strike"])
    king = float(df.loc[df["gex"].abs().idxmax(), "strike"])
    net_gex = float(df["gex"].sum())

    # Zero Gamma (cumulative GEX sign flip)
    df["cum_gex"] = df["gex"].cumsum()
    flips = df[(df["cum_gex"].shift(1) * df["cum_gex"]) <= 0]
    zero_g = float(flips.iloc[0]["strike"]) if not flips.empty else 0.0

    return spot, put_wall, king, call_wall, zero_g, net_gex

spot, put_wall, king, call_wall, zero_g, net_gex = fetch_cboe_spx()
basis = get_es_spx_basis()

csv_file = "gex_history.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp_et", "spx_spot", "basis", "put_wall", "king", "call_wall", "zero_gamma", "net_gex"])
    writer.writerow([
        now_ny.strftime("%Y-%m-%d %H:%M"),
        round(spot, 2),
        basis,
        round(put_wall, 1),
        round(king, 1),
        round(call_wall, 1),
        round(zero_g, 1),
        round(net_gex, 0),
    ])

print(f"Logged {now_ny.strftime('%H:%M ET')}: Put={put_wall}, King={king}, Call={call_wall}, ZeroG={zero_g}, Basis={basis}")
