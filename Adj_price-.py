#!/usr/bin/env python3
"""
Fetch Adj Close for:
- btc          -> BTC-USD
- eurusd       -> EURUSD=X
- market_proxy -> URTH
- sp500        -> ^GSPC
- eurostoxx50  -> ^STOXX50E

Output CSV columns (exact order):
Date, btc, eurusd, market_proxy, sp500, eurostoxx50

Usage:
  pip install pandas yfinance
  python fetch_adjclose_panel.py --start 2015-01-01 --end 2025-12-31 --out prices_adjclose.csv
"""

import argparse
from datetime import date
import pandas as pd
import yfinance as yf

# yfinance tickers for each output column
TICKERS = {
    "btc": "BTC-USD",
    "eurusd": "EURUSD=X",
    "market_proxy": "URTH",
    "sp500": "^GSPC",
    "eurostoxx50": "^STOXX50E",
}
COLUMN_ORDER = ["Date", "btc", "eurusd", "market_proxy", "sp500", "eurostoxx50"]


def fetch_adj_close(start: str, end: str) -> pd.DataFrame:
    # Download all tickers at once, then select Adj Close only
    raw = yf.download(
        list(TICKERS.values()),
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        threads=False,
        group_by="column",
    )
    if raw.empty:
        raise SystemExit("No data returned. Check tickers or date range.")

    # If yfinance returns a MultiIndex, select the 'Adj Close' level
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise SystemExit("Could not find 'Adj Close' in downloaded data.")
        df = raw["Adj Close"].copy()
    else:
        # Single ticker edge-case: ensure we have a DataFrame
        df = raw[["Adj Close"]].copy()
        df.columns = list(TICKERS.values())  # will be renamed below if single ticker

    # Remove any timezone info and ensure pure date index
    try:
        df.index = df.index.tz_localize(None)
    except Exception:
        pass
    df.index = pd.to_datetime(df.index)

    # Rename yfinance tickers -> our desired column names
    rename_map = {v: k for k, v in TICKERS.items()}
    df = df.rename(columns=rename_map)

    # Make sure all expected columns exist (if Yahoo lacks a series, you'll see it here)
    for col in TICKERS.keys():
        if col not in df.columns:
            df[col] = pd.NA

    # Sort by date, reset index, format Date
    df = df.sort_index()
    out = df.reset_index().rename(columns={"Date": "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")

    # Reorder and keep only requested columns
    out = out[["Date"] + list(TICKERS.keys())]
    # Final column order exactly as requested
    out.columns = COLUMN_ORDER
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01", help="YYYY-MM-DD (default: 2015-01-01)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="prices_adjclose.csv", help="Output CSV filename")
    args = parser.parse_args()

    end = args.end or date.today().strftime("%Y-%m-%d")
    panel = fetch_adj_close(args.start, end)

    # Light sanity checks
    expected = set(COLUMN_ORDER)
    got = set(panel.columns)
    if got != expected:
        raise SystemExit(f"Unexpected columns.\nExpected: {expected}\nGot: {got}")

    panel.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {panel.shape}")
    print("Head:")
    print(panel.head().to_string(index=False))


if __name__ == "__main__":
    main()
