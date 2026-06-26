#!/usr/bin/env python3
"""
Compute log returns from inputs/prices_adjclose.csv and write wide + long outputs.

Input (CSV in inputs/):
Date, btc, eurusd, market_proxy, sp500, eurostoxx50

Outputs (created in outputs/):
- returns_log_wide.csv  -> Date + *_ret_log columns
- returns_log.csv       -> long format: date, asset, ret_log, mkt_ret_log

Usage:
  pip install pandas numpy
  python calc_log_returns.py --in inputs/prices_adjclose.csv --outdir outputs
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


EXPECTED_COLS = ["Date", "btc", "eurusd", "market_proxy", "sp500", "eurostoxx50"]


def main():
    parser = argparse.ArgumentParser(description="Compute log returns from Adj Close panel.")
    parser.add_argument("--in", dest="inp", default="inputs/prices_adjclose.csv",
                        help="Path to input prices CSV (default: inputs/prices_adjclose.csv)")
    parser.add_argument("--outdir", default="outputs",
                        help="Directory to write outputs (default: outputs)")
    args = parser.parse_args()

    inp_path = Path(args.inp)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load & sanity checks
    df = pd.read_csv(inp_path, parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    got = list(df.columns)
    if got != EXPECTED_COLS:
        raise SystemExit(f"Unexpected columns.\nExpected exactly: {EXPECTED_COLS}\nGot: {got}")

    price_cols = ["btc", "eurusd", "market_proxy", "sp500", "eurostoxx50"]

    # Ensure floats and guard against non-positive prices
    for c in price_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] <= 0, c] = np.nan  # log undefined for non-positive prices

    # ---- Compute log returns: ln(P_t) - ln(P_{t-1}) per series
    log_prices = np.log(df[price_cols])
    log_returns = log_prices.diff()

    # ---- Wide output
    wide = pd.concat([df["Date"], log_returns], axis=1)
    wide["Date"] = pd.to_datetime(wide["Date"]).dt.strftime("%Y-%m-%d")
    wide.columns = ["Date"] + [f"{c}_ret_log" for c in price_cols]

    wide_path = outdir / "returns_log_wide.csv"
    wide.to_csv(wide_path, index=False)

    # ---- Long output (with market proxy duplicated as mkt_ret_log)
    long = wide.melt(id_vars="Date", var_name="asset_tmp", value_name="ret_log")
    long["asset"] = long["asset_tmp"].str.replace("_ret_log", "", regex=False)
    long = long.drop(columns=["asset_tmp"])
    long = long.rename(columns={"Date": "date"})
    # Attach market proxy return for the same date to each row (useful for the market model later)
    mkt = wide[["Date", "market_proxy_ret_log"]].rename(
        columns={"Date": "date", "market_proxy_ret_log": "mkt_ret_log"}
    )
    long = long.merge(mkt, on="date", how="left")

    # Optional: enforce column order & friendly types
    long = long[["date", "asset", "ret_log", "mkt_ret_log"]]
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")

    long_path = outdir / "returns_log.csv"
    long.to_csv(long_path, index=False)

    # ---- Console summary
    print(f"Wrote: {wide_path}  (shape={wide.shape})")
    print(f"Wrote: {long_path}  (shape={long.shape})")
    print("\nWide head:")
    print(wide.head().to_string(index=False))
    print("\nLong head:")
    print(long.head().to_string(index=False))


if __name__ == "__main__":
    main()
