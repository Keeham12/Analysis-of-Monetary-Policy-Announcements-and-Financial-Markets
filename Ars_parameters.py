#!/usr/bin/env python3
"""
Compute abnormal returns (AR) at tau ∈ {-1, 0, +1} for each event × asset,
using a market-model estimated on [-250, -30] TRADING DAYS (intersection with URTH).

Inputs (defaults; override via --events, --returns, --outdir):
- inputs/events_clean_fixed.csv  (cols expected at minimum: event_id, date, bank, stance)
- outputs/returns_log.csv        (long format: date, asset, ret_log, mkt_ret_log)

Outputs (written to --outdir, default: outputs/):
- event_ar.csv     : event_id, bank, stance, asset, market, tau, date_tau, AR, R_i, R_m, has_sigma, T_i
- est_params.csv   : event_id, bank, stance, asset, market, date_tau0, alpha_hat, beta_hat, sigma_hat,
                     T_i, Rm_bar, Sm2, has_sigma

Notes:
- tau grid is per-asset trading days: tau=0 is the FIRST tradable day ON/AFTER the event date for that asset.
- Estimation window uses ONLY days where BOTH asset and URTH returns are present.
- AR at a given tau is produced only if BOTH `ret_log` and `mkt_ret_log` exist on that tau date.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Map assets to "market" buckets (edit as you like)
ASSET_MARKET_MAP = {
    "btc": "crypto",
    "eurusd": "fx",
    "sp500": "equity",
    "eurostoxx50": "equity",
    # "market_proxy": "proxy"  # excluded from AR by default
}

# Assets to EXCLUDE from AR computation (e.g., the proxy itself)
EXCLUDE_ASSETS = {"market_proxy"}

# Minimum obs for Patell/BMP eligibility (you can tweak)
MIN_T_FOR_SIGMA = 150

def read_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names (lowercase)
    df.columns = [c.strip().lower() for c in df.columns]
    # Expected core columns
    # accept either 'date' or 'event_date'
    if "date" not in df.columns and "event_date" in df.columns:
        df = df.rename(columns={"event_date": "date"})
    needed = {"event_id", "date"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"Events file missing required columns: {missing}")
    # Optional columns
    for opt in ["bank", "stance"]:
        if opt not in df.columns:
            df[opt] = pd.NA

    # Parse dates (date only)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["event_id", "date", "bank", "stance"]].copy()

def read_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    # Sanity columns
    needed = {"date", "asset", "ret_log", "mkt_ret_log"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"returns_log.csv must have columns {needed}, got {set(df.columns)}")
    # Clean
    df = df.drop_duplicates(subset=["asset", "date"]).sort_values(["asset", "date"]).reset_index(drop=True)
    return df

def trading_tau_dates(asset_df: pd.DataFrame, event_date: pd.Timestamp) -> Dict[int, Optional[pd.Timestamp]]:
    """
    Build tau dates on the ASSET's trading-day calendar (asset_df has all asset dates, not intersection-filtered).
    tau=0: first asset date >= event_date
    tau=-1: previous asset trading date if exists
    tau=+1: next asset trading date if exists
    Returns dict { -1: date_or_None, 0: date_or_None, +1: date_or_None }
    """
    dates = asset_df["date"].values
    # find first index with date >= event_date
    idx0 = None
    # vectorized search: dates is numpy datetime64[ns]; event_date -> np.datetime64
    evt = np.datetime64(pd.to_datetime(event_date))
    # np.searchsorted assumes sorted ascending
    pos = np.searchsorted(dates, evt, side="left")
    if pos < len(dates):
        idx0 = pos
    else:
        # no date on/after event_date for this asset
        return {-1: None, 0: None, +1: None}

    tau0 = pd.to_datetime(dates[idx0])
    tau_m1 = pd.to_datetime(dates[idx0 - 1]) if idx0 - 1 >= 0 else None
    tau_p1 = pd.to_datetime(dates[idx0 + 1]) if idx0 + 1 < len(dates) else None
    return {-1: tau_m1, 0: tau0, +1: tau_p1}

def select_estimation_window(clean_df: pd.DataFrame, tau0_date: pd.Timestamp) -> Tuple[pd.DataFrame, int, bool, Optional[int]]:
    """
    From 'clean_df' (rows where both ret_log and mkt_ret_log are non-NaN),
    pick the [-250, -30] TRADING-DAY slice relative to the first clean date >= tau0_date.
    Returns (est_df, T_i, has_sigma, idx0_clean) where idx0_clean is the index of first clean >= tau0_date.
    """
    if clean_df.empty:
        return pd.DataFrame(), 0, False, None

    # first clean index >= tau0_date
    idx0 = np.searchsorted(clean_df["date"].values, np.datetime64(tau0_date), side="left")
    if idx0 >= len(clean_df):
        return pd.DataFrame(), 0, False, None

    start = idx0 - 250
    end_excl = idx0 - 29  # upper exclusive so last included is idx0-30
    if start < 0 or end_excl <= start:
        return pd.DataFrame(), 0, False, idx0

    est = clean_df.iloc[start:end_excl].copy()
    T_i = len(est)
    has_sigma = T_i >= MIN_T_FOR_SIGMA
    return est, T_i, has_sigma, idx0

def ols_market_model(est: pd.DataFrame) -> Tuple[float, float, float, float, float]:
    """
    Simple OLS: y = alpha + beta * x
    Returns (alpha_hat, beta_hat, sigma_hat, Rm_bar, Sm2)
    where sigma_hat uses ddof=2 (T-2) and Sm2 = sum (x - Rm_bar)^2
    """
    y = est["ret_log"].to_numpy()
    x = est["mkt_ret_log"].to_numpy()
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    cov_xy = float(np.sum((x - x_mean) * (y - y_mean)))
    var_x = float(np.sum((x - x_mean) ** 2))
    if var_x == 0.0:
        beta = 0.0
    else:
        beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    resid = y - (alpha + beta * x)
    T = len(y)
    sigma = float(np.sqrt(np.sum(resid**2) / max(T - 2, 1)))  # ddof=2; guard tiny T
    Sm2 = var_x  # sum of squares about mean
    return alpha, beta, sigma, x_mean, Sm2

def main():
    ap = argparse.ArgumentParser(description="Compute event ARs at tau -1/0/+1 with market-model estimation on trading days [-250,-30].")
    ap.add_argument("--events", default="inputs/events_clean_fixed.csv", help="Path to events CSV (default: inputs/events_clean_fixed.csv)")
    ap.add_argument("--returns", default="outputs/returns_log.csv", help="Path to long returns CSV (default: outputs/returns_log.csv)")
    ap.add_argument("--outdir", default="outputs", help="Where to write event_ar.csv and est_params.csv (default: outputs)")
    ap.add_argument("--include-proxy", action="store_true", help="Include market_proxy as an asset (off by default)")
    args = ap.parse_args()

    events_path = Path(args.events)
    returns_path = Path(args.returns)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    events = read_events(events_path)
    rets = read_returns(returns_path)

    # Split returns per asset and also keep "clean" intersection with proxy
    asset_groups: Dict[str, pd.DataFrame] = {}
    clean_groups: Dict[str, pd.DataFrame] = {}

    all_assets: List[str] = sorted(rets["asset"].unique().tolist())
    assets_for_ar = [a for a in all_assets if (a not in EXCLUDE_ASSETS) or args.include_proxy]

    for a in all_assets:
        g = rets.loc[rets["asset"] == a, ["date", "asset", "ret_log", "mkt_ret_log"]].copy()
        g = g.sort_values("date").reset_index(drop=True)
        asset_groups[a] = g
        clean = g.dropna(subset=["ret_log", "mkt_ret_log"]).copy()
        clean_groups[a] = clean

    # Prepare collectors
    est_rows = []
    ar_rows = []

    # Loop events × assets
    for _, ev in events.iterrows():
        ev_id = ev["event_id"]
        ev_date = pd.to_datetime(ev["date"])
        bank = ev.get("bank", pd.NA)
        stance = ev.get("stance", pd.NA)

        for a in assets_for_ar:
            asset_df = asset_groups[a]
            if asset_df.empty:
                continue

            # τ dates on asset calendar
            tau_dates = trading_tau_dates(asset_df, ev_date)
            tau0_date = tau_dates[0]
            if tau0_date is None:
                # no tradable day on/after event date for this asset
                continue

            # Estimation window on intersection calendar
            clean_df = clean_groups[a]
            est_df, T_i, has_sigma, idx0_clean = select_estimation_window(clean_df, tau0_date)
            # If there is NO clean date >= tau0_date, skip this (event, asset)
            if idx0_clean is None:
                continue

            # If no usable estimation rows, still write est_params (with NaNs) so you can see why
            if T_i > 0:
                alpha_hat, beta_hat, sigma_hat, Rm_bar, Sm2 = ols_market_model(est_df)
            else:
                alpha_hat = beta_hat = sigma_hat = Rm_bar = Sm2 = np.nan

            # Save estimation params row
            est_rows.append({
                "event_id": ev_id,
                "bank": bank,
                "stance": stance,
                "asset": a,
                "market": ASSET_MARKET_MAP.get(a, "other"),
                "date_tau0": tau0_date.strftime("%Y-%m-%d"),
                "alpha_hat": alpha_hat,
                "beta_hat": beta_hat,
                "sigma_hat": sigma_hat,
                "T_i": int(T_i),
                "Rm_bar": Rm_bar,
                "Sm2": Sm2,
                "has_sigma": int(bool(has_sigma and T_i > 0))
            })

            # Compute AR at tau ∈ {-1, 0, +1} where BOTH returns exist that day
            for tau in (-1, 0, +1):
                d = tau_dates[tau]
                if d is None:
                    continue
                row = asset_df.loc[asset_df["date"] == d]
                if row.empty:
                    continue
                R_i = row["ret_log"].iloc[0]
                R_m = row["mkt_ret_log"].iloc[0]

                # Only compute AR when both returns exist on that day
                if pd.isna(R_i) or pd.isna(R_m):
                    continue

                if np.isnan(alpha_hat) or np.isnan(beta_hat):
                    # No estimation — cannot form expected return; skip AR
                    continue

                AR = float(R_i - (alpha_hat + beta_hat * R_m))
                ar_rows.append({
                    "event_id": ev_id,
                    "bank": bank,
                    "stance": stance,
                    "asset": a,
                    "market": ASSET_MARKET_MAP.get(a, "other"),
                    "tau": int(tau),
                    "date_tau": pd.to_datetime(d).strftime("%Y-%m-%d"),
                    "AR": AR,
                    "R_i": float(R_i),
                    "R_m": float(R_m),
                    "has_sigma": int(bool(has_sigma and T_i > 0)),
                    "T_i": int(T_i)
                })

    # Write outputs
    est_df_out = pd.DataFrame(est_rows)
    ar_df_out = pd.DataFrame(ar_rows)

    est_path = outdir / "est_params.csv"
    ar_path = outdir / "event_ar.csv"
    est_df_out.to_csv(est_path, index=False)
    ar_df_out.to_csv(ar_path, index=False)

    # Console summary
    print(f"Wrote {est_path}   (rows={len(est_df_out)})")
    print(f"Wrote {ar_path}    (rows={len(ar_df_out)})")
    # Quick breakdown by asset
    if not ar_df_out.empty:
        print("\nAR rows by asset:")
        print(ar_df_out.groupby("asset")["AR"].count())
    # Eligibility stats
    if not est_df_out.empty:
        elig = est_df_out["has_sigma"].sum()
        total = len(est_df_out)
        print(f"\nEligibility (Patell/BMP): has_sigma={elig}/{total} "
              f"({elig/total*100:.1f}% with T_i ≥ {MIN_T_FOR_SIGMA})")

if __name__ == "__main__":
    main()
