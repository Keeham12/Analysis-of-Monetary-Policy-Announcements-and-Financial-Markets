#!/usr/bin/env python3
# compute_AAR_CAAR_tests.py  — writes market-level + by-asset panels (with composition counts)

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd

# try scipy for p-values; fall back to normal approx if missing
try:
    from scipy.stats import t as t_dist
    from scipy.stats import norm, binomtest
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

ASSET_COLS_EXPECTED = ["btc","eurusd","sp500","eurostoxx50"]  # used for composition columns

def two_sided_p_from_t(tval, df):
    if np.isnan(tval):
        return np.nan
    if SCIPY_OK and df is not None and df >= 1:
        return 2.0 * (1.0 - t_dist.cdf(abs(tval), df))
    z = float(abs(tval))
    return math.erfc(z / math.sqrt(2.0))

def two_sided_p_from_z(zval):
    if np.isnan(zval):
        return np.nan
    if SCIPY_OK:
        return 2.0 * (1.0 - norm.cdf(abs(zval)))
    z = float(abs(zval))
    return math.erfc(z / math.sqrt(2.0))

def exact_binom_p(k_pos, n):
    if n <= 0:
        return np.nan
    if SCIPY_OK:
        return float(binomtest(k_pos, n, 0.5, alternative="two-sided").pvalue)
    phat = k_pos / n
    z = (phat - 0.5) / math.sqrt(0.25 / n)
    return two_sided_p_from_z(z)

def _safe_mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if x.size else np.nan

def _safe_sd(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return np.nan
    return float(np.nanstd(x, ddof=1))

def patell_sar(ar, sigma_hat, T_i, Rm_tau, Rm_bar, Sm2):
    if not np.isfinite(sigma_hat) or sigma_hat <= 0 or Sm2 is None or Sm2 <= 0 or T_i is None or T_i <= 0:
        return np.nan
    pv = sigma_hat * np.sqrt(1.0 + 1.0 / T_i + ((Rm_tau - Rm_bar) ** 2) / Sm2)
    if pv == 0 or not np.isfinite(pv):
        return np.nan
    return float(ar / pv)

def add_composition_counts(record_dict, group, value_col="AR"):
    """Add N_btc, N_eurusd, N_sp500, N_eurostoxx50 counts of non-missing values to a record dict."""
    counts = group.groupby("asset")[value_col].apply(lambda s: s.dropna().shape[0]).to_dict()
    for a in ASSET_COLS_EXPECTED:
        record_dict[f"N_{a}"] = int(counts.get(a, 0))
    return record_dict

def build_panels(ar_path: Path, params_path: Path, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "main").mkdir(parents=True, exist_ok=True)

    ar = pd.read_csv(ar_path, parse_dates=["date_tau"])
    params = pd.read_csv(params_path)

    merge_keys = ["event_id", "asset"]
    keep_cols = [
        "alpha_hat","beta_hat","sigma_hat","T_i","Rm_bar","Sm2","has_sigma",
        "date_tau0","bank","stance","market"
    ]
    params_small = params[merge_keys + keep_cols].copy()
    df = ar.merge(params_small, on=merge_keys, how="left", suffixes=("", "_p"))
    for c in ["bank","stance","market"]:
        if f"{c}_p" in df.columns:
            df[c] = df[c].fillna(df[f"{c}_p"])
    df.drop(columns=[c for c in df.columns if c.endswith("_p")], inplace=True, errors="ignore")

    # Patell SAR per AR row
    df["SAR"] = df.apply(
        lambda r: patell_sar(r["AR"], r["sigma_hat"], r["T_i"], r["R_m"], r["Rm_bar"], r["Sm2"]),
        axis=1
    )

    # ===== AAR: market-level (with composition) =====
    records_aar = []
    for (mkt, bank, tau), g in df.groupby(["market","bank","tau"], dropna=False):
        vals = g["AR"].dropna().to_numpy()
        N = len(vals)
        AAR = _safe_mean(vals)
        s = _safe_sd(vals)
        SE = s / np.sqrt(N) if N >= 2 and np.isfinite(s) else np.nan
        t_stat = AAR / SE if np.isfinite(SE) and SE != 0 else np.nan
        t_p = two_sided_p_from_t(t_stat, N - 1) if np.isfinite(t_stat) else np.nan
        ci_lo = AAR - 1.96 * SE if np.isfinite(SE) else np.nan
        ci_hi = AAR + 1.96 * SE if np.isfinite(SE) else np.nan

        sar = g["SAR"].dropna().to_numpy()
        N_patell = len(sar)
        Z_patell = np.sum(sar) / np.sqrt(N_patell) if N_patell > 0 else np.nan
        p_patell = two_sided_p_from_z(Z_patell) if np.isfinite(Z_patell) else np.nan

        s_sar = _safe_sd(sar)
        mean_sar = _safe_mean(sar)
        SE_sar = s_sar / np.sqrt(N_patell) if N_patell >= 2 and np.isfinite(s_sar) else np.nan
        t_bmp = mean_sar / SE_sar if np.isfinite(SE_sar) and SE_sar != 0 else np.nan
        p_bmp = two_sided_p_from_t(t_bmp, N_patell - 1) if np.isfinite(t_bmp) else np.nan

        x = vals[~np.isclose(vals, 0.0, atol=0.0)]
        N_sign = int(x.size)
        K_pos = int(np.sum(x > 0))
        p_sign = exact_binom_p(K_pos, N_sign) if N_sign > 0 else np.nan

        rec = {
            "market": mkt, "bank": bank, "tau": tau, "N": N, "AAR": AAR,
            "SD": s, "SE": SE, "t": t_stat, "t_p": t_p,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "Patell_Z": Z_patell, "Patell_p": p_patell,
            "BMP_t": t_bmp, "BMP_p": p_bmp,
            "Sign_K_pos": K_pos, "Sign_N": N_sign, "Sign_p": p_sign
        }
        rec = add_composition_counts(rec, g, value_col="AR")
        records_aar.append(rec)

    aar_panel = pd.DataFrame.from_records(records_aar).sort_values(["market","bank","tau"])
    aar_panel.to_csv(outdir / "main" / "aar_panel.csv", index=False)

    # ===== AAR: by-asset =====
    records_aar_asset = []
    for (asset, mkt, bank, tau), g in df.groupby(["asset","market","bank","tau"], dropna=False):
        vals = g["AR"].dropna().to_numpy()
        N = len(vals)
        AAR = _safe_mean(vals)
        s = _safe_sd(vals)
        SE = s / np.sqrt(N) if N >= 2 and np.isfinite(s) else np.nan
        t_stat = AAR / SE if np.isfinite(SE) and SE != 0 else np.nan
        t_p = two_sided_p_from_t(t_stat, N - 1) if np.isfinite(t_stat) else np.nan
        ci_lo = AAR - 1.96 * SE if np.isfinite(SE) else np.nan
        ci_hi = AAR + 1.96 * SE if np.isfinite(SE) else np.nan

        sar = g["SAR"].dropna().to_numpy()
        N_patell = len(sar)
        Z_patell = np.sum(sar) / np.sqrt(N_patell) if N_patell > 0 else np.nan
        p_patell = two_sided_p_from_z(Z_patell) if np.isfinite(Z_patell) else np.nan

        s_sar = _safe_sd(sar)
        mean_sar = _safe_mean(sar)
        SE_sar = s_sar / np.sqrt(N_patell) if N_patell >= 2 and np.isfinite(s_sar) else np.nan
        t_bmp = mean_sar / SE_sar if np.isfinite(SE_sar) and SE_sar != 0 else np.nan
        p_bmp = two_sided_p_from_t(t_bmp, N_patell - 1) if np.isfinite(t_bmp) else np.nan

        x = vals[~np.isclose(vals, 0.0, atol=0.0)]
        N_sign = int(x.size)
        K_pos = int(np.sum(x > 0))
        p_sign = exact_binom_p(K_pos, N_sign) if N_sign > 0 else np.nan

        records_aar_asset.append({
            "asset": asset, "market": mkt, "bank": bank, "tau": tau, "N": N, "AAR": AAR,
            "SD": s, "SE": SE, "t": t_stat, "t_p": t_p,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "Patell_Z": Z_patell, "Patell_p": p_patell,
            "BMP_t": t_bmp, "BMP_p": p_bmp,
            "Sign_K_pos": K_pos, "Sign_N": N_sign, "Sign_p": p_sign
        })
    aar_by_asset = pd.DataFrame.from_records(records_aar_asset).sort_values(["asset","market","bank","tau"])
    aar_by_asset.to_csv(outdir / "main" / "aar_panel_by_asset.csv", index=False)

    # ===== CAAR windows =====
    WINDOWS = {
        "[-1,+1]": [-1, 0, +1],
        "[0,0]": [0],
        "[0,+1]": [0, +1],
        "[-1,0]": [-1, 0],
    }

    def build_window_rows(group):
        out = []
        for (ev, asset), g2 in group.groupby(["event_id","asset"]):
            for wname, days in WINDOWS.items():
                sub = g2[g2["tau"].isin(days)]
                if len(sub) != len(days) or sub["AR"].isna().any():
                    continue
                CAR_i = float(sub["AR"].sum())
                SCAR_i = np.nan if sub["SAR"].isna().any() else float(sub["SAR"].sum())
                out.append((wname, ev, asset, CAR_i, SCAR_i))
        if not out:
            return pd.DataFrame(columns=["window","event_id","asset","CAR_i","SCAR_i"])
        return pd.DataFrame(out, columns=["window","event_id","asset","CAR_i","SCAR_i"])

    # market-level (with composition)
    records_caar = []
    for (mkt, bank), g in df.groupby(["market","bank"], dropna=False):
        win_df = build_window_rows(g)
        if win_df.empty:
            continue
        for wname, gw in win_df.groupby("window"):
            cars = gw["CAR_i"].dropna().to_numpy()
            N = len(cars)
            CAAR = _safe_mean(cars)
            s = _safe_sd(cars)
            SE = s / np.sqrt(N) if N >= 2 and np.isfinite(s) else np.nan
            t_stat = CAAR / SE if np.isfinite(SE) and SE != 0 else np.nan
            t_p = two_sided_p_from_t(t_stat, N - 1) if np.isfinite(t_stat) else np.nan
            ci_lo = CAAR - 1.96 * SE if np.isfinite(SE) else np.nan
            ci_hi = CAAR + 1.96 * SE if np.isfinite(SE) else np.nan

            scar = gw["SCAR_i"].dropna().to_numpy()
            N_patell = len(scar)
            Z_patell = np.sum(scar) / np.sqrt(N_patell) if N_patell > 0 else np.nan
            p_patell = two_sided_p_from_z(Z_patell) if np.isfinite(Z_patell) else np.nan

            s_sar = _safe_sd(scar)
            mean_sar = _safe_mean(scar)
            SE_sar = s_sar / np.sqrt(N_patell) if N_patell >= 2 and np.isfinite(s_sar) else np.nan
            t_bmp = mean_sar / SE_sar if np.isfinite(SE_sar) and SE_sar != 0 else np.nan
            p_bmp = two_sided_p_from_t(t_bmp, N_patell - 1) if np.isfinite(t_bmp) else np.nan

            x = cars[~np.isclose(cars, 0.0, atol=0.0)]
            N_sign = int(x.size)
            K_pos = int(np.sum(x > 0))
            p_sign = exact_binom_p(K_pos, N_sign) if N_sign > 0 else np.nan

            rec = {
                "market": mkt, "bank": bank, "window": wname, "N": N, "CAAR": CAAR,
                "SD_CAR": s, "SE": SE, "t": t_stat, "t_p": t_p,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "Patell_Z": Z_patell, "Patell_p": p_patell,
                "BMP_t": t_bmp, "BMP_p": p_bmp,
                "Sign_K_pos": K_pos, "Sign_N": N_sign, "Sign_p": p_sign
            }
            # composition by asset within this (market, bank, window)
            rec = add_composition_counts(rec, gw.rename(columns={"CAR_i":"AR"}), value_col="AR")
            records_caar.append(rec)

    caar_panel = pd.DataFrame.from_records(records_caar).sort_values(["market","bank","window"])
    caar_panel.to_csv(outdir / "main" / "caar_panel.csv", index=False)

    # by-asset CAAR
    records_caar_asset = []
    for (asset, mkt, bank), g in df.groupby(["asset","market","bank"], dropna=False):
        win_df = build_window_rows(g)
        if win_df.empty:
            continue
        for wname, gw in win_df.groupby("window"):
            cars = gw["CAR_i"].dropna().to_numpy()
            N = len(cars)
            CAAR = _safe_mean(cars)
            s = _safe_sd(cars)
            SE = s / np.sqrt(N) if N >= 2 and np.isfinite(s) else np.nan
            t_stat = CAAR / SE if np.isfinite(SE) and SE != 0 else np.nan
            t_p = two_sided_p_from_t(t_stat, N - 1) if np.isfinite(t_stat) else np.nan
            ci_lo = CAAR - 1.96 * SE if np.isfinite(SE) else np.nan
            ci_hi = CAAR + 1.96 * SE if np.isfinite(SE) else np.nan

            scar = gw["SCAR_i"].dropna().to_numpy()
            N_patell = len(scar)
            Z_patell = np.sum(scar) / np.sqrt(N_patell) if N_patell > 0 else np.nan
            p_patell = two_sided_p_from_z(Z_patell) if np.isfinite(Z_patell) else np.nan

            s_sar = _safe_sd(scar)
            mean_sar = _safe_mean(scar)
            SE_sar = s_sar / np.sqrt(N_patell) if N_patell >= 2 and np.isfinite(s_sar) else np.nan
            t_bmp = mean_sar / SE_sar if np.isfinite(SE_sar) and SE_sar != 0 else np.nan
            p_bmp = two_sided_p_from_t(t_bmp, N_patell - 1) if np.isfinite(t_bmp) else np.nan

            x = cars[~np.isclose(cars, 0.0, atol=0.0)]
            N_sign = int(x.size)
            K_pos = int(np.sum(x > 0))
            p_sign = exact_binom_p(K_pos, N_sign) if N_sign > 0 else np.nan

            records_caar_asset.append({
                "asset": asset, "market": mkt, "bank": bank, "window": wname, "N": N, "CAAR": CAAR,
                "SD_CAR": s, "SE": SE, "t": t_stat, "t_p": t_p,
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "Patell_Z": Z_patell, "Patell_p": p_patell,
                "BMP_t": t_bmp, "BMP_p": p_bmp,
                "Sign_K_pos": K_pos, "Sign_N": N_sign, "Sign_p": p_sign
            })
    caar_by_asset = pd.DataFrame.from_records(records_caar_asset).sort_values(["asset","market","bank","window"])
    caar_by_asset.to_csv(outdir / "main" / "caar_panel_by_asset.csv", index=False)

    print("Wrote:")
    print(f"  {outdir / 'main' / 'aar_panel.csv'}")
    print(f"  {outdir / 'main' / 'aar_panel_by_asset.csv'}")
    print(f"  {outdir / 'main' / 'caar_panel.csv'}")
    print(f"  {outdir / 'main' / 'caar_panel_by_asset.csv'}")

def main():
    ap = argparse.ArgumentParser(description="Aggregate ARs to AAR/CAAR; output market-level and by-asset panels.")
    ap.add_argument("--ars", default="outputs/event_ar.csv", help="Path to event_ar.csv")
    ap.add_argument("--params", default="outputs/est_params.csv", help="Path to est_params.csv")
    ap.add_argument("--outdir", default="outputs", help="Output base directory (default: outputs)")
    args = ap.parse_args()
    build_panels(Path(args.ars), Path(args.params), Path(args.outdir))

if __name__ == "__main__":
    main()
