#!/usr/bin/env python3
"""
Make thesis tables (CSV) from your FDR-adjusted panels.

Outputs (default: outputs/tables/):
  - Table_A_AAR_tau0_by_asset_bank.csv          (A: announcement-day AAR)
  - Table_B_CAAR_m1p1_by_asset_bank.csv         (B: [-1,+1] window)
  - Table_C_BankComparison_tau0_and_m1p1.csv    (C: compact ECB vs Fed per asset)
  - Table_D_Stance_CAAR_m1p1_by_asset.csv       (D: stance heterogeneity)
  - Table_E_Period_CAAR_m1p1_by_asset.csv       (E: period heterogeneity)

Run:
  python make_thesis_tables.py \
    --main_fdr_dir outputs/main_fdr \
    --sub_stance_dir outputs/subsamples/stance \
    --sub_period_dir outputs/subsamples/period \
    --outdir outputs/tables
"""

import argparse
import math
from pathlib import Path
import pandas as pd
import numpy as np

PRIMARY_TEST = "BMP"  # which test's adjusted p to use for star display: "BMP", "Patell", "t", or "Corrado"

# ------------- helpers -----------------

def stars_from_padj(p):
    if not (isinstance(p, (int, float)) and math.isfinite(p)): return ""
    if p <= 0.01: return "***"
    if p <= 0.05: return "**"
    if p <= 0.10: return "*"
    return ""

def pick_star_col(df, prefix="BMP"):
    # Prefer explicit stars column if present, else compute from *_p_adj
    star_col = f"{prefix}_p_adj_stars"
    padj_col = f"{prefix}_p_adj"
    if star_col in df.columns:
        return star_col, padj_col
    # Sometimes the FDR script names were slightly different (lowercase, etc.)
    alt_star = f"{prefix.lower()}_p_adj_stars"
    alt_padj = f"{prefix.lower()}_p_adj"
    if alt_star in df.columns and alt_padj in df.columns:
        return alt_star, alt_padj
    # Fallback: compute stars from p_adj
    if padj_col in df.columns:
        df[star_col] = df[padj_col].apply(stars_from_padj)
        return star_col, padj_col
    if alt_padj in df.columns:
        df[alt_star] = df[alt_padj].apply(stars_from_padj)
        return alt_star, alt_padj
    # Last resort: no p_adj — compute from raw p if exists
    raw_col = f"{prefix}_p"
    if raw_col in df.columns:
        df[star_col] = df[raw_col].apply(stars_from_padj)
        return star_col, raw_col
    return None, None

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    return pd.read_csv(path)

def ensure_cols(df, needed):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns in {getattr(df, '__source__', 'DataFrame')}: {miss}")

def fmt_round(series, decimals=1):
    return series.round(decimals)

def prefer_corrado_augmented(main_dir: Path, base: str) -> Path:
    """
    Prefer Corrado-augmented + FDR filenames when present.
    """
    # Common file names from earlier scripts
    # AAR by-asset:
    candidates = [
        main_dir / f"{base}_fdr.csv",                  # preferred (FDR already applied)
        main_dir.parent / base.replace("_fdr", "") + ".csv",  # non-FDR fallback
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"Could not find any of: {candidates}")

# ------------- table builders -----------------

def build_table_A(aar_by_asset_fdr: pd.DataFrame, outpath: Path):
    df = aar_by_asset_fdr.copy()
    df = df[df["tau"] == 0]

    # Choose star source
    star_col, padj_col = pick_star_col(df, prefix=PRIMARY_TEST)
    # Optional Corrado (if present)
    corrado_padj = "Corrado_p_adj" if "Corrado_p_adj" in df.columns else None

    cols = ["asset", "bank", "N", "AAR_bps"]
    nice = df[cols].copy()
    nice["AAR_bps"] = fmt_round(nice["AAR_bps"], 1)

    # Attach p_adj columns for reference (optional)
    if padj_col and padj_col in df.columns:
        nice[f"{PRIMARY_TEST}_p_adj"] = df[padj_col].values
    if corrado_padj:
        nice["Corrado_p_adj"] = df[corrado_padj].values

    # Stars column
    if star_col:
        nice["Stars"] = df[star_col].values
    else:
        nice["Stars"] = ""

    nice = nice.sort_values(["asset", "bank"]).reset_index(drop=True)
    nice.to_csv(outpath, index=False)

def build_table_B(caar_by_asset_fdr: pd.DataFrame, outpath: Path):
    df = caar_by_asset_fdr.copy()
    df = df[df["window"] == "[-1,+1]"]

    star_col, padj_col = pick_star_col(df, prefix=PRIMARY_TEST)

    cols = ["asset", "bank", "N", "CAAR_bps"]
    nice = df[cols].copy()
    nice["CAAR_bps"] = fmt_round(nice["CAAR_bps"], 1)

    if padj_col and padj_col in df.columns:
        nice[f"{PRIMARY_TEST}_p_adj"] = df[padj_col].values
    if star_col:
        nice["Stars"] = df[star_col].values
    else:
        nice["Stars"] = ""

    nice = nice.sort_values(["asset", "bank"]).reset_index(drop=True)
    nice.to_csv(outpath, index=False)

def build_table_C(tableA: pd.DataFrame, tableB: pd.DataFrame, outpath: Path):
    """
    Compact bank comparison: one row per asset, columns = ECB_cell, Fed_cell
    cell = "AAR0_bps (CAAR_m1p1_bps)" and a star if AAR0 has a star.
    """
    A = tableA.copy()
    B = tableB.copy()

    A = A[["asset", "bank", "AAR_bps", "Stars"]].rename(columns={"Stars": "Star_AAR"})
    B = B[["asset", "bank", "CAAR_bps"]]

    merged = A.merge(B, on=["asset", "bank"], how="left")

    def make_cell(row):
        s = row.get("Star_AAR", "")
        star = s if isinstance(s, str) else ""
        return f"{row['AAR_bps']:.1f} ({row['CAAR_bps']:.1f}){star}"

    merged["cell"] = merged.apply(make_cell, axis=1)

    # Pivot to columns ECB/Fed
    wide = merged.pivot(index="asset", columns="bank", values="cell").reset_index()
    # Clean column order: asset, ECB, Fed (if present)
    cols = ["asset"] + [c for c in ["ECB", "Fed"] if c in wide.columns]
    wide = wide[cols]
    wide.to_csv(outpath, index=False)

def build_table_D(caar_stance_fdr: pd.DataFrame, outpath: Path):
    """
    Stance subsamples: two panels per bank (you’ll pivot further in LaTeX if desired).
    We’ll output long-wide-ish: asset, bank, then for each stance -> {bps, stars, N}.
    """
    df = caar_stance_fdr.copy()
    df = df[df["window"] == "[-1,+1]"].copy()

    star_col, padj_col = pick_star_col(df, prefix=PRIMARY_TEST)

    # Stance ordering if present
    stance_order = [x for x in ["Hawkish", "Hawk", "Neutral", "Dovish", "Dove"] if x in df["stance"].unique()]
    if not stance_order:
        stance_order = sorted(df["stance"].dropna().unique().tolist())

    rows = []
    for (asset, bank), g in df.groupby(["asset", "bank"], dropna=False):
        rec = {"asset": asset, "bank": bank}
        for st in stance_order:
            sub = g[g["stance"] == st]
            if sub.empty:
                rec[f"{st}_CAAR_bps"] = np.nan
                rec[f"{st}_Stars"] = ""
                rec[f"{st}_N"] = 0
            else:
                rec[f"{st}_CAAR_bps"] = float(fmt_round(sub["CAAR_bps"].iloc[0], 1))
                rec[f"{st}_N"] = int(sub["N"].iloc[0])
                if star_col and star_col in sub.columns:
                    rec[f"{st}_Stars"] = sub[star_col].iloc[0]
                elif padj_col and padj_col in sub.columns:
                    rec[f"{st}_Stars"] = stars_from_padj(sub[padj_col].iloc[0])
                else:
                    rec[f"{st}_Stars"] = ""
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["asset", "bank"])
    out.to_csv(outpath, index=False)

def build_table_E(caar_period_fdr: pd.DataFrame, outpath: Path):
    """
    Period subsamples: columns P1, P2, ... from your compute_subsample_panels.py labels.
    """
    df = caar_period_fdr.copy()
    df = df[df["window"] == "[-1,+1]"].copy()

    star_col, padj_col = pick_star_col(df, prefix=PRIMARY_TEST)

    periods = [p for p in sorted(df["period"].dropna().unique())]

    rows = []
    for (asset, bank), g in df.groupby(["asset", "bank"], dropna=False):
        rec = {"asset": asset, "bank": bank}
        for p in periods:
            sub = g[g["period"] == p]
            if sub.empty:
                rec[f"{p}_CAAR_bps"] = np.nan
                rec[f"{p}_Stars"] = ""
                rec[f"{p}_N"] = 0
            else:
                rec[f"{p}_CAAR_bps"] = float(fmt_round(sub["CAAR_bps"].iloc[0], 1))
                rec[f"{p}_N"] = int(sub["N"].iloc[0])
                if star_col and star_col in sub.columns:
                    rec[f"{p}_Stars"] = sub[star_col].iloc[0]
                elif padj_col and padj_col in sub.columns:
                    rec[f"{p}_Stars"] = stars_from_padj(sub[padj_col].iloc[0])
                else:
                    rec[f"{p}_Stars"] = ""
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["asset", "bank"])
    out.to_csv(outpath, index=False)

# ------------- main -----------------

def main():
    ap = argparse.ArgumentParser(description="Build thesis tables A–E as CSV.")
    ap.add_argument("--main_fdr_dir", default="outputs/main_fdr", help="Dir with fdr-adjusted main panels")
    ap.add_argument("--sub_stance_dir", default="outputs/subsamples/stance", help="Dir with stance subsample panels")
    ap.add_argument("--sub_period_dir", default="outputs/subsamples/period", help="Dir with period subsample panels")
    ap.add_argument("--outdir", default="outputs/tables", help="Where to write the CSV tables")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Load main panels (prefer FDR outputs) ---
    aar_by_asset_path  = Path(args.main_fdr_dir) / "aar_panel_by_asset_fdr.csv"
    caar_by_asset_path = Path(args.main_fdr_dir) / "caar_panel_by_asset_fdr.csv"
    if not aar_by_asset_path.exists() or not caar_by_asset_path.exists():
        raise SystemExit("Expected FDR-adjusted panel CSVs not found. Run the FDR script first.")

    aar_by_asset_fdr  = load_csv(aar_by_asset_path)
    caar_by_asset_fdr = load_csv(caar_by_asset_path)

    # --- Table A ---
    tableA_path = outdir / "Table_A_AAR_tau0_by_asset_bank.csv"
    build_table_A(aar_by_asset_fdr, tableA_path)

    # --- Table B ---
    tableB_path = outdir / "Table_B_CAAR_m1p1_by_asset_bank.csv"
    build_table_B(caar_by_asset_fdr, tableB_path)

    # --- Table C (uses A & B just written) ---
    A_df = load_csv(tableA_path)
    B_df = load_csv(tableB_path)
    tableC_path = outdir / "Table_C_BankComparison_tau0_and_m1p1.csv"
    build_table_C(A_df, B_df, tableC_path)

    # --- Stance ---
    stance_caar_fdr_path = Path(args.sub_stance_dir) / "caar_by_asset_stance_fdr.csv"
    if stance_caar_fdr_path.exists():
        stance_caar_fdr = load_csv(stance_caar_fdr_path)
        tableD_path = outdir / "Table_D_Stance_CAAR_m1p1_by_asset.csv"
        build_table_D(stance_caar_fdr, tableD_path)
    else:
        print(f"[WARN] Missing stance file: {stance_caar_fdr_path} (skipping Table D)")

    # --- Period ---
    period_caar_fdr_path = Path(args.sub_period_dir) / "caar_by_asset_period_fdr.csv"
    if period_caar_fdr_path.exists():
        period_caar_fdr = load_csv(period_caar_fdr_path)
        tableE_path = outdir / "Table_E_Period_CAAR_m1p1_by_asset.csv"
        build_table_E(period_caar_fdr, tableE_path)
    else:
        print(f"[WARN] Missing period file: {period_caar_fdr_path} (skipping Table E)")

    print("Tables written to:", outdir)

if __name__ == "__main__":
    main()
