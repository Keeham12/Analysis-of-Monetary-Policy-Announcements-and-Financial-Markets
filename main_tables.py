#!/usr/bin/env python3
"""
Build Section 5 tables (CSV):
  5.1 AAR by day (tau = -1,0,+1) per asset x bank
  5.2 CAAR windows [-1,+1], [0,0], [0,+1] per asset x bank
  5.3 Robustness across tests (p-adj columns) for tau=0 and window [-1,+1]
  5.4 Cross-group contrasts: (a) bank compact, (b) stance CAAR, (c) period CAAR

Inputs (defaults assume you already ran the FDR scripts):
  outputs/main_fdr/aar_panel_by_asset_fdr.csv
  outputs/main_fdr/caar_panel_by_asset_fdr.csv
  outputs/subsamples/stance/caar_by_asset_stance_fdr.csv   (optional)
  outputs/subsamples/period/caar_by_asset_period_fdr.csv   (optional)

Outputs:
  outputs/tables_sec5/Table_5_1_AAR_by_day.csv
  outputs/tables_sec5/Table_5_2_CAAR_windows.csv
  outputs/tables_sec5/Table_5_3_Robustness_tau0.csv
  outputs/tables_sec5/Table_5_3_Robustness_m1p1.csv
  outputs/tables_sec5/Table_5_4a_BankComparison.csv
  outputs/tables_sec5/Table_5_4b_Stance_CAAR.csv           (if stance file exists)
  outputs/tables_sec5/Table_5_4c_Period_CAAR.csv           (if period file exists)
"""

import argparse, math
from pathlib import Path
import pandas as pd
import numpy as np

PRIMARY = "BMP"  # which test to star

def stars(p):
    if not (isinstance(p,(int,float)) and math.isfinite(p)): return ""
    return "***" if p<=0.01 else ("**" if p<=0.05 else ("*" if p<=0.10 else ""))

def pick(df, base):
    """Return (stars_col, padj_col) if present; create stars_col from padj_col if missing."""
    sc, pc = f"{base}_p_adj_stars", f"{base}_p_adj"
    if sc in df.columns and pc in df.columns: return sc, pc
    if pc in df.columns:
        df[sc] = df[pc].apply(stars)
        return sc, pc
    # fallback to raw p if no adj present (shouldn't happen if you ran FDR)
    pc = f"{base}_p"
    if pc in df.columns:
        df[sc] = df[pc].apply(stars)
        return sc, pc
    return None, None

def build_5_1(aar_path: Path, outpath: Path):
    df = pd.read_csv(aar_path)
    sc, pc = pick(df, PRIMARY)
    keep = ["asset","bank","tau","N","AAR_bps"]
    out = df[keep].copy()
    out["AAR_bps"] = out["AAR_bps"].round(1)
    out["Stars"] = df[sc] if sc else ""
    out = out.sort_values(["asset","bank","tau"])
    out.to_csv(outpath, index=False)

def build_5_2(caar_path: Path, outpath: Path):
    df = pd.read_csv(caar_path)
    df = df[df["window"].isin(["[-1,+1]","[0,0]","[0,+1]"])].copy()
    sc, pc = pick(df, PRIMARY)
    keep = ["asset","bank","window","N","CAAR_bps"]
    out = df[keep].copy()
    out["CAAR_bps"] = out["CAAR_bps"].round(1)
    out["Stars"] = df[sc] if sc else ""
    # nice window order
    cat = pd.Categorical(out["window"], categories=["[-1,+1]","[0,0]","[0,+1]"], ordered=True)
    out["window"] = cat
    out = out.sort_values(["asset","bank","window"]).reset_index(drop=True)
    out.to_csv(outpath, index=False)

def build_robust(aar_path: Path, caar_path: Path, out_tau0: Path, out_m1p1: Path):
    # τ=0 robustness across tests
    aar = pd.read_csv(aar_path)
    tau0 = aar[aar["tau"]==0].copy()
    # include Corrado if present
    cols = ["asset","bank","N","AAR_bps",
            "t_p_adj" if "t_p_adj" in tau0 else "t_p",
            "Patell_p_adj" if "Patell_p_adj" in tau0 else "Patell_p",
            "BMP_p_adj" if "BMP_p_adj" in tau0 else "BMP_p",
            "Sign_p_adj" if "Sign_p_adj" in tau0 else "Sign_p"]
    if "Corrado_p_adj" in tau0: cols.append("Corrado_p_adj")
    elif "Corrado_p" in tau0:   cols.append("Corrado_p")
    out = tau0[cols].copy()
    # agreement count (# tests with p<=0.10)
    pcols = [c for c in out.columns if c.endswith("_p_adj") or c.endswith("_p")]
    out["Agree_<=10pct"] = (out[pcols] <= 0.10).sum(axis=1)
    out["AAR_bps"] = out["AAR_bps"].round(1)
    out = out.sort_values(["asset","bank"])
    out.to_csv(out_tau0, index=False)

    # window [-1,+1] robustness
    caar = pd.read_csv(caar_path)
    w = caar[caar["window"]=="[-1,+1]"].copy()
    cols = ["asset","bank","N","CAAR_bps",
            "t_p_adj" if "t_p_adj" in w else "t_p",
            "Patell_p_adj" if "Patell_p_adj" in w else "Patell_p",
            "BMP_p_adj" if "BMP_p_adj" in w else "BMP_p",
            "Sign_p_adj" if "Sign_p_adj" in w else "Sign_p"]
    outw = w[cols].copy()
    outw["Agree_<=10pct"] = (outw[[c for c in cols if c.endswith("_p_adj") or c.endswith("_p")]] <= 0.10).sum(axis=1)
    outw["CAAR_bps"] = outw["CAAR_bps"].round(1)
    outw = outw.sort_values(["asset","bank"])
    outw.to_csv(out_m1p1, index=False)

def build_5_4a(bank_compact_path: Path, outpath: Path):
    # you already created this earlier as Table_C_BankComparison_tau0_and_m1p1.csv
    df = pd.read_csv(bank_compact_path)
    df.to_csv(outpath, index=False)

def build_5_4b(stance_path: Path, outpath: Path):
    if not stance_path.exists(): return
    df = pd.read_csv(stance_path)
    df.to_csv(outpath, index=False)

def build_5_4c(period_path: Path, outpath: Path):
    if not period_path.exists(): return
    df = pd.read_csv(period_path)
    df.to_csv(outpath, index=False)

def main():
    ap = argparse.ArgumentParser(description="Make Section 5 tables (CSV).")
    ap.add_argument("--main_fdr_dir", default="outputs/main_fdr")
    ap.add_argument("--sub_stance_dir", default="outputs/subsamples/stance")
    ap.add_argument("--sub_period_dir", default="outputs/subsamples/period")
    ap.add_argument("--bank_compact", default="outputs/tables/Table_C_BankComparison_tau0_and_m1p1.csv")
    ap.add_argument("--outdir", default="outputs/tables_sec5")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    aar_path  = Path(args.main_fdr_dir) / "aar_panel_by_asset_fdr.csv"
    caar_path = Path(args.main_fdr_dir) / "caar_panel_by_asset_fdr.csv"
    stance_p  = Path(args.sub_stance_dir) / "caar_by_asset_stance_fdr.csv"
    period_p  = Path(args.sub_period_dir) / "caar_by_asset_period_fdr.csv"
    bank_c    = Path(args.bank_compact)

    build_5_1(aar_path, outdir / "Table_5_1_AAR_by_day.csv")
    build_5_2(caar_path, outdir / "Table_5_2_CAAR_windows.csv")
    build_robust(aar_path, caar_path,
                 outdir / "Table_5_3_Robustness_tau0.csv",
                 outdir / "Table_5_3_Robustness_m1p1.csv")
    build_5_4a(bank_c, outdir / "Table_5_4a_BankComparison.csv")
    build_5_4b(stance_p, outdir / "Table_5_4b_Stance_CAAR.csv")
    build_5_4c(period_p, outdir / "Table_5_4c_Period_CAAR.csv")

    print("Wrote Section 5 tables to:", outdir)

if __name__ == "__main__":
    main()
