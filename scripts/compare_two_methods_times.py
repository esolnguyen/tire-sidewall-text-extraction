#!/usr/bin/env python3
"""
compare_two_methods_times.py

Plot/compare avg execution times for two methods without showing filenames.

Usage:
  python3 scripts/compare_two_methods_times.py \
    --method-a g25_flash_only \
    --method-b g25_flash_pipepline \
    --input-dir eval_results/Entire \
    --out-prefix compare_g25

If you already have per-image CSVs use --csv-a and/or --csv-b.
"""
import argparse
import os
import glob
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# robust style
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid")
except Exception:
    plt.style.use("ggplot")


def compute_avg_from_jsons(input_dir, method_substr):
    pattern = os.path.join(input_dir, "*.json")
    files = sorted([p for p in glob.glob(pattern)
                   if method_substr in os.path.basename(p)])
    if not files:
        raise FileNotFoundError(
            f"No JSON files matching '{method_substr}' in {input_dir}")
    per_image = defaultdict(list)
    for fp in files:
        with open(fp, "r") as fh:
            j = json.load(fh)
        for item in j.get("detailed_results", []):
            fname = item.get("filename")
            et = item.get("execution_time")
            if fname and (et is not None):
                try:
                    per_image[fname].append(float(et))
                except Exception:
                    pass
    rows = []
    for fname, vals in sorted(per_image.items()):
        rows.append({"filename": fname, "avg_execution_time": sum(
            vals)/len(vals), "n_runs": len(vals)})
    return pd.DataFrame(rows)


def load_csv_if_exists(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        if "avg_execution_time" not in df.columns:
            for alt in ("avg_time", "avg_exec_time", "avg"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "avg_execution_time"})
                    break
        return df
    return None


def paired_arrays(df_a, df_b):
    df = pd.merge(df_a[['filename', 'avg_execution_time']], df_b[['filename', 'avg_execution_time']],
                  on='filename', how='inner', suffixes=('_a', '_b'))
    a = pd.to_numeric(df['avg_execution_time_a'],
                      errors='coerce').dropna().values
    b = pd.to_numeric(df['avg_execution_time_b'],
                      errors='coerce').dropna().values
    # ensure same order and same length
    return df['filename'].values, a, b


def summarize_and_plot(a, b, out_prefix):
    # stats
    def stat(x):
        return {"mean": float(np.nanmean(x)), "median": float(np.nanmedian(x)), "std": float(np.nanstd(x))}
    sa = stat(a)
    sb = stat(b)
    diff = b - a
    sd = stat(diff)
    # correlation + paired t-test if available
    try:
        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(a, b)
        t_res = stats.ttest_rel(b, a)
        ttext = f"pearson r={pearson_r:.4f} (p={pearson_p:.4g}); paired t: t={t_res.statistic:.4f}, p={t_res.pvalue:.4g}"
    except Exception:
        pearson_r = np.corrcoef(a, b)[0, 1] if len(a) > 1 else float('nan')
        ttext = "scipy not available: paired t-test skipped"
    # print summaries
    print("Method A: mean/median/std =",
          f"{sa['mean']:.4f}", f"{sa['median']:.4f}", f"{sa['std']:.4f}")
    print("Method B: mean/median/std =",
          f"{sb['mean']:.4f}", f"{sb['median']:.4f}", f"{sb['std']:.4f}")
    print("Diff (B-A): mean/median/std =",
          f"{sd['mean']:.4f}", f"{sd['median']:.4f}", f"{sd['std']:.4f}")
    print("Paired samples:", len(a))
    print(ttext)
    # plotting: boxplot, scatter, histogram of diffs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # boxplot
    axes[0].boxplot([a, b], labels=["A", "B"])
    axes[0].set_title("Distribution (boxplot)")
    axes[0].set_ylabel("Avg execution time (s)")
    # paired scatter
    axes[1].scatter(a, b, alpha=0.7)
    mn = min(np.nanmin(a), np.nanmin(b))
    mx = max(np.nanmax(a), np.nanmax(b))
    axes[1].plot([mn, mx], [mn, mx], color='gray', linestyle='--', linewidth=1)
    axes[1].set_xlabel("Method A avg time (s)")
    axes[1].set_ylabel("Method B avg time (s)")
    axes[1].set_title("Paired scatter (y=x line shown)")
    # histogram of differences
    axes[2].hist(diff, bins=30, color='tab:blue', alpha=0.8)
    axes[2].axvline(np.nanmean(diff), color='red', linestyle='--',
                    label=f"mean={np.nanmean(diff):.3f}")
    axes[2].set_title("Histogram of (B - A)")
    axes[2].legend()
    plt.tight_layout()
    png = f"{out_prefix}.png"
    fig.savefig(png, dpi=150)
    print(f"Saved comparison plot to {png}")
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "-i",
                   default=os.path.join("eval_results", "Entire"))
    p.add_argument("--method-a", default="g25_flash_only")
    p.add_argument("--method-b", default="g25_flash_pipepline")
    p.add_argument("--csv-a", help="optional precomputed CSV for method A")
    p.add_argument("--csv-b", help="optional precomputed CSV for method B")
    p.add_argument("--out-prefix", default="compare_g25")
    args = p.parse_args()

    df_a = load_csv_if_exists(args.csv_a) if args.csv_a else None
    if df_a is None:
        df_a = compute_avg_from_jsons(args.input_dir, args.method_a)
    df_b = load_csv_if_exists(args.csv_b) if args.csv_b else None
    if df_b is None:
        df_b = compute_avg_from_jsons(args.input_dir, args.method_b)

    fnames, a, b = paired_arrays(df_a, df_b)
    if len(a) == 0:
        raise RuntimeError(
            "No paired images found between the two methods (no common filenames).")
    summarize_and_plot(a, b, args.out_prefix)


if __name__ == "__main__":
    main()
