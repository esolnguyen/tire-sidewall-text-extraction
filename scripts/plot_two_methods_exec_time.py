#!/usr/bin/env python3
"""
plot_two_methods_exec_time.py

Plot per-image average execution time for two methods (e.g. 'g25_flash_only' and 'g25_flash_pipepline').

Usage examples:
  python3 scripts/plot_two_methods_exec_time.py \
    --method-a g25_flash_only \
    --method-b g25_flash_pipepline \
    --input-dir eval_results/Entire \
    --out png

Or, if you already created per-image CSVs:
  python3 scripts/plot_two_methods_exec_time.py \
    --csv-a avg_g25_flash_only.csv \
    --csv-b avg_g25_flash_pipeline.csv \
    --out png

Outputs:
 - Displays a matplotlib plot comparing avg execution times per image.
 - Optionally saves CSV merged result and/ or PNG chart.
"""
import argparse
import os
import glob
import json
import csv
from collections import defaultdict
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Robust plotting style: prefer seaborn if available, otherwise fall back to a matplotlib style
try:
    import seaborn as sns
    sns.set_theme(style='whitegrid')
except Exception:
    try:
        plt.style.use('ggplot')
    except Exception:
        # last resort: default style
        pass


def compute_avg_from_jsons(input_dir, method_substr):
    pattern = os.path.join(input_dir, "*.json")
    files = sorted([p for p in glob.glob(pattern)
                   if method_substr in os.path.basename(p)])
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {input_dir} matching '{method_substr}'")
    per_image = defaultdict(list)
    for fp in files:
        with open(fp, "r") as fh:
            j = json.load(fh)
        for item in j.get("detailed_results", []):
            fname = item.get("filename")
            et = item.get("execution_time")
            if fname is None or et is None:
                continue
            try:
                per_image[fname].append(float(et))
            except Exception:
                continue
    rows = []
    for fname, vals in sorted(per_image.items()):
        avg = sum(vals) / len(vals)
        rows.append(
            {"filename": fname, "avg_execution_time": avg, "n_runs": len(vals)})
    return pd.DataFrame(rows)


def load_csv_if_exists(path):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        # normalize column name
        if "avg_execution_time" not in df.columns and "avg_time" in df.columns:
            df = df.rename(columns={"avg_time": "avg_execution_time"})
        return df
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", "-i", default=os.path.join("eval_results", "Entire"),
                   help="Directory with JSON run files (default: eval_results/Entire)")
    p.add_argument("--method-a", required=False, default="g25_flash_only",
                   help="Substring to filter JSON files for method A (default g25_flash_only)")
    p.add_argument("--method-b", required=False, default="g25_flash_pipepline",
                   help="Substring to filter JSON files for method B (default g25_flash_pipepline)")
    p.add_argument(
        "--csv-a", help="Optional precomputed per-image CSV for method A")
    p.add_argument(
        "--csv-b", help="Optional precomputed per-image CSV for method B")
    p.add_argument("--out", choices=["png", "csv", "both", "show"], default="both",
                   help="What to output: png, csv, both, or just show (default: both)")
    p.add_argument("--out-prefix", default="g25_time_comparison",
                   help="Prefix for output files (default: g25_time_comparison)")
    args = p.parse_args()

    # load or compute method A
    df_a = load_csv_if_exists(args.csv_a) if args.csv_a else None
    if df_a is None:
        print(
            f"Computing per-image averages for method A using substring '{args.method_a}'...")
        df_a = compute_avg_from_jsons(args.input_dir, args.method_a)
    df_a = df_a.rename(
        columns={"avg_execution_time": "avg_a", "n_runs": "n_a"})

    # load or compute method B
    df_b = load_csv_if_exists(args.csv_b) if args.csv_b else None
    if df_b is None:
        print(
            f"Computing per-image averages for method B using substring '{args.method_b}'...")
        df_b = compute_avg_from_jsons(args.input_dir, args.method_b)
    df_b = df_b.rename(
        columns={"avg_execution_time": "avg_b", "n_runs": "n_b"})

    # Merge by filename (outer to keep any image present in either)
    df = pd.merge(df_a[["filename", "avg_a", "n_a"]], df_b[["filename", "avg_b", "n_b"]],
                  on="filename", how="outer").sort_values("filename").reset_index(drop=True)

    # Convert to numeric, keep NaN where missing
    df["avg_a"] = pd.to_numeric(df["avg_a"], errors="coerce")
    df["avg_b"] = pd.to_numeric(df["avg_b"], errors="coerce")

    # Prepare x-axis (index); optionally you can use filenames as x-axis labels but that can be crowded
    x = np.arange(len(df))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, df["avg_a"], marker="o", linestyle="-",
            label=args.method_a, color="tab:blue", alpha=0.9)
    ax.plot(x, df["avg_b"], marker="s", linestyle="-",
            label=args.method_b, color="tab:orange", alpha=0.9)

    ax.set_xlabel("Image (index)")
    ax.set_ylabel("Average execution time (s)")
    ax.set_title(
        f"Per-image avg execution time: {args.method_a} vs {args.method_b}")
    ax.legend()
    ax.set_xlim(-1, len(x))

    # Manage ticks: show filenames sparsely if many images
    if len(x) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(df["filename"], rotation=90, fontsize=8)
    else:
        step = max(1, len(x) // 20)
        ticks = x[::step]
        ax.set_xticks(ticks)
        ax.set_xticklabels(df["filename"].iloc[::step],
                           rotation=90, fontsize=8)

    plt.tight_layout()

    # Save outputs if requested
    if args.out in ("png", "both"):
        png_path = f"{args.out_prefix}.png"
        fig.savefig(png_path, dpi=150)
        print(f"Saved PNG chart to {png_path}")
    if args.out in ("csv", "both"):
        csv_path = f"{args.out_prefix}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved merged per-image CSV to {csv_path}")

    if args.out in ("show", "both", "png"):
        plt.show()


if __name__ == "__main__":
    main()
