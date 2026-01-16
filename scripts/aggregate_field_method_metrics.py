#!/usr/bin/env python3
"""
aggregate_field_method_metrics.py

Reads per-image CSV run files (like those in eval_results/Entire/), finds columns
ending with "_correct" and computes average correctness per Field grouped by Method.

Precision, Recall and F1 are set equal to the mean correctness (proxy). The script
can output a CSV with columns: Field, Method, Precision, Recall, F1.

Usage examples:
  python3 scripts/aggregate_field_method_metrics.py
  python3 scripts/aggregate_field_method_metrics.py --input-dir eval_results/Entire --output aggregated_metrics.csv

"""
import argparse
import glob
import os
import csv
from collections import defaultdict
from statistics import mean


def infer_method_from_filename(fn: str) -> str:
    bn = os.path.basename(fn).lower()
    # common normalizations used in this workspace
    if 'g25_flash_pipepline' in bn:
        return 'G25_flash_pipeline'
    if 'g25_flash_only' in bn:
        return 'G25_flash_only'
    # fallback: filename without extension
    return os.path.splitext(os.path.basename(fn))[0]


def parse_bool_like(v):
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ('true', '1', 't', 'yes', 'y'):
        return True
    if s in ('false', '0', 'f', 'no', 'n'):
        return False
    # try numeric
    try:
        return float(s) != 0.0
    except Exception:
        return False


def collect_per_file_means(filepaths, field_suffix='_correct'):
    """Return list of tuples: (filepath, method, {field: mean})"""
    results = []
    for fp in filepaths:
        try:
            with open(fp, newline='') as fh:
                reader = csv.DictReader(fh)
                counts = defaultdict(int)
                totals = defaultdict(int)
                total_rows = 0
                for r in reader:
                    total_rows += 1
                    for k, v in r.items():
                        if k and k.endswith(field_suffix):
                            totals[k] += 1
                            if parse_bool_like(v):
                                counts[k] += 1
                means = {}
                if total_rows == 0:
                    # no rows -> zeroes
                    for k in totals.keys():
                        means[k] = 0.0
                else:
                    for k in set(list(totals.keys()) + list(counts.keys())):
                        means[k] = counts.get(k, 0) / total_rows
        except Exception as e:
            print(f"Warning: failed to read '{fp}': {e}")
            continue
        method = infer_method_from_filename(fp)
        results.append((fp, method, means))
    return results


def aggregate_by_method(per_file_results):
    # per_file_results: list of (filepath, method, {field: mean})
    by_method = defaultdict(lambda: defaultdict(list))
    for fp, method, means in per_file_results:
        for field_col, m in means.items():
            by_method[method][field_col].append(m)
    # compute mean across files for each method+field
    agg = {}
    for method, field_map in by_method.items():
        agg[method] = {fc: mean(vals) for fc, vals in field_map.items()}
    return agg


def write_output(agg, output_path, field_suffix='_correct'):
    # output rows: Field, Method, Precision, Recall, F1
    rows = []
    # collect all fields
    fields = set()
    for method, fm in agg.items():
        fields.update(fm.keys())
    # sort fields by name without suffix
    fields = sorted(list(fields), key=lambda x: x.replace(field_suffix, ''))

    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Field', 'Method', 'Precision', 'Recall', 'F1'])
        for fc in fields:
            field_label = fc.replace(field_suffix, '').capitalize()
            first = True
            for method in sorted(agg.keys()):
                val = agg[method].get(fc, 0.0)
                if first:
                    writer.writerow(
                        [field_label, method, f"{val:.4f}", f"{val:.4f}", f"{val:.4f}"])
                    first = False
                else:
                    writer.writerow(
                        ['', method, f"{val:.4f}", f"{val:.4f}", f"{val:.4f}"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', '-i', default=os.path.join(os.getcwd(), 'eval_results', 'Entire'),
                   help='Directory with per-run CSVs (default: eval_results/Entire)')
    p.add_argument('--include', default=None,
                   help='Optional substring to filter which files to include (e.g. g25_flash)')
    p.add_argument('--output', '-o', default='aggregated_field_method_metrics.csv',
                   help='Output CSV filename (default: aggregated_field_method_metrics.csv)')
    p.add_argument('--per-run', action='store_true',
                   help='Also print per-run values instead of only aggregated')
    args = p.parse_args()

    pattern = os.path.join(args.input_dir, '*.csv')
    files = sorted(glob.glob(pattern))
    if args.include:
        files = [f for f in files if args.include in os.path.basename(f)]

    if not files:
        print(
            f"No CSV files found in {args.input_dir} matching include='{args.include}'")
        return

    per_file_results = collect_per_file_means(files)

    if args.per_run:
        print('Per-run means (Field columns are *_correct):')
        for fp, method, means in per_file_results:
            print(f"File: {os.path.basename(fp)}  Method: {method}")
            for fc, m in sorted(means.items()):
                print(f"  {fc.replace('_correct','').capitalize()}: {m:.4f}")

    agg = aggregate_by_method(per_file_results)

    write_output(agg, args.output)
    print(f"Wrote aggregated metrics to {args.output}")


if __name__ == '__main__':
    main()
