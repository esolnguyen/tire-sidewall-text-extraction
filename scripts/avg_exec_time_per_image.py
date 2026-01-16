#!/usr/bin/env python3
"""
avg_exec_time_per_image.py

Reads JSON run files (e.g. in eval_results/Entire/) and computes the average
execution_time per image (keyed by filename) across all runs found.

Outputs CSV with columns: filename, avg_execution_time, n_runs

Usage:
  python3 scripts/avg_exec_time_per_image.py
  python3 scripts/avg_exec_time_per_image.py --input-dir eval_results/Entire --output avg_execution_time_per_image.csv

"""
import argparse
import glob
import json
import os
import csv
from collections import defaultdict


def collect_execution_times(input_dir, include_substr=None):
    pattern = os.path.join(input_dir, '*.json')
    files = sorted(glob.glob(pattern))
    if include_substr:
        files = [f for f in files if include_substr in os.path.basename(f)]
    per_image = defaultdict(list)
    for fp in files:
        try:
            with open(fp) as fh:
                j = json.load(fh)
        except Exception as e:
            print(f"Warning: failed to load {fp}: {e}")
            continue
        for item in j.get('detailed_results', []):
            fname = item.get('filename')
            et = item.get('execution_time')
            if fname is None or et is None:
                continue
            # convert to float if possible
            try:
                per_image[fname].append(float(et))
            except Exception:
                continue
    return per_image


def write_csv(per_image, output_path):
    with open(output_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['filename', 'avg_execution_time', 'n_runs'])
        for fname in sorted(per_image.keys()):
            vals = per_image[fname]
            if not vals:
                continue
            avg = sum(vals) / len(vals)
            writer.writerow([fname, f"{avg:.6f}", len(vals)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', '-i', default=os.path.join('eval_results', 'Entire'),
                   help='Directory with JSON run files (default: eval_results/Entire)')
    p.add_argument('--include', default=None,
                   help='Optional substring filter for filenames (e.g. g25_flash)')
    p.add_argument('--output', '-o', default='avg_execution_time_per_image.csv',
                   help='Output CSV path')
    args = p.parse_args()

    per_image = collect_execution_times(
        args.input_dir, include_substr=args.include)
    if not per_image:
        print(
            f"No execution times found in {args.input_dir} (include={args.include})")
        return
    write_csv(per_image, args.output)
    print(f"Wrote {len(per_image)} image averages to {args.output}")


if __name__ == '__main__':
    main()
