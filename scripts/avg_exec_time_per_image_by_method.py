#!/usr/bin/env python3
"""
avg_exec_time_per_image_by_method.py

Compute per-image average execution_time across runs for a specific method group.
Example:
  python3 scripts/avg_exec_time_per_image_by_method.py --method-substr g25_flash_only
  python3 scripts/avg_exec_time_per_image_by_method.py --method-substr g25_flash_pipepline --output avg_pipeline.csv

Outputs CSV with columns: filename, avg_execution_time, n_runs, std_dev
"""
import argparse
import glob
import json
import os
import csv
import math
from collections import defaultdict


def safe_mean(xs):
    return sum(xs)/len(xs) if xs else 0.0


def safe_std(xs):
    if not xs:
        return 0.0
    m = safe_mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/len(xs))


def compute_avg_per_image(input_dir, method_substr):
    pattern = os.path.join(input_dir, '*.json')
    files = sorted(glob.glob(pattern))
    files = [f for f in files if method_substr in os.path.basename(f)]
    if not files:
        raise FileNotFoundError(
            f'No json files found in {input_dir} matching "{method_substr}"')

    per_image = defaultdict(list)  # filename -> list of execution times
    for fp in files:
        with open(fp, 'r') as fh:
            jobj = json.load(fh)
        for item in jobj.get('detailed_results', []):
            fname = item.get('filename')
            et = item.get('execution_time')
            if fname is None or et is None:
                continue
            try:
                per_image[fname].append(float(et))
            except Exception:
                continue
    return per_image


def write_csv(per_image, out_path):
    with open(out_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ['filename', 'avg_execution_time', 'n_runs', 'std_dev'])
        for fname in sorted(per_image.keys()):
            vals = per_image[fname]
            avg = safe_mean(vals)
            std = safe_std(vals)
            writer.writerow([fname, f'{avg:.6f}', len(vals), f'{std:.6f}'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', '-i', default=os.path.join('eval_results', 'Entire'),
                   help='Directory containing JSON run files (default: eval_results/Entire)')
    p.add_argument('--method-substr', '-m', required=True,
                   help='Substring to select runs, e.g. "g25_flash_only" or "g25_flash_pipepline"')
    p.add_argument('--output', '-o', default=None,
                   help='Output CSV filename (default will be avg_<method>.csv)')
    args = p.parse_args()

    out = args.output or f'avg_{args.method_substr}.csv'
    per_image = compute_avg_per_image(args.input_dir, args.method_substr)
    write_csv(per_image, out)
    print(f'Wrote {len(per_image)} image averages to {out}')


if __name__ == '__main__':
    main()
