#!/usr/bin/env python
"""Compare Dual eval agg results against the State baseline (eval_best.ckpt) with the 80% rule.
Usage: python scripts/compare.py <eval_dir> [baseline_dir]"""
import os
import sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTS = ["C32", "HOP62", "HepG2-C3A", "Hs 766T", "PANC-1"]

dual_dir = sys.argv[1]
base_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "reference", "official_eval")

HIGHER_BETTER_EXCLUDE = {"mse", "mae", "mse_delta", "mae_delta", "de_nsig_counts_real", "de_nsig_counts_pred"}

rows = []
for ct in CTS:
    try:
        new = pd.read_csv(f"{dual_dir}/{ct}_agg_results.csv", index_col=0).loc["mean"]
        old = pd.read_csv(f"{base_dir}/{ct}_agg_results.csv", index_col=0).loc["mean"]
    except FileNotFoundError as e:
        print("missing:", e)
        continue
    for m in old.index:
        o, n = old[m], new.get(m, np.nan)
        if pd.isna(o) or pd.isna(n):
            continue
        if m in HIGHER_BETTER_EXCLUDE:
            better = n < o
            floor = n <= o / 0.8 if o != 0 else True  # not worse than 1/0.8x
        else:
            better = n > o
            floor = n >= 0.8 * o
        rows.append((ct, m, o, n, better, floor))

df = pd.DataFrame(rows, columns=["ct", "metric", "state", "dual", "better", "above_floor"])
n_total = len(df)
if n_total == 0:
    print("no metrics compared (missing agg_results.csv?)")
    sys.exit(1)
n_better = int(df["better"].sum())
print(f"metrics compared: {n_total}")
print(f"dual better: {n_better} ({n_better/n_total*100:.1f}%)  [target >= 80%]")
viol = df[~df["above_floor"]]
print(f"below floor (worse than 80% of state): {len(viol)}")
if len(viol):
    print(viol.to_string(index=False))
print()
pivot = df.pivot_table(index="metric", values="better", aggfunc="mean")
worst = df.assign(ratio=df["dual"] / df["state"].replace(0, np.nan)).sort_values("ratio").head(15)
print("worst 15 metric-celltype by dual/state ratio:")
print(worst[["ct", "metric", "state", "dual"]].to_string(index=False))
