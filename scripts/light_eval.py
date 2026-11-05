#!/usr/bin/env python
"""Memory-light, resumable eval: expression metrics computed from per-pert means (chunked),
DE metrics via cell-eval pipeline from precomputed DE CSVs. Mirrors cell-eval profile=full
(minus pearson_edistance/clustering_agreement, as in the baseline).
Usage: light_eval.py <ct> <pred.h5> <eval_dir> <out_csv>
"""
import sys, os, time
import numpy as np
import pandas as pd
import h5py

ct, pred_h5, ev, out_csv = sys.argv[1:5]
ct_data = {"HepG2-C3A": "HepG2/C3A"}.get(ct, ct)
CTRL = "[('DMSO_TF', 0.0, 'uM')]"

f = h5py.File(pred_h5, "r")
cn = f["cell_name"].asstr()[:]
pn = f["pert_name"].asstr()[:]
m = np.where(cn == ct_data)[0]
pn_m = pn[m]
perts = np.array(sorted(p for p in pd.unique(pn_m) if p != CTRL))
G = f["pred"].shape[1]
print(ct, "cells:", len(m), "perts:", len(perts), flush=True)

cache = f"{ev}/{ct}_means.npz"
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    pm_pred, pm_real, cm = z["pm_pred"], z["pm_real"], z["cm"]
else:
    idx = np.argsort(m)
    m_sorted = m[idx]
    pn_s = pn_m[idx]
    pm_pred = np.zeros((len(perts), G), dtype=np.float64)
    pm_real = np.zeros((len(perts), G), dtype=np.float64)
    cnt = np.zeros(len(perts), dtype=np.int64)
    cm = np.zeros(G, dtype=np.float64)
    nctrl = 0
    pidx = {p: i for i, p in enumerate(perts)}
    B = 200000
    for i0 in range(0, len(m_sorted), B):
        sel = m_sorted[i0:i0 + B]
        lab = pn_s[i0:i0 + B]
        Xp = np.clip(f["pred"][sel], 0.0, 14.0)
        Xr = np.clip(f["real"][sel], 0.0, 14.0)
        for p in np.unique(lab):
            if p == CTRL:
                continue
            mm = lab == p
            j = pidx[p]
            pm_pred[j] += Xp[mm].sum(0)
            pm_real[j] += Xr[mm].sum(0)
            cnt[j] += mm.sum()
        if (i0 // B) % 10 == 0:
            print("means", i0, flush=True)
    pm_pred /= np.maximum(cnt, 1)[:, None]
    pm_real /= np.maximum(cnt, 1)[:, None]
    # control means computed separately per matrix
    cm_pred = np.zeros(G); cm_real = np.zeros(G); nctrl = 0
    for i0 in range(0, len(m_sorted), B):
        sel = m_sorted[i0:i0 + B]
        lab = pn_s[i0:i0 + B]
        mm = lab == CTRL
        if mm.sum():
            Xp = np.clip(f["pred"][sel[mm]], 0.0, 14.0); Xr = np.clip(f["real"][sel[mm]], 0.0, 14.0)
            cm_pred += Xp.sum(0); cm_real += Xr.sum(0); nctrl += mm.sum()
    cm_pred /= max(nctrl, 1); cm_real /= max(nctrl, 1)
    cm = np.stack([cm_pred, cm_real])
    np.savez(cache, pm_pred=pm_pred, pm_real=pm_real, cm=cm, perts=perts)
cm_pred, cm_real = cm[0], cm[1]

from scipy.stats import pearsonr
import sklearn.metrics as skm

eff_pred = pm_pred - cm_pred
eff_real = pm_real - cm_real

res = {}
res["pearson_delta"] = float(np.mean([pearsonr(eff_pred[i], eff_real[i]).correlation for i in range(len(perts))]))
res["mse"] = float(np.mean([skm.mean_squared_error(pm_pred[i], pm_real[i]) for i in range(len(perts))]))
res["mae"] = float(np.mean([skm.mean_absolute_error(pm_pred[i], pm_real[i]) for i in range(len(perts))]))
res["mse_delta"] = float(np.mean([skm.mean_squared_error(eff_pred[i], eff_real[i]) for i in range(len(perts))]))
res["mae_delta"] = float(np.mean([skm.mean_absolute_error(eff_pred[i], eff_real[i]) for i in range(len(perts))]))
for metric_name, metric in [("discrimination_score_l1", "cityblock"), ("discrimination_score_l2", "euclidean"), ("discrimination_score_cosine", "cosine")]:
    D = skm.pairwise_distances(eff_real, eff_pred, metric=metric)
    ranks = np.argsort(D, axis=0).argsort(axis=0)[np.arange(len(perts)), np.arange(len(perts))]
    res[metric_name] = float(np.mean(1.0 - ranks / len(perts)))
print("expression metrics done", flush=True)

# ---- DE metrics from CSVs (direct formulas, matches cell-eval exactly) ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this package's scripts/
from de_metrics_direct import compute_de_metrics

res.update(compute_de_metrics(f"{ev}/{ct}_real_de.csv", f"{ev}/{ct}_pred_de.csv"))
print("de metrics done", flush=True)

pd.Series(res).to_csv(out_csv)
print("LIGHT_EVAL_DONE", out_csv, flush=True)
for k in sorted(res):
    print(f"{k},{res[k]}")
