#!/usr/bin/env python
"""Chunked (resumable) pred-DE computation with pdex, then merge.
Usage: chunked_de.py <ct> <pred.h5> <workdir> <out_csv> [threads] [chunk_size]
real/pred mode chosen by arg6: matrix=pred|real
"""
import sys, os, time
import numpy as np
import pandas as pd
import h5py
import anndata as ad

ct, pred_h5, workdir, out_csv = sys.argv[1:5]
ct_data = {"HepG2-C3A": "HepG2/C3A"}.get(ct, ct)
threads = int(sys.argv[5]) if len(sys.argv) > 5 else 6
chunk_size = int(sys.argv[6]) if len(sys.argv) > 6 else 60
matrix = sys.argv[7] if len(sys.argv) > 7 else "pred"
os.makedirs(workdir, exist_ok=True)
CTRL = "[('DMSO_TF', 0.0, 'uM')]"

f = h5py.File(pred_h5, "r")
cn = f["cell_name"].asstr()[:]
pn = f["pert_name"].asstr()[:]
m = np.where(cn == ct_data)[0]
perts_ct = pd.Categorical(pn[m]).categories.tolist()
targets = [p for p in perts_ct if p != CTRL]
print(ct, "cells:", len(m), "targets:", len(targets), flush=True)

import polars as pl
from pdex import pdex

G = f["pred"].shape[1]
var = pd.DataFrame(index=[str(i) for i in range(G)])
obs_all = pd.DataFrame({
    "drugname_drugconc": pd.Categorical(pn[m]),
    "cell_name": pd.Categorical(cn[m]),
})
obs_all.index = [f"cell_{i:07d}" for i in range(len(m))]

order = np.argsort(m)
m_sorted = m[order]
obs_all = obs_all.iloc[order].reset_index(drop=True)
pn_m = pn[m_sorted]

chunks = [targets[i:i + chunk_size] for i in range(0, len(targets), chunk_size)]
for ci, ch in enumerate(chunks):
    cf = f"{workdir}/chunk_{ci*chunk_size:04d}_{min((ci+1)*chunk_size, len(targets)):04d}.csv"
    # if os.path.exists(cf):  # anti-SIGKILL resume (disabled in clean copy)
    #     continue
    t0 = time.time()
    sel = np.where(np.isin(pn_m, ch) | (pn_m == CTRL))[0]
    X = np.clip(f[matrix][m_sorted[sel]], 0.0, 14.0)
    obs_sub = obs_all.iloc[sel].copy()
    obs_sub["drugname_drugconc"] = obs_sub["drugname_drugconc"].cat.remove_unused_categories()
    adata = ad.AnnData(X=X, obs=obs_sub, var=var)
    frame = pdex(
        adata=adata,
        mode="ref",
        reference=CTRL,
        groupby="drugname_drugconc",
        threads=threads,
        exp_post_agg=True,
        is_log1p=True,
    )
    frame.write_csv(cf)
    print(f"chunk {ci}/{len(chunks)} {time.time()-t0:.0f}s", flush=True)
    del X, adata, frame

import glob
files = sorted(glob.glob(f"{workdir}/chunk_*.csv"))
frames = [pl.read_csv(fp) for fp in files]
full = pl.concat(frames)
full.write_csv(out_csv)
print("DE_DONE", out_csv, full.shape, flush=True)
