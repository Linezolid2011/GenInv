#!/usr/bin/env python
"""DE metrics directly from pred/real DE CSVs, replicating cell-eval formulas exactly."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_de_metrics(real_csv, pred_csv, fdr_threshold=0.05):
    real = pd.read_csv(real_csv)
    pred = pd.read_csv(pred_csv)
    real["abs_lfc"] = real["log2_fold_change"].abs()
    pred["abs_lfc"] = pred["log2_fold_change"].abs()
    targets = sorted(real["target"].unique())
    K = [50, 100, 200, 500]

    res = {f"overlap_at_{k}": [] for k in K}
    res.update({f"precision_at_{k}": [] for k in K})
    res.update(overlap_at_N=[], precision_at_N=[], de_direction_match=[], de_spearman_lfc_sig=[],
               de_sig_genes_recall=[], de_nsig_counts_real=[], de_nsig_counts_pred=[],
               pr_auc=[], roc_auc=[])
    nsig_real_all, nsig_pred_all = [], []

    for t in targets:
        r = real[real["target"] == t]
        p = pred[pred["target"] == t]
        rs = r[r["fdr"] < fdr_threshold]
        ps = p[p["fdr"] < fdr_threshold]
        rs_sorted = rs.sort_values("abs_lfc", ascending=False)
        ps_sorted = ps.sort_values("abs_lfc", ascending=False)
        nr, np_ = len(rs), len(ps)
        nsig_real_all.append(nr)
        nsig_pred_all.append(np_)
        for k in K:
            k_eff_r = min(k, nr)
            k_eff_p = min(k, np_)
            res[f"overlap_at_{k}"].append(
                len(set(rs_sorted["feature"][:k_eff_r]) & set(ps_sorted["feature"][:k_eff_r])) / k_eff_r if k_eff_r else 0.0)
            res[f"precision_at_{k}"].append(
                len(set(rs_sorted["feature"][:k_eff_p]) & set(ps_sorted["feature"][:k_eff_p])) / k_eff_p if k_eff_p else 0.0)
        k_r, k_p = nr, np_
        res["overlap_at_N"].append(
            len(set(rs_sorted["feature"][:k_r]) & set(ps_sorted["feature"][:k_r])) / k_r if k_r else 0.0)
        res["precision_at_N"].append(
            len(set(rs_sorted["feature"][:k_p]) & set(ps_sorted["feature"][:k_p])) / k_p if k_p else 0.0)
        # direction match: real sig joined with full pred frame
        pj = p.set_index("feature")["log2_fold_change"]
        common = rs["feature"].isin(pj.index)
        feat = rs["feature"][common]
        res["de_direction_match"].append(
            float((np.sign(rs.set_index("feature").loc[feat, "log2_fold_change"]) == np.sign(pj.loc[feat])).mean()) if len(feat) else np.nan)
        # spearman lfc on real sig genes, missing pred -> 0
        lfc_r = rs.set_index("feature")["log2_fold_change"]
        lfc_p = pj.reindex(lfc_r.index).fillna(0.0)
        if lfc_r.nunique() > 1 and lfc_p.nunique() > 0:
            res["de_spearman_lfc_sig"].append(float(lfc_r.corr(lfc_p, method="spearman")))
        else:
            res["de_spearman_lfc_sig"].append(np.nan)
        # recall
        res["de_sig_genes_recall"].append(len(set(rs["feature"]) & set(ps["feature"])) / nr if nr else 0.0)
        # auc
        labels = (r["fdr"] < fdr_threshold).astype(float).values
        q = p.set_index("feature")["fdr"].reindex(r["feature"]).fillna(1.0).clip(1e-10, 1.0)
        scores = (-np.log10(q)).values
        if 0 < labels.sum() < len(labels):
            res["pr_auc"].append(float(average_precision_score(labels, scores)))
            res["roc_auc"].append(float(roc_auc_score(labels, scores)))
        else:
            res["pr_auc"].append(np.nan)
            res["roc_auc"].append(np.nan)

    out = {}
    for k, v in res.items():
        out[k] = float(np.nanmean(v)) if len(v) else np.nan
    out["de_nsig_counts_real"] = float(np.mean(nsig_real_all))
    out["de_nsig_counts_pred"] = float(np.mean(nsig_pred_all))
    # effect size: spearman of nsig counts across targets
    out["de_spearman_sig"] = float(pd.Series(nsig_real_all).corr(pd.Series(nsig_pred_all), method="spearman"))
    return out


if __name__ == "__main__":
    real_csv, pred_csv = sys.argv[1], sys.argv[2]
    out = compute_de_metrics(real_csv, pred_csv)
    for k in sorted(out):
        print(f"{k},{out[k]}")
