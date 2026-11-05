#!/usr/bin/env python
"""Inference-time probe: is the inverse map g(x_pert, p_bar) drug-specific?

For each test batch (one cell line), compute:
  - dual correct:  g(x_pert, p_true)   vs real control
  - dual wrong:    g(x_pert, p_wrong)  vs real control   (R draws)
  - primal correct: f(x_ctrl, p_true)  vs real pert      (sanity)
  - primal wrong:   f(x_ctrl, p_wrong) vs real pert      (sanity)
Metric: Pearson between predicted and real pseudobulk (X_hvg space).

Usage: probe_inv_specificity.py <ct> <run_dir> <ckpt_path> <toml> <out_csv> [n_batches] [R]
"""
import sys, os
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ct, run_dir, ckpt_path, toml_path, out_csv = sys.argv[1:6]
n_batches = int(sys.argv[6]) if len(sys.argv) > 6 else 40
R = int(sys.argv[7]) if len(sys.argv) > 7 else 4

sys.path.insert(0, ROOT)

from cell_load.data_modules import PerturbationDataModule
from cell_load.config import ExperimentConfig

cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
cfg["data"]["kwargs"]["toml_config_path"] = toml_path

dm = PerturbationDataModule.load_state(f"{run_dir}/data_module.torch")
dm.toml_config_path = toml_path
dm.config = ExperimentConfig.from_toml(toml_path)
dm.config.validate()
dm.train_datasets, dm.val_datasets, dm.test_datasets = [], [], []
dm.perturbation_features_file = os.environ.get(
    "FORCE_PERT_MAP", os.path.join(ROOT, "configs", "pert_onehot_map.pt"))
dm._setup_global_maps()
dm.setup(stage="test")

from dual_state.model import DualStateTransitionPerturbationModel
var_dims = dm.get_var_dims()
model = DualStateTransitionPerturbationModel.load_from_checkpoint(
    ckpt_path, weights_only=False,
    input_dim=var_dims["input_dim"], output_dim=var_dims["output_dim"],
    pert_dim=var_dims["pert_dim"], batch_dim=var_dims["batch_dim"],
    gene_dim=var_dims["hvg_dim"] if cfg["data"]["kwargs"]["output_space"] == "gene" else var_dims["gene_dim"],
    hvg_dim=var_dims["hvg_dim"],
)
model.eval().cuda()
S = model.cell_sentence_len
G = var_dims["output_dim"]
pert_map = model.pert_onehot_map if hasattr(model, "pert_onehot_map") else None
if pert_map is None:
    pert_map = torch.load(dm.perturbation_features_file, map_location="cpu", weights_only=False)
pert_names = list(pert_map.keys())
print(f"model loaded. S={S} G={G} perts={len(pert_names)}", flush=True)

rng = np.random.default_rng(0)

def wrong_onehot(true_oh):
    # true_oh: [B*S, pert_dim] one-hot rows; return same shape with a different pert
    idx_true = true_oh.argmax(dim=-1).cpu().numpy()
    out = true_oh.clone()
    for i in np.unique(idx_true):
        cand = [j for j in range(true_oh.shape[-1]) if j != i]
        j = rng.choice(cand)
        out[idx_true == i] = 0.0
        out[idx_true == i, j] = 1.0
    return out

def pb(x):  # pseudobulk over all cells (variable-length, padded=False)
    return x.reshape(-1, G).mean(dim=0).detach().cpu().numpy()

def pear(a, b):
    return float(np.corrcoef(a, b)[0, 1])

rows = []
loader = dm.test_dataloader()
with torch.no_grad():
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        for k in ["pert_cell_emb", "ctrl_cell_emb", "pert_emb", "batch"]:
            batch[k] = batch[k].cuda() if torch.is_tensor(batch[k]) else batch[k]
        perts_here = batch.get("pert_name")
        # delta space: predicted change relative to the INPUT population
        in_pert = pb(batch["pert_cell_emb"])   # dual input
        in_ctrl = pb(batch["ctrl_cell_emb"])   # primal input
        d_real_dual = pb(batch["ctrl_cell_emb"]) - in_pert
        d_real_primal = pb(batch["pert_cell_emb"]) - in_ctrl
        true_pear_dual = pear(pb(model._dual_forward(batch, "pert_cell_emb", padded=False)) - in_pert, d_real_dual)
        true_pear_primal = pear(pb(model.forward(batch, padded=False)) - in_ctrl, d_real_primal)
        wd, wp = [], []
        for r in range(R):
            b2 = dict(batch)
            b2["pert_emb"] = wrong_onehot(batch["pert_emb"])
            wd.append(pear(pb(model._dual_forward(b2, "pert_cell_emb", padded=False)) - in_pert, d_real_dual))
            wp.append(pear(pb(model.forward(b2, padded=False)) - in_ctrl, d_real_primal))
        rows.append(dict(batch=bi, pert=str(perts_here[0] if isinstance(perts_here, (list, tuple)) else perts_here),
                         dual_correct=true_pear_dual, dual_wrong=float(np.mean(wd)),
                         primal_correct=true_pear_primal, primal_wrong=float(np.mean(wp))))
        print(f"batch {bi}: dual {true_pear_dual:.4f}/{np.mean(wd):.4f}  primal {true_pear_primal:.4f}/{np.mean(wp):.4f}", flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(out_csv, index=False)
print("SUMMARY", flush=True)
for c in ["dual_correct", "dual_wrong", "primal_correct", "primal_wrong"]:
    print(f"  {c}: {df[c].mean():.4f} (n={len(df)})", flush=True)
print("PROBE_DONE", out_csv, flush=True)
