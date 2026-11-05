#!/usr/bin/env python
"""Chunked, resumable prediction for one cell line (survives SIGKILLs).
Usage: chunked_predict.py <ct> <run_dir> <ckpt_name> <toml> <out_h5>
"""
import sys, os, time
import numpy as np
import torch
import yaml

ct, run_dir, ckpt_name, toml_path, out_h5 = sys.argv[1:6]

from cell_load.data_modules import PerturbationDataModule
from cell_load.config import ExperimentConfig

cfg = yaml.safe_load(open(f"{run_dir}/config.yaml"))
cfg["data"]["kwargs"]["toml_config_path"] = toml_path

dm = PerturbationDataModule.load_state(f"{run_dir}/data_module.torch")
dm.toml_config_path = toml_path
dm.config = ExperimentConfig.from_toml(toml_path)
dm.config.validate()
dm.train_datasets, dm.val_datasets, dm.test_datasets = [], [], []
_force_map = os.environ.get("FORCE_PERT_MAP")
if _force_map:
    # make _setup_global_maps take the featurization path (official 1138-pert map),
    # otherwise it regenerates a map from only the perts present in this h5ad
    dm.perturbation_features_file = _force_map
dm._setup_global_maps()
dm.setup(stage="test")

from state.tx.models.state_transition import StateTransitionPerturbationModel
var_dims = dm.get_var_dims()
model = StateTransitionPerturbationModel.load_from_checkpoint(
    f"{run_dir}/checkpoints/{ckpt_name}", weights_only=False,
    input_dim=var_dims["input_dim"], output_dim=var_dims["output_dim"],
    pert_dim=var_dims["pert_dim"], batch_dim=var_dims["batch_dim"],
    gene_dim=var_dims["hvg_dim"] if cfg["data"]["kwargs"]["output_space"] == "gene" else var_dims["gene_dim"],
    hvg_dim=var_dims["hvg_dim"],
)
model.eval().cuda()

loader = dm.test_dataloader()
n_batches = len(loader)
print("test batches:", n_batches, flush=True)

import h5py
G = var_dims["output_dim"]
KEYS = ["pert_name", "cell_name", "plate", "ctrl_barcode", "pert_barcode"]
# NOTE(anti-SIGKILL resume logic, disabled for the clean copy):
# if os.path.exists(out_h5):
#     hf = h5py.File(out_h5, "a")
#     done = int(hf.attrs.get("done_batches", 0))
#     print("resume from batch", done, flush=True)
# else:
if True:
    hf = h5py.File(out_h5, "w")
    hf.create_dataset("pred", shape=(0, G), maxshape=(None, G), dtype="float32", chunks=(256, G))
    hf.create_dataset("real", shape=(0, G), maxshape=(None, G), dtype="float32", chunks=(256, G))
    dt = h5py.special_dtype(vlen=str)
    for k in KEYS:
        hf.create_dataset(k, shape=(0,), maxshape=(None,), dtype=dt)
    done = 0


def ensure_list(values, n):
    if values is None:
        return ["?"] * n
    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            if isinstance(v, (list, tuple)):
                out.extend([str(x) for x in v])
            else:
                out.append(str(v))
        return out if len(out) == n else (out * (n // max(len(out), 1) + 1))[:n]
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
        return [str(x) for x in values.tolist()]
    return [str(values)] * n


t0 = time.time()
with torch.no_grad():
    for i, batch in enumerate(loader):
        # if i < done:  # anti-SIGKILL resume (disabled in clean copy)
        #     continue
        batch = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.predict_step(batch, i, padded=False)
        pred = out["preds"].reshape(-1, G).float().cpu().numpy()
        real = out["pert_cell_emb"].reshape(-1, G).float().cpu().numpy()
        nb = pred.shape[0]
        for name, arr in [("pred", pred), ("real", real)]:
            hf[name].resize(hf[name].shape[0] + nb, axis=0)
            hf[name][-nb:] = arr
        meta = {
            "pert_name": ensure_list(out.get("pert_name"), nb),
            "cell_name": ensure_list(out.get("celltype_name"), nb),
            "plate": ensure_list(batch.get("batch_name"), nb),
            "ctrl_barcode": ensure_list(out.get("ctrl_cell_barcode"), nb),
            "pert_barcode": ensure_list(out.get("pert_cell_barcode"), nb),
        }
        for k, vals in meta.items():
            hf[k].resize(hf[k].shape[0] + nb, axis=0)
            hf[k][-nb:] = vals
        hf.flush()
        done = i + 1
        # hf.attrs["done_batches"] = done  # anti-SIGKILL resume marker (disabled in clean copy)
        if done % 100 == 0:
            print(f"{done}/{n_batches} {time.time()-t0:.0f}s", flush=True)

hf.attrs["done_batches"] = done
hf.close()
print("CHUNKED_PREDICT_DONE", done, flush=True)
