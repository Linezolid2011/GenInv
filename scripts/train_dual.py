#!/usr/bin/env python
"""Driver: train ST-HVG-Tahoe with the dual-learning objective.

Default behavior matches the official run: from scratch (init_from=null),
lr 3e-4, max_steps 200000, val_freq 4000, num_workers 36 (from config.yaml).
Pass --ckpt to fine-tune from a checkpoint instead.

Reuses state's official training entry (run_tx_train) with two patches:
1. model registry: "state" -> DualStateTransitionPerturbationModel
2. data.kwargs.perturbation_features_file -> official pert_onehot_map.pt,
   guaranteeing perturbation one-hot order identical to the checkpoint.

Usage (from repo root, state venv):
  python scripts/train_dual.py --name dual_v1 --arch cls --dual-cls-weight 0.1 \
      --var-anchor-weight 0.1 --devices 4 --batch-size 16
"""
import argparse
import copy
import os
import sys

SELF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (has configs/, scripts/)
OFFICIAL_CONFIG = os.path.join(SELF, "configs", "official_config.yaml")  # official State training config template

sys.path.insert(0, SELF)

import yaml
from omegaconf import OmegaConf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--lambda-dual", type=float, default=1.0)
    ap.add_argument("--lambda-cycle", type=float, default=0.0)
    ap.add_argument("--dual-shuffle", action="store_true",
                    help="wrong-drug ablation: shuffle pert labels within batch for the dual branch")
    ap.add_argument("--lr", type=float, default=3e-4)  # official: 3e-4
    ap.add_argument("--max-steps", type=int, default=200000)  # official: 200000
    ap.add_argument("--val-freq", type=int, default=4000)  # official: 4000
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="per-device batch size; official runs 1 device x 64")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="dataloader workers per device; default keeps official config value (36)")
    ap.add_argument("--ckpt", default=None,
                    help="init checkpoint for fine-tuning; default None = train from scratch (official behavior)")
    ap.add_argument("--resume-config", default=None, help="optional: reuse a previous Dual run's config.yaml")
    ap.add_argument("--arch", choices=["inv", "cls", "clsout", "clsout2", "invsplit"], default="inv",
                    help="inv: inverse-mapping dual (DualState); cls: DualCross hidden-diff classifier; clsout: output-delta classifier (shortcut-free); invsplit: inverse dual with perturbation-level split")
    ap.add_argument("--dual-cls-weight", type=float, default=0.1)
    ap.add_argument("--dual-pert-frac", type=float, default=0.2,
                    help="invsplit: fraction of perturbations that are dual-only")
    ap.add_argument("--dual-split-seed", type=int, default=42)
    ap.add_argument("--var-anchor-weight", type=float, default=0.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.resume_config or OFFICIAL_CONFIG))
    cfg["data"]["kwargs"]["toml_config_path"] = os.environ.get("DUAL_TOML", f"{SELF}/configs/generalization_tahoe50.toml")
    cfg["data"]["kwargs"]["perturbation_features_file"] = f"{SELF}/configs/pert_onehot_map.pt"
    if args.num_workers is not None:
        cfg["data"]["kwargs"]["num_workers"] = args.num_workers  # else keep official value (36)
    cfg["model"]["kwargs"]["init_from"] = args.ckpt  # None = from scratch, as in the official config
    cfg["model"]["kwargs"]["lambda_dual"] = args.lambda_dual
    cfg["model"]["kwargs"]["dual_shuffle_labels"] = args.dual_shuffle
    cfg["model"]["kwargs"]["lambda_cycle"] = args.lambda_cycle
    cfg["model"]["kwargs"]["dual_cls_weight"] = args.dual_cls_weight
    cfg["model"]["kwargs"]["dual_pert_frac"] = args.dual_pert_frac
    cfg["model"]["kwargs"]["dual_split_seed"] = args.dual_split_seed
    cfg["model"]["kwargs"]["var_anchor_weight"] = args.var_anchor_weight
    cfg["training"]["lr"] = args.lr
    cfg["training"]["max_steps"] = args.max_steps
    cfg["training"]["val_freq"] = args.val_freq
    cfg["training"]["ckpt_every_n_steps"] = args.val_freq
    cfg["training"]["devices"] = max(args.devices, 1)
    cfg["training"]["strategy"] = "ddp_find_unused_parameters_true" if args.devices > 1 else "auto"
    cfg["training"]["batch_size"] = args.batch_size
    cfg["use_wandb"] = False
    cfg["output_dir"] = f"{SELF}/runs"
    cfg["name"] = args.name
    cfg["overwrite"] = False

    # patch model registry
    import state.tx.utils as tx_utils
    from dual_state.model import DualCrossStateTransitionPerturbationModel, DualInvSplitStateTransitionPerturbationModel, DualOutCls2StateTransitionPerturbationModel, DualOutClsStateTransitionPerturbationModel, DualStateTransitionPerturbationModel

    model_cls = {
        "inv": DualStateTransitionPerturbationModel,
        "cls": DualCrossStateTransitionPerturbationModel,
        "clsout": DualOutClsStateTransitionPerturbationModel,
        "clsout2": DualOutCls2StateTransitionPerturbationModel,
        "invsplit": DualInvSplitStateTransitionPerturbationModel,
    }[args.arch]

    _orig_get = tx_utils.get_lightning_module

    def patched(model_type, data_config, model_config, training_config, var_dims):
        if model_type.lower() == "state":
            module_config = {**model_config, **training_config}
            module_config["embed_key"] = data_config["embed_key"]
            module_config["output_space"] = data_config["output_space"]
            module_config["gene_names"] = var_dims["gene_names"]
            module_config["batch_size"] = training_config["batch_size"]
            module_config["control_pert"] = data_config.get("control_pert", "non-targeting")
            gene_dim = var_dims["hvg_dim"] if data_config["output_space"] == "gene" else var_dims["gene_dim"]
            return model_cls(
                input_dim=var_dims["input_dim"],
                gene_dim=gene_dim,
                hvg_dim=var_dims["hvg_dim"],
                output_dim=var_dims["output_dim"],
                pert_dim=var_dims["pert_dim"],
                batch_dim=var_dims["batch_dim"],
                **module_config,
            )
        return _orig_get(model_type, data_config, model_config, training_config, var_dims)

    tx_utils.get_lightning_module = patched
    # run_tx_train imported get_lightning_module by name at module import time;
    # patch the reference inside _train as well
    import state._cli._tx._train as train_mod

    train_mod.get_lightning_module = patched

    from state._cli._tx._train import run_tx_train

    run_tx_train(OmegaConf.create(cfg))


if __name__ == "__main__":
    main()
