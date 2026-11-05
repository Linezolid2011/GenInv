# GenInv

Generative **inverse-mapping dual learning** for single-cell perturbation response
prediction, built on top of the official **State** transition model (arc-institute)
and the Tahoe-50 ("State-Tahoe-Filtered") training/evaluation pipeline.

A perturbation experiment measures a control population and a perturbed population
under the same drug, so the same data supports both directions. GenInv trains the
primal map `f: (x_ctrl, p) → x_pert` (standard State) *plus* an inverse map
`g: (x_pert, p̄) → x_ctrl` that shares the gene decoder but uses a separate inverse
perturbation encoder; both directions use the same energy-MMD objective on the same
batches, so no extra data is needed.

This repository contains **code + vendored State + the official reference metrics**,
and documents how to build the Python environment it was developed with. Datasets and
checkpoints are not included.

## Layout

```
├── dual_state/                 # model classes: DualState (GenInv) + discriminative baselines
│   └── model.py
├── scripts/
│   ├── train_dual.py           #   patched official State trainer (arch: inv/cls/clsout/clsout2/invsplit)
│   ├── eval_run.sh             #   5-cell-line eval (strip → predict → DE → metrics → compare)
│   ├── eval_run_one.sh         #   same pipeline restricted to a single cell line
│   ├── chunked_predict.py      #   chunked prediction for one cell line (resumable)
│   ├── chunked_de.py           #   chunked Wilcoxon DE on predictions
│   ├── light_eval.py           #   expression + DE metrics (cell-eval-compatible)
│   ├── de_metrics_direct.py    #   DE metric formulas (imported by light_eval)
│   ├── compare.py              #   compare an eval dir against the official State baseline
│   ├── strip_ckpt.py           #   drop dual-only weights so stock State can load the ckpt
│   └── probe_inv_specificity.py#   inference-time probe: does the inverse map read the drug label?
├── configs/
│   ├── generalization_tahoe50.toml      # 50-line split, data on the shared filesystem
│   ├── generalization_tahoe50_shm.toml  # same split, data staged in /dev/shm
│   ├── toml_shm_<CT>.toml               # per-cell-line eval configs (5 held-out lines)
│   ├── official_config.yaml             # official State training config template
│   └── pert_onehot_map.pt               # official 1138-dim perturbation one-hot map
├── reference/official_eval/    # official State baseline: {CT}_agg_results.csv (tracked)
│                               #   {CT}_real_de.csv ground truth (not tracked, see below)
├── state/                      # vendored official State repo (imported as `state`)
├── environment/                # pip freeze + rebuild script
├── ops/                        # long-run helpers (launcher, supervisor, auto-eval)
└── README.md
```

## Environment

`environment/pip-freeze.txt` pins the 152 packages used for all reported runs
(Python 3.11.15, torch 2.7.1+cu126). Rebuild it with:

```bash
bash environment/setup_env.sh            # or: environment/setup_env.sh /path/to/python3.11
./venvs/state/bin/python -c "import torch, state; print(torch.__version__, torch.cuda.is_available())"
```

`setup_env.sh` creates `venvs/state`, installs the pinned packages, then installs the
vendored `state/` package in editable mode. Any venv satisfying the freeze works —
the scripts only assume `state` is importable and (by default) look for the interpreter
at `venvs/state/bin/python`; override with `PY=/path/to/python` for `eval_run*.sh`.

## Data

Datasets are not part of this repository. The pipeline reads two dataset roots:

| Purpose | Contents | Configured in |
|---|---|---|
| Training set | 50 per-context h5ad (`State-Tahoe-Filtered`), plus `.rechunked` markers | `configs/generalization_tahoe50.toml` |
| Training set, `/dev/shm` staging copy | same 50 files | `configs/generalization_tahoe50_shm.toml` |
| Evaluation set | 5 full cell-line h5ad + `hvg_names_ordered.json` | `scripts/eval_run.sh` / `eval_run_one.sh` (`SRC_H5`) |

`eval_run.sh` stages each cell line into `/dev/shm/dual_pd_<CT>/` on demand and deletes
the prediction file after that line is scored. Override the eval root without editing
files:

```bash
SRC_H5=/path/to/state_tahoe_h5ad bash scripts/eval_run.sh <run> <ckpt> <gpu>
```

For fast training, stage the data in memory first:

```bash
mkdir -p /dev/shm/tahoe50 && cp -a /path/to/tahoe50/. /dev/shm/tahoe50/
```

### Absolute paths kept on purpose

Everything in the source tree resolves relative to the repository root. The only
absolute paths are dataset locations (they are machine-specific and cannot be
relativized without copying the data in):

| File | Line | Value | Notes |
|---|---|---|---|
| `configs/generalization_tahoe50.toml` | 4 | `tahoe_holdout = <training-data-root>/` | edit this line to point at your copy |
| `scripts/eval_run.sh` | 17 | `SRC_H5` default = `<eval-data-root>` | override with the `SRC_H5` env var |
| `scripts/eval_run_one.sh` | 21 | `SRC_H5` default = `<eval-data-root>` | override with the `SRC_H5` env var |

The `/dev/shm/...` entries in `configs/generalization_tahoe50_shm.toml` and
`configs/toml_shm_<CT>.toml`, and in the shell scripts, are pipeline scratch space
(populated by the scripts themselves), not data locations.

## Quickstart

Run from the repository root: `data_module.torch` stores section names relative to the
working directory, so scripts that call `load_state` must see `configs/` there.

```bash
# GenInv, 200k steps, one A100-80G, batch 64
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline CELL_LOAD_H5_RDCC_NBYTES=8388608 \
DUAL_TOML=configs/generalization_tahoe50_shm.toml \
./venvs/state/bin/python scripts/train_dual.py --name geninv_run --arch inv --lambda-dual 1.0 \
  --lambda-cycle 0.0 --var-anchor-weight 0.0 --max-steps 200000 --val-freq 4000 \
  --num-workers 24 --devices 1 --batch-size 64

# vanilla State baseline (same trainer, dual weights off)
#   --arch cls --dual-cls-weight 0.0 --var-anchor-weight 0.0

# Evaluation on the 5 held-out cell lines
bash scripts/eval_run.sh geninv_run runs/geninv_run/checkpoints/step200000.ckpt 0
# -> eval_geninv_run/{CT}_agg_results.csv + compare_vs_official.txt

# Evaluation on a single cell line
bash scripts/eval_run_one.sh geninv_run runs/geninv_run/checkpoints/step200000.ckpt PANC-1 0
```

`--arch`: `inv` (GenInv, default), `cls` / `clsout` / `clsout2` (discriminative dual
variants, reported as negative results), `invsplit` (perturbation-split dual, negative
result).

Runtime is roughly 2 s/step on one A100-80G with the data in `/dev/shm`, i.e. ~4.5 days
for 200k steps; a full 5-line evaluation takes 5–6 h on one GPU.

### Long runs — `ops/`

```bash
# crash-restart wrapper (setsid-detached); writes runs/<name>.log
setsid nohup bash ops/run_train.sh <name> <gpu> >> ops/run_train.log 2>&1 &
# checkpoint + metric snapshots, and a memory guard
setsid nohup ./venvs/state/bin/python ops/supervise.py --name <name> >> ops/supervise.log 2>&1 &
# automatic 5-line evaluation at 100k and 200k, on a second GPU
setsid nohup ./venvs/state/bin/python ops/watch_eval.py --name <name> --gpu 1 >> ops/watch_eval.log 2>&1 &
```

The supervisor keeps an append-only copy of `metrics.csv` (Lightning truncates it when
training resumes, and `last.ckpt` is only written every `val_freq`) and materialises
`checkpoints/step<N>.ckpt` at multiples of 20k, which is what the eval watcher waits for.

## Reference tables

`reference/official_eval/` holds the official State evaluation baselines:

| File | Size | Tracked |
|---|---|---|
| `{CT}_agg_results.csv` (5 cell lines) | ~3.5 KB each | yes |
| `{CT}_real_de.csv` (5 cell lines) | ~225 MB each (~1.1 GB total) | **no** |

The `real_de` tables are the ground truth for the DE metrics (`light_eval.py` needs them)
and are plain CSVs of the official run's per-gene DE statistics; they are reproduced by
running the official State checkpoint through `scripts/eval_run.sh` once. They are too
large for Git, so they are gitignored.

If they are missing, `eval_run.sh` stops with an explicit `FATAL: missing ground-truth DE
table` message. Provide them by either:

- pointing `BASE_EVAL` at a directory that contains `{CT}_real_de.csv`, or
- generating them with the official State checkpoint and this pipeline.

`compare.py` only needs the small `{CT}_agg_results.csv`, so the baseline comparison works
out of the box.

## Results

Five held-out cell lines, Cell-Eval-compatible metrics, 200k steps.

| model | Pearson-Δ | DE direction | DE LFC Spearman | DE recall |
|---|---|---|---|---|
| vanilla State (from scratch) | 0.9220 | 0.8982 | 0.6121 | 0.7012 |
| **GenInv** | **0.9312** | **0.9171** | **0.6646** | **0.7285** |

GenInv improves on all four metrics. Across independent runs the spread is small
(Pearson-Δ 0.9300–0.9329, direction 0.9121–0.9171, LFC Spearman 0.6487–0.6646,
recall 0.7232–0.7298).

## Notes / gotchas

- **Working directory**: scripts that call `load_state` (`chunked_predict.py`,
  `probe_inv_specificity.py`) must run from the repository root; the saved
  `data_module.torch` embeds section names relative to it.
- **Strip before eval**: GenInv checkpoints contain `pert_encoder_inv.*`. `eval_run.sh`
  runs `strip_ckpt.py` automatically; if you evaluate by hand, strip first.
- **Eval cache**: `{CT}_means.npz` is cached per eval directory. Never evaluate two
  different checkpoints into the same directory — use a fresh `eval_*` directory.
- **Read metrics by column name** (`pearson_delta`, `de_direction_match`,
  `de_spearman_lfc_sig`, `de_sig_genes_recall`), never by position: the column set changes
  between runs.
- The rich progress bar is disabled when stdout is not a tty; follow
  `runs/<name>/version_0/metrics.csv` instead.
- `metrics.csv` is truncated when training resumes from a checkpoint.

## License

The vendored `state/` directory is the official State code and carries its own terms:
see `state/LICENSE`, `state/MODEL_LICENSE.md` and `state/MODEL_ACCEPTABLE_USE_POLICY.md`.
