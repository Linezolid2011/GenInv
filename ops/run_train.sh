#!/bin/bash
# Launch a training run with a crash-restart wrapper (200k steps by default).
#
# Usage:
#   bash ops/run_train.sh <name> [gpu] [extra train_dual.py args...]
#
# Detach it so it survives the session:
#   setsid nohup bash ops/run_train.sh <name> <gpu> >> ops/run_train.log 2>&1 &
#
# Data: defaults to /dev/shm/tahoe50 (fast). Use
#   DUAL_TOML=configs/generalization_tahoe50.toml bash ops/run_train.sh ...
# to train straight off the shared filesystem instead.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: run_train.sh <run_name> [gpu] [extra args...]}"
GPU="${2:-0}"
shift 2 2>/dev/null || shift 1
PY="$ROOT/venvs/state/bin/python"
mkdir -p "$ROOT/runs"

cd "$ROOT" || exit 1
for t in $(seq 1 300); do
  CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=offline CELL_LOAD_H5_RDCC_NBYTES=8388608 \
  DUAL_TOML="${DUAL_TOML:-configs/generalization_tahoe50_shm.toml}" \
  "$PY" scripts/train_dual.py --name "$NAME" --arch inv --lambda-dual 1.0 \
    --lambda-cycle 0.0 --var-anchor-weight 0.0 \
    --max-steps 200000 --val-freq 4000 --num-workers 24 --devices 1 --batch-size 64 \
    "$@" >> "$ROOT/runs/$NAME.log" 2>&1 && break
  echo "[$(date -Is)] restart $t" >> "$ROOT/runs/$NAME.log"
  sleep 30
done
echo "[$(date -Is)] TRAIN_LOOP_EXIT" >> "$ROOT/runs/$NAME.log"
