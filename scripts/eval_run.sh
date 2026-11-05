#!/bin/bash
# Full eval pipeline for a trained run:
#   strip ckpt -> chunked_predict (5 cell lines) -> chunked_de -> light_eval -> compare vs official baseline.
# Pred matrices are staged in /dev/shm and deleted right after each cell line is evaluated.
# Usage: scripts/eval_run.sh <run_name> <ckpt_path> [gpu] [eval_dir]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/venvs/state/bin/python}"
BASE_EVAL="${BASE_EVAL:-$ROOT/reference/official_eval}"

NAME="$1"; CKPT="$2"; GPU="${3:-0}"
RUN="$ROOT/runs/$NAME"
EV="${4:-$ROOT/eval_$NAME}"
mkdir -p "$EV"

# --- populate per-cell-line shm eval data (volatile /dev/shm may have been wiped) ---
SRC_H5="${SRC_H5:-/vepfs-mlp2/queue013/public/ybh/ybh4/data/state_tahoe_h5ad}"  # dataset path (kept absolute by design; override via env)
for CT in C32 HOP62 HepG2-C3A "Hs 766T" PANC-1; do
  SHM_DIR="/dev/shm/dual_pd_$CT"
  mkdir -p "$SHM_DIR"
  if [ ! -f "$SHM_DIR/$CT.h5ad" ]; then
    echo "populating $SHM_DIR from $SRC_H5"
    cp -a "$SRC_H5/$CT.h5ad" "$SHM_DIR/$CT.h5ad" || echo "WARN: shm populate failed for $CT"
    cp -a "$SRC_H5/hvg_names_ordered.json" "$SHM_DIR/" 2>/dev/null || true
  fi
done

STRIPPED="$RUN/checkpoints/$(basename "${CKPT%.ckpt}")_stripped.ckpt"
if [ ! -f "$STRIPPED" ]; then
  for t in $(seq 1 50); do
    "$PY" "$ROOT/scripts/strip_ckpt.py" "$CKPT" "$STRIPPED" && break; sleep 10
  done
fi

for CT in C32 HOP62 HepG2-C3A "Hs 766T" PANC-1; do
  SAFE=$(echo "$CT" | tr ' ' '_')
  PRED="/dev/shm/pred_${NAME}_${SAFE}.h5"
  echo "=== $CT: predict ==="
  for t in $(seq 1 200); do
    CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/chunked_predict.py" "$CT" "$RUN" \
      "$(basename "$STRIPPED")" "$ROOT/configs/toml_shm_$CT.toml" "$PRED" && break; sleep 10
  done
  echo "=== $CT: DE ==="
  for t in $(seq 1 100); do
    "$PY" "$ROOT/scripts/chunked_de.py" "$CT" "$PRED" "$RUN/de_$SAFE" "$EV/${CT}_pred_de.csv" 6 20 pred && break; sleep 10
  done
  if [ ! -f "$EV/${CT}_real_de.csv" ]; then
    if [ -f "$BASE_EVAL/${CT}_real_de.csv" ]; then
      cp "$BASE_EVAL/${CT}_real_de.csv" "$EV/${CT}_real_de.csv"
    else
      echo "FATAL: missing ground-truth DE table $BASE_EVAL/${CT}_real_de.csv" >&2
      echo "       (the large reference/*_real_de.csv files are not tracked in git;" >&2
      echo "        set BASE_EVAL, or regenerate them with the official State evaluation)" >&2
      exit 1
    fi
  fi
  echo "=== $CT: light_eval ==="
  for t in $(seq 1 100); do
    "$PY" "$ROOT/scripts/light_eval.py" "$CT" "$PRED" "$EV" "$EV/${CT}_light_agg.csv" && break; sleep 10
  done
  rm -f "$PRED"
done

# reshape light_agg series -> <ct>_agg_results.csv (single "mean" row) so compare.py can read it
"$PY" - "$EV" <<'EOF'
import sys, pandas as pd
ev = sys.argv[1]
for ct in ["C32", "HOP62", "HepG2-C3A", "Hs 766T", "PANC-1"]:
    s = pd.read_csv(f"{ev}/{ct}_light_agg.csv", index_col=0).iloc[:, 0]
    s.to_frame("mean").T.to_csv(f"{ev}/{ct}_agg_results.csv")
print("reshaped to agg_results.csv")
EOF

"$PY" "$ROOT/scripts/compare.py" "$EV" | tee "$EV/compare_vs_official.txt"
echo "EVAL_RUN_DONE $NAME"
