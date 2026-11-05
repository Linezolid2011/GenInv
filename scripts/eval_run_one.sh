#!/bin/bash
# Single-cell-line eval pipeline (PANC-1 only).
#
# Mirrors scripts/eval_run.sh exactly (strip -> chunked_predict -> chunked_de ->
# light_eval -> compare), but restricted to one cell line so we can validate a
# rebuilt dataset without spending hours on the other four.
#
# Usage: scripts/eval_run_one.sh <run_name> <ckpt_path> <cell_line> [gpu] [eval_dir]
#   cell_line must be one of: C32 HOP62 HepG2-C3A "Hs 766T" PANC-1
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/venvs/state/bin/python}"
BASE_EVAL="${BASE_EVAL:-$ROOT/reference/official_eval}"

NAME="$1"; CKPT="$2"; CT="${3:?cell line required}"; GPU="${4:-0}"
RUN="$ROOT/runs/$NAME"
EV="${5:-$ROOT/eval_${NAME}_$(echo "$CT" | tr ' ' '_')}"
SAFE=$(echo "$CT" | tr ' ' '_')
mkdir -p "$EV"

SRC_H5="${SRC_H5:-/vepfs-mlp2/queue013/public/ybh/ybh4/data/state_tahoe_h5ad}"

# populate only this cell line's shm dir
SHM_DIR="/dev/shm/dual_pd_$CT"
mkdir -p "$SHM_DIR"
if [ ! -f "$SHM_DIR/$CT.h5ad" ]; then
  echo "populating $SHM_DIR from $SRC_H5"
  cp -a "$SRC_H5/$CT.h5ad" "$SHM_DIR/$CT.h5ad" || { echo "FATAL: shm populate failed"; exit 1; }
  cp -a "$SRC_H5/hvg_names_ordered.json" "$SHM_DIR/" 2>/dev/null || true
fi
echo "shm data: $(ls -la "$SHM_DIR/$CT.h5ad" | awk '{print $5" bytes"}')"

STRIPPED="$RUN/checkpoints/$(basename "${CKPT%.ckpt}")_stripped.ckpt"
if [ ! -f "$STRIPPED" ]; then
  for t in $(seq 1 50); do
    "$PY" "$ROOT/scripts/strip_ckpt.py" "$CKPT" "$STRIPPED" && break; sleep 10
  done
fi

PRED="/dev/shm/pred_${NAME}_${SAFE}.h5"
echo "=== $CT: predict ==="
for t in $(seq 1 200); do
  CUDA_VISIBLE_DEVICES=$GPU "$PY" "$ROOT/scripts/chunked_predict.py" "$CT" "$RUN" \
    "$(basename "$STRIPPED")" "$ROOT/configs/toml_shm_$CT.toml" "$PRED" && break
  echo "retry $t"; sleep 10
done

echo "=== $CT: DE ==="
for t in $(seq 1 100); do
  "$PY" "$ROOT/scripts/chunked_de.py" "$CT" "$PRED" "$RUN/de_$SAFE" "$EV/${CT}_pred_de.csv" 6 20 pred && break
  echo "retry $t"; sleep 10
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
  "$PY" "$ROOT/scripts/light_eval.py" "$CT" "$PRED" "$EV" "$EV/${CT}_light_agg.csv" && break
  echo "retry $t"; sleep 10
done
rm -f "$PRED"

"$PY" - "$EV" "$CT" <<'EOF'
import sys, pandas as pd
ev, ct = sys.argv[1], sys.argv[2]
s = pd.read_csv(f"{ev}/{ct}_light_agg.csv", index_col=0).iloc[:, 0]
s.to_frame("mean").T.to_csv(f"{ev}/{ct}_agg_results.csv")
print("reshaped ->", f"{ev}/{ct}_agg_results.csv")
EOF

"$PY" "$ROOT/scripts/compare.py" "$EV" | tee "$EV/compare_vs_official.txt"
echo "SINGLE_EVAL_DONE $CT"
