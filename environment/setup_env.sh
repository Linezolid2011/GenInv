#!/bin/bash
# Rebuild the Python environment used by GenInv-cleaned-final.
#
# The repo ships a pre-built env at venvs/state (with an embedded interpreter at
# python/), so this script is only needed on a new machine or after a Python upgrade.
#
# Usage:
#   bash environment/setup_env.sh                 # uses python3.11 from PATH
#   bash environment/setup_env.sh /path/to/python3.11
#
# Requires: network access to PyPI (and the PyTorch cu126 wheel index), or a local mirror.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYBIN="${1:-python3.11}"
VENV="$ROOT/venvs/state"
REQ="$ROOT/environment/pip-freeze.txt"

if [ ! -f "$REQ" ]; then
  echo "missing $REQ" >&2
  exit 1
fi

echo "==> creating venv at $VENV with $PYBIN"
"$PYBIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install -U pip wheel

# `arc-state` is installed editable from ./state (see below), not from PyPI.
TMPREQ="$(mktemp)"
grep -v '^arc-state' "$REQ" > "$TMPREQ"

echo "==> installing dependencies (this pulls torch 2.7.1 + CUDA 12.6 wheels)"
"$VENV/bin/pip" install -r "$TMPREQ" \
  --extra-index-url https://download.pytorch.org/whl/cu126
rm -f "$TMPREQ"

echo "==> installing vendored State (editable)"
"$VENV/bin/pip" install -e "$ROOT/state" --no-deps

echo "==> verifying"
"$VENV/bin/python" -c "import sys, torch, state; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('state', state.__file__)"
