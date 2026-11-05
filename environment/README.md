# Environment

The released environment is described by `pip-freeze.txt`: Python 3.11.15 with
torch 2.7.1+cu126 (CUDA 12.6 wheels) and the 152 packages used for all reported runs.

## Rebuild

```bash
bash environment/setup_env.sh              # uses python3.11 from PATH
bash environment/setup_env.sh /path/to/python3.11
```

The script creates `venvs/state`, installs `pip-freeze.txt` (minus the editable
`arc-state` line, which is installed from `../state`), then runs a smoke import:

```bash
./venvs/state/bin/python -c "import torch, state; print(torch.__version__, torch.cuda.is_available())"
```

## Using another environment

Any venv that satisfies `pip-freeze.txt` and has the vendored package installed
(`pip install -e state --no-deps`) will work. The shell pipelines default to
`venvs/state/bin/python`; point `PY` elsewhere to override:

```bash
PY=/path/to/python bash scripts/eval_run.sh <run> <ckpt> <gpu>
```

## Notes

- `CUDA_VISIBLE_DEVICES` selects the GPU; the environment is built against CUDA 12.6.
- `venvs/` holds several GB of binaries and is not tracked by Git.
- Dataset locations are the only absolute paths in the source tree; see the
  "Absolute paths kept on purpose" section of the top-level README.
