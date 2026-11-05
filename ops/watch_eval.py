#!/usr/bin/env python
"""Run the 5-cell-line evaluation when a run reaches each target step.

Waits for `runs/<name>/checkpoints/step<target>.ckpt` (created by ops/supervise.py),
then runs `scripts/eval_run.sh <name> <ckpt> <eval_gpu> eval_<name>_step<k>k`.
Writes `runs/<name>/EVAL_<target>_DONE` on success.

Usage:
  setsid nohup venvs/state/bin/python ops/watch_eval.py --name <name> --gpu 1 \
      >> ops/watch_eval.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu", default="1", help="GPU used for evaluation (keep training on another)")
    ap.add_argument("--targets", default="100000,200000")
    ap.add_argument("--poll", type=int, default=120)
    return ap.parse_args()


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main():
    a = parse_args()
    run_dir = os.path.join(ROOT, "runs", a.name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    targets = [int(t) for t in a.targets.split(",") if t.strip()]
    log(f"armed for {a.name}: targets={targets} eval_gpu={a.gpu}")

    for target in targets:
        marker = os.path.join(run_dir, f"EVAL_{target}_DONE")
        if os.path.exists(marker):
            log(f"{target}: already done")
            continue
        ck = os.path.join(ckpt_dir, f"step{target}.ckpt")
        log(f"{target}: waiting for {ck}")
        while not os.path.exists(ck):
            time.sleep(a.poll)
        k = target // 1000
        ev = os.path.join(ROOT, f"eval_{a.name}_step{k}k")
        log(f"{target}: eval start -> {ev} (gpu {a.gpu})")
        logf = os.path.join(ROOT, "ops", f"eval_{a.name}_{k}k.log")
        with open(logf, "ab") as lf:
            rc = subprocess.run(
                ["bash", "scripts/eval_run.sh", a.name, ck, str(a.gpu), ev],
                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
            ).returncode
        if rc == 0:
            open(marker, "w").write("ok\n")
            log(f"{target}: EVAL DONE -> {ev}")
        else:
            log(f"{target}: eval FAILED rc={rc} (see {logf})")
            sys.exit(1)
    log("all targets done")


if __name__ == "__main__":
    main()
