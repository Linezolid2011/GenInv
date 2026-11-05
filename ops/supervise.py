#!/usr/bin/env python
"""Supervisor for a training run launched by ops/run_train.sh.

Every POLL seconds it:
  * snapshots version_0/metrics.csv (content-hash dedup, pruned) -> version_0/metrics_snapshots/
    (Lightning truncates metrics.csv when a run resumes from a checkpoint, so this is the
     only way to keep the full val curve)
  * copies checkpoints/last.ckpt -> step<N>.ckpt when its global_step is a multiple of --ckpt-every
  * logs the training main's RSS/PSS and SIGKILLs it when PSS exceeds --pss-limit-g
    (the run_train.sh wrapper restarts and resumes from last.ckpt within ~30 s)

Usage:
  setsid nohup venvs/state/bin/python ops/supervise.py --name <name> \
      >> ops/supervise.log 2>&1 &
"""
import argparse
import hashlib
import os
import re
import signal
import subprocess
import time

import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--ckpt-every", type=int, default=20000)
    ap.add_argument("--pss-limit-g", type=float, default=75.0)
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--max-steps", type=int, default=200000)
    return ap.parse_args()


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def pss_gib(pid):
    total_kb = 0
    try:
        with open(f"/proc/{pid}/smaps") as f:
            for line in f:
                if line.startswith("Pss:"):
                    total_kb += int(line.split()[1])
    except OSError:
        return None
    return total_kb / 1e6


def rss_gib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        return None


def ppid_of(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[1])
    except Exception:
        return -1


def cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def find_main(name):
    """The training python main: comm startswith 'python', cmdline has train_dual.py --name <name>,
    and its parent is the restart loop in run_train.sh (parent cmdline contains 'run_train')."""
    out = subprocess.run(["pgrep", "-f", f"train_dual.py --name {name}"],
                         capture_output=True, text=True)
    for pid_s in out.stdout.split():
        pid = int(pid_s)
        try:
            with open(f"/proc/{pid}/comm") as f:
                if not f.read().strip().startswith("python"):
                    continue
        except OSError:
            continue
        cl = cmdline(pid)
        if "train_dual.py" not in cl:
            continue
        pcl = cmdline(ppid_of(pid))
        if "run_train" in pcl or "seq 1 300" in pcl:
            return pid
    return None


def ckpt_step(path):
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("global_step", 0))
    except Exception as e:
        log(f"ckpt read error {path}: {e}")
        return None


def snapshot_metrics(version_dir, state):
    src = os.path.join(version_dir, "metrics.csv")
    try:
        with open(src, "rb") as f:
            data = f.read()
    except OSError:
        return
    h = hashlib.md5(data).hexdigest()
    if state.get("md5") == h:
        return
    state["md5"] = h
    outdir = os.path.join(version_dir, "metrics_snapshots")
    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, f"{time.strftime('%m%d_%H%M')}.csv")
    with open(dst + ".tmp", "wb") as f:
        f.write(data)
    os.rename(dst + ".tmp", dst)
    snaps = sorted(os.listdir(outdir))
    for s in snaps[:-400]:
        try:
            os.remove(os.path.join(outdir, s))
        except OSError:
            pass


def main():
    a = parse_args()
    run_dir = os.path.join(a.root, "runs", a.name)
    version_dir = os.path.join(run_dir, "version_0")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    msnap = {"md5": None}
    missed = set()
    last_kill = 0.0
    log(f"supervising {a.name} (root={a.root}, ckpt_every={a.ckpt_every}, pss_limit={a.pss_limit_g}G)")

    while True:
        snapshot_metrics(version_dir, msnap)

        last = os.path.join(ckpt_dir, "last.ckpt")
        if os.path.exists(last):
            step = ckpt_step(last)
            if step is not None:
                for t in range(a.ckpt_every, a.max_steps + 1, a.ckpt_every):
                    dst = os.path.join(ckpt_dir, f"step{t}.ckpt")
                    if os.path.exists(dst):
                        continue
                    if step == t:
                        tmp = dst + ".tmp"
                        subprocess.run(["cp", last, tmp], check=False)
                        os.rename(tmp, dst)
                        log(f"saved step{t}.ckpt")
                    elif step > t and (t not in missed):
                        log(f"MISSED step{t} (last.ckpt at {step})")
                        missed.add(t)

        pid = find_main(a.name)
        if pid:
            pss, rss = pss_gib(pid), rss_gib(pid)
            log(f"{a.name}: pid={pid} rss={rss and round(rss,1)}GiB pss={pss and round(pss,1)}GiB")
            if pss and pss > a.pss_limit_g and time.time() - last_kill > 3600:
                log(f"PSS {pss:.1f}G > {a.pss_limit_g}G -> SIGKILL (wrapper will resume)")
                try:
                    os.kill(pid, signal.SIGKILL)
                    last_kill = time.time()
                except OSError as e:
                    log(f"kill failed: {e}")
        else:
            log(f"{a.name}: no training main found")

        time.sleep(a.poll)


if __name__ == "__main__":
    main()
