#!/usr/bin/env python
# scripts/evo_cache_monitor.py
"""
DeepFluHGT  —  Evo2 full-cache monitor (logging enhancement for the frozen period)
====================================================================
Records hourly heartbeats for the running full-cache job; when the job completes,
automatically generates cache_manifest.json (provenance that Methods can cite directly).

Heartbeat rows (logs/evo_cache_monitor.log, CSV):
  timestamp, processed_windows, total_windows, cache_size_bytes, gpu0_mem_mib

manifest field (evo_cache/Contig_500k/cache_manifest.json):
  dataset / protocol(window,stride) / embedding_dim / dtype /
  per-split {n_vectors, sha256, bytes, mtime} / completed / exit_code

usage (long-running daemon):
  nohup python -u \
      scripts/evo_cache_monitor.py >> logs/evo_cache_monitor.log 2>&1 &
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time

BASE = "."
JOB_LOG = os.path.join(BASE, "logs", "evo_contig_full_cache.log")
OUT_DIR = os.path.join(BASE, "evo_cache", "Contig_500k")
INTERVAL = 3600

PROGRESS_RE = re.compile(r"windows (\d+)/(\d+)")


def heartbeat() -> bool:
    """Record one heartbeat. Returns False to indicate the job has already finished (the monitor should exit)."""
    processed = total = ""
    try:
        with open(JOB_LOG, errors="ignore") as f:
            for line in f:
                m = PROGRESS_RE.search(line)
                if m:
                    processed, total = m.group(1), m.group(2)
    except FileNotFoundError:
        pass
    cache_bytes = 0
    if os.path.isdir(OUT_DIR):
        cache_bytes = sum(os.path.getsize(os.path.join(OUT_DIR, fn))
                          for fn in os.listdir(OUT_DIR) if fn.endswith(".pt"))
    try:
        mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.split("\n")[0].strip()
    except Exception:
        mem = "?"
    with open(JOB_LOG, errors="ignore") as f:
        content_tail = f.read()[-400:]
    done = "FULL_CACHE_EXIT" in content_tail
    print(f"{time.strftime('%F %T')},{processed or 0},{total or 0},"
          f"{cache_bytes},{mem}", flush=True)
    return not done


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(exit_code: str) -> None:
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    splits = {}
    for split in ("train", "val", "test"):
        p = os.path.join(OUT_DIR, f"{split}.pt")
        if not os.path.isfile(p):
            continue
        cache = torch.load(p, map_location="cpu", weights_only=False)
        vec = next(iter(cache.values()))
        splits[split] = {
            "n_vectors": len(cache),
            "dim": int(vec.shape[0]),
            "bytes": os.path.getsize(p),
            "sha256": sha256_of(p),
        }
        del cache
    manifest = {
        "dataset": "Contig_500k",
        "model": "evo2_7b",
        "layer": "blocks.28.mlp.l3",
        "protocol": "window=3000,stride=2700,overlap=300,terminal_partial=kept,"
                    "aggregation=mean,equal_weight",
        "dtype": "float16",
        "completed": exit_code == "0" and len(splits) == 3,
        "exit_code": exit_code,
        "splits": splits,
        "created": time.strftime("%F %T"),
    }
    with open(os.path.join(OUT_DIR, "cache_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[monitor] manifest → {OUT_DIR}/cache_manifest.json "
          f"(completed={manifest['completed']})", flush=True)


def main():
    print(f"[monitor] start, interval={INTERVAL}s", flush=True)
    while heartbeat():
        time.sleep(INTERVAL)
    # job finished → generate manifest
    exit_code = "?"
    try:
        with open(JOB_LOG, errors="ignore") as f:
            m = re.search(r"FULL_CACHE_EXIT=(\d+)", f.read())
        if m:
            exit_code = m.group(1)
    except FileNotFoundError:
        pass
    write_manifest(exit_code)
    print("[monitor] exit", flush=True)


if __name__ == "__main__":
    main()
