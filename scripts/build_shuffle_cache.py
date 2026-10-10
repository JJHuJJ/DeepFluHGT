#!/usr/bin/env python
"""v7 M3 control: dataset-level permutation of the evo cache (seed 42); keys permuted, values unchanged.
usage: python scripts/build_shuffle_cache.py <src.pt> <dst.pt>"""
import sys, hashlib, numpy as np, torch
src, dst = sys.argv[1], sys.argv[2]
cache = torch.load(src, map_location="cpu", weights_only=False)
keys = sorted(cache.keys())
perm = np.random.RandomState(42).permutation(len(keys))
shuffled = {keys[int(p)]: cache[k] for p, k in zip(perm, keys)}
assert len(shuffled) == len(cache)
tmp = dst + ".tmp"; torch.save(shuffled, tmp)
import os; os.replace(tmp, dst)
h = hashlib.sha256(np.asarray(perm).tobytes()).hexdigest()
oh = hashlib.sha256(open(src, "rb").read()).hexdigest()
print(f"[shuffle] {len(shuffled)} vectors; perm_sha={h[:16]} orig_sha={oh[:16]}")
import json
mf = dst.rsplit("/", 1)[0] + "/shuffle_manifest.json"
m = {}
if __import__("os").path.exists(mf):
    m = json.load(open(mf))
m[dst.rsplit("/", 1)[1]] = {"shuffle_seed": 42, "permutation_sha256": h,
                            "original_cache_sha256": oh, "n": len(shuffled),
                            "note": "dataset-level key permutation; values untouched"}
json.dump(m, open(mf, "w"), indent=2)
