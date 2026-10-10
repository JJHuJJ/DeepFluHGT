# C-01 unit test (CODE_AUDIT_V7, CRITICAL): mixup_batch must carry and correctly mix evo_embeddings
# Pure CPU (CUDA_VISIBLE_DEVICES=""); does not touch any running processes / checkpoints / evo_cache.
#
# Assertions:
#   1. When input batch.evo_embeddings is Tensor[B, D], the output shape/type is unchanged
#      and strictly equals the manual mix with the mixup-internal λ and the same shuffle
#      index: lam*evo[i] + (1-lam)*evo[perm[i]] (λ/perm are recovered from the bio_features
#      fingerprint, without depending on mixup-internal RNG implementation details).
#   2. When input evo_embeddings=None the output stays None (training without evo enabled
#      is unaffected, M0 semantics).
#   3. The original batch is not modified in place; metadata seq_ids/sample_types are
#      passed through unchanged.
#   4. The B<=1 early-exit path returns as-is.
import json
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.contracts import HGTBatch
from training_engine.impl.mixup import mixup_batch

B, L, D_EVO, F_BIO = 8, 32, 16, 260
torch.manual_seed(20260820)

results = {"trials": []}


def make_batch(with_evo: bool) -> HGTBatch:
    g = torch.Generator().manual_seed(7)
    bio = torch.zeros(B, F_BIO, dtype=torch.float32)
    for i in range(B):
        # fingerprint: row i has a 1 at column i → after mixing, two non-zero entries (values lam and 1-lam) uniquely recover perm
        bio[i, i] = 1.0
    return HGTBatch(
        input_ids=torch.randint(1, 6, (B, L), generator=g),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        bio_features=bio,
        evo_embeddings=(torch.randn(B, D_EVO, generator=g).float()
                        if with_evo else None),
        labels=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        breakpoint_targets=torch.zeros(B, L, dtype=torch.float32),
        region_targets=torch.zeros(B, L, dtype=torch.float32),
        pe_case=torch.randint(0, 8, (B,), generator=g),
        hgt_mode=torch.randint(0, 4, (B,), generator=g),
        n_events=torch.randint(0, 3, (B,), generator=g),
        seq_ids=[f"seq_{i}" for i in range(B)],
        sample_types=["POS" if i % 2 else "NEG" for i in range(B)],
    )


# ── Case 1: batch with evo (multiple draws, ensure a non-degenerate λ) ──
# λ from Beta(0.2,0.2) is often extremely close to 1 → the mixing check degenerates
# (identity). Draw multiple trials; require at least one trial with 1-lam ≥ 0.05
# before considering coverage sufficient.
def run_trial(seed: int):
    torch.manual_seed(seed)
    batch = make_batch(with_evo=True)
    evo_before = batch.evo_embeddings.clone()
    bio_before = batch.bio_features.clone()

    mixed, lam = mixup_batch(batch, alpha=0.2)

    assert 0.5 <= lam <= 1.0, f"lam should be constrained to [0.5, 1], got {lam}"

    # perm recovery: row i's fingerprint non-zero entries after mixing = {i: lam, perm[i]: 1-lam}
    # (or a fixed point {i: 1.0}).
    # Threshold 1e-9: 1-lam can in principle be arbitrarily small; with a single
    # non-zero entry whose value ≈ lam, the other side's weight is < 1e-9, whose
    # impact on the manual mix value is far below the allclose tolerance, so
    # perm = i is acceptable (the mix is nearly the identity).
    perm = []
    for i in range(B):
        row = mixed.bio_features[i]
        nz = (row > 1e-9).nonzero(as_tuple=True)[0].tolist()
        assert nz, f"row {i} fingerprint all-zero, cannot recover perm"
        if len(nz) == 1:
            assert nz[0] == i, f"row {i} single non-zero entry should be at column i, got {nz}"
            v = row[i].item()
            assert abs(v - 1.0) < 1e-5 or abs(v - lam) < 1e-5, \
                f"row {i} single non-zero value {v} is neither the fixed point 1.0 nor lam={lam}"
            perm.append(i)   # fixed point, or an approximate fixed point with (1-lam)<1e-9
        else:
            assert len(nz) == 2, f"row {i} should have exactly 2 non-zero entries, got {nz}"
            j = nz[0] if nz[0] != i else nz[1]
            vals = {round(row[i].item(), 9), round(row[j].item(), 9)}
            assert abs(max(vals) - lam) < 1e-4 and abs(min(vals) - (1 - lam)) < 1e-4, \
                f"row {i} fingerprint values {vals} do not match lam={lam}"
            perm.append(j)
    perm = torch.tensor(perm, dtype=torch.long)

    # core assertion: evo mix value == lam*evo[i] + (1-lam)*evo[perm[i]] (same λ, same index)
    expected_evo = lam * batch.evo_embeddings + (1 - lam) * batch.evo_embeddings[perm]
    assert mixed.evo_embeddings is not None, "C-01: evo_embeddings was dropped"
    assert mixed.evo_embeddings.shape == (B, D_EVO), \
        f"evo shape changed: {tuple(mixed.evo_embeddings.shape)} != {(B, D_EVO)}"
    assert mixed.evo_embeddings.dtype == torch.float32
    assert torch.allclose(mixed.evo_embeddings, expected_evo, atol=1e-6), \
        "evo mix value inconsistent with the manual mix (same λ/index)"

    # cross-validation: labels should also equal the mix with the same λ and perm (confirming the recovered perm is correct)
    exp_labels = lam * batch.labels.float() + (1 - lam) * batch.labels[perm].float()
    assert torch.allclose(mixed.labels, exp_labels, atol=1e-6)

    # the original batch was not modified in place
    assert torch.equal(batch.evo_embeddings, evo_before)
    assert torch.equal(batch.bio_features, bio_before)

    # metadata passed through unchanged
    assert mixed.seq_ids == batch.seq_ids and mixed.sample_types == batch.sample_types

    return {
        "seed": seed,
        "lam": round(lam, 6),
        "perm": perm.tolist(),
        "shape": list(mixed.evo_embeddings.shape),
        "max_abs_err_vs_manual": float(
            (mixed.evo_embeddings - expected_evo).abs().max().item()),
        "input_untouched": True,
    }


n_degenerate_guard = 0
for _seed in range(1, 25):
    trial = run_trial(_seed)
    results["trials"].append(trial)
    if 1.0 - trial["lam"] >= 0.05:
        n_degenerate_guard += 1
    if len(results["trials"]) >= 3 and n_degenerate_guard >= 1:
        break
assert n_degenerate_guard >= 1, \
    "all λ across 24 trials > 0.95; the mixing check degenerates (check the RNG or increase the number of trials)"
results["with_evo"] = {
    "n_trials": len(results["trials"]),
    "n_non_degenerate": n_degenerate_guard,
    "worst_max_abs_err_vs_manual": max(t["max_abs_err_vs_manual"]
                                       for t in results["trials"]),
}

# ── Case 2: evo_embeddings=None (M0 / not-yet-enabled path) ──
batch_none = make_batch(with_evo=False)
mixed_none, lam_none = mixup_batch(batch_none, alpha=0.2)
assert mixed_none.evo_embeddings is None, \
    "C-01: with input None the output should stay None (training without evo enabled must not change semantics)"
results["none_evo"] = {"evo_embeddings": None, "lam": round(lam_none, 6)}

# ── Case 3: B<=1 early exit ──
single = make_batch(with_evo=True)
single.labels = single.labels[:1]
single.bio_features = single.bio_features[:1]
single.evo_embeddings = single.evo_embeddings[:1]
# truncating the remaining HGTBatch fields is harmless for the assertions (mixup early-exits on labels.size(0) alone)
mixed_s, lam_s = mixup_batch(single, alpha=0.2)
assert lam_s == 1.0 and mixed_s is single and mixed_s.evo_embeddings is not None
results["single_sample"] = {"returned_original": True, "lam": 1.0}

# ── evidence written to disk ──
results["status"] = "PASS"
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "evidence", "bugfix", "mixup_evo_after.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"PASS: {len(results['trials'])} trials "
      f"({results['with_evo']['n_non_degenerate']} non-degenerate lam), "
      f"worst |err| vs manual mix = "
      f"{results['with_evo']['worst_max_abs_err_vs_manual']:.2e}; "
      f"None->None; B=1 passthrough")
print(f"evidence -> {out}")
