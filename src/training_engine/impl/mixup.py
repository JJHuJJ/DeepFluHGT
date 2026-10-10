# training_engine/impl/mixup.py
"""
DeepFluHGT  —  Mixup data augmentation
==================================
Apply Mixup to an HGTBatch, adapted to the data structure.

BUG-010 fix: hgt_mode / pe_case are one-hot encoded before mixing; n_events is mixed as float.
After mixing, hgt_mode changes shape from [B] long to [B, 4] float,
and pe_case from [B] long to [B, 8] float.
The downstream loss/api.py direction loss already supports soft targets.

C-01 fix (CODE_AUDIT_V7, CRITICAL): evo_embeddings are now passed through  —  mixed with the same λ and the
same perm index (lam*evo[i] + (1-lam)*evo[perm[i]]); kept None when the input is None.
Before the fix, mixup rebuilt HGTBatch dropping this field → in use_evo=true training (M2), after
mixup_warmup ended, every train step gave the Evo2 branch zero gradient while eval always carried evo input,
so train/eval input distributions were misaligned with no logged exception. seq_ids/sample_types metadata are
likewise passed through unchanged (not consumed on the training path, kept for provenance).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from data_pipeline.contracts import HGTBatch

# hgt_mode: 0=insertion, 1=replacement, 2=deletion, 3=none
N_HGT_MODES = 4
# pe_case: 0..7 (PE_CASE_MAP has 8 kinds in total)
N_PE_CASES = 8


def mixup_batch(
    batch: HGTBatch,
    alpha: float = 0.2,
) -> Tuple[HGTBatch, float]:
    """
    Apply Mixup data augmentation to an HGTBatch.

    Strategy:
      - Beta(α, α) sampling λ
      - ensure λ ≥ 0.5 (the larger side of the swap always stays the primary sample)
      - randompaired batch withinsample

    Parameters
    ----------
    batch : HGTBatch
    alpha : float
        Beta distributionparameters (default 0.2, 0 = disable)

    Returns
    -------
    (mixed_batch, lam) : the mixed HGTBatch + mixing coefficient
    """
    B = batch.labels.size(0)
    if B <= 1:
        return batch, 1.0

    device = batch.labels.device

    # m-07 fix + NEW-02 fix: use the global RNG for diversity,
    # while keeping reproducibility (the global RNG is already controlled by seed_all)
    gen = torch.Generator(device=device)
    # Base on the current global seed + a batch offset so each batch pairs differently
    _batch_offset = int(torch.rand(1, device=device).item() * 2**31)
    gen.manual_seed((torch.initial_seed() or 42) + _batch_offset)

    # Beta sampling
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        return batch, 1.0

    # Ensure λ ≥ 0.5
    lam = max(lam, 1.0 - lam)

    # randompairedindex
    perm = torch.randperm(B, device=device, generator=gen)

    # Mixing function
    def _mix(a, b):
        if isinstance(a, torch.Tensor) and a.is_floating_point():
            return lam * a + (1 - lam) * b
        return a

    # BUG-010 fix: one-hot encode hgt_mode / pe_case before mixing
    hgt_onehot = F.one_hot(batch.hgt_mode, N_HGT_MODES).float()    # [B, 4]
    mixed_hgt = lam * hgt_onehot + (1 - lam) * hgt_onehot[perm]     # [B, 4]

    pe_onehot = F.one_hot(batch.pe_case, N_PE_CASES).float()        # [B, 8]
    mixed_pe = lam * pe_onehot + (1 - lam) * pe_onehot[perm]        # [B, 8]

    # Mix n_events as float
    mixed_n = lam * batch.n_events.float() + (1 - lam) * batch.n_events[perm].float()

    # C-01 fix (CODE_AUDIT_V7): mix evo_embeddings with the same λ and the same perm.
    # _mix on a float tensor is lam*a + (1-lam)*b, semantically consistent with labels/bio_features;
    # kept None when the input is None (evo disabled); the model follows the no-evo path.
    mixed_evo = None
    if isinstance(batch.evo_embeddings, torch.Tensor):
        mixed_evo = _mix(batch.evo_embeddings, batch.evo_embeddings[perm])

    # Build the mixed HGTBatch
    mixed = HGTBatch(
        input_ids=batch.input_ids,                          # token IDs not mixed
        attention_mask=batch.attention_mask,                # mask not mixed
        labels=_mix(batch.labels.float(), batch.labels[perm].float()),
        breakpoint_targets=_mix(batch.breakpoint_targets, batch.breakpoint_targets[perm]),
        region_targets=_mix(batch.region_targets, batch.region_targets[perm]),
        bio_features=_mix(batch.bio_features, batch.bio_features[perm]),
        pe_case=mixed_pe,                                   # [B, 8] soft one-hot
        hgt_mode=mixed_hgt,                                 # [B, 4] soft one-hot
        n_events=mixed_n,                                   # [B] float
        evo_embeddings=mixed_evo,                           # [B, D_evo] float (C-01)
        seq_ids=batch.seq_ids,                              # metadata passed through unchanged (C-01)
        sample_types=batch.sample_types,
    )

    return mixed, lam
