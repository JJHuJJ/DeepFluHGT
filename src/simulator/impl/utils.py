# simulator/impl/utils.py
"""
Utility functions: sequencing error simulation, GC bias filtering, abundance sampling
"""

import random
import math
import numpy as np


def apply_sequencing_errors(seq: str, base_error_rate: float = 0.005) -> str:
    """
    Apply random point mutations to a sequence to simulate sequencing errors.
    The error rate grows linearly with position (3' read-end quality degradation).
    """
    if base_error_rate <= 0:
        return seq
    bases = list(seq)
    L = len(bases)
    for i in range(L):
        # Error rate rises toward the 3' end
        pos_rate = base_error_rate * (1.0 + i / L)
        if random.random() < pos_rate:
            cur = bases[i]
            bases[i] = random.choice([b for b in 'ATCG' if b != cur])
    return ''.join(bases)


def gc_bias_weight(gc_content: float) -> float:
    """
    Illumina GC bias model: the optimal GC content is ~50%; the further from it, the lower the weight.
    """
    return math.exp(-((gc_content - 0.5) ** 2) / 0.03)


def gc_bias_filter(seq: str) -> bool:
    """
    Decide whether to keep a sequence under the GC bias model.
    Returns True to keep it, False to drop it (resample).
    """
    if not seq:
        return True
    gc = (seq.count('G') + seq.count('C')) / len(seq)
    return random.random() < gc_bias_weight(gc)


def coverage_scale(num_genomes: int,
                   mean: float = 0.0,
                   sigma: float = 0.5) -> list:
    """
    Draw lognormal abundance weights, simulating the relative abundance of each genome in an environmental sample.
    """
    return np.random.lognormal(mean=mean, sigma=sigma,
                               size=num_genomes).tolist()


def coverage_zipf(num_genomes: int, alpha: float = 1.0) -> list:
    """
    Zipf abundance distribution: a few species have very high abundance, while many
    species have low abundance.

    Parameters
    ----------
    num_genomes : number of genomes
    alpha : Zipf exponent (larger = more skewed)
    """
    ranks = np.arange(1, num_genomes + 1)
    weights = 1.0 / np.power(ranks, alpha)
    weights /= weights.sum()
    np.random.shuffle(weights)
    return weights.tolist()


def batch_effect_shift(weights: list, shift_sigma: float = 0.1) -> list:
    """
    Apply a batch-effect perturbation to abundance weights.
    Simulates small abundance-distribution differences between samples.

    Parameters
    ----------
    weights : base abundance weights
    shift_sigma : perturbation strength (0 = no perturbation)
    """
    if shift_sigma <= 0:
        return weights
    perturbation = np.random.lognormal(0, shift_sigma, len(weights))
    return (np.array(weights) * perturbation).tolist()


# =========================================================================== #
# Label Noise
# =========================================================================== #

def apply_label_noise(label: int, noise_rate: float = 0.0) -> int:
    """
    Flip the label (0 ↔ 1) with probability noise_rate.
    Simulates annotation error.
    """
    if noise_rate <= 0:
        return label
    if random.random() < noise_rate:
        return 1 - label
    return label


def shift_boundary(
    hgt_start: int,
    hgt_end: int,
    seq_len: int,
    max_shift: int = 0,
) -> tuple:
    """
    Randomly shift the HGT boundary by up to ±max_shift bp.
    Simulates non-deterministic annotation (imprecise boundary localization).

    Returns
    -------
    (new_start, new_end) — guaranteed 0 <= new_start < new_end <= seq_len
    """
    if max_shift <= 0:
        return hgt_start, hgt_end
    shift_s = random.randint(-max_shift, max_shift)
    shift_e = random.randint(-max_shift, max_shift)
    new_s = max(0, min(hgt_start + shift_s, seq_len - 1))
    new_e = max(new_s + 1, min(hgt_end + shift_e, seq_len))
    return new_s, new_e
