# simulator/impl/error_models.py
"""
Platform-specific sequencing error models
==========================================
Provides realistic error-model simulation for different sequencing platforms.

Platform characteristics
--------
Illumina : substitution-dominated, 3' read-end quality degradation, elevated error
rate in homopolymers and GC-rich regions
Nanopore : high error rate (5-15%), indel-dominated (especially in homopolymers),
weak position dependence
PacBio : CCS/HiFi low error rate (1-2%), random substitutions, no clear context dependence
"""

import random


# =========================================================================== #
# Helper: homopolymer run detection
# =========================================================================== #

def _homopolymer_factor(seq: str, pos: int) -> float:
    """
    Compute the homopolymer inflation factor at position pos.
    The longer the run of identical bases, the higher the error rate.
    Returns a multiplier >= 1.0.
    """
    base = seq[pos]
    run_len = 1
    # Extend left
    i = pos - 1
    while i >= 0 and seq[i] == base:
        run_len += 1
        i -= 1
    # Extend right
    i = pos + 1
    while i < len(seq) and seq[i] == base:
        run_len += 1
        i += 1
    # run 1 → 1.0, run 2 → 1.3, run 3 → 1.6, ... capped at ~2.5
    return 1.0 + 0.3 * min(run_len, 5)


def _local_gc_ratio(seq: str, pos: int, window: int = 20) -> float:
    """
    Compute the local GC content near position pos (within the window).
    """
    start = max(0, pos - window // 2)
    end = min(len(seq), pos + window // 2)
    region = seq[start:end]
    if not region:
        return 0.5
    return sum(region.count(b) for b in 'GCgc') / len(region)


# =========================================================================== #
# Illumina Error Model
# =========================================================================== #

def apply_illumina_errors(seq: str, base_error_rate: float = 0.005) -> str:
    """
    Illumina error model: substitution-dominated. Features:

    1. 3' read-end quality degradation (position-dependent, error rate grows linearly)
    2. Homopolymer runs raise the error rate (A₆ → error-prone)
    3. GC-rich regions (>65% GC) get a 1.5x error-rate increase
    4. Substitutions only (no indels)
    """
    if base_error_rate <= 0:
        return seq
    bases = list(seq)
    L = len(bases)
    for i in range(L):
        # Position-dependent: error rate rises toward the 3' end
        pos_factor = 1.0 + i / L
        # Homopolymer inflation
        hp_factor = _homopolymer_factor(seq, i)
        # GC-rich inflation
        gc_ratio = _local_gc_ratio(seq, i)
        gc_factor = 1.5 if gc_ratio > 0.65 else 1.0
        # Combined error rate
        rate = base_error_rate * pos_factor * hp_factor * gc_factor
        if random.random() < rate:
            cur = bases[i]
            bases[i] = random.choice([b for b in 'ATCG' if b != cur])
    return ''.join(bases)


# =========================================================================== #
# Nanopore Error Model
# =========================================================================== #

def apply_nanopore_errors(
    seq: str,
    base_error_rate: float = 0.08,
    indel_fraction: float = 0.6,
) -> str:
    """
    Nanopore error model: high error rate, indel-dominated.

    Features:
    1. Indels account for ~60% of errors, substitutions ~40%
    2. Indel rate rises sharply within homopolymer runs
    3. Weak position dependence
    4. May change the sequence length (callers must trim/pad)

    Parameters
    ----------
    seq : input sequence
    base_error_rate : baseline error rate (Nanopore typically 5-15%, default 8%)
    indel_fraction : fraction of errors that are indels (default 60%)
    """
    if base_error_rate <= 0:
        return seq
    result = []
    i = 0
    L = len(seq)
    while i < L:
        hp_factor = _homopolymer_factor(seq, min(i, L - 1))
        rate = base_error_rate * hp_factor
        r = random.random()
        if r < rate * indel_fraction * 0.5:
            # insertion: add a random base before the current base
            result.append(random.choice('ATCG'))
            result.append(seq[i])
        elif r < rate * indel_fraction:
            # deletion: skip the current base
            pass
        elif r < rate:
            # substitution
            cur = seq[i]
            result.append(random.choice([b for b in 'ATCG' if b != cur]))
        else:
            result.append(seq[i])
        i += 1
    return ''.join(result)


# =========================================================================== #
# PacBio (CCS/HiFi) Error Model
# =========================================================================== #

def apply_pacbio_errors(seq: str, base_error_rate: float = 0.01) -> str:
    """
    PacBio CCS/HiFi error model: low error rate, random substitutions.

    Features:
    1. Error rate 1-2% (after CCS consensus)
    2. Random substitutions, no clear context dependence
    3. Very few indels (<5% of errors)
    """
    if base_error_rate <= 0:
        return seq
    result = []
    i = 0
    L = len(seq)
    while i < L:
        if random.random() < base_error_rate:
            # 95% substitution, 5% indel
            r2 = random.random()
            if r2 < 0.025:
                # insertion
                result.append(random.choice('ATCG'))
                result.append(seq[i])
            elif r2 < 0.05:
                # deletion: skip current base
                pass
            else:
                # substitution
                cur = seq[i]
                result.append(random.choice([b for b in 'ATCG' if b != cur]))
        else:
            result.append(seq[i])
        i += 1
    return ''.join(result)


# =========================================================================== #
# Dispatch Function
# =========================================================================== #

_PLATFORM_MAP = {
    'illumina': apply_illumina_errors,
    'nanopore': apply_nanopore_errors,
    'pacbio': apply_pacbio_errors,
}


def apply_platform_errors(
    seq: str,
    platform: str = 'illumina',
    base_error_rate: float = 0.005,
) -> str:
    """
    Apply sequencing errors for the given platform. Unknown platforms fall back to Illumina.

    Parameters
    ----------
    seq : input sequence
    platform : "illumina" | "nanopore" | "pacbio"
    base_error_rate : baseline error rate (typical values differ by platform)
    """
    if base_error_rate <= 0:
        return seq
    func = _PLATFORM_MAP.get(platform, apply_illumina_errors)
    return func(seq, base_error_rate)
