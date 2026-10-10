# simulator/impl/evolution.py
"""
Evolutionary divergence models
============
Modular evolution models that simulate the evolutionary distance of a virus
fragment prior to an HGT event.

Jukes-Cantor (JC69)
-------------------
The simplest nucleotide substitution model:
  - 4 bases at equal frequencies (25% each)
  - All substitution rates are equal
  - P(change | distance d) = 3/4 * (1 - exp(-4d/3))

Usage: before inserting a virus fragment into a host, apply evolutionary
      divergence so the fragment no longer matches the original reference sequence
      exactly.
"""

import random
import math


def apply_jukes_cantor(seq: str, distance: float = 0.0) -> str:
    """
    Apply the Jukes-Cantor (JC69) substitution model to a sequence.

    Parameters
    ----------
    seq : input sequence string
    distance : evolutionary distance (substitutions per site)
               0.0 = no change
               0.01 = ~1% divergence (close relative)
               0.05 = ~5% divergence (intermediate)
               0.10 = ~10% divergence (distant relative)

    Returns
    -------
    The evolved sequence (substitutions may have occurred)
    """
    if distance <= 0:
        return seq

    # JC69 substitution probability
    p_change = 0.75 * (1.0 - math.exp(-4.0 / 3.0 * distance))

    bases = list(seq)
    for i in range(len(bases)):
        if bases[i] not in 'ATCG':
            continue
        if random.random() < p_change:
            cur = bases[i]
            bases[i] = random.choice([b for b in 'ATCG' if b != cur])
    return ''.join(bases)


def apply_kimura_2p(
    seq: str,
    distance: float = 0.0,
    ts_ratio: float = 2.0,
) -> str:
    """
    Kimura 2-parameter (K2P) substitution model.

    Unlike JC69, K2P distinguishes transitions from transversions:
    - Transitions (A↔G, C↔T) occur at a higher rate
    - Transversions (A↔C, A↔T, G↔C, G↔T) occur at a lower rate

    Parameters
    ----------
    seq : input sequence
    distance : evolutionary distance
    ts_ratio : transition/transversion ratio (default 2.0)

    Returns
    -------
    The evolved sequence
    """
    if distance <= 0:
        return seq

    # K2P probability (correct Kimura 2-parameter formulas)
    # alpha = transition rate, beta = transversion rate
    # ts_ratio = alpha / beta => alpha = ts_ratio * beta
    # Normalisation: alpha + 2*beta = 1 => beta = 1/(2*(ts_ratio+1))
    beta = 1.0 / (2.0 * (ts_ratio + 1.0))
    p_ts = 0.25 + 0.25 * math.exp(-4.0 * beta * distance) \
         - 0.50 * math.exp(-2.0 * (ts_ratio * beta + beta) * distance)
    p_tv = 0.25 - 0.25 * math.exp(-4.0 * beta * distance)

    # Transition pairs
    _ts_map = {'A': 'G', 'G': 'A', 'C': 'T', 'T': 'C'}
    _tv_options = {
        'A': ['C', 'T'], 'G': ['C', 'T'],
        'C': ['A', 'G'], 'T': ['A', 'G'],
    }

    bases = list(seq)
    for i in range(len(bases)):
        cur = bases[i]
        if cur not in 'ATCG':
            continue
        r = random.random()
        if r < p_ts:
            bases[i] = _ts_map[cur]
        elif r < p_ts + p_tv:
            bases[i] = random.choice(_tv_options[cur])
    return ''.join(bases)


def apply_gc_amelioration(
    seq: str,
    target_gc: float,
    distance: float = 0.0,
) -> str:
    """
    GC amelioration model.

    Simulates the process by which foreign DNA, after long-term residence in a host
    genome, gradually adapts toward the host GC content.
    This is an important evolutionary signature of horizontal gene transfer: ancient
    HGT fragments have GC content closer to the host, while recently inserted
    fragments retain the original (usually viral) GC composition.

    Mechanism
    ----
    Apply directed mutations at random positions so that the local GC content drifts
    toward target_gc.
    The number of mutations is proportional to the distance — the larger the
      evolutionary distance, the stronger the amelioration.
      Each mutated position picks its replacement direction from the current GC state:
      - If local GC < target_gc → tend to replace with G/C
      - If local GC > target_gc → tend to replace with A/T

    Parameters
    ----------
    seq : input sequence (usually a virus fragment that already underwent divergence)
    target_gc : host GC content (0.0–1.0)
    distance : evolutionary distance (substitutions/site)
                 Controls the amelioration strength: 0 = no amelioration, larger = closer to host GC

    Returns
    -------
    The ameliorated sequence
    """
    if distance <= 0 or len(seq) < 10:
        return seq

    # Amelioration strength: mutation fraction proportional to distance, capped at 10% of bases
    n_mutations = min(
        int(len(seq) * distance * 0.3), # 30% of divergence rate
        len(seq) // 10, # at most 10% of bases
    )
    if n_mutations < 1:
        return seq

    # Compute the current GC content
    current_gc = sum(1 for b in seq if b in 'GCgc') / len(seq)
    if abs(current_gc - target_gc) < 0.005:
        return seq # already close to the target GC

    bases = list(seq)
    # Randomly choose mutation positions
    positions = random.sample(range(len(bases)), min(n_mutations, len(bases)))

    for pos in positions:
        cur = bases[pos]
        if cur not in 'ATCG':
            continue
        if current_gc < target_gc:
            # Need a GC increase: replace A/T with G/C
            if cur in 'AT':
                bases[pos] = random.choice('GC')
        else:
            # Need a GC decrease: replace G/C with A/T
            if cur in 'GC':
                bases[pos] = random.choice('AT')

    return ''.join(bases)
