# simulator/impl/dataset_generator.py
"""
Dataset orchestrator — receives parameters from SimulatorConfig and generates the
full dataset.
Refactored from v4.0 simulate_dataset_v3.py, with the CLI dependency removed.
"""

import json
import math
import multiprocessing as mp
import os
import random

import numpy as np
from Bio.Seq import Seq

from .microbe_sampler import MicrobeSampler
from .virus_fragment import load_virus_genomes, generate_virus_pool
from .genome_splitter import split_microbe_genomes, split_virus_genomes
from .sequence_generator import (
    HGTSequenceGenerator,
    classify_pe_case,
    classify_pe_case_multi,
    compute_insert_segments,
    compute_insert_segments_multi,
    compute_read_segments,
    build_composite_host,
    generate_virus_only,
    generate_pcr_chimera,
    generate_negative_hard,
    generate_host_only,
    generate_neg_conserved,
    generate_neg_homology,
    generate_neg_low_complexity,
    generate_neg_gc_matched_chimeric,
    generate_neg_amalgamated_hgt,
    generate_neg_partial_virus_hit,
    generate_pe_neg_host_diversity,
    generate_pe_neg_gc_shift_region,
    generate_pe_neg_codon_boundary,
    apply_contamination,
    apply_assembly_artifact,
    apply_microhomology,
)


def _compute_pe_case_pos_range(case_type, virus_len, read_len, insert_len):
    """
    Compute the valid range of hgt_start for a given PE case.

    Returns (lo, hi) such that lo <= hgt_start <= hi is valid, or None if impossible.
    """
    RL = read_len
    IL = insert_len
    VL = virus_len

    if case_type == 'R1_HOST_R2_VIRUS':
        lo = max(RL, IL - VL)
        hi = IL - RL
    elif case_type == 'R1_HOST_R2_MIXED':
        lo = max(RL, IL - RL - VL + 1)
        hi = IL - VL - 1
    elif case_type == 'R1_VIRUS_R2_HOST':
        if VL > IL - RL:
            return None
        lo = 0
        hi = 0
    elif case_type == 'R1_VIRUS_R2_MIXED':
        if VL <= RL or VL > IL - RL:
            return None
        lo = 0
        hi = 0
    elif case_type == 'R1_MIXED_R2_HOST':
        hi = min(RL - 1, IL - RL - VL)
        lo = 1
        if lo > hi:
            return None
    elif case_type == 'R1_MIXED_R2_VIRUS':
        lo = max(1, IL - VL)
        hi = min(RL - 1, IL - RL)
        if lo > hi:
            return None
    elif case_type == 'R1_MIXED_R2_MIXED':
        candidates = []
        s1_lo = max(1, IL - RL - VL + 1)
        s1_hi = min(RL - 1, IL - RL)
        if s1_lo <= s1_hi:
            candidates.append((s1_lo, s1_hi))
        if VL > RL:
            s2_lo = 1
            s2_hi = min(RL - 1, IL - VL - 1)
            if s2_lo <= s2_hi:
                candidates.append((s2_lo, s2_hi))
        if VL >= IL:
            s3_lo = max(RL, IL - RL)
            s3_hi = min(IL - VL, IL - 1)
            if s3_lo <= s3_hi:
                candidates.append((s3_lo, s3_hi))
        if not candidates:
            return None
        lo = min(c[0] for c in candidates)
        hi = max(c[1] for c in candidates)
    else:
        return None

    if lo > hi or lo < 0 or hi >= IL:
        return None
    return (lo, hi)
from .utils import apply_sequencing_errors, gc_bias_filter
from .error_models import apply_platform_errors
from .evolution import apply_jukes_cantor, apply_gc_amelioration
from .utils import apply_label_noise, shift_boundary
from ..contracts import SimulatorConfig, SimulationOutput


# ============================================================================= #
# Difficulty grading & label confidence
# ============================================================================= #

def _compute_difficulty(
    virus_len: int,
    n_events: int,
    error_rate: float,
    has_contamination: bool,
    evolutionary_distance: float,
    has_artifact: bool = False,
    neg_type: str = '',
    read_len: int = 150,
) -> str:
    """
    Compute the difficulty grade from sample features.

    Scoring dimensions
    --------
    - Virus fragment length: short fragments are harder to detect
    - Number of events: multiple events add complexity
    - Error rate: high error rates mask the signal
    - Contamination / artifacts: interfere with detection
    - Evolutionary distance: distant origins are harder to detect via sequence
    matching
    - Hard-negative sample type
    """
    score = 0.0
    # Virus fragment length (relative to read_len)
    if virus_len > 0:
        ratio = virus_len / read_len if read_len > 0 else 1.0
        if ratio < 0.3:
            score += 0.30 # extremely short fragment, nearly invisible
        elif ratio < 0.6:
            score += 0.20 # short fragment
        elif ratio < 1.0:
            score += 0.10
        else:
            score += 0.02
    # Multiple events
    if n_events >= 3:
        score += 0.20
    elif n_events >= 2:
        score += 0.10
    # Error rate
    if error_rate > 0.02:
        score += 0.20
    elif error_rate > 0.005:
        score += 0.10
    # Contamination
    if has_contamination:
        score += 0.15
    # Evolutionary distance
    if evolutionary_distance > 0.1:
        score += 0.15
    elif evolutionary_distance > 0.02:
        score += 0.08
    # Assembly artifact
    if has_artifact:
        score += 0.10
    # Hard negatives
    if neg_type in ('NEG_HARD', 'NEG_CONSERVED', 'NEG_HOMOLOGY',
                     'NEG_LOW_COMPLEXITY', 'NEG_GC_MATCHED_CHIMERIC',
                     'NEG_AMALGAMATED_HGT', 'NEG_PARTIAL_VIRUS_HIT',
                     'NEG_PE_HOST_DIVERSITY', 'NEG_PE_GC_SHIFT_REGION',
                     'NEG_PE_CODON_BOUNDARY'):
        score += 0.15

    if score < 0.15:
        return 'easy'
    elif score < 0.35:
        return 'medium'
    elif score < 0.55:
        return 'hard'
    else:
        return 'extreme'


def _compute_label_confidence(
    label: int,
    boundary_shift_max: int = 0,
    has_label_noise: bool = False,
    neg_type: str = '',
    has_contamination: bool = False,
    evolutionary_distance: float = 0.0,
) -> float:
    """
    Compute label confidence (soft label).

    Confidence reflects annotation reliability:
    - Clear positive/negative: 1.0 (fully certain)
    - Boundary shift: slightly lowered (0.90-0.97)
    - Hard negatives: lower (0.05-0.20, indicating some ambiguity)
    - Label noise already applied: lowered
    - Contamination present: slightly lowered
    - Large evolutionary distance: slightly lowered

    Use: during training, BCE(target=confidence) can replace BCE(target=label) to
    implement label smoothing.
    """
    confidence = 1.0 if label == 1 else 0.0

    # Boundary shift lowers positive confidence
    if boundary_shift_max > 0 and label == 1:
        confidence = 0.90 + 0.07 * (1.0 - min(boundary_shift_max / 10.0, 1.0))

    # Label-noise effect
    if has_label_noise:
        # If noise was already applied, lower the confidence
        if label == 1:
            confidence = min(confidence, 0.92)
        else:
            confidence = max(confidence, 0.08)

    # Hard negatives: they themselves have some probability of mislabeling
    if label == 0 and neg_type in ('NEG_HARD', 'NEG_CONSERVED', 'NEG_HOMOLOGY',
                                     'NEG_GC_MATCHED_CHIMERIC', 'NEG_AMALGAMATED_HGT',
                                     'NEG_PARTIAL_VIRUS_HIT', 'NEG_PE_HOST_DIVERSITY',
                                     'NEG_PE_GC_SHIFT_REGION', 'NEG_PE_CODON_BOUNDARY'):
        confidence = 0.05 + 0.10 * random.random()
    elif label == 0 and neg_type == 'NEG_LOW_COMPLEXITY':
        confidence = 0.02 + 0.05 * random.random()

    # Contamination slightly lowers confidence
    if has_contamination:
        confidence = confidence * 0.95 if label == 1 else confidence + 0.03

    # With a large evolutionary distance, positive confidence drops slightly
    if evolutionary_distance > 0.05 and label == 1:
        confidence *= 0.97

    return round(max(0.0, min(1.0, confidence)), 4)


def _enrich_meta(meta: dict, cfg: dict, neg_type: str = '',
                 has_contamination: bool = False,
                 artifact_type: str = '',
                 evolutionary_distance: float = None) -> dict:
    """
    Add difficulty and label_confidence fields to sample metadata.
    """
    label = meta.get('label', 0)
    n_events = meta.get('n_events', 1)
    # Compute the average virus fragment length
    events = meta.get('events', [])
    virus_len = events[0]['virus_len'] if events else 0
    read_len = cfg.get('read_len', 150)

    evo_dist_for_meta = (evolutionary_distance if evolutionary_distance is not None
                         else cfg.get('evolutionary_distance', 0.0))

    difficulty = _compute_difficulty(
        virus_len=virus_len,
        n_events=n_events,
        error_rate=cfg.get('error_rate', 0.005),
        has_contamination=has_contamination,
        evolutionary_distance=evo_dist_for_meta,
        has_artifact=bool(artifact_type and artifact_type != 'none'),
        neg_type=neg_type,
        read_len=read_len,
    )
    label_confidence = _compute_label_confidence(
        label=label,
        boundary_shift_max=cfg.get('boundary_shift_max', 0),
        has_label_noise=cfg.get('label_noise', 0) > 0,
        neg_type=neg_type,
        has_contamination=has_contamination,
        evolutionary_distance=evo_dist_for_meta,
    )
    meta['difficulty'] = difficulty
    meta['label_confidence'] = label_confidence
    if evolutionary_distance is not None:
        meta['evolutionary_distance'] = round(evolutionary_distance, 6)
    if artifact_type:
        meta['artifact_type'] = artifact_type
    return meta


# ============================================================================= #
# Provenance-log writing helpers
# ============================================================================= #

PROV_HEADER = (
    "seq_id\tlabel\thgt_mode\tpe_case\tread\tseg_order\t"
    "seg_type\tsource_type\tsource_file\tsource_seq_id\t"
    "genome_start\tgenome_end\tstrand\tread_start\tread_end\tnote\n"
)


def _prov_rows(sid, label, hgt_mode, pe_case, read_name, segs, note=''):
    """Format one read's segment list into provenance TSV rows."""
    lines = []
    for i, seg in enumerate(segs):
        lines.append(
            f"{sid}\t{label}\t{hgt_mode}\t{pe_case}\t{read_name}\t{i}\t"
            f"{seg['seg_type']}\t{seg['source_type']}\t"
            f"{seg['source_file']}\t{seg['source_seq_id']}\t"
            f"{seg['genome_start']}\t{seg['genome_end']}\t"
            f"{seg['strand']}\t{seg['read_start']}\t{seg['read_end']}\t"
            f"{note}\n"
        )
    return ''.join(lines)


# ============================================================================= #
# Built-in default distributions (used when not configured via YAML)
# ============================================================================= #

_DEFAULT_PE_CASE_RATIOS = {
    'R1_HOST_R2_VIRUS': 0.24,
    'R1_HOST_R2_MIXED': 0.12,
    'R1_VIRUS_R2_HOST': 0.24,
    'R1_VIRUS_R2_MIXED': 0.10,
    'R1_MIXED_R2_HOST': 0.12,
    'R1_MIXED_R2_VIRUS': 0.10,
    'R1_MIXED_R2_MIXED': 0.08,
}

_DEFAULT_NEG_TYPE_RATIOS = {
    'NEG': 0.15,
    'NEG_VIRUS': 0.08,
    'NEG_HARD': 0.08,
    'NEG_PCR': 0.04,
    'NEG_CONSERVED': 0.06,
    'NEG_HOMOLOGY': 0.06,
    'NEG_LOW_COMPLEXITY': 0.04,
    'NEG_GC_MATCHED_CHIMERIC': 0.12,
    'NEG_AMALGAMATED_HGT': 0.08,
    'NEG_PARTIAL_VIRUS_HIT': 0.07,
    'NEG_PE_HOST_DIVERSITY': 0.10,
    'NEG_PE_GC_SHIFT_REGION': 0.06,
    'NEG_PE_CODON_BOUNDARY': 0.06,
}

_DEFAULT_CONTIG_HGT_MODE_RANGES = {
    'insertion': (0.70, 0.90),
    'replacement': (0.08, 0.25),
    'deletion': (0.01, 0.10),
}

_DEFAULT_PE_VIRUS_FRAG_TIERS = [
    (50, 200, 0.15),
    (200, 1000, 0.30),
    (1000, 5000, 0.55),
]

_DEFAULT_CONTIG_VIRUS_FRAG_TIERS = [
    (300, 800, 0.20),
    (800, 1500, 0.55),
    (1500, 2341, 0.25),
]


def _noisy_tiers(base_tiers, noise=0.05):
    """Add random jitter to stratification weights to avoid tidy proportions."""
    noisy = []
    for lo, hi, w in base_tiers:
        noisy_w = max(0.01, w + random.gauss(0, w * noise))
        noisy.append((lo, hi, noisy_w))
    return noisy


def _sample_evo_distance(evo_tiers, fallback_distance,
                         distribution='uniform', gamma_shape=2.0):
    """
    Randomly sample one distance value from the evolutionary-distance strata.
    Degrades to a fixed fallback_distance when evo_tiers is None.

    Parameters
    ----------
    evo_tiers : list | None — [[lo, hi, weight], ...]
    fallback_distance : float — fixed evolutionary distance (used without tiers)
    distribution : str — 'uniform' | 'gamma'
    gamma_shape : float — gamma distribution shape parameter (gamma mode only)

    Returns
    -------
    float — the sampled evolutionary distance for this draw (substitutions/site)
    """
    if not evo_tiers:
        return fallback_distance
    # Choose a tier by weight
    weights = [t[2] for t in evo_tiers]
    tier = random.choices(evo_tiers, weights=weights, k=1)[0]
    lo, hi = tier[0], tier[1]

    if distribution == 'gamma' and hi > lo:
        # Gamma distribution: mapped to [lo, hi]
        # with shape=gamma_shape, scale=(hi-lo)/gamma_shape
        # the mean ≈ lo + (hi-lo), but right-skewed (more near relatives)
        scale = (hi - lo) / gamma_shape
        raw = lo + random.gammavariate(gamma_shape, scale)
        return min(raw, hi) # truncate to the tier's upper bound
    else:
        return random.uniform(lo, hi)


def _apply_evo_pipeline(vseq, bg_seq, _sample_evo,
                        gc_amel, mh_len):
    """
    Apply the full evolution pipeline to a virus fragment:
      1. GC amelioration (if enabled)
      2. Microhomology (if enabled)

    Returns
    -------
    (processed_vseq, mh_info)
    """
    # GC amelioration
    if gc_amel and _sample_evo > 0:
        host_gc = sum(1 for b in bg_seq if b in 'GCgc') / max(len(bg_seq), 1)
        vseq = apply_gc_amelioration(vseq, host_gc, _sample_evo)

    # Microhomology
    vseq, mh_info = apply_microhomology(bg_seq, vseq, mh_len)
    return vseq, mh_info


# ============================================================================= #
# Paired-End data chunk (child process)
# ============================================================================= #

def _generate_chunk_pe(proc_id, cfg):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    sampler = MicrobeSampler(cfg['microbe_dir'],
                             use_coverage=cfg['use_coverage'],
                             allowed_files=cfg.get('microbe_allowed_files'))
    if cfg.get('virus_genomes_subset'):
        genomes = cfg['virus_genomes_subset']
    else:
        genomes = load_virus_genomes(cfg['virus_fasta'])
    virus_pool = generate_virus_pool(
        genomes, cfg.get('virus_fasta_label', cfg['virus_fasta']),
        cfg['virus_fragment_min'], cfg['virus_fragment_max'],
        cfg['virus_fragments_per_genome'],
        tiers=_noisy_tiers(cfg.get('virus_frag_tiers_pe') or _DEFAULT_PE_VIRUS_FRAG_TIERS),
    )
    hgt_gen = HGTSequenceGenerator(
        cfg['insertion_prob'], cfg['replacement_prob'], cfg['deletion_prob'])

    read_len = cfg['read_len']
    err_rate = cfg['error_rate']
    platform = cfg.get('seq_platform', 'illumina')
    evo_dist = cfg.get('evolutionary_distance', 0.0)
    evo_tiers = cfg.get('evolutionary_distance_tiers')
    host_mut = cfg.get('host_mutation_rate', 0.0)
    contam_rate = cfg.get('contamination_rate', 0.0)
    lbl_noise = cfg.get('label_noise', 0.0)
    bnd_shift = cfg.get('boundary_shift_max', 0)
    gc_amel = cfg.get('gc_amelioration', False)
    evo_dist_distrib = cfg.get('evo_distance_distribution', 'uniform')
    evo_gamma_shape = cfg.get('evo_distance_gamma_shape', 2.0)
    mh_len = cfg.get('microhomology_len', 0)

    r1_tmp = f"{cfg['temp_prefix']}_R1_{proc_id}.fasta"
    r2_tmp = f"{cfg['temp_prefix']}_R2_{proc_id}.fasta"
    bed_tmp = f"{cfg['temp_prefix']}_bed_{proc_id}.tsv"
    prov_tmp = f"{cfg['temp_prefix']}_prov_{proc_id}.tsv"

    # Negative-type distribution (YAML or built-in default)
    _neg_ratios = cfg.get('neg_type_ratios') or _DEFAULT_NEG_TYPE_RATIOS
    _neg_types = list(_neg_ratios.keys())
    _neg_weights = list(_neg_ratios.values())

    meta_pos = []

    with open(r1_tmp, 'w') as f1, \
         open(r2_tmp, 'w') as f2, \
         open(bed_tmp, 'w') as bf, \
         open(prov_tmp, 'w') as pf:

        # ---- positive samples -------------------------------------------------------
        # PE case quotas: allocated per the target distribution
        pe_quotas = {}
        _q_total = 0
        for _i, (case, ratio) in enumerate((cfg.get('pe_case_ratios') or _DEFAULT_PE_CASE_RATIOS).items()):
            # Base quota + Poisson jitter (σ≈√n, avoiding tidy integers)
            base = cfg['n_pos'] * ratio
            noisy = max(1, int(np.random.normal(base, max(1, base * 0.05))))
            pe_quotas[case] = noisy
            _q_total += noisy
        # Correct the total: rescale so the sum equals n_pos
        _scale = cfg['n_pos'] / _q_total
        pe_quotas = {c: max(1, round(v * _scale)) for c, v in pe_quotas.items()}
        _q_remain = cfg['n_pos'] - sum(pe_quotas.values())
        for _q_case in sorted(pe_quotas.keys()):
            if _q_remain <= 0:
                break
            pe_quotas[_q_case] += 1
            _q_remain -= 1

        pos_done = 0
        _pos_attempts = 0
        _force_accept = False
        _hard_limit = cfg['n_pos'] * 100 # hard upper limit: prevents an infinite loop
        while pos_done < cfg['n_pos']:
            _pos_attempts += 1
            if _pos_attempts > _hard_limit:
                print(f"[WARNING] PE positive: hit hard limit after {_hard_limit} "
                      f"attempts, generated {pos_done}/{cfg['n_pos']}")
                break
            if not _force_accept and _pos_attempts > cfg['n_pos'] * 20:
                _force_accept = True
            # -- Host background: multi-source or single-source --
            use_multi_host = (cfg.get('multi_source_host_enabled')
                              and random.random() < cfg.get('multi_source_host_prob', 0))
            if use_multi_host:
                n_src = random.randint(2, cfg.get('multi_source_host_max', 3))
                bg, bg_prov_list = build_composite_host(
                    sampler, 2 * read_len, n_src,
                    use_coverage=cfg['use_coverage'])
            else:
                bg, bg_prov_single = sampler.sample_fragment_with_provenance(
                    2 * read_len, use_coverage=cfg['use_coverage'])
                bg_prov_list = [bg_prov_single]

            if cfg['gc_bias'] and not gc_bias_filter(bg):
                continue
            # Host population diversity: simulate natural base mutations between strains
            if host_mut > 0:
                bg = apply_jukes_cantor(bg, host_mut)
            # Note: sequencing errors are applied uniformly after the insert is
            # generated (so virus regions also receive errors)

            # -- Decide: sandwich / multi-event / single-event --
            use_sandwich = (cfg.get('sandwich_enabled')
                            and random.random() < cfg.get('sandwich_prob', 0))
            use_multi = (cfg.get('multi_event_enabled')
                         and random.random() < cfg.get('multi_event_prob', 0))

            if use_sandwich:
                # -- Sandwich: force 2 virus events with a small gap --
                vseq1, vprov1 = random.choice(virus_pool)
                vseq2, vprov2 = random.choice(virus_pool)
                _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                vseq1 = apply_jukes_cantor(vseq1, _sample_evo)
                vseq2 = apply_jukes_cantor(vseq2, _sample_evo)
                vseq1, _ = _apply_evo_pipeline(vseq1, bg, _sample_evo, gc_amel, mh_len)
                vseq2, _ = _apply_evo_pipeline(vseq2, bg, _sample_evo, gc_amel, mh_len)
                _orig_vlens = [len(vseq1), len(vseq2)]
                try:
                    insert, events, modes, event_meta = \
                        hgt_gen.generate_multi_hgt_sample(
                            bg, [vseq1, vseq2],
                            min_gap=cfg.get('sandwich_min_gap', 5))
                except ValueError:
                    continue

                insert_len = len(insert)
                if insert_len < read_len:
                    continue

                virus_regions = events
                case_type, r1_vs, r1_ve, r2_vs, r2_ve = \
                    classify_pe_case_multi(virus_regions, read_len, insert_len)
                if case_type == 'INTERNAL':
                    continue
                if not _force_accept and pe_quotas.get(case_type, 0) <= 0:
                    continue
                if not _force_accept:
                    pe_quotas[case_type] -= 1

                insert = apply_platform_errors(insert, platform, err_rate)
                r1_seq = insert[:read_len]
                r2_seq = str(Seq(insert[-read_len:]).reverse_complement())

                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"
                f1.write(f">{sid} HGT case={case_type}\n{r1_seq}\n")
                f2.write(f">{sid} HGT case={case_type}\n{r2_seq}\n")

                for ei, ((hs, he), m) in enumerate(zip(events, modes)):
                    bf.write(
                        f"{sid}\t{ei}\t1\t{insert_len}\t{hs}\t{he}\t"
                        f"{he-hs}\t{m}\t{case_type}\t"
                        f"{r1_vs}\t{r1_ve}\t{r2_vs}\t{r2_ve}\n")

                # Provenance (vseq_idx alignment)
                _vp_all = [vprov1, vprov2]
                _vp_sorted = [_vp_all[em['vseq_idx']] for em in event_meta]
                _ov_sorted = [_orig_vlens[em['vseq_idx']] for em in event_meta]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, _vp_sorted, event_meta, insert_len)
                r1_segs = compute_read_segments(
                    insert_segs, 0, read_len, is_rc=False)
                r2_segs = compute_read_segments(
                    insert_segs, insert_len - read_len, read_len, is_rc=True)
                pf.write(_prov_rows(sid, 1, '+'.join(modes), case_type, 'R1', r1_segs))
                pf.write(_prov_rows(sid, 1, '+'.join(modes), case_type, 'R2', r2_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': 2,
                    'hgt_modes': modes,
                    'pe_case': case_type,
                    'events': [{'hgt_start': hs, 'hgt_end': he,
                                'virus_len': he - hs, 'hgt_mode': m,
                                'original_virus_len': _ov_sorted[ei]}
                               for ei, ((hs, he), m) in enumerate(zip(events, modes))],
                    'insert_len': insert_len,
                    'r1_vs': r1_vs, 'r1_ve': r1_ve,
                    'r2_vs': r2_vs, 'r2_ve': r2_ve,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

            elif use_multi:
                # -- Multi-event HGT --
                n_events = random.randint(2, cfg.get('multi_event_max', 3))
                _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                virus_seqs, virus_provs, _orig_vlens = [], [], []
                for _ in range(n_events):
                    vseq, vprov = random.choice(virus_pool)
                    vseq = apply_jukes_cantor(vseq, _sample_evo)
                    vseq, _ = _apply_evo_pipeline(vseq, bg, _sample_evo, gc_amel, mh_len)
                    _orig_vlens.append(len(vseq))
                    if len(vseq) > len(bg):
                        use_v = vseq[:len(bg) // (n_events + 1)]
                        if vprov['strand'] == '-':
                            vprov = {**vprov,
                                     'genome_start': vprov['genome_end'] - len(use_v)}
                        else:
                            vprov = {**vprov,
                                     'genome_end': vprov['genome_start'] + len(use_v)}
                    else:
                        use_v = vseq
                    virus_seqs.append(use_v)
                    virus_provs.append(vprov)

                try:
                    insert, events, modes, event_meta = \
                        hgt_gen.generate_multi_hgt_sample(
                            bg, virus_seqs,
                            min_gap=cfg.get('multi_event_min_gap', 20))
                except ValueError:
                    continue

                insert_len = len(insert)
                if insert_len < read_len:
                    continue

                case_type, r1_vs, r1_ve, r2_vs, r2_ve = \
                    classify_pe_case_multi(events, read_len, insert_len)
                if case_type == 'INTERNAL':
                    continue
                if not _force_accept and pe_quotas.get(case_type, 0) <= 0:
                    continue
                if not _force_accept:
                    pe_quotas[case_type] -= 1

                insert = apply_platform_errors(insert, platform, err_rate)
                r1_seq = insert[:read_len]
                r2_seq = str(Seq(insert[-read_len:]).reverse_complement())

                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"
                f1.write(f">{sid} HGT case={case_type}\n{r1_seq}\n")
                f2.write(f">{sid} HGT case={case_type}\n{r2_seq}\n")

                for ei, ((hs, he), m) in enumerate(zip(events, modes)):
                    bf.write(
                        f"{sid}\t{ei}\t1\t{insert_len}\t{hs}\t{he}\t"
                        f"{he-hs}\t{m}\t{case_type}\t"
                        f"{r1_vs}\t{r1_ve}\t{r2_vs}\t{r2_ve}\n")

                # Provenance (vseq_idx alignment)
                _vp_sorted = [virus_provs[em['vseq_idx']] for em in event_meta]
                _ov_sorted = [_orig_vlens[em['vseq_idx']] for em in event_meta]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, _vp_sorted, event_meta, insert_len)
                r1_segs = compute_read_segments(
                    insert_segs, 0, read_len, is_rc=False)
                r2_segs = compute_read_segments(
                    insert_segs, insert_len - read_len, read_len, is_rc=True)
                pf.write(_prov_rows(sid, 1, '+'.join(modes), case_type, 'R1', r1_segs))
                pf.write(_prov_rows(sid, 1, '+'.join(modes), case_type, 'R2', r2_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': n_events,
                    'hgt_modes': modes,
                    'pe_case': case_type,
                    'events': [{'hgt_start': hs, 'hgt_end': he,
                                'virus_len': he - hs, 'hgt_mode': m,
                                'original_virus_len': _orig_vlens[ei]}
                               for ei, ((hs, he), m) in enumerate(zip(events, modes))],
                    'insert_len': insert_len,
                    'r1_vs': r1_vs, 'r1_ve': r1_ve,
                    'r2_vs': r2_vs, 'r2_ve': r2_ve,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

            else:
                # -- Single-event HGT (PE case sampled directly) --
                # Strategy: first pick the target PE case, then constrain the
                # hgt_start range, and finally generate
                _avail_cases = [c for c, q in pe_quotas.items() if q > 0] if not _force_accept else None

                _generated = False
                _max_case_attempts = 15 if _avail_cases else 1

                for _ca in range(_max_case_attempts):
                    vseq, vprov = random.choice(virus_pool)
                    _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                    vseq = apply_jukes_cantor(vseq, _sample_evo)
                    vseq, _ = _apply_evo_pipeline(vseq, bg, _sample_evo, gc_amel, mh_len)
                    _orig_vlen = len(vseq)
                    if len(vseq) > len(bg):
                        use_v = vseq[:len(bg) // 2]
                        if vprov['strand'] == '-':
                            vprov = {**vprov, 'genome_start': vprov['genome_end'] - len(use_v)}
                        else:
                            vprov = {**vprov, 'genome_end': vprov['genome_start'] + len(use_v)}
                    else:
                        use_v = vseq

                    vl = len(use_v)

                    if _avail_cases:
                        random.shuffle(_avail_cases)
                        target_case = None
                        target_pos_range = None
                        for _tc in _avail_cases:
                            for _est_il in [2 * read_len + vl, 3 * read_len, 2 * read_len + vl // 2]:
                                _pr = _compute_pe_case_pos_range(_tc, vl, read_len, _est_il)
                                if _pr is not None:
                                    target_case = _tc
                                    target_pos_range = _pr
                                    break
                            if target_case:
                                break
                        if target_case is None:
                            continue
                    else:
                        target_case = None
                        target_pos_range = None

                    if target_case and target_pos_range:
                        lo, hi = target_pos_range
                        lo = max(0, min(lo, len(bg) - 1))
                        hi = max(lo, min(hi, len(bg) - 1))
                        if lo > hi:
                            continue
                        constrained_pos = random.randint(lo, hi)
                        try:
                            insert, (hgt_s, hgt_e), hgt_mode, bg_roff = \
                                hgt_gen.generate_positive_sample_at(bg, use_v, constrained_pos)
                        except (ValueError, IndexError):
                            continue
                    else:
                        try:
                            insert, (hgt_s, hgt_e), hgt_mode, bg_roff = \
                                hgt_gen.generate_positive_sample(bg, use_v)
                        except ValueError:
                            continue

                    insert_len = len(insert)
                    if insert_len < read_len:
                        continue

                    case_type, r1_vs, r1_ve, r2_vs, r2_ve = classify_pe_case(
                        hgt_s, hgt_e, read_len, insert_len)
                    if case_type == 'INTERNAL':
                        continue
                    if not _force_accept and pe_quotas.get(case_type, 0) <= 0:
                        continue
                    if not _force_accept:
                        pe_quotas[case_type] -= 1

                    _generated = True
                    break

                if not _generated:
                    continue

                insert = apply_platform_errors(insert, platform, err_rate)
                r1_seq = insert[:read_len]
                r2_seq = str(Seq(insert[-read_len:]).reverse_complement())

                v_len = hgt_e - hgt_s
                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"

                f1.write(f">{sid} HGT case={case_type}\n{r1_seq}\n")
                f2.write(f">{sid} HGT case={case_type}\n{r2_seq}\n")
                bf.write(
                    f"{sid}\t0\t1\t{insert_len}\t{hgt_s}\t{hgt_e}\t{v_len}\t"
                    f"{hgt_mode}\t{case_type}\t"
                    f"{r1_vs}\t{r1_ve}\t{r2_vs}\t{r2_ve}\n"
                )

                # Provenance
                _ev_meta = [{'hgt_s': hgt_s, 'hgt_e': hgt_e,
                             'bg_left_end': hgt_s,
                             'bg_right_offset': bg_roff}]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, [vprov], _ev_meta, insert_len)
                r1_segs = compute_read_segments(insert_segs, 0, read_len, is_rc=False)
                r2_segs = compute_read_segments(insert_segs,
                                                 insert_len - read_len,
                                                 read_len, is_rc=True)
                pf.write(_prov_rows(sid, 1, hgt_mode, case_type, 'R1', r1_segs))
                pf.write(_prov_rows(sid, 1, hgt_mode, case_type, 'R2', r2_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': 1,
                    'hgt_modes': [hgt_mode],
                    'pe_case': case_type,
                    'events': [{'hgt_start': hgt_s, 'hgt_end': hgt_e,
                                'virus_len': v_len, 'hgt_mode': hgt_mode,
                                'original_virus_len': _orig_vlen}],
                    'insert_len': insert_len,
                    'r1_vs': r1_vs, 'r1_ve': r1_ve,
                    'r2_vs': r2_vs, 'r2_ve': r2_ve,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

        # ---- Background pool (used for PCR chimeras) -----------------------------
        bg_pool_with_prov = [
            sampler.sample_fragment_with_provenance(
                read_len, use_coverage=cfg['use_coverage'])
            for _ in range(1000)
        ]

        # ---- negative samples -------------------------------------------------------
        neg_done = 0
        while neg_done < cfg['n_neg']:
            bg, bg_prov = sampler.sample_fragment_with_provenance(
                2 * read_len, use_coverage=cfg['use_coverage'])
            if cfg['gc_bias'] and not gc_bias_filter(bg):
                continue
            # Note: sequencing errors are applied uniformly to the final seq

            neg_type = random.choices(_neg_types, weights=_neg_weights, k=1)[0]

            if neg_type == 'NEG':
                seq = bg
                insert_segs = generate_host_only(bg_prov, 2 * read_len)
            elif neg_type == 'NEG_VIRUS':
                seq, insert_segs = generate_virus_only(virus_pool, 2 * read_len)
            elif neg_type == 'NEG_HARD':
                seq, insert_segs = generate_negative_hard(
                    bg, bg_prov, virus_pool, bg_pool_with_prov)
            elif neg_type == 'NEG_CONSERVED':
                seq, insert_segs = generate_neg_conserved(
                    bg_pool_with_prov, 2 * read_len)
            elif neg_type == 'NEG_HOMOLOGY':
                seq, insert_segs = generate_neg_homology(
                    bg_pool_with_prov, virus_pool, 2 * read_len)
            elif neg_type == 'NEG_LOW_COMPLEXITY':
                seq, insert_segs = generate_neg_low_complexity(2 * read_len, bg_pool_with_prov)
            elif neg_type == 'NEG_GC_MATCHED_CHIMERIC':
                seq, insert_segs = generate_neg_gc_matched_chimeric(
                    bg, bg_prov, virus_pool, 2 * read_len)
            elif neg_type == 'NEG_AMALGAMATED_HGT':
                seq, insert_segs = generate_neg_amalgamated_hgt(
                    bg, bg_prov, virus_pool, 2 * read_len)
            elif neg_type == 'NEG_PARTIAL_VIRUS_HIT':
                seq, insert_segs = generate_neg_partial_virus_hit(
                    bg_pool_with_prov, virus_pool, 2 * read_len)
            elif neg_type == 'NEG_PE_HOST_DIVERSITY':
                seq, insert_segs = generate_pe_neg_host_diversity(
                    bg_pool_with_prov, virus_pool, 2 * read_len)
            elif neg_type == 'NEG_PE_GC_SHIFT_REGION':
                seq, insert_segs = generate_pe_neg_gc_shift_region(
                    bg, bg_prov, 2 * read_len)
            elif neg_type == 'NEG_PE_CODON_BOUNDARY':
                seq, insert_segs = generate_pe_neg_codon_boundary(
                    bg, bg_prov, 2 * read_len)
            else: # NEG_PCR
                seq, insert_segs = generate_pcr_chimera(
                    bg_pool_with_prov, 2 * read_len)

            # Apply sequencing errors uniformly (so all sample types receive platform errors)
            seq = apply_platform_errors(seq, platform, err_rate)

            # Contamination injection
            if contam_rate > 0:
                seq, insert_segs = apply_contamination(
                    seq, insert_segs, virus_pool, contam_rate)

            # Label noise
            eff_label = apply_label_noise(0, lbl_noise)

            r1_seq = seq[:read_len]
            r2_seq = str(Seq(seq[-read_len:]).reverse_complement())

            gid = cfg['start_neg_id'] + neg_done
            sid = f"neg_{gid:08d}"

            f1.write(f">{sid} {neg_type}\n{r1_seq}\n")
            f2.write(f">{sid} {neg_type}\n{r2_seq}\n")
            bf.write(
                f"{sid}\t{eff_label}\t0\t{2*read_len}\t-1\t-1\t0\t"
                f"none\t{neg_type}\t-1\t-1\t-1\t-1\n"
            )

            r1_segs = compute_read_segments(
                insert_segs, 0, read_len, is_rc=False)
            r2_segs = compute_read_segments(
                insert_segs, len(seq) - read_len, read_len, is_rc=True)
            pf.write(_prov_rows(sid, eff_label, 'none', neg_type, 'R1',
                                r1_segs, note=neg_type))
            pf.write(_prov_rows(sid, eff_label, 'none', neg_type, 'R2',
                                r2_segs, note=neg_type))
            neg_done += 1

    return meta_pos, bed_tmp, prov_tmp, (r1_tmp, r2_tmp)


# ============================================================================= #
# Contig data chunk (child process)
# ============================================================================= #

def _generate_chunk_contig(proc_id, cfg):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    sampler = MicrobeSampler(cfg['microbe_dir'],
                             use_coverage=cfg['use_coverage'],
                             allowed_files=cfg.get('microbe_allowed_files'))
    if cfg.get('virus_genomes_subset'):
        genomes = cfg['virus_genomes_subset']
    else:
        genomes = load_virus_genomes(cfg['virus_fasta'])
    virus_pool = generate_virus_pool(
        genomes, cfg.get('virus_fasta_label', cfg['virus_fasta']),
        cfg['virus_fragment_min'], cfg['virus_fragment_max'],
        cfg['virus_fragments_per_genome'],
        tiers=_noisy_tiers(cfg.get('virus_frag_tiers_contig') or _DEFAULT_CONTIG_VIRUS_FRAG_TIERS),
    )
    # Contig HGT mode: sampled randomly from the specified range (each run gets
    # natural jitter)
    _mode_ranges = cfg.get('contig_hgt_mode_ranges') or _DEFAULT_CONTIG_HGT_MODE_RANGES
    _ins = random.uniform(*_mode_ranges['insertion'])
    _rep = random.uniform(*_mode_ranges['replacement'])
    _del = random.uniform(*_mode_ranges['deletion'])
    hgt_gen = HGTSequenceGenerator(_ins, _rep, _del)

    err_rate = cfg['error_rate']
    c_min = cfg['contig_min_length']
    c_max = cfg['contig_max_length']
    platform = cfg.get('seq_platform', 'illumina')
    evo_dist = cfg.get('evolutionary_distance', 0.0)
    evo_tiers = cfg.get('evolutionary_distance_tiers')
    host_mut = cfg.get('host_mutation_rate', 0.0)
    contam_rate = cfg.get('contamination_rate', 0.0)
    lbl_noise = cfg.get('label_noise', 0.0)
    bnd_shift = cfg.get('boundary_shift_max', 0)
    asm_rate = cfg.get('assembly_artifact_rate', 0.0)
    gc_amel = cfg.get('gc_amelioration', False)
    evo_dist_distrib = cfg.get('evo_distance_distribution', 'uniform')
    evo_gamma_shape = cfg.get('evo_distance_gamma_shape', 2.0)
    mh_len = cfg.get('microhomology_len', 0)

    fasta_tmp = f"{cfg['temp_prefix']}_contig_{proc_id}.fasta"
    bed_tmp = f"{cfg['temp_prefix']}_bed_{proc_id}.tsv"
    prov_tmp = f"{cfg['temp_prefix']}_prov_{proc_id}.tsv"

    meta_pos = []

    # Negative-type distribution (YAML or built-in default)
    _neg_ratios = cfg.get('neg_type_ratios') or _DEFAULT_NEG_TYPE_RATIOS
    _neg_types = list(_neg_ratios.keys())
    _neg_weights = list(_neg_ratios.values())

    with open(fasta_tmp, 'w') as fa, \
         open(bed_tmp, 'w') as bf, \
         open(prov_tmp, 'w') as pf:

        # ---- positive samples -------------------------------------------------------
        pos_done = 0
        while pos_done < cfg['n_pos']:
            target_len = int(math.exp(
                random.uniform(math.log(c_min), math.log(c_max))))
            bg_len = min(target_len, sampler.max_len)
            if bg_len < 50:
                continue

            # -- Host background: multi-source or single-source --
            use_multi_host = (cfg.get('multi_source_host_enabled')
                              and random.random() < cfg.get('multi_source_host_prob', 0))
            if use_multi_host:
                n_src = random.randint(2, cfg.get('multi_source_host_max', 3))
                bg, bg_prov_list = build_composite_host(
                    sampler, bg_len, n_src,
                    use_coverage=cfg['use_coverage'])
            else:
                bg, bg_prov_single = sampler.sample_fragment_with_provenance(
                    bg_len, use_coverage=cfg['use_coverage'])
                bg_prov_list = [bg_prov_single]

            if cfg['gc_bias'] and not gc_bias_filter(bg):
                continue
            # Host population diversity
            if host_mut > 0:
                bg = apply_jukes_cantor(bg, host_mut)

            # -- Decide: sandwich / multi-event / single-event --
            use_sandwich = (cfg.get('sandwich_enabled')
                            and random.random() < cfg.get('sandwich_prob', 0))
            use_multi = (cfg.get('multi_event_enabled')
                         and random.random() < cfg.get('multi_event_prob', 0))

            if use_sandwich:
                vseq1, vprov1 = random.choice(virus_pool)
                vseq2, vprov2 = random.choice(virus_pool)
                _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                vseq1 = apply_jukes_cantor(vseq1, _sample_evo)
                vseq2 = apply_jukes_cantor(vseq2, _sample_evo)
                vseq1, _ = _apply_evo_pipeline(vseq1, bg, _sample_evo, gc_amel, mh_len)
                vseq2, _ = _apply_evo_pipeline(vseq2, bg, _sample_evo, gc_amel, mh_len)
                _orig_vlens = [len(vseq1), len(vseq2)]
                try:
                    contig, events, modes, event_meta = \
                        hgt_gen.generate_multi_hgt_sample(
                            bg, [vseq1, vseq2],
                            min_gap=cfg.get('sandwich_min_gap', 5))
                except ValueError:
                    continue

                contig = apply_platform_errors(contig, platform, err_rate)
                # Apply assembly artifact
                if asm_rate > 0:
                    contig, _ = apply_assembly_artifact(contig, asm_rate)
                seq_len = len(contig)
                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"

                fa.write(f">{sid} HGT len={seq_len}\n{contig}\n")
                for ei, ((hs, he), m) in enumerate(zip(events, modes)):
                    bf.write(f"{sid}\t{ei}\t1\t{seq_len}\t{hs}\t{he}\t"
                             f"{he-hs}\t{m}\n")

                # fix: virus_provs by vseq_idx with event_meta alignment
                _vp_all = [vprov1, vprov2]
                _vp_sorted = [_vp_all[em['vseq_idx']] for em in event_meta]
                _ov_sorted = [_orig_vlens[em['vseq_idx']] for em in event_meta]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, _vp_sorted, event_meta, seq_len)
                contig_segs = compute_read_segments(
                    insert_segs, 0, seq_len, is_rc=False)
                pf.write(_prov_rows(sid, 1, '+'.join(modes), 'contig',
                                    'CONTIG', contig_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': 2,
                    'hgt_modes': modes,
                    'events': [{'hgt_start': hs, 'hgt_end': he,
                                'virus_len': he - hs, 'hgt_mode': m,
                                'original_virus_len': _ov_sorted[ei]}
                               for ei, ((hs, he), m) in enumerate(zip(events, modes))],
                    'seq_len': seq_len,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

            elif use_multi:
                n_events = random.randint(2, cfg.get('multi_event_max', 3))
                _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                virus_seqs, virus_provs, _orig_vlens = [], [], []
                for _ in range(n_events):
                    vseq, vprov = random.choice(virus_pool)
                    vseq = apply_jukes_cantor(vseq, _sample_evo)
                    vseq, _ = _apply_evo_pipeline(vseq, bg, _sample_evo, gc_amel, mh_len)
                    _orig_vlens.append(len(vseq))
                    if len(vseq) > bg_len:
                        use_v = vseq[:bg_len // (n_events + 1)]
                        if vprov['strand'] == '-':
                            vprov = {**vprov,
                                     'genome_start': vprov['genome_end'] - len(use_v)}
                        else:
                            vprov = {**vprov,
                                     'genome_end': vprov['genome_start'] + len(use_v)}
                    else:
                        use_v = vseq
                    virus_seqs.append(use_v)
                    virus_provs.append(vprov)

                try:
                    contig, events, modes, event_meta = \
                        hgt_gen.generate_multi_hgt_sample(
                            bg, virus_seqs,
                            min_gap=cfg.get('multi_event_min_gap', 20))
                except ValueError:
                    continue

                contig = apply_platform_errors(contig, platform, err_rate)
                # Apply assembly artifact
                if asm_rate > 0:
                    contig, _ = apply_assembly_artifact(contig, asm_rate)
                seq_len = len(contig)
                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"

                fa.write(f">{sid} HGT len={seq_len}\n{contig}\n")
                for ei, ((hs, he), m) in enumerate(zip(events, modes)):
                    bf.write(f"{sid}\t{ei}\t1\t{seq_len}\t{hs}\t{he}\t"
                             f"{he-hs}\t{m}\n")

                # fix: virus_provs by vseq_idx with event_meta alignment
                _vp_sorted = [virus_provs[em['vseq_idx']] for em in event_meta]
                _ov_sorted = [_orig_vlens[em['vseq_idx']] for em in event_meta]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, _vp_sorted, event_meta, seq_len)
                contig_segs = compute_read_segments(
                    insert_segs, 0, seq_len, is_rc=False)
                pf.write(_prov_rows(sid, 1, '+'.join(modes), 'contig',
                                    'CONTIG', contig_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': n_events,
                    'hgt_modes': modes,
                    'events': [{'hgt_start': hs, 'hgt_end': he,
                                'virus_len': he - hs, 'hgt_mode': m,
                                'original_virus_len': _ov_sorted[ei]}
                               for ei, ((hs, he), m) in enumerate(zip(events, modes))],
                    'seq_len': seq_len,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

            else:
                # -- Single-event HGT (original logic) --
                vseq, vprov = random.choice(virus_pool)
                _sample_evo = _sample_evo_distance(evo_tiers, evo_dist, evo_dist_distrib, evo_gamma_shape)
                vseq = apply_jukes_cantor(vseq, _sample_evo)
                vseq, _ = _apply_evo_pipeline(vseq, bg, _sample_evo, gc_amel, mh_len)
                _orig_vlen = len(vseq)
                if len(vseq) > bg_len:
                    use_v = vseq[:bg_len // 2] if bg_len > 10 else vseq[:5]
                    if vprov['strand'] == '-':
                        vprov = {**vprov,
                                 'genome_start': vprov['genome_end'] - len(use_v)}
                    else:
                        vprov = {**vprov,
                                 'genome_end': vprov['genome_start'] + len(use_v)}
                else:
                    use_v = vseq

                try:
                    contig, (hgt_s, hgt_e), hgt_mode, bg_roff = \
                        hgt_gen.generate_positive_sample(bg, use_v)
                except ValueError:
                    continue

                contig = apply_platform_errors(contig, platform, err_rate)
                # Apply assembly artifact
                if asm_rate > 0:
                    contig, _ = apply_assembly_artifact(contig, asm_rate)
                seq_len = len(contig)
                v_len = hgt_e - hgt_s
                gid = cfg['start_pos_id'] + pos_done
                sid = f"pos_{gid:08d}"

                fa.write(f">{sid} HGT len={seq_len}\n{contig}\n")
                bf.write(f"{sid}\t0\t1\t{seq_len}\t{hgt_s}\t{hgt_e}\t"
                         f"{v_len}\t{hgt_mode}\n")

                _ev_meta = [{'hgt_s': hgt_s, 'hgt_e': hgt_e,
                             'bg_left_end': hgt_s,
                             'bg_right_offset': bg_roff}]
                insert_segs = compute_insert_segments_multi(
                    bg_prov_list, [vprov], _ev_meta, seq_len)
                contig_segs = compute_read_segments(
                    insert_segs, 0, seq_len, is_rc=False)
                pf.write(_prov_rows(sid, 1, hgt_mode, 'contig', 'CONTIG',
                                    contig_segs))

                meta_pos.append(_enrich_meta({
                    'seq_id': sid, 'label': 1,
                    'n_events': 1,
                    'hgt_modes': [hgt_mode],
                    'events': [{'hgt_start': hgt_s, 'hgt_end': hgt_e,
                                'virus_len': v_len, 'hgt_mode': hgt_mode,
                                'original_virus_len': _orig_vlen}],
                    'seq_len': seq_len,
                }, cfg, evolutionary_distance=_sample_evo))
                pos_done += 1

        # ---- Background pool -------------------------------------------------------
        bg_pool_with_prov = [
            sampler.sample_fragment_with_provenance(
                min(2000, sampler.max_len),
                use_coverage=cfg['use_coverage'])
            for _ in range(200)
        ]

        # ---- negative samples -------------------------------------------------------
        neg_done = 0
        while neg_done < cfg['n_neg']:
            target_len = int(math.exp(
                random.uniform(math.log(c_min), math.log(c_max))))
            bg_len = min(target_len, sampler.max_len)
            if bg_len < 50:
                continue

            bg, bg_prov = sampler.sample_fragment_with_provenance(
                bg_len, use_coverage=cfg['use_coverage'])
            if cfg['gc_bias'] and not gc_bias_filter(bg):
                continue
            # Note: sequencing errors are applied uniformly to the final seq

            neg_type = random.choices(_neg_types, weights=_neg_weights, k=1)[0]

            if neg_type == 'NEG':
                seq = bg
                insert_segs = generate_host_only(bg_prov, bg_len)
            elif neg_type == 'NEG_VIRUS':
                seq, insert_segs = generate_virus_only(virus_pool, bg_len)
            elif neg_type == 'NEG_HARD':
                seq, insert_segs = generate_negative_hard(
                    bg, bg_prov, virus_pool, bg_pool_with_prov)
            elif neg_type == 'NEG_CONSERVED':
                seq, insert_segs = generate_neg_conserved(
                    bg_pool_with_prov, bg_len)
            elif neg_type == 'NEG_HOMOLOGY':
                seq, insert_segs = generate_neg_homology(
                    bg_pool_with_prov, virus_pool, bg_len)
            elif neg_type == 'NEG_LOW_COMPLEXITY':
                seq, insert_segs = generate_neg_low_complexity(bg_len, bg_pool_with_prov)
            elif neg_type == 'NEG_GC_MATCHED_CHIMERIC':
                seq, insert_segs = generate_neg_gc_matched_chimeric(
                    bg, bg_prov, virus_pool, bg_len)
            elif neg_type == 'NEG_AMALGAMATED_HGT':
                seq, insert_segs = generate_neg_amalgamated_hgt(
                    bg, bg_prov, virus_pool, bg_len)
            elif neg_type == 'NEG_PARTIAL_VIRUS_HIT':
                seq, insert_segs = generate_neg_partial_virus_hit(
                    bg_pool_with_prov, virus_pool, bg_len)
            elif neg_type == 'NEG_PE_HOST_DIVERSITY':
                seq, insert_segs = generate_pe_neg_host_diversity(
                    bg_pool_with_prov, virus_pool, bg_len)
            elif neg_type == 'NEG_PE_GC_SHIFT_REGION':
                seq, insert_segs = generate_pe_neg_gc_shift_region(
                    bg, bg_prov, bg_len)
            elif neg_type == 'NEG_PE_CODON_BOUNDARY':
                seq, insert_segs = generate_pe_neg_codon_boundary(
                    bg, bg_prov, bg_len)
            else: # NEG_PCR
                seq, insert_segs = generate_pcr_chimera(
                    bg_pool_with_prov, bg_len)

            # Apply sequencing errors uniformly
            seq = apply_platform_errors(seq, platform, err_rate)

            # Contamination injection
            if contam_rate > 0:
                seq, insert_segs = apply_contamination(
                    seq, insert_segs, virus_pool, contam_rate)

            # Label noise
            eff_label = apply_label_noise(0, lbl_noise)

            seq_len = len(seq)
            gid = cfg['start_neg_id'] + neg_done
            sid = f"neg_{gid:08d}"

            fa.write(f">{sid} {neg_type} len={seq_len}\n{seq}\n")
            bf.write(f"{sid}\t{eff_label}\t0\t{seq_len}\t-1\t-1\t0\t{neg_type}\n")
            contig_segs_neg = compute_read_segments(
                insert_segs, 0, seq_len, is_rc=False)
            pf.write(_prov_rows(sid, eff_label, 'none', neg_type, 'CONTIG',
                                contig_segs_neg, note=neg_type))
            neg_done += 1

    return meta_pos, bed_tmp, prov_tmp, fasta_tmp


# ============================================================================= #
# Merge helpers
# ============================================================================= #

def _merge_tsv(temps, out_path, header):
    with open(out_path, 'w') as out:
        out.write(header)
        for tmp in temps:
            with open(tmp) as f:
                for line in f:
                    if not line.startswith('#'):
                        out.write(line)
            os.remove(tmp)


def _merge_fasta(temps, out_path):
    with open(out_path, 'w') as out:
        for tmp in temps:
            with open(tmp) as f:
                out.write(f.read())
            os.remove(tmp)


def _merge_paired_fasta(temp_pairs, r1_path, r2_path):
    with open(r1_path, 'w') as o1, open(r2_path, 'w') as o2:
        for r1_tmp, r2_tmp in temp_pairs:
            with open(r1_tmp) as f: o1.write(f.read())
            with open(r2_tmp) as f: o2.write(f.read())
            os.remove(r1_tmp)
            os.remove(r2_tmp)


# ============================================================================= #
# simulated data statistics report
# ============================================================================= #

def _print_statistics(mode, all_meta_pos, breakpoints_bed_path):
    """Print the complete statistics report for the simulated dataset."""
    from collections import Counter

    n_pos = len(all_meta_pos)
    total = n_pos

    # Count negative samples from the BED file
    neg_types = Counter()
    with open(breakpoints_bed_path) as bf:
        for line in bf:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            label = int(parts[2])
            if label == 0:
                total += 1
                # PE: col 9 = pe_case (has neg_type); Contig: col 7 = hgt_mode (has neg_type)
                if mode == 'paired_end' and len(parts) >= 9:
                    neg_types[parts[8]] += 1
                elif mode == 'contig' and len(parts) >= 8:
                    neg_types[parts[7]] += 1
    n_neg = total - n_pos

    # Positive-sample statistics
    pe_cases = Counter()
    hgt_modes_all = Counter()
    n_events_dist = Counter()
    virus_lens = [] # Effective length (hgt_e - hgt_s; actual length after insertion)
    orig_virus_lens = [] # Original length (fragment length as sampled from the virus pool, before truncation)
    for m in all_meta_pos:
        if mode == 'paired_end' and 'pe_case' in m:
            pe_cases[m['pe_case']] += 1
        n_events_dist[m['n_events']] += 1
        for ev in m.get('events', []):
            hgt_modes_all[ev['hgt_mode']] += 1
            virus_lens.append(ev['virus_len'])
            if 'original_virus_len' in ev:
                orig_virus_lens.append(ev['original_virus_len'])

    # -- Print report --
    print()
    print("=" * 64)
    print(" simulated data statistics report (Simulation Statistics)")
    print("=" * 64)
    print(f" mode: {mode}")
    print(f" total samples: {total:,}")
    print()

    # 1. Positive/negative sample ratio
    print(" [1] Positive/negative sample ratio")
    print(f" {'type':<12} {'count':>10} {'of total':>8} {'of class':>8}")
    print(f" {'─'*42}")
    pos_pct = n_pos / total * 100 if total else 0
    neg_pct = n_neg / total * 100 if total else 0
    print(f" {'positive samples':<12} {n_pos:>10,} {pos_pct:>7.1f}% {'100.0%':>8}")
    print(f" {'negative samples':<12} {n_neg:>10,} {neg_pct:>7.1f}% {'100.0%':>8}")
    print(f" {'total':<12} {total:>10,} {'100.0%':>8}")
    print()

    # 2. PE case distribution (PE mode only)
    if mode == 'paired_end' and pe_cases:
        print(" [2] PE Case distribution (positive samples)")
        print(f" {'PE Case':<25} {'count':>8} {'of positive samples':>8} {'of total':>8}")
        print(f" {'─'*52}")
        for case in sorted(pe_cases.keys()):
            cnt = pe_cases[case]
            print(f" {case:<25} {cnt:>8,} {cnt/n_pos*100:>7.1f}% {cnt/total*100:>7.1f}%")
        print()

    # 3. Negative-sample type distribution
    if neg_types:
        print(f" [{'3' if mode == 'paired_end' else '2'}] Negative-sample type distribution")
        print(f" {'type':<15} {'count':>8} {'of negatives':>8} {'of total':>8}")
        print(f" {'─'*42}")
        for nt in sorted(neg_types.keys()):
            cnt = neg_types[nt]
            print(f" {nt:<15} {cnt:>8,} {cnt/n_neg*100:>7.1f}% {cnt/total*100:>7.1f}%")
        print()

    # 4. HGT event mode distribution
    idx = 4 if mode == 'paired_end' else 3
    if hgt_modes_all:
        total_events = sum(hgt_modes_all.values())
        print(f" [{idx}] HGT event mode distribution (total {total_events:,} events)")
        print(f" {'mode':<25} {'count':>8} {'fraction':>8}")
        print(f" {'─'*42}")
        for m in sorted(hgt_modes_all.keys()):
            cnt = hgt_modes_all[m]
            print(f" {m:<25} {cnt:>8,} {cnt/total_events*100:>7.1f}%")
        print()

    # 5. Multi-event distribution
    idx += 1
    print(f" [{idx}] Event-count distribution (positive samples)")
    print(f" {'events':<10} {'samples':>8} {'of positive samples':>8}")
    print(f" {'─'*28}")
    for ne in sorted(n_events_dist.keys()):
        cnt = n_events_dist[ne]
        print(f" {ne:<10} {cnt:>8,} {cnt/n_pos*100:>7.1f}%")
    print()

    # 6. Virus fragment length distribution
    idx += 1

    def _print_length_stats(label, lens_list):
        """Print summary statistics for one group of lengths."""
        if not lens_list:
            return
        arr = np.array(lens_list)
        print(f" {label}")
        print(f" {'statistic':<12} {'value':>10}")
        print(f" {'─'*24}")
        print(f" {'min':<12} {arr.min():>10,} bp")
        print(f" {'25%':<12} {np.percentile(arr, 25):>10,.0f} bp")
        print(f" {'median':<12} {np.median(arr):>10,.0f} bp")
        print(f" {'75%':<12} {np.percentile(arr, 75):>10,.0f} bp")
        print(f" {'max':<12} {arr.max():>10,} bp")
        print(f" {'mean':<12} {arr.mean():>10,.1f} bp")
        print(f" {'standard deviation':<12} {arr.std():>10,.1f} bp")

        bins = [0, 200, 500, 1000, 2000, 5000, 10000, 50000, float('inf')]
        bin_labels = ['0-200', '200-500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '10k-50k', '>50k']
        print(f"\n length distribution by bin:")
        print(f" {'range':<12} {'count':>8} {'fraction':>8}")
        print(f" {'─'*30}")
        for lo, hi, bl in zip(bins[:-1], bins[1:], bin_labels):
            cnt = int(((arr >= lo) & (arr < hi)).sum())
            if cnt > 0:
                print(f" {bl:<12} {cnt:>8,} {cnt/len(lens_list)*100:>7.1f}%")
        print()

    if mode == 'paired_end':
        # PE mode: show the original length (used for tier validation) and the effective length (actual in-read)
        if orig_virus_lens:
            print(f" [{idx}] Original virus fragment length distribution (sampled from the virus pool, total {len(orig_virus_lens):,})")
            _print_length_stats("original length (before truncation; used for tier validation)", orig_virus_lens)
        idx += 1
        if virus_lens:
            print(f" [{idx}] Effective virus fragment length distribution (in-read after insertion, total {len(virus_lens):,})")
            _print_length_stats("effective length (hgt_end - hgt_start)", virus_lens)
    else:
        # Contig mode: effective length ≈ original length (contigs are long enough)
        if virus_lens:
            print(f" [{idx}] Virus fragment length distribution (total {len(virus_lens):,} fragments)")
            _print_length_stats("virus fragment length (hgt_end - hgt_start)", virus_lens)
        if orig_virus_lens and virus_lens:
            # If the original and effective lengths differ, also show the truncation fraction
            orig_arr = np.array(orig_virus_lens)
            eff_arr = np.array(virus_lens)
            truncated = int((orig_arr != eff_arr).sum())
            if truncated > 0:
                print(f" note: {truncated}/{len(orig_virus_lens)} fragments truncated "
                      f"(original length > contig background length)")
                print()

    # Difficulty distribution (based on positive samples)
    idx += 1
    if all_meta_pos:
        diff_dist = Counter()
        conf_values = []
        for m in all_meta_pos:
            if 'difficulty' in m:
                diff_dist[m['difficulty']] += 1
            if 'label_confidence' in m:
                conf_values.append(m['label_confidence'])
        if diff_dist:
            print(f" [{idx}] difficulty distribution (positive samples)")
            print(f" {'difficulty':<12} {'count':>8} {'of positive samples':>10}")
            print(f" {'─'*32}")
            for d in ['easy', 'medium', 'hard', 'extreme']:
                cnt = diff_dist.get(d, 0)
                if cnt > 0:
                    print(f" {d:<12} {cnt:>8,} {cnt/n_pos*100:>9.1f}%")
            print()
        if conf_values:
            conf_arr = np.array(conf_values)
            print(f" label-confidence stats: mean={conf_arr.mean():.3f} "
                  f"min={conf_arr.min():.3f} max={conf_arr.max():.3f}")
            print()

    print("=" * 64)


def _split_seed_offset(split_name: str) -> int:
    """Deterministic seed offset: does not rely on Python hash() (PYTHONHASHSEED is randomized)."""
    return {'train': 100000, 'val': 200000, 'test': 300000}[split_name]


# ============================================================================= #
# DatasetGenerator — core orchestrator
# ============================================================================= #

class DatasetGenerator:
    """Generate a complete dataset from a SimulatorConfig (supports multiprocessing)."""

    def __init__(self, config: SimulatorConfig):
        self.config = config

    def generate(self) -> SimulationOutput:
        cfg = self.config
        if cfg.genome_isolation:
            return self._generate_with_isolation()

        random.seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)
        tmp_dir = os.path.join(cfg.output_dir, '_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        n_procs = cfg.processes or mp.cpu_count()
        print(f"[DatasetGenerator] mode={cfg.mode} n_pos={cfg.n_pos} "
              f"n_neg={cfg.n_neg} processes={n_procs}")

        pos_per = math.ceil(cfg.n_pos / n_procs)
        neg_per = math.ceil(cfg.n_neg / n_procs)

        base = dict(
            microbe_dir = cfg.microbe_dir,
            virus_fasta = cfg.virus_fasta,
            virus_fragment_min = cfg.virus_fragment_min,
            virus_fragment_max = cfg.virus_fragment_max,
            virus_fragments_per_genome = cfg.virus_fragments_per_genome,
            insertion_prob = cfg.insertion_prob,
            replacement_prob = cfg.replacement_prob,
            deletion_prob = cfg.deletion_prob,
            error_rate = cfg.error_rate,
            read_len = cfg.read_len,
            contig_min_length = cfg.contig_min_length,
            contig_max_length = cfg.contig_max_length,
            gc_bias = cfg.gc_bias,
            use_coverage = cfg.use_coverage,
            temp_prefix = os.path.join(tmp_dir, 'chunk'),
            multi_event_enabled = cfg.multi_event_enabled,
            multi_event_prob = cfg.multi_event_prob,
            multi_event_max = cfg.multi_event_max,
            multi_event_min_gap = cfg.multi_event_min_gap,
            multi_source_host_enabled = cfg.multi_source_host_enabled,
            multi_source_host_prob = cfg.multi_source_host_prob,
            multi_source_host_max = cfg.multi_source_host_max,
            sandwich_enabled = cfg.sandwich_enabled,
            sandwich_prob = cfg.sandwich_prob,
            sandwich_min_gap = cfg.sandwich_min_gap,
            # Stratified/distribution configuration (None = use built-in defaults)
            virus_frag_tiers_pe = cfg.virus_frag_tiers_pe,
            virus_frag_tiers_contig = cfg.virus_frag_tiers_contig,
            pe_case_ratios = cfg.pe_case_ratios,
            neg_type_ratios = cfg.neg_type_ratios,
            contig_hgt_mode_ranges = cfg.contig_hgt_mode_ranges,
            # Biological-realism augmentation parameters
            seq_platform = cfg.seq_platform,
            coverage_distribution = cfg.coverage_distribution,
            coverage_zipf_alpha = cfg.coverage_zipf_alpha,
            batch_effect_sigma = cfg.batch_effect_sigma,
            contamination_rate = cfg.contamination_rate,
            contaminant_dir = cfg.contaminant_dir,
            assembly_artifact_rate = cfg.assembly_artifact_rate,
            label_noise = cfg.label_noise,
            boundary_shift_max = cfg.boundary_shift_max,
            evolutionary_distance = cfg.evolutionary_distance,
            evolutionary_distance_tiers = cfg.evolutionary_distance_tiers,
            host_mutation_rate = cfg.host_mutation_rate,
            open_set_virus = cfg.open_set_virus,
            gc_amelioration = cfg.gc_amelioration,
            evo_distance_distribution = cfg.evo_distance_distribution,
            evo_distance_gamma_shape = cfg.evo_distance_gamma_shape,
            microhomology_len = cfg.microhomology_len,
        )

        worker = (_generate_chunk_pe
                  if cfg.mode == 'paired_end'
                  else _generate_chunk_contig)

        pool = mp.Pool(processes=n_procs)
        futures = []
        s_pos = s_neg = 0
        for i in range(n_procs):
            pchunk = min(pos_per, cfg.n_pos - s_pos)
            nchunk = min(neg_per, cfg.n_neg - s_neg)
            if pchunk <= 0 and nchunk <= 0:
                break
            pcfg = {**base,
                    'n_pos': max(pchunk, 0),
                    'n_neg': max(nchunk, 0),
                    'start_pos_id': s_pos,
                    'start_neg_id': s_neg,
                    'seed': cfg.seed + i + 1}
            futures.append(pool.apply_async(worker, args=(i, pcfg)))
            s_pos += pchunk
            s_neg += nchunk

        pool.close()
        pool.join()

        all_meta_pos, temp_beds, temp_provs, temp_fastas = [], [], [], []
        for fut in futures:
            meta_pos, bed_tmp, prov_tmp, fasta_info = fut.get()
            all_meta_pos.extend(meta_pos)
            temp_beds.append(bed_tmp)
            temp_provs.append(prov_tmp)
            temp_fastas.append(fasta_info)

        fasta_files = {}
        if cfg.mode == 'paired_end':
            bed_header = (
                "#seq_id\tevent_idx\tlabel\tinsert_len\thgt_start\thgt_end\tvirus_len"
                "\thgt_mode\tpe_case\tr1_vs\tr1_ve\tr2_vs\tr2_ve\n"
            )
            breakpoints_bed = os.path.join(cfg.output_dir, 'breakpoints.bed')
            _merge_tsv(temp_beds, breakpoints_bed, bed_header)
            r1_path = os.path.join(cfg.output_dir, 'R1.fasta')
            r2_path = os.path.join(cfg.output_dir, 'R2.fasta')
            _merge_paired_fasta(temp_fastas, r1_path, r2_path)
            fasta_files = {'R1': r1_path, 'R2': r2_path}
        else:
            bed_header = (
                "#seq_id\tevent_idx\tlabel\tseq_len\thgt_start\thgt_end\tvirus_len\thgt_mode\n"
            )
            breakpoints_bed = os.path.join(cfg.output_dir, 'breakpoints.bed')
            _merge_tsv(temp_beds, breakpoints_bed, bed_header)
            ds_path = os.path.join(cfg.output_dir, 'dataset.fasta')
            _merge_fasta(temp_fastas, ds_path)
            fasta_files = {'dataset': ds_path}

        provenance_tsv = os.path.join(cfg.output_dir, 'provenance.tsv')
        _merge_tsv(temp_provs, provenance_tsv, PROV_HEADER)

        metadata_json = os.path.join(cfg.output_dir, 'metadata.json')
        with open(metadata_json, 'w') as f:
            json.dump({'mode': cfg.mode, 'positive': all_meta_pos}, f, indent=2)

        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        # -- Statistics report --
        _print_statistics(cfg.mode, all_meta_pos, breakpoints_bed)

        return SimulationOutput(
            fasta_files=fasta_files,
            breakpoints_bed=breakpoints_bed,
            provenance_tsv=provenance_tsv,
            metadata_json=metadata_json,
            sample_count=cfg.n_pos + cfg.n_neg,
            pos_count=cfg.n_pos,
            neg_count=cfg.n_neg,
        )

    # ============================================================================= #
    # Genome-isolation generation path
    # ============================================================================= #

    def _generate_with_isolation(self) -> SimulationOutput:
        """
        Genome-isolation generation: first split the microbe files and virus strains,
        then generate samples independently for each of train/val/test.
        """
        cfg = self.config
        random.seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Genome split ratios
        mb_ratios = cfg.genome_isolation_microbe_ratios or [0.8, 0.1, 0.1]
        vi_ratios = cfg.genome_isolation_virus_ratios or [0.7, 0.15, 0.15]

        print(f"[GenomeIsolation] Splitting microbe genomes "
              f"({mb_ratios[0]:.0%}/{mb_ratios[1]:.0%}/{mb_ratios[2]:.0%})...")
        microbe_splits = split_microbe_genomes(
            cfg.microbe_dir, mb_ratios[0], mb_ratios[1], mb_ratios[2], cfg.seed)

        print(f"[GenomeIsolation] Splitting virus strains "
              f"({vi_ratios[0]:.0%}/{vi_ratios[1]:.0%}/{vi_ratios[2]:.0%})...")
        virus_splits = split_virus_genomes(
            cfg.virus_fasta, vi_ratios[0], vi_ratios[1], vi_ratios[2], cfg.seed)

        # output manifest
        manifest = {
            'microbe': {k: {'count': len(v), 'files': v}
                        for k, v in microbe_splits.items()},
            'virus': {k: {'count': len(v), 'strains': sorted(set(
                sid for _, sid in v))}
                      for k, v in virus_splits.items()},
            'ratios': {
                'microbe': mb_ratios,
                'virus': vi_ratios,
            },
            'seed': cfg.seed,
        }
        manifest_path = os.path.join(cfg.output_dir, 'genome_split_manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[GenomeIsolation] Manifest saved to {manifest_path}")

        # Assign per-split sample counts by train_ratio/val_ratio/test_ratio
        split_names = ['train', 'val', 'test']
        split_sample_ratios = [cfg.train_ratio, cfg.val_ratio, cfg.test_ratio]

        n_procs = cfg.processes or mp.cpu_count()
        worker = (_generate_chunk_pe
                  if cfg.mode == 'paired_end'
                  else _generate_chunk_contig)

        all_fasta_files = {}
        last_bed = ''
        last_prov = ''
        last_meta = ''
        total_pos = 0
        total_neg = 0

        for split_name, sample_ratio in zip(split_names, split_sample_ratios):
            if sample_ratio <= 0:
                continue

            n_pos_split = max(1, round(cfg.n_pos * sample_ratio))
            n_neg_split = max(1, round(cfg.n_neg * sample_ratio))
            total_pos += n_pos_split
            total_neg += n_neg_split

            split_dir = os.path.join(cfg.output_dir, split_name)
            os.makedirs(split_dir, exist_ok=True)
            tmp_dir = os.path.join(split_dir, '_tmp')
            os.makedirs(tmp_dir, exist_ok=True)

            print(f"\n[GenomeIsolation] Generating {split_name}: "
                  f"{n_pos_split} pos + {n_neg_split} neg "
                  f"(microbe={len(microbe_splits[split_name])} files, "
                  f"virus={len(virus_splits[split_name])} seqs)")

            pos_per = math.ceil(n_pos_split / n_procs)
            neg_per = math.ceil(n_neg_split / n_procs)

            base = dict(
                microbe_dir = cfg.microbe_dir,
                virus_fasta = cfg.virus_fasta,
                virus_fasta_label = cfg.virus_fasta,
                virus_fragment_min = cfg.virus_fragment_min,
                virus_fragment_max = cfg.virus_fragment_max,
                virus_fragments_per_genome = cfg.virus_fragments_per_genome,
                insertion_prob = cfg.insertion_prob,
                replacement_prob = cfg.replacement_prob,
                deletion_prob = cfg.deletion_prob,
                error_rate = cfg.error_rate,
                read_len = cfg.read_len,
                contig_min_length = cfg.contig_min_length,
                contig_max_length = cfg.contig_max_length,
                gc_bias = cfg.gc_bias,
                use_coverage = cfg.use_coverage,
                temp_prefix = os.path.join(tmp_dir, 'chunk'),
                multi_event_enabled = cfg.multi_event_enabled,
                multi_event_prob = cfg.multi_event_prob,
                multi_event_max = cfg.multi_event_max,
                multi_event_min_gap = cfg.multi_event_min_gap,
                multi_source_host_enabled = cfg.multi_source_host_enabled,
                multi_source_host_prob = cfg.multi_source_host_prob,
                multi_source_host_max = cfg.multi_source_host_max,
                sandwich_enabled = cfg.sandwich_enabled,
                sandwich_prob = cfg.sandwich_prob,
                sandwich_min_gap = cfg.sandwich_min_gap,
                virus_frag_tiers_pe = cfg.virus_frag_tiers_pe,
                virus_frag_tiers_contig = cfg.virus_frag_tiers_contig,
                pe_case_ratios = cfg.pe_case_ratios,
                neg_type_ratios = cfg.neg_type_ratios,
                contig_hgt_mode_ranges = cfg.contig_hgt_mode_ranges,
                seq_platform = cfg.seq_platform,
                coverage_distribution = cfg.coverage_distribution,
                coverage_zipf_alpha = cfg.coverage_zipf_alpha,
                batch_effect_sigma = cfg.batch_effect_sigma,
                contamination_rate = cfg.contamination_rate,
                contaminant_dir = cfg.contaminant_dir,
                assembly_artifact_rate = cfg.assembly_artifact_rate,
                label_noise = cfg.label_noise,
                boundary_shift_max = cfg.boundary_shift_max,
                evolutionary_distance = cfg.evolutionary_distance,
                evolutionary_distance_tiers = cfg.evolutionary_distance_tiers,
                host_mutation_rate = cfg.host_mutation_rate,
                open_set_virus = cfg.open_set_virus,
                gc_amelioration = cfg.gc_amelioration,
                evo_distance_distribution = cfg.evo_distance_distribution,
                evo_distance_gamma_shape = cfg.evo_distance_gamma_shape,
                microhomology_len = cfg.microhomology_len,
                # Genome-isolation parameters
                microbe_allowed_files = microbe_splits[split_name],
                virus_genomes_subset = virus_splits[split_name],
            )

            pool = mp.Pool(processes=n_procs)
            futures = []
            s_pos = s_neg = 0
            for i in range(n_procs):
                pchunk = min(pos_per, n_pos_split - s_pos)
                nchunk = min(neg_per, n_neg_split - s_neg)
                if pchunk <= 0 and nchunk <= 0:
                    break
                pcfg = {**base,
                        'n_pos': max(pchunk, 0),
                        'n_neg': max(nchunk, 0),
                        'start_pos_id': s_pos,
                        'start_neg_id': s_neg,
                        'seed': cfg.seed + _split_seed_offset(split_name) + i + 1}
                futures.append(pool.apply_async(worker, args=(i, pcfg)))
                s_pos += pchunk
                s_neg += nchunk

            pool.close()
            pool.join()

            all_meta_pos, temp_beds, temp_provs, temp_fastas = [], [], [], []
            for fut in futures:
                meta_pos, bed_tmp, prov_tmp, fasta_info = fut.get()
                all_meta_pos.extend(meta_pos)
                temp_beds.append(bed_tmp)
                temp_provs.append(prov_tmp)
                temp_fastas.append(fasta_info)

            if cfg.mode == 'paired_end':
                bed_header = (
                    "#seq_id\tevent_idx\tlabel\tinsert_len\thgt_start\thgt_end\tvirus_len"
                    "\thgt_mode\tpe_case\tr1_vs\tr1_ve\tr2_vs\tr2_ve\n"
                )
                breakpoints_bed = os.path.join(split_dir, 'breakpoints.bed')
                _merge_tsv(temp_beds, breakpoints_bed, bed_header)
                r1_path = os.path.join(split_dir, 'R1.fasta')
                r2_path = os.path.join(split_dir, 'R2.fasta')
                _merge_paired_fasta(temp_fastas, r1_path, r2_path)
                all_fasta_files[f'{split_name}_R1'] = r1_path
                all_fasta_files[f'{split_name}_R2'] = r2_path
            else:
                bed_header = (
                    "#seq_id\tevent_idx\tlabel\tseq_len\thgt_start\thgt_end\tvirus_len\thgt_mode\n"
                )
                breakpoints_bed = os.path.join(split_dir, 'breakpoints.bed')
                _merge_tsv(temp_beds, breakpoints_bed, bed_header)
                ds_path = os.path.join(split_dir, 'dataset.fasta')
                _merge_fasta(temp_fastas, ds_path)
                all_fasta_files[split_name] = ds_path

            provenance_tsv = os.path.join(split_dir, 'provenance.tsv')
            _merge_tsv(temp_provs, provenance_tsv, PROV_HEADER)

            metadata_json = os.path.join(split_dir, 'metadata.json')
            with open(metadata_json, 'w') as f:
                json.dump({'mode': cfg.mode, 'positive': all_meta_pos,
                           'split': split_name}, f, indent=2)

            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

            last_bed = breakpoints_bed
            last_prov = provenance_tsv
            last_meta = metadata_json

            _print_statistics(cfg.mode, all_meta_pos, breakpoints_bed)

        return SimulationOutput(
            fasta_files=all_fasta_files,
            breakpoints_bed=last_bed,
            provenance_tsv=last_prov,
            metadata_json=last_meta,
            split_dir=cfg.output_dir,
            sample_count=total_pos + total_neg,
            pos_count=total_pos,
            neg_count=total_neg,
        )
