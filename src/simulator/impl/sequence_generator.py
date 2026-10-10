# simulator/impl/sequence_generator.py
"""
HGT sequence generation engine
─────────────────────────────────────────────────────────────────────────────
Main changes
  * generate_positive_sample now returns bg_right_offset:
      records the start offset of the right-side host in the original bg sequence,
      used for provenance coordinate computation
  * classify_pe_case naming unified as R1_{type}_R2_{type}, type ∈ {HOST, VIRUS, MIXED}
      covering all 7 valid positive-sample scenarios + INTERNAL (gap region only)
  * New compute_insert_segments / compute_read_segments:
      decompose the post-integration insert into genome-coordinate segments,
      used for provenance logging
  * negative-sample generation functions now support returning provenance info
─────────────────────────────────────────────────────────────────────────────
"""

import random
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════ #
#  HGT event generator
# ═══════════════════════════════════════════════════════════════════════════ #

class HGTSequenceGenerator:
    """
    Integrate a virus fragment into a host background sequence to simulate an HGT event.

    Three operation modes
    ------------
    insertion           : virus inserted directly into the host sequence (no host bases lost)
    replacement         : a stretch of host sequence is excised and filled with virus of
                          equal or arbitrary length
    deletion_associated : 1-50 bp of host sequence around the integration site is randomly
                          deleted before the virus is inserted (modeling integration
                          instability)
    """

    def __init__(self, insertion_prob: float = 0.6,
                 replacement_prob: float = 0.3,
                 deletion_prob: float = 0.1):
        total = insertion_prob + replacement_prob + deletion_prob
        self.p_ins = insertion_prob / total
        self.p_rep = replacement_prob / total
        self.p_del = deletion_prob / total

    def generate_positive_sample(
        self, bg_seq: str, virus_seq: str
    ) -> Tuple[str, Tuple[int, int], str, int]:
        """
        Execute one HGT event on bg_seq.

        Parameters
        ----------
        bg_seq    : host background sequence
        virus_seq : virus fragment

        Returns
        -------
        new_seq          : full sequence after integration
        (hgt_s, hgt_e)   : half-open interval [start, end) of the virus in new_seq
        mode             : operation mode string
        bg_right_offset  : start offset of the right-side host fragment in the original bg_seq
                           (for provenance: right-side host = bg_seq[bg_right_offset:])
        """
        bg_len = len(bg_seq)
        v_len  = len(virus_seq)

        if bg_len < 10:
            raise ValueError(f"Background sequence too short: {bg_len}")

        # No left margin (allow pos=0 to cover the R1_VIRUS case)
        # Keep a small buffer on the right
        right_buffer = max(1, min(10, bg_len // 20))
        pos    = random.randint(0, max(1, bg_len - right_buffer - 1))
        r      = random.random()

        if r < self.p_ins:
            #  —  —  Insertion  —  —  all host bases preserved
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[pos:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "insertion"
            bg_right_offset = pos                       # bg[pos:] follows the virus

        elif r < self.p_ins + self.p_rep:
            #  —  —  Replacement  —  —  excise a host stretch and fill with virus
            rep_end         = min(pos + v_len, bg_len)
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[rep_end:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "replacement"
            bg_right_offset = rep_end                   # bg[rep_end:] follows the virus

        else:
            #  —  —  Deletion-Associated  —  —  delete first, then insert the virus
            del_len         = random.randint(1, min(50, bg_len - pos))
            del_end         = min(pos + del_len, bg_len)
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[del_end:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "deletion_associated"
            bg_right_offset = del_end                   # bg[del_end:] follows the virus

        return new_seq, (hgt_s, hgt_e), mode, bg_right_offset

    def generate_positive_sample_at(
        self, bg_seq: str, virus_seq: str, pos: int
    ) -> Tuple[str, Tuple[int, int], str, int]:
        """
        Execute one HGT event at the specified position pos in bg_seq.

        Parameters
        ----------
        bg_seq    : host background sequence
        virus_seq : virus fragment
        pos       : start position of the insertion/replacement/deletion

        Returns
        -------
        Same as generate_positive_sample
        """
        bg_len = len(bg_seq)
        v_len  = len(virus_seq)
        r      = random.random()

        if r < self.p_ins:
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[pos:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "insertion"
            bg_right_offset = pos
        elif r < self.p_ins + self.p_rep:
            rep_end         = min(pos + v_len, bg_len)
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[rep_end:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "replacement"
            bg_right_offset = rep_end
        else:
            del_len         = random.randint(1, min(50, bg_len - pos))
            del_end         = min(pos + del_len, bg_len)
            new_seq         = bg_seq[:pos] + virus_seq + bg_seq[del_end:]
            hgt_s, hgt_e    = pos, pos + v_len
            mode            = "deletion_associated"
            bg_right_offset = del_end

        return new_seq, (hgt_s, hgt_e), mode, bg_right_offset

    # ------------------------------------------------------------------ #
    #  Multi-event HGT generation
    # ------------------------------------------------------------------ #

    def generate_multi_hgt_sample(
        self, bg_seq: str, virus_seqs: List[str],
        min_gap: int = 20,
    ) -> Tuple[str, List[Tuple[int, int]], List[str], List[dict]]:
        """
        Execute multiple HGT events on bg_seq.

        Parameters
        ----------
        bg_seq     : host background sequence (may be a composite host)
        virus_seqs : list of virus fragments (one virus sequence per insertion event)
        min_gap    : minimum spacing between adjacent events (bp)

        Returns
        -------
        final_seq  : full sequence after integration
        events     : list of (hgt_start, hgt_end) per event (final_seq coordinates)
        modes      : list of mode strings per event
        event_meta : list of per-event metadata dicts, each containing:
                     hgt_start, hgt_end, mode, bg_left_end, bg_right_offset
                      (all in final_seq coordinates except the bg_* fields, which use
                       original bg coordinates)
        """
        n_events = len(virus_seqs)
        if n_events == 0:
            return bg_seq, [], [], []

        bg_len = len(bg_seq)
        if bg_len < 10:
            raise ValueError(f"Background sequence too short: {bg_len}")

        # ── Step 1: determine each event's insertion position in the original bg_seq ──
        # Pre-assign a position per event (original bg coordinate system)
        total_virus_space = sum(len(v) for v in virus_seqs)
        margin = max(0, min(10, bg_len // 20))  # multi-event keeps some margin
        usable = bg_len - 2 * margin
        if usable < total_virus_space + min_gap * (n_events - 1):
            # Not enough space: assign evenly spaced positions
            spacing = max(min_gap, usable // (n_events + 1))
            raw_positions = [margin + spacing * (i + 1) for i in range(n_events)]
        else:
            # Assign positions randomly
            raw_positions = []
            occupied: List[Tuple[int, int]] = []
            for i, vseq in enumerate(virus_seqs):
                v_len = len(vseq)
                placed = False
                for _ in range(100):
                    pos = random.randint(margin, max(margin + 1, bg_len - margin - 1))
                    # Check for conflicts with already-placed events
                    conflict = False
                    for (os, oe) in occupied:
                        if pos - min_gap < oe and pos + v_len + min_gap > os:
                            conflict = True
                            break
                    if not conflict:
                        raw_positions.append(pos)
                        occupied.append((pos, pos + v_len))
                        placed = True
                        break
                if not placed:
                    # Fall back to even spacing
                    spacing = max(min_gap, usable // (n_events + 1))
                    raw_positions = [margin + spacing * (i + 1)
                                     for i in range(n_events)]
                    break

        # Sort by position, and sort virus_seqs to match
        order = sorted(range(n_events), key=lambda i: raw_positions[i])
        sorted_positions = [raw_positions[i] for i in order]
        sorted_viruses   = [virus_seqs[i] for i in order]

        # ── Step 2: execute HGT events in order, tracking coordinate offsets ──────
        current_seq = bg_seq
        offset = 0  # cumulative offset (current sequence length minus bg_len)
        raw_events: List[dict] = []  # raw information recorded at each step

        for i, (pos, vseq) in enumerate(zip(sorted_positions, sorted_viruses)):
            v_len = len(vseq)
            adjusted_pos = pos + offset  # position within current_seq

            r = random.random()
            if r < self.p_ins:
                # Insertion
                new_seq = current_seq[:adjusted_pos] + vseq + current_seq[adjusted_pos:]
                hgt_s = adjusted_pos
                hgt_e = adjusted_pos + v_len
                mode = "insertion"
                bg_right_offset = pos  # right-side host start in original bg coordinates
                offset += v_len
            elif r < self.p_ins + self.p_rep:
                # Replacement
                rep_end = min(adjusted_pos + v_len, len(current_seq))
                replaced_len = rep_end - adjusted_pos
                new_seq = current_seq[:adjusted_pos] + vseq + current_seq[rep_end:]
                hgt_s = adjusted_pos
                hgt_e = adjusted_pos + v_len
                mode = "replacement"
                bg_right_offset = pos + replaced_len  # bg coordinates: right-side host start
                offset += v_len - replaced_len
            else:
                # Deletion-associated
                remaining = len(current_seq) - adjusted_pos
                if remaining < 1:
                    # adjusted_pos is past the end of the sequence; degenerate to insertion
                    new_seq = current_seq[:adjusted_pos] + vseq + current_seq[adjusted_pos:]
                    hgt_s = adjusted_pos
                    hgt_e = adjusted_pos + v_len
                    mode = "insertion"
                    bg_right_offset = pos
                    offset += v_len
                else:
                    del_len = random.randint(1, min(50, remaining))
                    del_end = min(adjusted_pos + del_len, len(current_seq))
                    new_seq = current_seq[:adjusted_pos] + vseq + current_seq[del_end:]
                    hgt_s = adjusted_pos
                    hgt_e = adjusted_pos + v_len
                    mode = "deletion_associated"
                    bg_right_offset = pos + del_len  # original bg coordinates
                    offset += v_len - del_len

            # bg_left_end: end of the left-side host in original bg coordinates = pos
            raw_events.append({
                'hgt_s': hgt_s,
                'hgt_e': hgt_e,
                'mode': mode,
                'bg_left_end': pos,         # bg coordinates
                'bg_right_offset': bg_right_offset,  # bg coordinates
                'v_len': v_len,
                'vseq_idx': order[i],       # index into virus_provs (for provenance alignment)
            })
            current_seq = new_seq

        # ── Step 3: convert all coordinates to the final_seq coordinate system ────
        # Since events are processed left to right by position, the later events'
        # hgt_s/hgt_e are already correct
        final_seq = current_seq
        events = [(e['hgt_s'], e['hgt_e']) for e in raw_events]
        modes  = [e['mode'] for e in raw_events]
        event_meta = raw_events

        return final_seq, events, modes, event_meta


# ═══════════════════════════════════════════════════════════════════════════ #
#  Paired-End scenario classifier
# ═══════════════════════════════════════════════════════════════════════════ #

def classify_pe_case(
    hgt_start: int, hgt_end: int,
    read_len: int, insert_len: int
) -> Tuple[str, int, int, int, int]:
    """
    Classify the PE scenario from the position of the virus within the insert.

    Naming convention
    --------
    R1_{R1_type}_R2_{R2_type}, where type ∈ {HOST, VIRUS, MIXED}

        HOST  : the read is entirely host sequence (no virus)
        VIRUS : the read is entirely virus sequence (no host breakpoint)
        MIXED : the read contains both host and virus (a breakpoint is present)

    Valid positive-sample scenarios (7 kinds)
    ----------------------
    R1_HOST_R2_VIRUS   R1_HOST_R2_MIXED
    R1_VIRUS_R2_HOST   R1_VIRUS_R2_MIXED
    R1_MIXED_R2_HOST   R1_MIXED_R2_VIRUS
    R1_MIXED_R2_MIXED

    Special scenario
    --------
    INTERNAL  : the virus falls entirely in the gap between R1 and R2
                (only occurs when insert > 2×read_len; the caller should
                skip or resample)

    Coordinate systems
    ------
    R1 covers insert[0 : read_len] (forward strand);
    R2 is taken as the RC of insert[insert_len-read_len : insert_len],
    R2 coordinate-system position k ↔ insert position (insert_len - 1 - k)

    Returns
    -------
    case_type    : str
    r1_vs, r1_ve : virus coordinate interval within R1 (-1 if absent)
    r2_vs, r2_ve : virus coordinate interval in the R2 RC coordinate system (-1 if absent)
    """
    r2_raw_start = insert_len - read_len   # start of the R2 region within the insert

    # ── R1 ──────────────────────────────────────────────────────────────────
    r1_ov_s = max(0,        hgt_start)
    r1_ov_e = min(read_len, hgt_end)
    in_r1   = r1_ov_s < r1_ov_e

    if in_r1:
        r1_vs, r1_ve = r1_ov_s, r1_ov_e
        if r1_vs == 0 and r1_ve == read_len:
            r1_type = "VIRUS"
        else:
            r1_type = "MIXED"
    else:
        r1_vs = r1_ve = -1
        r1_type = "HOST"

    # ── R2 (insert coordinates) → RC coordinates ──────────────────────────────
    r2_ov_s_ins = max(r2_raw_start, hgt_start)
    r2_ov_e_ins = min(insert_len,   hgt_end)
    in_r2       = r2_ov_s_ins < r2_ov_e_ins

    if in_r2:
        # RC coordinates: position 0 = end of the insert
        r2_vs = insert_len - r2_ov_e_ins
        r2_ve = insert_len - r2_ov_s_ins
        if r2_vs == 0 and r2_ve == read_len:
            r2_type = "VIRUS"
        else:
            r2_type = "MIXED"
    else:
        r2_vs = r2_ve = -1
        r2_type = "HOST"

    # ── Classify ───────────────────────────────────────────────────────────────
    if not in_r1 and not in_r2:
        case_type = "INTERNAL"
    else:
        case_type = f"R1_{r1_type}_R2_{r2_type}"

    return case_type, r1_vs, r1_ve, r2_vs, r2_ve


def classify_pe_case_multi(
    virus_regions: List[Tuple[int, int]],
    read_len: int,
    insert_len: int,
) -> Tuple[str, int, int, int, int]:
    """
    Multi-region version of PE scenario classification.

    Computes the union of the virus regions' overlaps with R1/R2, then applies the
    same logic as classify_pe_case to decide the HOST/VIRUS/MIXED type of R1/R2.

    Parameters
    ----------
    virus_regions : list of (hgt_start, hgt_end) in insert coordinates
    read_len      : single-end read length
    insert_len    : total insert length

    Returns
    -------
    A tuple in the same format as classify_pe_case:
    (case_type, r1_vs, r1_ve, r2_vs, r2_ve)
    r1_vs/r1_ve and r2_vs/r2_ve are the union bounds of the virus overlap
    """
    r2_raw_start = insert_len - read_len

    # ── R1: compute the union of overlaps with [0, read_len) ───────────────────
    r1_intervals = []
    for (vs, ve) in virus_regions:
        ov_s = max(0, vs)
        ov_e = min(read_len, ve)
        if ov_s < ov_e:
            r1_intervals.append((ov_s, ov_e))

    if r1_intervals:
        r1_vs = min(s for s, e in r1_intervals)
        r1_ve = max(e for s, e in r1_intervals)
        if r1_vs == 0 and r1_ve == read_len:
            r1_type = "VIRUS"
        else:
            r1_type = "MIXED"
        in_r1 = True
    else:
        r1_vs = r1_ve = -1
        r1_type = "HOST"
        in_r1 = False

    # ── R2: compute the union of overlaps with [r2_raw_start, insert_len) → RC coords ──
    r2_intervals_ins = []
    for (vs, ve) in virus_regions:
        ov_s = max(r2_raw_start, vs)
        ov_e = min(insert_len, ve)
        if ov_s < ov_e:
            r2_intervals_ins.append((ov_s, ov_e))

    if r2_intervals_ins:
        # Convert to RC coordinates
        r2_rc_intervals = [(insert_len - e, insert_len - s)
                           for s, e in r2_intervals_ins]
        r2_vs = min(s for s, e in r2_rc_intervals)
        r2_ve = max(e for s, e in r2_rc_intervals)
        if r2_vs == 0 and r2_ve == read_len:
            r2_type = "VIRUS"
        else:
            r2_type = "MIXED"
        in_r2 = True
    else:
        r2_vs = r2_ve = -1
        r2_type = "HOST"
        in_r2 = False

    # ── Classify ──────────────────────────────────────────────────────────────
    if not in_r1 and not in_r2:
        case_type = "INTERNAL"
    else:
        case_type = f"R1_{r1_type}_R2_{r2_type}"

    return case_type, r1_vs, r1_ve, r2_vs, r2_ve


# ═══════════════════════════════════════════════════════════════════════════ #
#  Provenance Segment Decomposition
# ═══════════════════════════════════════════════════════════════════════════ #

def compute_insert_segments(
    bg_prov: dict,
    virus_prov: dict,
    hgt_pos: int,
    bg_right_offset: int,
    hgt_start: int,
    hgt_end: int,
    insert_len: int,
) -> List[dict]:
    """
    Decompose the post-integration insert into a list of genome-coordinate segments
    (sorted by insert coordinate, ascending).

    Per-segment dict fields
    ------------------
    seg_type      : 'HOST' or 'VIRUS'
    source_type   : 'microbe' or 'virus'
    source_file   : source FASTA file path
    source_seq_id : source sequence ID
    genome_start  : start coordinate in the source genome (0-based, half-open interval)
    genome_end    : end coordinate in the source genome
    strand        : '+' or '-'
    insert_start  : start coordinate within the insert sequence
    insert_end    : end coordinate within the insert sequence

    bg_prov  : source-info dict returned by MicrobeSampler
    virus_prov : source-info dict of the corresponding fragment from VirusPool
    hgt_pos  : position of the integration point within bg (= hgt_start, also the
               left-side host length)
    bg_right_offset : start offset of the right-side host within bg (see
                      generate_positive_sample)
    """
    segs: List[dict] = []
    bg_g_start = bg_prov['genome_start']

    #  —  —  Left-side host bg[0 : hgt_pos]  —  —
    if hgt_pos > 0:
        segs.append({
            'seg_type':     'HOST',
            'source_type':  'microbe',
            'source_file':  bg_prov['file'],
            'source_seq_id': bg_prov['seq_id'],
            'genome_start': bg_g_start,
            'genome_end':   bg_g_start + hgt_pos,
            'strand':       '+',
            'insert_start': 0,
            'insert_end':   hgt_pos,   # == hgt_start
        })

    #  —  —  virus  —  — 
    segs.append({
        'seg_type':     'VIRUS',
        'source_type':  'virus',
        'source_file':  virus_prov['file'],
        'source_seq_id': virus_prov['seq_id'],
        'genome_start': virus_prov['genome_start'],
        'genome_end':   virus_prov['genome_end'],
        'strand':       virus_prov['strand'],
        'insert_start': hgt_start,
        'insert_end':   hgt_end,
    })

    #  —  —  Right-side host bg[bg_right_offset :]  —  —
    right_len = insert_len - hgt_end
    if right_len > 0:
        segs.append({
            'seg_type':     'HOST',
            'source_type':  'microbe',
            'source_file':  bg_prov['file'],
            'source_seq_id': bg_prov['seq_id'],
            'genome_start': bg_g_start + bg_right_offset,
            'genome_end':   bg_g_start + bg_right_offset + right_len,
            'strand':       '+',
            'insert_start': hgt_end,
            'insert_end':   insert_len,
        })

    return segs


def compute_insert_segments_multi(
    bg_prov_list: List[dict],
    virus_prov_list: List[dict],
    event_meta: List[dict],
    insert_len: int,
) -> List[dict]:
    """
    Multi-event version of provenance segment decomposition.

    Decomposes the post-integration insert into a list of genome-coordinate segments
    (sorted by insert coordinate, ascending). Supports:
      - multiple virus insertion events
      - composite host background (from multiple microbe genomes)

    Parameters
    ----------
    bg_prov_list   : list of host source-info dicts.
                     Single-source host: [bg_prov] (one element)
                     Composite host: multiple elements, each with composite_start /
                     composite_end
    virus_prov_list: source info of each virus fragment (one-to-one with event_meta)
    event_meta     : the event_meta list returned by generate_multi_hgt_sample
    insert_len     : total length of the final sequence

    Returns
    -------
    segs : list of segments sorted by insert_start ascending, same format as
           compute_insert_segments
    """
    if not event_meta:
        # No events: the entire sequence is host
        return _build_host_segs_for_range(
            bg_prov_list, 0, insert_len, 0, insert_len)

    segs: List[dict] = []
    # Sort by hgt_s
    sorted_events = sorted(
        range(len(event_meta)),
        key=lambda i: event_meta[i]['hgt_s'])

    cursor = 0  # current position within the insert
    bg_cursor = 0  # current position within the original bg sequence

    for idx in sorted_events:
        em = event_meta[idx]
        vprov = virus_prov_list[idx]
        hgt_s = em['hgt_s']
        hgt_e = em['hgt_e']

        # ── Host region left of the virus ──
        if hgt_s > cursor:
            host_insert_start = cursor
            host_insert_end = hgt_s
            # bg coordinates: from bg_cursor to bg_left_end
            bg_start = bg_cursor
            bg_end = em['bg_left_end']
            host_segs = _build_host_segs_for_range(
                bg_prov_list, bg_start, bg_end,
                host_insert_start, host_insert_end)
            segs.extend(host_segs)

        # ── Virus region ──
        segs.append({
            'seg_type':     'VIRUS',
            'source_type':  'virus',
            'source_file':  vprov['file'],
            'source_seq_id': vprov['seq_id'],
            'genome_start': vprov['genome_start'],
            'genome_end':   vprov['genome_end'],
            'strand':       vprov['strand'],
            'insert_start': hgt_s,
            'insert_end':   hgt_e,
        })

        cursor = hgt_e
        bg_cursor = em['bg_right_offset']

    # ── Final host stretch ──
    if cursor < insert_len:
        host_segs = _build_host_segs_for_range(
            bg_prov_list, bg_cursor, bg_cursor + (insert_len - cursor),
            cursor, insert_len)
        segs.extend(host_segs)

    # Sort by insert_start
    segs.sort(key=lambda x: x['insert_start'])
    return segs


def _build_host_segs_for_range(
    bg_prov_list: List[dict],
    bg_start: int, bg_end: int,
    insert_start: int, insert_end: int,
) -> List[dict]:
    """
    Build provenance segments for the host region within [bg_start, bg_end).
    Supports:
      - Single-source host: bg_prov_list has exactly one element
      - Composite host: bg_prov_list has multiple elements, each with
        composite_start/composite_end

    bg_start/bg_end : range in the original bg sequence coordinate system
    insert_start/insert_end : range in the insert sequence coordinate system
    """
    if bg_end <= bg_start:
        return []

    # Fast path for a single-source host
    if len(bg_prov_list) == 1:
        bp = bg_prov_list[0]
        return [{
            'seg_type':     'HOST',
            'source_type':  'microbe',
            'source_file':  bp['file'],
            'source_seq_id': bp['seq_id'],
            'genome_start': bp['genome_start'] + bg_start,
            'genome_end':   bp['genome_start'] + bg_end,
            'strand':       '+',
            'insert_start': insert_start,
            'insert_end':   insert_end,
        }]

    # Composite host: determine which source fragments are involved, based on the
    # bg coordinate range
    result: List[dict] = []
    remaining_insert = insert_start

    for bp in bg_prov_list:
        cs = bp.get('composite_start', 0)
        ce = bp.get('composite_end', 0)
        # This fragment's range [cs, ce) within the composite host
        # intersected with the requested [bg_start, bg_end)
        ov_s = max(cs, bg_start)
        ov_e = min(ce, bg_end)
        if ov_s >= ov_e:
            continue

        seg_len = ov_e - ov_s
        # Compute the offset within this fragment
        offset_in_seg = ov_s - cs
        result.append({
            'seg_type':     'HOST',
            'source_type':  'microbe',
            'source_file':  bp['file'],
            'source_seq_id': bp['seq_id'],
            'genome_start': bp['genome_start'] + offset_in_seg,
            'genome_end':   bp['genome_start'] + offset_in_seg + seg_len,
            'strand':       '+',
            'insert_start': remaining_insert,
            'insert_end':   remaining_insert + seg_len,
        })
        remaining_insert += seg_len

    return result


def compute_read_segments(
    insert_segs: List[dict],
    read_raw_start: int,
    read_len: int,
    is_rc: bool = False,
) -> List[dict]:
    """
    From the insert segment list, extract the read's overlapped region and convert
    coordinates.

    Parameters
    ----------
    insert_segs    : return value of compute_insert_segments
    read_raw_start : read start within the insert (R1=0, R2=insert_len-read_len)
    read_len       : read length (usually 150)
    is_rc          : whether to take the RC (True for R2)

    Returns
    -------
    Segment list in read coordinates; each segment gains new fields:
        read_start, read_end : coordinates within the read sequence
         (for R2, already converted to the RC coordinate system, strand also flipped)
    """
    result: List[dict] = []
    read_raw_end = read_raw_start + read_len

    for seg in insert_segs:
        ov_s = max(seg['insert_start'], read_raw_start)
        ov_e = min(seg['insert_end'],   read_raw_end)
        if ov_s >= ov_e:
            continue

        # Position in the read's raw coordinate system (before RC)
        raw_r_s = ov_s - read_raw_start
        raw_r_e = ov_e - read_raw_start

        # Offset within the segment (used to compute the genome sub-interval)
        off_s = ov_s - seg['insert_start']
        off_e = ov_e - seg['insert_start']

        if seg['strand'] == '+':
            g_s = seg['genome_start'] + off_s
            g_e = seg['genome_start'] + off_e
            eff_strand = '+'
        else:  # '-' strand (an already-RC'd virus fragment)
            g_s = seg['genome_end'] - off_e
            g_e = seg['genome_end'] - off_s
            eff_strand = '-'

        if is_rc:
            # R2 takes the RC: read coordinates flip, strand direction reverses
            read_s = read_len - raw_r_e
            read_e = read_len - raw_r_s
            eff_strand = '-' if eff_strand == '+' else '+'
        else:
            read_s = raw_r_s
            read_e = raw_r_e

        result.append({
            'seg_type':     seg['seg_type'],
            'source_type':  seg['source_type'],
            'source_file':  seg['source_file'],
            'source_seq_id': seg['source_seq_id'],
            'genome_start': g_s,
            'genome_end':   g_e,
            'strand':       eff_strand,
            'read_start':   read_s,
            'read_end':     read_e,
        })

    # Sort by read coordinate ascending
    result.sort(key=lambda x: x['read_start'])
    return result


# ═══════════════════════════════════════════════════════════════════════════ #
#  Composite Host Builder
# ═══════════════════════════════════════════════════════════════════════════ #

def build_composite_host(
    sampler,
    target_len: int,
    n_sources: int,
    use_coverage: bool = False,
) -> Tuple[str, List[dict]]:
    """
    Sample fragments from several different microbe genomes and concatenate them
    into a composite host sequence.

    Parameters
    ----------
    sampler     : MicrobeSampler instance
    target_len  : target total length
    n_sources   : number of sources (2-3)
    use_coverage: whether to use abundance weights

    Returns
    -------
    composite_seq : the concatenated sequence
    bg_prov_list  : provenance-info list per source; each dict contains:
                    file, seq_id, genome_start, genome_end, strand,
                    composite_start, composite_end
    """
    # Randomly assign each source's length (mean ± 20%)
    avg = target_len / n_sources
    raw_lens = [max(10, int(random.gauss(avg, avg * 0.2)))
                for _ in range(n_sources)]
    # Adjust so the total equals target_len
    total = sum(raw_lens)
    diff = target_len - total
    raw_lens[-1] = max(10, raw_lens[-1] + diff)

    parts: List[str] = []
    bg_prov_list: List[dict] = []
    composite_pos = 0

    for part_len in raw_lens:
        seq, prov = sampler.sample_fragment_with_provenance(
            part_len, use_coverage=use_coverage)
        actual_len = len(seq)
        parts.append(seq)

        bg_prov_list.append({
            'file':            prov['file'],
            'seq_id':          prov['seq_id'],
            'genome_start':    prov['genome_start'],
            'genome_end':      prov['genome_start'] + actual_len,
            'strand':          '+',
            'composite_start': composite_pos,
            'composite_end':   composite_pos + actual_len,
        })
        composite_pos += actual_len

    composite_seq = ''.join(parts)
    return composite_seq, bg_prov_list


# ═══════════════════════════════════════════════════════════════════════════ #
#  Negative-sample generation (with provenance return)
# ═══════════════════════════════════════════════════════════════════════════ #

def generate_long_sequence_from_fragments(
    frags: List, target_len: int
) -> str:
    """
    Build a sequence of arbitrary target length by concatenating multiple fragments.
    frags can be a list of str or a list of (seq, prov) tuples (the seq part is used).
    """
    parts, filled = [], 0
    while filled < target_len:
        item = random.choice(frags)
        seq  = item[0] if isinstance(item, tuple) else item
        need = target_len - filled
        parts.append(seq[:need])
        filled += min(len(seq), need)
    return ''.join(parts)


def generate_virus_only(
    virus_pool: List, target_len: int
) -> Tuple[str, List[dict]]:
    """
    Pure-virus negative sample; supports arbitrary length (auto-concatenates
    multiple fragments).

    virus_pool : list of (seq, prov_dict)

    Returns
    -------
    seq       : the concatenated sequence
    prov_segs : provenance segment list (insert coordinate system)
    """
    parts, prov_segs, filled = [], [], 0
    while filled < target_len:
        seq, prov = random.choice(virus_pool)
        need      = target_len - filled
        use_len   = min(len(seq), need)
        parts.append(seq[:use_len])
        gs = prov['genome_start']
        ge = prov['genome_end']
        if prov['strand'] == '-':
            gs = prov['genome_end'] - use_len
        else:
            ge = prov['genome_start'] + use_len
        prov_segs.append({
            'seg_type':     'VIRUS',
            'source_type':  'virus',
            'source_file':  prov['file'],
            'source_seq_id': prov['seq_id'],
            'genome_start': gs,
            'genome_end':   ge,
            'strand':       prov['strand'],
            'insert_start': filled,
            'insert_end':   filled + use_len,
        })
        filled += use_len
    return ''.join(parts), prov_segs


def generate_pcr_chimera(
    bg_pool_with_prov: List[Tuple[str, dict]], target_len: int
) -> Tuple[str, List[dict]]:
    """
    PCR chimera: randomly pick two different host sequences, concatenate them, and
    trim to the target length.

    bg_pool_with_prov : list of (seq, prov_dict)

    Returns
    -------
    chimera   : the chimeric sequence
    prov_segs : provenance segment list
    """
    n = len(bg_pool_with_prov)
    if n < 2:
        # Degenerate case: reverse of itself
        seq, prov = bg_pool_with_prov[0]
        chimera   = (seq + seq[::-1])[:target_len]
        prov_segs = [{
            'seg_type': 'HOST', 'source_type': 'microbe',
            'source_file': prov['file'], 'source_seq_id': prov['seq_id'],
            'genome_start': prov['genome_start'],
            'genome_end':   prov['genome_start'] + len(chimera),
            'strand': '+', 'insert_start': 0, 'insert_end': len(chimera),
        }]
        return chimera, prov_segs

    i1, i2 = random.sample(range(n), 2)
    seq1, prov1 = bg_pool_with_prov[i1]
    seq2, prov2 = bg_pool_with_prov[i2]
    split  = random.randint(1, max(1, min(len(seq1), len(seq2)) - 1))
    part1  = seq1[:split]
    part2  = seq2[split:split + (target_len - split)]
    chimera = (part1 + part2)[:target_len]

    prov_segs = []
    len1 = min(split, target_len)
    if len1 > 0:
        prov_segs.append({
            'seg_type': 'HOST', 'source_type': 'microbe',
            'source_file': prov1['file'], 'source_seq_id': prov1['seq_id'],
            'genome_start': prov1['genome_start'],
            'genome_end':   prov1['genome_start'] + len1,
            'strand': '+', 'insert_start': 0, 'insert_end': len1,
        })
    len2 = len(chimera) - len1
    if len2 > 0:
        prov_segs.append({
            'seg_type': 'HOST', 'source_type': 'microbe',
            'source_file': prov2['file'], 'source_seq_id': prov2['seq_id'],
            'genome_start': prov2['genome_start'] + split,
            'genome_end':   prov2['genome_start'] + split + len2,
            'strand': '+', 'insert_start': len1, 'insert_end': len(chimera),
        })
    return chimera, prov_segs


def generate_negative_hard(
    bg_seq: str,
    bg_prov: dict,
    virus_pool: List[Tuple[str, dict]],
    bg_pool_with_prov: Optional[List[Tuple[str, dict]]] = None,
) -> Tuple[str, List[dict]]:
    """
    Hard negative sample: virus-virus concatenation (50%) or PCR chimera (50%).

    Returns  (seq, prov_segs)
    """
    target_len = len(bg_seq)
    if random.random() < 0.5 or not bg_pool_with_prov:
        # virus-virus concatenation
        seq, segs = generate_virus_only(virus_pool, target_len)
    else:
        seq, segs = generate_pcr_chimera(bg_pool_with_prov, target_len)
    return seq, segs


def generate_host_only(
    bg_prov: dict, target_len: int
) -> List[dict]:
    """
    Provenance segments for a pure-host negative sample (the sequence itself is
    sampled externally; only prov_segs are built here).
    """
    return [{
        'seg_type':     'HOST',
        'source_type':  'microbe',
        'source_file':  bg_prov['file'],
        'source_seq_id': bg_prov['seq_id'],
        'genome_start': bg_prov['genome_start'],
        'genome_end':   bg_prov['genome_start'] + target_len,
        'strand':       '+',
        'insert_start': 0,
        'insert_end':   target_len,
    }]


def _trim_or_pad(seq: str, target_len: int) -> str:
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + 'N' * (target_len - len(seq))


def apply_microhomology(
    bg_seq: str,
    virus_seq: str,
    mh_len: int = 0,
) -> Tuple[str, str]:
    """
    Add microhomology at the HGT junction.

    Microhomology is a hallmark of non-homologous end joining (NHEJ) and
    microhomology-mediated end joining (MMEJ) repair in eukaryotic/prokaryotic
    organisms. When foreign DNA integrates, the two ends of the virus fragment may
    share a 1-5 bp short-sequence match with the host junction.

    This function takes mh_len bp from the head and tail of the virus fragment and
    replaces the corresponding host bases with matching sequence, creating a short
    homologous region at the junction. This makes breakpoint detection harder —
    the model must distinguish genuinely virus-derived bases from bases matched by
    microhomology.

    Parameters
    ----------
    bg_seq    : host background sequence
    virus_seq : virus fragment sequence
    mh_len    : microhomology length (bp, 0 = disabled)

    Returns
    -------
    (modified_virus, mh_info)
      modified_virus : the modified virus sequence (microhomology added at both ends)
      mh_info        : dict with 'left_mh' and 'right_mh' (the actual matching sequences)
    """
    mh_info = {'left_mh': '', 'right_mh': ''}
    if mh_len <= 0 or len(virus_seq) < mh_len * 2 + 5:
        return virus_seq, mh_info

    bases = list(virus_seq)

    # Left microhomology: the first mh_len bp of the virus match the host bases at
    # the junction
    # The junction means the host bases at the virus insertion position pos
    left_mh_len = min(mh_len, len(bg_seq))
    if left_mh_len > 0:
        # Take bases from the tail of the host bg_seq (approximating host sequence
        # near the insertion locus)
        # Use a random host fragment to simulate the junction
        left_host_mh = bg_seq[:left_mh_len].upper()
        for i in range(left_mh_len):
            bases[i] = left_host_mh[i]
        mh_info['left_mh'] = ''.join(bases[:left_mh_len])

    # Right microhomology: the last mh_len bp of the virus match the host bases to
    # the right of the junction
    right_mh_len = min(mh_len, len(bg_seq))
    if right_mh_len > 0:
        right_host_mh = bg_seq[-right_mh_len:].upper()
        for i in range(right_mh_len):
            bases[-(right_mh_len - i)] = right_host_mh[i]
        mh_info['right_mh'] = ''.join(bases[-right_mh_len:])

    return ''.join(bases), mh_info


# ═══════════════════════════════════════════════════════════════════════════ #
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════ #

def get_gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(seq.count(b) for b in 'GCgc') / len(seq)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Hard Negative Generators
# ═══════════════════════════════════════════════════════════════════════════ #

def generate_neg_conserved(
    bg_pool_with_prov: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_CONSERVED: conserved-region negative sample.

    Sample fragments from two different microbe genomes and concatenate them into a
    chimera. These fragments may come from highly conserved regions such as rRNA or
    housekeeping genes — highly similar across species yet not HGT.

    Simulation strategy: pick two host fragments with similar GC content and join
    them, so the sequence compositionally looks like a cross-species transfer when
    it is merely conserved homology.

    Parameters
    ----------
    bg_pool_with_prov : list of (seq, prov_dict)
    target_len        : target length

    Returns
    -------
    (seq, prov_segs)
    """
    n = len(bg_pool_with_prov)
    if n < 2:
        seq, prov = bg_pool_with_prov[0]
        result = seq[:target_len]
        return result, [dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov['file'], source_seq_id=prov['seq_id'],
            genome_start=prov['genome_start'],
            genome_end=prov['genome_start'] + len(result),
            strand='+', insert_start=0, insert_end=len(result),
        )]

    # Sample several candidates and pick the pair with the closest GC content
    candidates = random.sample(range(n), min(8, n))
    best_pair = None
    best_gc_diff = float('inf')
    for _ in range(min(4, len(candidates) // 2)):
        i1, i2 = random.sample(candidates, 2)
        gc1 = get_gc_content(bg_pool_with_prov[i1][0][:target_len])
        gc2 = get_gc_content(bg_pool_with_prov[i2][0][:target_len])
        diff = abs(gc1 - gc2)
        if diff < best_gc_diff:
            best_gc_diff = diff
            best_pair = (i1, i2)

    i1, i2 = best_pair
    seq1, prov1 = bg_pool_with_prov[i1]
    seq2, prov2 = bg_pool_with_prov[i2]

    # Join in the middle
    split = random.randint(target_len // 4, 3 * target_len // 4)
    part1 = seq1[:split]
    part2 = seq2[split:split + (target_len - split)]
    result = (part1 + part2)[:target_len]
    actual_split = min(split, target_len)

    prov_segs = []
    if actual_split > 0:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov1['file'], source_seq_id=prov1['seq_id'],
            genome_start=prov1['genome_start'],
            genome_end=prov1['genome_start'] + actual_split,
            strand='+', insert_start=0, insert_end=actual_split,
        ))
    len2 = len(result) - actual_split
    if len2 > 0:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov2['file'], source_seq_id=prov2['seq_id'],
            genome_start=prov2['genome_start'] + split,
            genome_end=prov2['genome_start'] + split + len2,
            strand='+', insert_start=actual_split, insert_end=len(result),
        ))
    return result, prov_segs


def generate_neg_homology(
    bg_pool_with_prov: List[Tuple[str, dict]],
    virus_pool: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_HOMOLOGY: evolutionarily homologous negative sample.

    Sample a fragment from the host, then apply a few directed mutations to nudge its
    composition closer to virus-like features (GC shift, specific k-mer patterns),
    yet the sequence genuinely originates from the host — not HGT.

    This class of sample directly challenges detectors based on k-mer matching or
    compositional differences.

    Parameters
    ----------
    bg_pool_with_prov : list of (seq, prov_dict)
    virus_pool        : list of (seq, prov_dict)  —  used to obtain virus composition features
    target_len        : target length

    Returns
    -------
    (seq, prov_segs)
    """
    seq, prov = random.choice(bg_pool_with_prov)
    seq = seq[:target_len]
    bases = list(seq)

    # Compute the virus pool's average GC content
    if virus_pool:
        v_gc = sum(get_gc_content(v[0][:200]) for v in virus_pool[:50]) / min(50, len(virus_pool))
    else:
        v_gc = 0.5

    # Directed mutation: replace a few bases so GC content approaches the virus's
    n_mutations = max(1, int(target_len * 0.03))  # ~3% directed mutations
    positions = random.sample(range(len(bases)), min(n_mutations, len(bases)))
    for pos in positions:
        cur_gc = get_gc_content(''.join(bases))
        if cur_gc < v_gc:
            # Need GC up: replace with G or C
            bases[pos] = random.choice('GC')
        elif cur_gc > v_gc:
            # Need GC down: replace with A or T
            bases[pos] = random.choice('AT')
        else:
            bases[pos] = random.choice([b for b in 'ATCG' if b != bases[pos]])

    result = ''.join(bases)
    return result, [dict(
        seg_type='HOST', source_type='microbe',
        source_file=prov['file'], source_seq_id=prov['seq_id'],
        genome_start=prov['genome_start'],
        genome_end=prov['genome_start'] + len(result),
        strand='+', insert_start=0, insert_end=len(result),
    )]


def generate_neg_low_complexity(
    target_len: int,
    bg_pool_with_prov: Optional[List[Tuple[str, dict]]] = None,
) -> Tuple[str, List[dict]]:

    """
    NEG_LOW_COMPLEXITY: low-complexity negative sample.

    Built from repeat units (di/tri/tetra-nucleotide repeats) and poly-A/T.
    When bg_pool_with_prov is available, preferentially extract real low-complexity
    regions from host genomes (e.g. SSR / Simple Sequence Repeats); otherwise fall
    back to synthetic repeat sequences.
    These sequences can fool k-mer-frequency or composition-based detectors because
    they may produce spurious cross-species similarity signals.

    Parameters
    ----------
    target_len        : target length
    bg_pool_with_prov : host sequence pool (optional, for extracting real
                        low-complexity regions)

    Returns
    -------
    (seq, prov_segs)  —  prov_segs has source_type='low_complexity'
    """
    # If a host pool exists, with 50% probability extract a real low-complexity
    # region from it
    if bg_pool_with_prov and len(bg_pool_with_prov) >= 2 and random.random() < 0.5:
        return _generate_host_derived_low_complexity(bg_pool_with_prov, target_len)

    # Fall back to a synthetic repeat sequence
    repeat_units = [
        'AT', 'GC', 'TA', 'CG',
        'ATC', 'GAT', 'TAC', 'GCA',
        'ATCG', 'GCAT', 'TACG', 'AGCT',
    ]
    # Mostly a single repeat; occasionally insert random sequence
    parts = []
    filled = 0
    while filled < target_len:
        if random.random() < 0.15:
            # Insert random sequence to break up the repeat
            rand_len = random.randint(5, 20)
            rand_seq = ''.join(random.choice('ATCG') for _ in range(rand_len))
            parts.append(rand_seq)
            filled += rand_len
        else:
            # Repeat unit
            unit = random.choice(repeat_units)
            repeat_count = random.randint(5, max(6, target_len // len(unit) // 3))
            repeat_seq = unit * repeat_count
            parts.append(repeat_seq)
            filled += len(repeat_seq)

    seq = ''.join(parts)[:target_len]

    # Occasionally insert a poly-A/T run inside the sequence
    if target_len > 50 and random.random() < 0.4:
        poly_len = random.randint(10, min(30, target_len // 5))
        poly_base = random.choice('AT')
        poly = poly_base * poly_len
        pos = random.randint(0, target_len - poly_len)
        seq = seq[:pos] + poly + seq[pos + poly_len:]

    return seq, [dict(
        seg_type='LOW_COMPLEXITY', source_type='synthetic',
        source_file='synthetic', source_seq_id='repeat',
        genome_start=0, genome_end=len(seq),
        strand='+', insert_start=0, insert_end=len(seq),
    )]


def _generate_host_derived_low_complexity(
    bg_pool_with_prov: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    Build a negative sample by extracting a real low-complexity region from a host
    genome.

    Strategy:
      1. Scan several fragments from the host pool to find ones containing
         low-complexity regions (identified via sliding-window GC variance:
         high-variance regions contain repeats)
      2. Extract the low-complexity region and join it with a fragment from another
         host
      3. This mimics real genomic SSRs (Simple Sequence Repeats) and the scenario
         where they are mistakenly aligned or k-mer matched to cross-species
         sequences

    Parameters
    ----------
    bg_pool_with_prov : list of (seq, prov_dict)
    target_len        : target length

    Returns
    -------
    (seq, prov_segs)
    """
    # Scan candidates: look for fragments with high variance (suggesting repeat regions)
    candidates = random.sample(range(len(bg_pool_with_prov)),
                               min(20, len(bg_pool_with_prov)))
    best_idx = None
    best_repeat_score = -1

    for idx in candidates:
        seq, _ = bg_pool_with_prov[idx]
        if len(seq) < 50:
            continue
        # Detect repeats: compute compositional uniformity with a sliding window
        window = 10
        gc_deviations = []
        for i in range(0, min(len(seq) - window, 200), window // 2):
            subseq = seq[i:i + window]
            gc = sum(1 for b in subseq if b in 'GCgc') / len(subseq)
            gc_deviations.append(gc)
        if len(gc_deviations) >= 3:
            # High variance = repeat structure present
            import statistics
            score = statistics.variance(gc_deviations)
            if score > best_repeat_score:
                best_repeat_score = score
                best_idx = idx

    if best_idx is None:
        # Fallback: pick a host fragment at random
        best_idx = random.choice(candidates)

    seq1, prov1 = bg_pool_with_prov[best_idx]
    seq1 = seq1[:target_len]

    # With 50% probability join with another host (increasing cross-species similarity)
    if random.random() < 0.5 and len(bg_pool_with_prov) >= 2:
        other_indices = [i for i in range(len(bg_pool_with_prov)) if i != best_idx]
        other_idx = random.choice(other_indices)
        seq2, prov2 = bg_pool_with_prov[other_idx]

        split = random.randint(target_len // 3, 2 * target_len // 3)
        part1 = seq1[:split]
        part2 = seq2[split:split + (target_len - split)]
        result = (part1 + part2)[:target_len]
        actual_split = min(split, target_len)

        prov_segs = []
        if actual_split > 0:
            prov_segs.append(dict(
                seg_type='LOW_COMPLEXITY', source_type='microbe',
                source_file=prov1['file'], source_seq_id=prov1['seq_id'],
                genome_start=prov1['genome_start'],
                genome_end=prov1['genome_start'] + actual_split,
                strand='+', insert_start=0, insert_end=actual_split,
            ))
        len2 = len(result) - actual_split
        if len2 > 0:
            prov_segs.append(dict(
                seg_type='LOW_COMPLEXITY', source_type='microbe',
                source_file=prov2['file'], source_seq_id=prov2['seq_id'],
                genome_start=prov2['genome_start'] + split,
                genome_end=prov2['genome_start'] + split + len2,
                strand='+', insert_start=actual_split, insert_end=len(result),
            ))
        return result, prov_segs
    else:
        return seq1, [dict(
            seg_type='LOW_COMPLEXITY', source_type='microbe',
            source_file=prov1['file'], source_seq_id=prov1['seq_id'],
            genome_start=prov1['genome_start'],
            genome_end=prov1['genome_start'] + len(seq1),
            strand='+', insert_start=0, insert_end=len(seq1),
        )]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Contamination simulation
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_contamination(
    primary_seq: str,
    primary_prov_segs: List[dict],
    contaminant_pool: List[Tuple[str, dict]],
    contamination_rate: float = 0.05,
) -> Tuple[str, List[dict]]:
    """
    Inject a contaminant into the sequence, simulating experimental contamination /
    index hopping.

    Randomly select a region within the sequence (length ≈ contamination_rate ×
    total_len) and replace it with sequence sampled from contaminant_pool.

    Parameters
    ----------
    primary_seq       : original sequence
    primary_prov_segs : original provenance segments
    contaminant_pool  : list of (seq, prov_dict) contaminant sequence pool
    contamination_rate: contamination fraction (0.01–0.1)

    Returns
    -------
    (contaminated_seq, updated_prov_segs)
    """
    if contamination_rate <= 0 or not contaminant_pool:
        return primary_seq, primary_prov_segs

    seq_len = len(primary_seq)
    contam_len = max(1, int(seq_len * contamination_rate))
    contam_start = random.randint(0, max(0, seq_len - contam_len))

    # Sample from the contaminant pool
    c_seq, c_prov = random.choice(contaminant_pool)
    actual_len = min(contam_len, len(c_seq))
    contam_fragment = c_seq[:actual_len]

    # Replace the sequence region
    result_seq = primary_seq[:contam_start] + contam_fragment + primary_seq[contam_start + actual_len:]
    result_seq = result_seq[:seq_len]  # keep the original length

    # Build the new provenance segments
    new_segs = []
    for seg in primary_prov_segs:
        seg_s = seg.get('insert_start', 0)
        seg_e = seg.get('insert_end', seq_len)
        # Entirely before the contaminated region
        if seg_e <= contam_start:
            new_segs.append(seg)
        # Entirely after the contaminated region
        elif seg_s >= contam_start + actual_len:
            # Offset unchanged, since contamination is a replacement (total length
            # unchanged)
            new_segs.append(seg)
        else:
            # Partial overlap: keep the part before the contaminated region
            if seg_s < contam_start:
                new_segs.append({**seg, 'insert_end': contam_start})

    # Insert contaminant provenance (stride-aware truncation)
    gs = c_prov['genome_start']
    ge = c_prov['genome_end']
    if c_prov.get('strand') == '-':
        gs = c_prov['genome_end'] - actual_len
    else:
        ge = c_prov['genome_start'] + actual_len
    new_segs.append(dict(
        seg_type='CONTAMINANT', source_type='contaminant',
        source_file=c_prov['file'], source_seq_id=c_prov['seq_id'],
        genome_start=gs,
        genome_end=ge,
        strand=c_prov.get('strand', '+'),
        insert_start=contam_start, insert_end=contam_start + actual_len,
    ))

    # Original segments after the contaminated region
    for seg in primary_prov_segs:
        seg_s = seg.get('insert_start', 0)
        seg_e = seg.get('insert_end', seq_len)
        if seg_e > contam_start + actual_len and seg_s < contam_start + actual_len:
            overlap_end = seg_e
            new_segs.append(dict(
                **{k: v for k, v in seg.items() if k not in ('insert_start', 'insert_end', 'genome_start')},
                genome_start=seg['genome_start'] + (contam_start + actual_len - seg_s),
                insert_start=contam_start + actual_len,
                insert_end=overlap_end,
            ))
        elif seg_s >= contam_start + actual_len:
            pass  # already added above

    new_segs.sort(key=lambda x: x.get('insert_start', 0))
    return result_seq, new_segs


# ═══════════════════════════════════════════════════════════════════════════ #
#  Assembly Artifacts (Contig mode only)
# ═══════════════════════════════════════════════════════════════════════════ #

def apply_assembly_artifact(
    seq: str,
    artifact_rate: float = 0.0,
) -> Tuple[str, str]:
    """
    Apply an assembly artifact to the contig with probability artifact_rate.

    Artifact types:
      - truncation : truncate (trim a stretch from the end)
      - n_gap      : N-gap (replace a middle region with NNNN)
      - misassembly: misjoin (append an unrelated region to the end)

    Parameters
    ----------
    seq           : input contig sequence
    artifact_rate : per-contig probability of applying an artifact

    Returns
    -------
    (modified_seq, artifact_type)
    artifact_type ∈ {'none', 'truncation', 'n_gap', 'misassembly'}
    """
    if artifact_rate <= 0 or random.random() >= artifact_rate:
        return seq, 'none'

    L = len(seq)
    artifact = random.choice(['truncation', 'n_gap', 'misassembly'])

    if artifact == 'truncation':
        # Truncation: trim 5-20% from the end
        trim_len = random.randint(max(1, L // 20), max(2, L // 5))
        return seq[:L - trim_len], 'truncation'

    elif artifact == 'n_gap':
        # N-gap: insert an N region in the middle
        gap_len = random.randint(5, max(6, L // 10))
        gap_start = random.randint(0, max(0, L - gap_len))
        result = seq[:gap_start] + 'N' * gap_len + seq[gap_start + gap_len:]
        return result, 'n_gap'

    else:  # misassembly
        # Misjoin: replace the end of the sequence with a reversed fragment
        # (mimicking a chimera)
        join_point = random.randint(L // 3, 2 * L // 3)
        # Take a fragment from another random region of the sequence and append it
        frag_start = random.randint(0, max(0, join_point - 50))
        frag_len = L - join_point
        frag = seq[frag_start:frag_start + frag_len]
        if len(frag) < frag_len:
            frag = frag + 'N' * (frag_len - len(frag))
        result = seq[:join_point] + frag
        return result, 'misassembly'


# ═════════════════════════════════════════════════════════════════════════════ #
#  Additional Hard Negative Generators (based on gaps found in real-benchmark evaluation)
# ═════════════════════════════════════════════════════════════════════════════ #

def _gc_level_precise(seq_list: list, target_gc: float, max_iter: int = 5) -> list:
    """Precise GC leveling: tune seq_list's GC to target_gc, ensuring each swap is
    effective.
    Algorithm: compute the gap between the current and target GC, swap only between
    AT↔GC, and iterate with validation.
    """
    seq = seq_list[:]
    n = len(seq)
    if n == 0:
        return seq
    for _ in range(max_iter):
        current_gc = get_gc_content(''.join(seq))
        diff = target_gc - current_gc
        if abs(diff) < 0.005:
            break
        n_needed = int(abs(diff) * n / 2) + 1
        if diff > 0:
            at_positions = [i for i in range(n) if seq[i] in 'AT']
            random.shuffle(at_positions)
            for pos in at_positions[:n_needed]:
                seq[pos] = random.choice(list('GC'))
        else:
            gc_positions = [i for i in range(n) if seq[i] in 'GC']
            random.shuffle(gc_positions)
            for pos in gc_positions[:n_needed]:
                seq[pos] = random.choice(list('AT'))
    return seq


def _kmer_to_idx(kmer: str, k: int = 4) -> int:
    """4-mer → index [0, 4**k). Returns -1 if N is present."""
    idx = 0
    for b in kmer:
        v = {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(b, -1)
        if v < 0:
            return -1
        idx = idx * 4 + v
    return idx


def _kmer_amelioration(vfrag, host_ref, k: int = 4, max_iter: int = 30,
                       l1_threshold: float = 0.3, rng=None):
    """P1a-B (RFC-P1-CONTIG-CHEAT-001): level the virus fragment's k-mers toward the
    host reference.

    Non-GC-preserving swaps (any base A/C/G/T), directly optimizing 4-mer L1.
    GC is re-leveled afterwards in _gc_level_precise (called within
    generate_neg_gc_matched_chimeric).
    Weakens the k-mer shortcut in NTv3 encodings (the root cause of F3 contig
    cheating).

    Greedy: at each step randomly pick candidate positions, try all 4 base swaps,
    and accept a swap if it lowers the 4-mer L1 (vfrag vs host_ref). Iterate until
    L1 < l1_threshold, max_iter, or no improvement.
    """
    import numpy as np
    if rng is None:
        rng = random
    # Scheme B: non-GC-preserving  —  arbitrary base swaps toward host 4-mers
    # (directly optimizes the 4-mer profile)
    bases = 'ACGT'
    cur = list(vfrag)
    L = len(cur)
    dim = 4 ** k

    def _freq(seq_list):
        counts = np.zeros(dim, dtype=np.float64)
        for i in range(len(seq_list) - k + 1):
            idx = _kmer_to_idx(''.join(seq_list[i:i + k]), k)
            if idx >= 0:
                counts[idx] += 1
        s = counts.sum()
        return counts / s if s > 0 else counts

    l1 = float(np.abs(_freq(cur) - host_ref).sum())
    for _ in range(max_iter):
        if l1 < l1_threshold:
            break
        improved = False
        n_cand = min(L, 100)
        for pos in rng.sample(range(L), n_cand):
            orig = cur[pos]
            best_b, best_l1 = orig, l1
            for b in bases:
                if b == orig:
                    continue
                cur[pos] = b
                new_l1 = float(np.abs(_freq(cur) - host_ref).sum())
                if new_l1 < best_l1:
                    best_l1 = new_l1
                    best_b = b
            cur[pos] = best_b
            if best_l1 < l1 - 1e-6:
                l1 = best_l1
                improved = True
        if not improved:
            break
    return cur


# P1a-C: standard codon table (64 codons → AA)
_CODON_TABLE = {'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L','ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V','TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P','ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E','TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R','AGT':'S','AGC':'S','AGA':'R','AGG':'R'}


def _compute_codon_usage(seq: str) -> dict:
    """Compute codon usage frequencies from the sequence (64 codons → freq).
    Reading frame 0."""
    from collections import Counter
    counts = Counter()
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3].upper()
        if codon in _CODON_TABLE:
            counts[codon] += 1
    total = sum(counts.values()) or 1
    return {c: counts.get(c, 0) / total for c in _CODON_TABLE}


def _codon_amelioration(vfrag, host_codon_usage, replacement_rate=0.6, rng=None):
    """P1a-C: codon-level leveling  —  synonymous replacement toward host codon usage
    (amino acids preserved, reading frame 0).

    For each codon in vfrag's reading frame 0, with probability replacement_rate
    replace it with the host's most preferred synonymous codon.
    Shifts 4-mers toward the host (leveling the k-mer shortcut) while preserving
    amino acids (coding region). May change GC (re-leveled afterwards).
    """
    from collections import defaultdict
    if rng is None:
        rng = random
    aa_to_codons = defaultdict(list)
    for codon, aa in _CODON_TABLE.items():
        aa_to_codons[aa].append(codon)
    aa_best = {aa: max(codons, key=lambda c: host_codon_usage.get(c, 0))
               for aa, codons in aa_to_codons.items()}
    cur = list(vfrag)
    for i in range(0, len(cur) - 2, 3):
        codon = ''.join(cur[i:i + 3]).upper()
        aa = _CODON_TABLE.get(codon)
        if aa is None or aa == '*':
            continue
        best = aa_best[aa]
        if codon != best and rng.random() < replacement_rate:
            for j, b in enumerate(best):
                cur[i + j] = b
    return cur


def generate_neg_gc_matched_chimeric(
    bg_seq: str,
    bg_prov: dict,
    virus_pool: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_GC_MATCHED_CHIMERIC: GC-matched chimeric negative sample.

    The model's largest error source: level the virus fragment's GC to match the
    host, then insert it into the host; the sequence "contains a non-host fragment"
    (different k-mers/composition) yet no GC shift exists.

    The model must learn that "non-host k-mers present" alone does not imply HGT+;
    it must also detect the GC shift and breakpoint discontinuity.

    Generation strategy (v2 revised):
    1. Sample a target_len region from the host background
    2. Sample a fragment (500-2000 bp) from the virus pool
    3. First apply silent mutations (AT↔AT, GC↔GC) to preserve k-mer diversity
    4. Compute local host GC at the insertion point (500 bp windows on both sides),
       and precisely GC-level the virus fragment toward it
    5. Validate that the post-leveling GC deviation < 0.01; otherwise iterate
    6. Replace a stretch of host with the GC-leveled virus fragment

    Returns (seq, prov_segs)
    """
    if not virus_pool:
        seq = bg_seq[:target_len]
        return seq, [dict(
            seg_type='HOST', source_type='microbe',
            source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
            genome_start=bg_prov['genome_start'],
            genome_end=bg_prov['genome_start'] + len(seq),
            strand='+', insert_start=0, insert_end=len(seq),
        )]

    bg = list(bg_seq[:target_len])
    if len(bg) < target_len:
        bg.extend(['N'] * (target_len - len(bg)))
    bg = bg[:target_len]

    vfrag_seq, vfrag_prov = random.choice(virus_pool)
    min_insert = min(50, len(vfrag_seq))
    upper = max(min_insert, min(2000, target_len // 3))
    insert_len = min(random.randint(min_insert, upper), len(vfrag_seq))
    vfrag = list(vfrag_seq[:insert_len].upper())

    head = target_len // 5
    insert_pos = random.randint(head, max(head, target_len - head - insert_len))

    # silent mutations: GC-preserving swaps (AT↔AT, GC↔GC) to preserve k-mer diversity
    n_silent = int(len(vfrag) * random.uniform(0.02, 0.08))
    for _ in range(n_silent):
        pos = random.randint(0, len(vfrag) - 1)
        base = vfrag[pos]
        if base in 'AT':
            vfrag[pos] = 'T' if base == 'A' else 'A'
        elif base in 'GC':
            vfrag[pos] = 'C' if base == 'G' else 'G'

    # local host GC: use flanking region around insert position
    flank = min(500, insert_pos, target_len - insert_pos - insert_len)
    left_region = bg[max(0, insert_pos - flank):insert_pos]
    right_region = bg[insert_pos + insert_len:insert_pos + insert_len + flank]
    flank_seq = ''.join(left_region + right_region)
    local_host_gc = get_gc_content(flank_seq) if len(flank_seq) > 50 else get_gc_content(''.join(bg))

    # precise GC leveling
    vfrag = _gc_level_precise(vfrag, local_host_gc, max_iter=8)

    # P1a-B: non-GC-preserving k-mer amelioration (arbitrary swaps toward host
    # 4-mers) + GC re-leveling
    try:
        import numpy as np
        _dim = 256
        _href = np.zeros(_dim, dtype=np.float64)
        for _i in range(len(flank_seq) - 3):
            _idx = _kmer_to_idx(flank_seq[_i:_i + 4])
            if _idx >= 0:
                _href[_idx] += 1
        _s = _href.sum()
        if _s > 0:
            _href /= _s
            vfrag = _kmer_amelioration(vfrag, _href, max_iter=100, l1_threshold=0.3, rng=random)
            vfrag = _gc_level_precise(vfrag, local_host_gc, max_iter=8)  # non-GC-preserving swaps changed GC; re-level
    except Exception:
        pass

    vfrag_matched = ''.join(vfrag)

    result = bg[:insert_pos] + list(vfrag_matched) + bg[insert_pos + insert_len:]
    result = result[:target_len]
    result_str = ''.join(result)

    prov_segs = []
    if insert_pos > 0:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
            genome_start=bg_prov['genome_start'],
            genome_end=bg_prov['genome_start'] + insert_pos,
            strand='+', insert_start=0, insert_end=insert_pos,
        ))
    prov_segs.append(dict(
        seg_type='VIRUS_GC_MATCHED', source_type='virus_gc_matched',
        source_file=vfrag_prov.get('file', ''),
        source_seq_id=vfrag_prov.get('seq_id', ''),
        genome_start=0, genome_end=insert_len,
        strand='+', insert_start=insert_pos, insert_end=insert_pos + insert_len,
    ))
    tail_start = insert_pos + insert_len
    if tail_start < target_len:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
            genome_start=bg_prov['genome_start'] + tail_start,
            genome_end=bg_prov['genome_start'] + target_len,
            strand='+', insert_start=tail_start, insert_end=target_len,
        ))

    return result_str, prov_segs


def generate_neg_amalgamated_hgt(
    bg_seq: str,
    bg_prov: dict,
    virus_pool: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_AMALGAMATED_HGT: ancient, already-amalgamated HGT negative sample.

    Simulates the amelioration process described by Lawrence & Ochman (1997):
    a gene acquired by HGT converges to the host's GC and codon preferences over
    ~10 Myr.
    Features of this class of sequence:
    - Overall GC matches the host (no obvious GC shift)
    - k-mer frequencies differ slightly from the host but have largely converged
    - No real breakpoint (the sequence is contiguous)
    - Yet may retain faint foreign k-mer features

    Key differences from true HGT positives:
    - No breakpoint discontinuity
    - GC shift has vanished
    - k-mer patterns have converged

    Generation strategy:
    1. Sample a target_len region from the host
    2. Choose a stretch (1000-3000 bp) as the "ancient HGT region"
    3. Apply strong GC amelioration (→ host GC) plus weak k-mer perturbation to
       that region
    4. Insert no foreign fragment; only modify the composition so it "looks
       somewhat foreign"

    Returns (seq, prov_segs)
    """
    bg = list(bg_seq[:target_len])
    if len(bg) < target_len:
        bg.extend(['N'] * (target_len - len(bg)))
    bg = bg[:target_len]

    # Scale region_len with target_len to avoid overflow on short contigs
    region_min = max(target_len // 15, target_len // 10)
    region_max = max(region_min, min(target_len // 3, 3000))
    region_len = random.randint(region_min, region_max)
    head = target_len // 5
    region_start = random.randint(head, max(head, target_len - head - region_len))
    region = bg[region_start:region_start + region_len]

    host_gc = get_gc_content(''.join(bg))
    region_gc = get_gc_content(''.join(region))

    gc_diff = host_gc - region_gc
    if abs(gc_diff) > 0.01:
        n_gc_changes = int(abs(gc_diff) * len(region) * 0.8)
        adjust_bases = 'GC' if gc_diff > 0 else 'AT'
        positions = random.sample(range(len(region)), min(n_gc_changes, len(region)))
        for pos in positions:
            region[pos] = random.choice(list(adjust_bases))

    n_compositional = int(len(region) * random.uniform(0.01, 0.04))
    positions = random.sample(range(len(region)), min(n_compositional, len(region)))
    for pos in positions:
        region[pos] = random.choice(list('ATGC'))

    result = bg[:region_start] + region + bg[region_start + region_len:]
    result = result[:target_len]
    result_str = ''.join(result)

    return result_str, [dict(
        seg_type='HOST_AMALGAMATED', source_type='microbe_amalgamated',
        source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
        genome_start=bg_prov['genome_start'],
        genome_end=bg_prov['genome_start'] + len(result_str),
        strand='+', insert_start=0, insert_end=len(result_str),
    )]


def generate_neg_partial_virus_hit(
    bg_pool_with_prov: List[Tuple[str, dict]],
    virus_pool: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_PARTIAL_VIRUS_HIT: host region partially homologous to a virus.

    A microbe genome may contain a region with distant homology to a virus sequence
    (e.g., conserved RNA polymerase domains, nucleotide-binding proteins); this is
    not HGT, yet k-mer matching triggers a false call.

    Generation strategy:
    1. Sample a target_len fragment from the host
    2. Apply directed mutations in a middle region (10-30%) to pull its k-mers
       toward the virus
    3. But keep the overall sequence frame and coverage consistent with pure host
    4. Key: the whole sequence still best-matches the host in BLAST

    Returns (seq, prov_segs)
    """
    if not bg_pool_with_prov or not virus_pool:
        seq, prov = random.choice(bg_pool_with_prov) if bg_pool_with_prov else ('A' * target_len, {})
        return seq[:target_len], [dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov.get('file', ''), source_seq_id=prov.get('seq_id', ''),
            genome_start=prov.get('genome_start', 0),
            genome_end=prov.get('genome_start', 0) + target_len,
            strand='+', insert_start=0, insert_end=target_len,
        )]

    bg_seq, bg_prov = random.choice(bg_pool_with_prov)
    bg = list(bg_seq[:target_len])
    if len(bg) < target_len:
        bg.extend(['N'] * (target_len - len(bg)))
    bg = bg[:target_len]

    vfrag_seq, _ = random.choice(virus_pool)
    vfrag = vfrag_seq[:min(200, len(vfrag_seq))].upper()
    v_kmers = set()
    for i in range(len(vfrag) - 3):
        v_kmers.add(vfrag[i:i+4])

    mimic_start = random.randint(target_len // 5, 2 * target_len // 5)
    mimic_len = random.randint(target_len // 10, target_len // 5)
    mimic_end = min(mimic_start + mimic_len, target_len)

    n_mimic = int((mimic_end - mimic_start) * random.uniform(0.05, 0.15))
    positions = random.sample(range(mimic_start, mimic_end), min(n_mimic, mimic_end - mimic_start))
    for pos in positions:
        if pos + 4 <= len(bg):
            target_kmer = random.choice(list(v_kmers)) if v_kmers else 'ATCG'
            for j in range(4):
                bg[pos + j] = target_kmer[j]

    result_str = ''.join(bg)

    return result_str, [dict(
        seg_type='HOST_VIRUS_HOMOLOG', source_type='microbe_virus_homolog',
        source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
        genome_start=bg_prov['genome_start'],
        genome_end=bg_prov['genome_start'] + len(result_str),
        strand='+', insert_start=0, insert_end=len(result_str),
    )]


# ═════════════════════════════════════════════════════════════════════════════ #
#  PE-specific Hard Negative Generators
#  Addressing the core sim-to-real gap issues in PE mode:
#    1. R1/R2 share the same origin but the composition looks inconsistent
#       (spanning a GC-shift region, a conserved region, etc.)
#    2. Embedding differences between host regions are misread as an HGT signal
# ═════════════════════════════════════════════════════════════════════════════ #

def generate_pe_neg_host_diversity(
    bg_pool_with_prov: List[Tuple[str, dict]],
    virus_pool: List[Tuple[str, dict]],
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_PE_HOST_DIVERSITY: host-diversity negative sample (PE-specific).

    Simulates data where hosts come from a limited genome set, so the model overfits
    to particular host embeddings. This negative sample joins across genomes,
    forcing the model not to rely on specific host embedding patterns.

    Generation strategy:
    1. Sample one stretch from each of two different microbe genomes
    2. Join them into one sequence (R1 from genome A, R2 from genome B)
    3. Both are host, but from different species → looks like HGT but is not

    Returns (seq, prov_segs)
    """
    if len(bg_pool_with_prov) < 2:
        seq, prov = random.choice(bg_pool_with_prov)
        result = seq[:target_len]
        return result, [dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov['file'], source_seq_id=prov['seq_id'],
            genome_start=prov['genome_start'],
            genome_end=prov['genome_start'] + len(result),
            strand='+', insert_start=0, insert_end=len(result),
        )]

    idx1, idx2 = random.sample(range(len(bg_pool_with_prov)), 2)
    seq1, prov1 = bg_pool_with_prov[idx1]
    seq2, prov2 = bg_pool_with_prov[idx2]

    split = random.randint(target_len // 3, 2 * target_len // 3)
    part1 = seq1[:split]
    part2 = seq2[:target_len - split]
    result = (part1 + part2)[:target_len]

    prov_segs = []
    actual_split = min(split, len(part1))
    if actual_split > 0:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov1['file'], source_seq_id=prov1['seq_id'],
            genome_start=prov1['genome_start'],
            genome_end=prov1['genome_start'] + actual_split,
            strand='+', insert_start=0, insert_end=actual_split,
        ))
    len2 = len(result) - actual_split
    if len2 > 0:
        prov_segs.append(dict(
            seg_type='HOST', source_type='microbe',
            source_file=prov2['file'], source_seq_id=prov2['seq_id'],
            genome_start=prov2['genome_start'],
            genome_end=prov2['genome_start'] + len2,
            strand='+', insert_start=actual_split, insert_end=len(result),
        ))

    return result, prov_segs


def generate_pe_neg_gc_shift_region(
    bg_seq: str,
    bg_prov: dict,
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_PE_GC_SHIFT_REGION: intragenomic host GC-shift region negative sample
    (PE-specific).

    Microbe genomes contain natural GC-shift regions (e.g., GC-rich rRNA operons,
    GC-poor AT-rich islands); when R1 and R2 span such a region their GC differs,
    and the model may misread this as "R1/R2 origins inconsistent" → false HGT+.

    Generation strategy:
    1. Pick a stretch from a host sequence
    2. Apply a directed GC shift to the second half (mimicking natural
       intragenomic GC variation)
    3. R1/R2 come from the same host but with different GC → the model must learn
       to distinguish this

    Returns (seq, prov_segs)
    """
    bg = list(bg_seq[:target_len])
    if len(bg) < target_len:
        bg.extend(['N'] * (target_len - len(bg)))
    bg = bg[:target_len]

    shift_start = random.randint(target_len // 3, target_len // 2)
    shift_len = random.randint(target_len // 6, target_len // 3)
    shift_end = min(shift_start + shift_len, target_len)

    gc_shift = random.choice([-1, 1]) * random.uniform(0.05, 0.15)
    region = bg[shift_start:shift_end]
    current_gc = get_gc_content(''.join(region))
    target_gc = max(0.15, min(0.85, current_gc + gc_shift))

    n_changes = int(abs(gc_shift) * len(region) / 2)
    adjust_bases = 'GC' if gc_shift > 0 else 'AT'
    positions = random.sample(range(len(region)), min(n_changes, len(region)))
    for pos in positions:
        region[pos] = random.choice(list(adjust_bases))

    result = bg[:shift_start] + region + bg[shift_end:]
    result = result[:target_len]
    result_str = ''.join(result)

    return result_str, [dict(
        seg_type='HOST_GC_SHIFT', source_type='microbe_gc_shift',
        source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
        genome_start=bg_prov['genome_start'],
        genome_end=bg_prov['genome_start'] + len(result_str),
        strand='+', insert_start=0, insert_end=len(result_str),
    )]


def generate_pe_neg_codon_boundary(
    bg_seq: str,
    bg_prov: dict,
    target_len: int,
) -> Tuple[str, List[dict]]:
    """
    NEG_PE_CODON_BOUNDARY: codon-preference boundary negative sample (PE-specific).

    Different genes have different codon preferences, and codon usage jumps at gene
    boundaries within a genome. When R1/R2 span a gene boundary, their
    k-mer/composition patterns look inconsistent and may be misjudged as HGT.

    Generation strategy:
    1. Pick a stretch from a host sequence
    2. Apply a codon-preference shift to a middle region (synonymous-mutation
       replacement)
    3. Keep GC unchanged (only the triplet usage pattern changes)

    Returns (seq, prov_segs)
    """
    bg = list(bg_seq[:target_len])
    if len(bg) < target_len:
        bg.extend(['N'] * (target_len - len(bg)))
    bg = bg[:target_len]

    region_start = random.randint(target_len // 4, target_len // 2)
    region_len = random.randint(target_len // 8, target_len // 4)
    region_end = min(region_start + region_len, target_len)

    region = bg[region_start:region_end]
    n_codon_swaps = len(region) // 6

    gc_synonymous = {
        'TTT': ['TTC'], 'TTC': ['TTT'], 'TTA': ['TTG'], 'TTG': ['TTA'],
        'CTT': ['CTC', 'CTG'], 'CTC': ['CTT', 'CTG'], 'CTA': ['CTG'], 'CTG': ['CTC', 'CTA'],
        'ATT': ['ATC', 'ATA'], 'ATC': ['ATT', 'ATA'], 'ATA': ['ATT', 'ATC'],
        'GTT': ['GTC', 'GTG'], 'GTC': ['GTT', 'GTG'], 'GTA': ['GTG'], 'GTG': ['GTC', 'GTA'],
        'TCT': ['TCC', 'TCG'], 'TCC': ['TCT', 'TCG'], 'TCA': ['TCG', 'TCT'],
        'TCG': ['TCC', 'TCA'], 'CCT': ['CCC', 'CCG'], 'CCC': ['CCT', 'CCG'],
        'CCA': ['CCG'], 'CCG': ['CCC', 'CCA'],
        'ACT': ['ACC', 'ACG'], 'ACC': ['ACT', 'ACG'], 'ACA': ['ACG'], 'ACG': ['ACC', 'ACA'],
        'GCT': ['GCC', 'GCG'], 'GCC': ['GCT', 'GCG'], 'GCA': ['GCG'], 'GCG': ['GCC', 'GCA'],
        'CGT': ['CGC', 'CGG'], 'CGC': ['CGT', 'CGG'], 'CGA': ['CGG'], 'CGG': ['CGC', 'CGA'],
        'AGT': ['AGC'], 'AGC': ['AGT'], 'AGA': ['AGG'], 'AGG': ['AGA'],
        'GGT': ['GGC', 'GGG'], 'GGC': ['GGT', 'GGG'], 'GGA': ['GGG'], 'GGG': ['GGC', 'GGA'],
        'AAT': ['AAC'], 'AAC': ['AAT'], 'AAG': ['AAA'], 'AAA': ['AAG'],
        'GAT': ['GAC'], 'GAC': ['GAT'], 'GAG': ['GAA'], 'GAA': ['GAG'],
        'CAT': ['CAC'], 'CAC': ['CAT'], 'CAG': ['CAA'], 'CAA': ['CAG'],
        'TAT': ['TAC'], 'TAC': ['TAT'],
    }

    for _ in range(n_codon_swaps):
        pos = random.randint(0, len(region) - 3)
        codon_str = ''.join(region[pos:pos + 3])
        if 'N' in codon_str:
            continue
        syn = gc_synonymous.get(codon_str, [])
        if len(syn) > 0:
            new_codon = random.choice(syn)
            if new_codon != codon_str:
                for j in range(3):
                    region[pos + j] = new_codon[j]

    result = bg[:region_start] + region + bg[region_end:]
    result = result[:target_len]
    result_str = ''.join(result)

    return result_str, [dict(
        seg_type='HOST_CODON_SHIFT', source_type='microbe_codon_shift',
        source_file=bg_prov['file'], source_seq_id=bg_prov['seq_id'],
        genome_start=bg_prov['genome_start'],
        genome_end=bg_prov['genome_start'] + len(result_str),
        strand='+', insert_start=0, insert_end=len(result_str),
    )]
