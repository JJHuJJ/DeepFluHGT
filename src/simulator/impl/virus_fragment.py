# simulator/impl/virus_fragment.py
"""
Virus genome loading and fragment generation (with provenance tracking)
"""

import random
from Bio import SeqIO
from Bio.Seq import Seq


def load_virus_genomes(fasta_file: str):
    """
    Load all virus genome sequences from a FASTA file.
    Automatically replace IUPAC ambiguity codes (R, Y, S, W, K, M, B, D, H, V) with
    random concrete bases.

    Returns
    -------
    genomes : list of (seq_str, seq_id)
    """
    # IUPAC ambiguity codes → random choice of valid bases
    _iupac_map = {
        'R': 'AG', 'Y': 'CT', 'S': 'GC', 'W': 'AT',
        'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT',
        'H': 'ACT', 'V': 'ACG',
    }
    genomes = []
    n_fixed = 0
    for rec in SeqIO.parse(fasta_file, "fasta"):
        s = str(rec.seq).upper()
        if len(s) > 0:
            # Replace ambiguity codes with random bases
            fixed = []
            for b in s:
                if b in 'ATCGN':
                    fixed.append(b)
                elif b in _iupac_map:
                    fixed.append(random.choice(_iupac_map[b]))
                    n_fixed += 1
                else:
                    fixed.append('N')
            genomes.append((''.join(fixed), rec.id))
    if not genomes:
        raise RuntimeError(f"No sequences found in {fasta_file}")
    if n_fixed > 0:
        print(f"[VirusDB] Replaced {n_fixed} IUPAC ambiguity code(s) with random bases")
    print(f"[VirusDB] Loaded {len(genomes)} virus genome(s) from {fasta_file}")
    return genomes


def generate_virus_fragments(genomes, min_len, max_len,
                              num_per_genome=500):
    """
    Backward-compatible wrapper: returns only the list of sequence strings (no provenance information).
    """
    pool_with_prov = generate_virus_pool(
        genomes, 'unknown_virus_file', min_len, max_len, num_per_genome)
    return [seq for seq, _ in pool_with_prov]


def generate_virus_pool(genomes, fasta_file: str,
                         min_len: int, max_len: int,
                         num_per_genome: int = 500,
                         tiers=None):
    """
    Randomly sample fragments from the virus genomes while recording provenance.

    Parameters
    ----------
    genomes : return value of load_virus_genomes, a list of (seq, seq_id)
    fasta_file : original FASTA file path (written into prov['file'])
    min_len / max_len : fragment length range (used when tiers=None)
    num_per_genome : number of fragments sampled per genome
    tiers : list of (min_len, max_len, weight) or None
                     stratified sampling: pick a tier by weight, then sample uniformly within the
                     tier range; when provided, min_len / max_len are ignored

    Returns
    -------
    pool : list of (fragment_seq: str, prov: dict)
           prov field: file, seq_id, genome_start, genome_end, strand
    """
    pool = []
    skipped_count = 0
    tier_skipped = {} # tier_range -> count of genomes that skipped it
    for genome_seq, genome_id in genomes:
        g_len = len(genome_seq)

        # Pre-filter compatible tiers: tier_min <= g_len
        if tiers:
            valid_tiers = [(lo, hi, w) for lo, hi, w in tiers if lo <= g_len]
            if not valid_tiers:
                skipped_count += 1
                continue
            for lo, hi, _ in tiers:
                if lo > g_len:
                    tier_skipped.setdefault((lo, hi), 0)
                    tier_skipped[(lo, hi)] += 1

        for _ in range(num_per_genome):
            # Stratified sampling: choose only among compatible tiers
            if tiers:
                tier_weights = [t[2] for t in valid_tiers]
                tier = random.choices(valid_tiers, weights=tier_weights, k=1)[0]
                # Clamp the upper bound to the genome length
                hi = min(tier[1], g_len)
                if tier[0] > hi:
                    continue # Defensive skip: should not happen (valid_tiers already filtered)
                L = random.randint(tier[0], hi)
            else:
                L = random.randint(min_len, max_len)

            if L >= g_len:
                frag = genome_seq
                v_start = 0
                v_end = g_len
                is_rc = False
            else:
                v_start = random.randint(0, g_len - L)
                v_end = v_start + L
                frag = genome_seq[v_start:v_end]
                is_rc = False

            strand = '+'
            if random.random() < 0.5:
                frag = str(Seq(frag).reverse_complement())
                strand = '-'

            prov = {
                'file': fasta_file,
                'seq_id': genome_id,
                'genome_start': v_start,
                'genome_end': v_end,
                'strand': strand,
            }
            pool.append((frag, prov))

    if tiers:
        tier_desc = ', '.join(f'{lo}-{hi}bp ({w:.0%})' for lo, hi, w in tiers)
        print(f"[VirusDB] Generated {len(pool)} fragment(s), "
              f"tiered: [{tier_desc}]")
        if skipped_count:
            print(f"[VirusDB] {skipped_count} genome(s) skipped "
                  f"(shorter than all tier minimums)")
        for (lo, hi), cnt in sorted(tier_skipped.items()):
            print(f"[VirusDB] Tier {lo}-{hi}bp: {cnt} genome(s) too short, excluded")
    else:
        print(f"[VirusDB] Generated {len(pool)} fragment(s), "
              f"length range [{min_len}, {max_len}] bp")
    return pool
