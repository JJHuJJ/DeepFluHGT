# simulator/impl/genome_splitter.py
"""
Genome splitting module — partitions microbe files and virus strains into
non-overlapping train/val/test sets.
The split runs before data generation, ensuring that train/val/test use entirely
different genome sources.
"""

import os
import random
import re
from Bio import SeqIO

# IUPAC ambiguity codes → random choice of valid bases
_IUPAC_MAP = {
    'R': 'AG', 'Y': 'CT', 'S': 'GC', 'W': 'AT',
    'K': 'GT', 'M': 'AC', 'B': 'CGT', 'D': 'AGT',
    'H': 'ACT', 'V': 'ACG',
}


def split_microbe_genomes(microbe_dir: str,
                          train_ratio: float,
                          val_ratio: float,
                          test_ratio: float,
                          seed: int) -> dict:
    """
    Randomly split the files under the microbe genome directory by ratio.

    Parameters
    ----------
    microbe_dir : str — microbe genome directory
    train_ratio, val_ratio, test_ratio : float — ratios (must sum to 1)
    seed : int — random seed (deterministic partition)

    Returns
    -------
    dict : {'train': [fname, ...], 'val': [...], 'test': [...]}
    """
    files = sorted(
        f for f in os.listdir(microbe_dir)
        if f.endswith(('.fa', '.fasta', '.fna'))
    )
    if not files:
        raise RuntimeError(f"No genome files found in {microbe_dir}")

    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    return {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:],
    }


def _parse_strain_name(seq_id: str) -> str:
    """Extract and normalize the strain name from a sequence ID."""
    m = re.search(r'(A/\S+?\(\w+\))', seq_id)
    if not m:
        return seq_id.split()[0]
    name = m.group(1).lower()
    name = re.sub(r'/(\d{2})\(', lambda m: f'/19{m.group(1)}(', name)
    return name


def split_virus_genomes(virus_fasta: str,
                        train_ratio: float,
                        val_ratio: float,
                        test_ratio: float,
                        seed: int) -> dict:
    """
    Group virus genomes by strain, then split the groups by ratio.

    Returns
    -------
    dict : {'train': [(seq_str, seq_id), ...], 'val': [...], 'test': [...]}
           The return format matches load_virus_genomes and can be passed directly to
           generate_virus_pool.
    """
    # Group by strain (consistent with load_virus_genomes: IUPAC replacement + upper)
    strain_seqs = {} # strain -> [(seq_str, seq_id)]
    for rec in SeqIO.parse(virus_fasta, "fasta"):
        s = str(rec.seq).upper()
        if len(s) == 0:
            continue
        # IUPAC ambiguity codes → random concrete bases
        fixed = []
        for b in s:
            if b in 'ATCGN':
                fixed.append(b)
            elif b in _IUPAC_MAP:
                fixed.append(random.choice(_IUPAC_MAP[b]))
            else:
                fixed.append('N')
        strain = _parse_strain_name(rec.description)
        strain_seqs.setdefault(strain, []).append((''.join(fixed), rec.id))

    if not strain_seqs:
        raise RuntimeError(f"No sequences found in {virus_fasta}")

    strains = list(strain_seqs.keys())
    random.seed(seed)
    random.shuffle(strains)

    n = len(strains)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = min(n_train, n - 2)
    n_val = min(n_val, n - n_train - 1)

    result = {}
    splits = {
        'train': strains[:n_train],
        'val': strains[n_train:n_train + n_val],
        'test': strains[n_train + n_val:],
    }
    for split_name, split_strains in splits.items():
        genomes = []
        for strain in split_strains:
            genomes.extend(strain_seqs[strain])
        result[split_name] = genomes

    return result
