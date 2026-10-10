# data_pipeline/tests/test_bio_features.py
"""
Biological feature extraction tests.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data_pipeline.impl.bio_features import BioFeatureExtractor


def test_output_shape():
    """Output dimension = 260."""
    bio = BioFeatureExtractor()
    feats = bio.extract("ATCGATCG" * 10)
    assert feats.shape == (260,), f"shape={feats.shape}"
    return True


def test_gc_content():
    """GC content accuracy."""
    bio = BioFeatureExtractor()
    # 50% GC
    feats = bio.extract("ATCG" * 25)
    gc = feats[0]
    assert abs(gc - 0.5) < 0.01, f"GC={gc}"
    return True


def test_gc_all_a():
    """all A sequence: GC=0."""
    bio = BioFeatureExtractor()
    feats = bio.extract("A" * 100)
    assert feats[0] == 0.0
    return True


def test_gc_all_gc():
    """all GC sequence: GC=1."""
    bio = BioFeatureExtractor()
    feats = bio.extract("GC" * 50)
    assert feats[0] == 1.0
    return True


def test_kmer_frequency_sums_to_one():
    """4-mer frequencies sum to ≈ 1.0."""
    bio = BioFeatureExtractor()
    feats = bio.extract("ATCGATCG" * 20)
    kmer_sum = feats[4:].sum()  # skip the first 4 global statistics
    assert abs(kmer_sum - 1.0) < 0.01, f"kmer_sum={kmer_sum}"
    return True


def test_entropy():
    """Shannon entropy: low for a pure sequence, high for a mixed sequence."""
    bio = BioFeatureExtractor()
    feats_uniform = bio.extract("ATCG" * 50)
    feats_pure = bio.extract("A" * 200)

    entropy_uniform = feats_uniform[2]
    entropy_pure = feats_pure[2]

    assert entropy_uniform > entropy_pure, \
        f"uniform={entropy_uniform}, pure={entropy_pure}"
    assert entropy_pure < 0.01  # pure-sequence entropy ≈ 0
    return True


def test_feature_dim_constant():
    """feature_dim == 260."""
    bio = BioFeatureExtractor()
    assert bio.feature_dim == 260
    return True


def test_custom_k():
    """Custom k=3 → 4 + 64 = 68 dims."""
    bio = BioFeatureExtractor(kmer_size=3)
    feats = bio.extract("ATCG" * 10)
    assert feats.shape == (68,), f"shape={feats.shape}"
    return True


def test_short_sequence():
    """Short sequence (<k): k-mers all zero."""
    bio = BioFeatureExtractor()
    feats = bio.extract("AT")
    assert feats.shape == (260,)
    # k-mers all zero (sequence length < 4)
    assert feats[4:].sum() == 0.0
    return True


def test_n_bases_excluded():
    """k-mers containing N are excluded."""
    bio = BioFeatureExtractor()
    feats_clean = bio.extract("ATCG" * 10)
    feats_with_n = bio.extract("ATCN" * 10)

    # k-mer frequencies containing N sum to < 1.0
    kmer_sum = feats_with_n[4:].sum()
    assert kmer_sum < 1.0, f"kmer_sum with N={kmer_sum}"
    return True


def run():
    tests = [
        test_output_shape,
        test_gc_content,
        test_gc_all_a,
        test_gc_all_gc,
        test_kmer_frequency_sums_to_one,
        test_entropy,
        test_feature_dim_constant,
        test_custom_k,
        test_short_sequence,
        test_n_bases_excluded,
    ]
    results = []
    for t in tests:
        try:
            ok = t()
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {t.__name__}")
            results.append(ok)
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            results.append(False)
    return all(results)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
