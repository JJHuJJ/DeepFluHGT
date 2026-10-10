# simulator/contracts.py
"""
DeepFluHGT — Simulator module data structures
==========================================
Defines all input/output data structures of the simulator module.
All other modules interact with the simulator exclusively through these structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimulatorConfig:
    """Complete simulator configuration — controls all parameters of data generation."""

    # -- Required parameters --
    mode: str = "paired_end" # "paired_end" | "contig"
    virus_fasta: str = "" # Virus genome FASTA path
    microbe_dir: str = "" # Microbe genome directory
    output_dir: str = "./simulated_data" # Output directory

    # -- Random seed --
    seed: int = 42

    # -- Paired-end (PE) mode parameters --
    read_len: int = 150 # Read length for single-end reads (bp)

    # -- Contig mode parameters --
    contig_min_length: int = 300 # Shortest contig (bp)
    contig_max_length: int = 100000 # Longest contig (bp)

    # -- HGT event parameters --
    insertion_prob: float = 0.6 # Insertion event probability
    replacement_prob: float = 0.3 # Replacement event probability
    deletion_prob: float = 0.1 # Deletion-associated event probability
    multi_event_enabled: bool = True # Enable multi-event HGT
    multi_event_prob: float = 0.2 # Multi-event trigger probability
    multi_event_max: int = 3 # Maximum number of events
    multi_event_min_gap: int = 20 # Minimum spacing between events (bp)
    sandwich_enabled: bool = True # Enable sandwich mode
    sandwich_prob: float = 0.15 # Sandwich trigger probability
    sandwich_min_gap: int = 5 # Minimum spacing between the two sandwich events (bp)

    # -- Virus fragment parameters --
    virus_fragment_min: int = 20 # Shortest virus fragment (bp)
    virus_fragment_max: int = 200 # Longest virus fragment (bp)
    virus_fragments_per_genome: int = 500 # Number of fragments generated per genome

    # -- Multi-source host sampling parameters --
    multi_source_host_enabled: bool = True # Enable multi-source host sampling
    multi_source_host_prob: float = 0.2 # Multi-source host sampling probability
    multi_source_host_max: int = 3 # Maximum number of sources

    # -- Sample counts --
    n_pos: int = 10000 # Number of positive samples
    n_neg: int = 10000 # Number of negative samples

    # -- Generation parameters --
    error_rate: float = 0.005 # Sequencing error rate
    gc_bias: bool = False # Enable GC bias filtering
    use_coverage: bool = False # Enable abundance-weighted sampling
    processes: int = 1 # Number of parallel worker processes

    # -- Data splitting --
    do_split: bool = True # Whether to automatically split into train/val/test
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # -- Stratified/distribution configuration (passed in from YAML; None = use built-in defaults) --
    virus_frag_tiers_pe: Optional[list] = None # [[min,max,weight], ...]
    virus_frag_tiers_contig: Optional[list] = None
    pe_case_ratios: Optional[dict] = None # {case_name: ratio}
    neg_type_ratios: Optional[dict] = None # {type_name: ratio}
    contig_hgt_mode_ranges: Optional[dict] = None # {mode: [lo, hi]}

    # -- Sequencing platform error model --
    seq_platform: str = "illumina" # "illumina" | "nanopore" | "pacbio"

    # -- Abundance (coverage) distribution augmentation --
    coverage_distribution: str = "lognormal" # "lognormal" | "zipf"
    coverage_zipf_alpha: float = 1.0 # Zipf exponent
    batch_effect_sigma: float = 0.0 # Batch effect strength (0 = disabled)

    # -- Contamination simulation --
    contamination_rate: float = 0.0 # Contamination fraction (0 = disabled, typically 0.01–0.1)
    contaminant_dir: str = "" # Contaminant source genome directory (empty = sampled from the microbe pool)

    # -- Assembly artifacts (contig mode only) --
    assembly_artifact_rate: float = 0.0 # Probability of applying an artifact to each contig

    # -- Label noise --
    label_noise: float = 0.0 # Label flip fraction (0 = disabled)
    boundary_shift_max: int = 0 # Maximum HGT boundary shift (bp, 0 = disabled)

    # -- Evolutionary divergence --
    evolutionary_distance: float = 0.0 # Viral evolutionary distance (substitutions/site, 0 = disabled)
    host_mutation_rate: float = 0.0 # Host population diversity (substitutions/site, 0 = disabled, typical 0.001-0.005)
    # Stratified evolutionary distance: when not None, overrides evolutionary_distance
    # Format: [[lo, hi, weight], ...] where lo/hi = substitutions/site and weight = sampling weight
    # example: [[0.0, 0.01, 0.4], [0.01, 0.05, 0.35], [0.05, 0.15, 0.25]]
    # Interpretation: recent HGT 0-1%, intermediate 1-5%, ancient 5-15%
    evolutionary_distance_tiers: Optional[list] = None
    gc_amelioration: bool = False # Enable GC amelioration (GC of ancient HGT fragments drifts toward the host)
    # Sampling distribution for the evolutionary distance: "uniform" | "gamma"
    # Within each tier the gamma distribution yields a right-skewed sample (more close-range, fewer long-range events), better matching HGT age distributions
    evo_distance_distribution: str = "uniform"
    evo_distance_gamma_shape: float = 2.0 # Gamma distribution shape parameter (k>1 = left-skewed, k<1 = right-skewed)

    # -- Microhomology --
    microhomology_len: int = 0 # Microhomology length at HGT junctions (0 = disabled, typical 1-5)

    # -- Open set / OOD --
    open_set_virus: bool = False # Test set uses virus strains unseen during training

    # -- Genome isolation (prevents data leakage) --
    # C-02 fix: default True; train/val/test are split by genome file so a genome never leaks across splits
    genome_isolation: bool = True
    genome_isolation_microbe_ratios: Optional[list] = None # [train, val, test] default [0.8, 0.1, 0.1]
    genome_isolation_virus_ratios: Optional[list] = None # [train, val, test] default [0.7, 0.15, 0.15]

    def validate(self) -> None:
        """Validate the configuration parameters."""
        assert self.mode in ("paired_end", "contig"), \
            f"mode must be 'paired_end' or 'contig', got '{self.mode}'"
        assert self.n_pos >= 0, f"n_pos must be >= 0, got {self.n_pos}"
        assert self.n_neg >= 0, f"n_neg must be >= 0, got {self.n_neg}"
        probs = self.insertion_prob + self.replacement_prob + self.deletion_prob
        assert abs(probs - 1.0) < 1e-6, \
            f"HGT event probs must sum to 1.0, got {probs}"
        assert self.read_len > 0, f"read_len must be > 0, got {self.read_len}"
        assert self.contig_min_length > 0, f"contig_min_length must be > 0"
        assert self.contig_max_length >= self.contig_min_length, \
            f"contig_max_length must be >= contig_min_length"
        if self.do_split:
            ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
            assert abs(ratio_sum - 1.0) < 1e-6, \
                f"split ratios must sum to 1.0, got {ratio_sum}"
        # Validation of newly added parameters
        assert self.seq_platform in ("illumina", "nanopore", "pacbio"), \
            f"seq_platform must be 'illumina'|'nanopore'|'pacbio', got '{self.seq_platform}'"
        assert self.coverage_distribution in ("lognormal", "zipf"), \
            f"coverage_distribution must be 'lognormal'|'zipf', got '{self.coverage_distribution}'"
        assert 0.0 <= self.contamination_rate <= 1.0, \
            f"contamination_rate must be in [0, 1], got {self.contamination_rate}"
        assert 0.0 <= self.assembly_artifact_rate <= 1.0, \
            f"assembly_artifact_rate must be in [0, 1], got {self.assembly_artifact_rate}"
        assert 0.0 <= self.label_noise < 0.5, \
            f"label_noise must be in [0, 0.5), got {self.label_noise}"
        assert self.boundary_shift_max >= 0, \
            f"boundary_shift_max must be >= 0, got {self.boundary_shift_max}"
        assert self.evolutionary_distance >= 0, \
            f"evolutionary_distance must be >= 0, got {self.evolutionary_distance}"
        assert self.host_mutation_rate >= 0, \
            f"host_mutation_rate must be >= 0, got {self.host_mutation_rate}"
        if self.evolutionary_distance_tiers is not None:
            assert isinstance(self.evolutionary_distance_tiers, list), \
                "evolutionary_distance_tiers must be a list"
            assert len(self.evolutionary_distance_tiers) > 0, \
                "evolutionary_distance_tiers must not be empty"
            for tier in self.evolutionary_distance_tiers:
                assert len(tier) == 3, \
                    f"each tier must be [lo, hi, weight], got {tier}"
                assert 0.0 <= tier[0] < tier[1], \
                    f"tier lo must be < hi, got [{tier[0]}, {tier[1]})]"
                assert tier[2] >= 0, \
                    f"tier weight must be >= 0, got {tier[2]}"
            total_w = sum(t[2] for t in self.evolutionary_distance_tiers)
            assert total_w > 0, \
                f"tier weights must sum to > 0, got {total_w}"
        assert self.evo_distance_distribution in ("uniform", "gamma"), \
            f"evo_distance_distribution must be 'uniform'|'gamma', got '{self.evo_distance_distribution}'"
        assert self.evo_distance_gamma_shape > 0, \
            f"evo_distance_gamma_shape must be > 0, got {self.evo_distance_gamma_shape}"
        assert self.microhomology_len >= 0, \
            f"microhomology_len must be >= 0, got {self.microhomology_len}"


@dataclass
class ProvenanceSegment:
    """A single provenance segment traced back to its source."""
    seg_type: str # HOST / VIRUS
    source_type: str # microbe / virus
    source_file: str # source FASTA filepath
    source_seq_id: str # sourcesequence ID
    genome_start: int # Start in the source genome (0-based)
    genome_end: int # End in the source genome (half-open)
    strand: str # + / -
    insert_start: int # Start position within the insert
    insert_end: int # End position within the insert
    read_start: int = -1 # Start position within the read (-1 = not present in this read)
    read_end: int = -1 # End position within the read


@dataclass
class HGTEvent:
    """Description of a single HGT event."""
    hgt_start: int # Start in the final sequence (0-based)
    hgt_end: int # End in the final sequence (half-open)
    virus_len: int # Virus fragment length
    hgt_mode: str # insertion / replacement / deletion_associated


@dataclass
class SampleMetadata:
    """Complete metadata for a single sample."""
    seq_id: str # sequence ID
    label: int # 0 (negative) / 1 (positive)
    n_events: int # Number of HGT events
    hgt_modes: List[str] # Mode of each event
    pe_case: str = "" # PE case type (PE mode only)
    events: List[HGTEvent] = field(default_factory=list)
    insert_len: int = 0 # insert length (PE mode only)
    seq_len: int = 0 # Sequence length (contig mode only)
    r1_vs: int = -1 # Start of the virus region in R1
    r1_ve: int = -1 # End of the virus region in R1
    r2_vs: int = -1 # Start of the virus region in R2
    r2_ve: int = -1 # End of the virus region in R2
    difficulty: str = "" # "easy" | "medium" | "hard" | "extreme"
    label_confidence: float = 1.0 # soft label ∈ [0, 1]
    artifact_type: str = "" # assembly artifact type (contig mode)


@dataclass
class SimulationOutput:
    """Simulator output."""
    fasta_files: Dict[str, str] = field(default_factory=dict) # {name: path}
    breakpoints_bed: str = ""
    provenance_tsv: str = ""
    metadata_json: str = ""
    split_dir: str = "" # Split output directory (when do_split=True)
    sample_count: int = 0
    pos_count: int = 0
    neg_count: int = 0
