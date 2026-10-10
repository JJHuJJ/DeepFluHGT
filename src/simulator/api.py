# simulator/api.py
"""
DeepFluHGT — Simulator module public interface
=========================================
All other modules interact with the simulator through this file.
Provides four standard interfaces: initialize, simulate_dataset, validate_output, test
"""

from __future__ import annotations

import os

from .contracts import SimulatorConfig, SimulationOutput
from .impl.dataset_generator import DatasetGenerator
from .impl.dataset_splitter import DatasetSplitter


def initialize(config: SimulatorConfig) -> None:
    """
    Initialization check: validate the configuration and ensure input paths exist.
    Call before simulate_dataset.
    """
    config.validate()

    if not os.path.isfile(config.virus_fasta):
        raise FileNotFoundError(f"Virus FASTA not found: {config.virus_fasta}")
    if not os.path.isdir(config.microbe_dir):
        raise NotADirectoryError(
            f"Microbe genome directory not found: {config.microbe_dir}")

    os.makedirs(config.output_dir, exist_ok=True)
    print("[simulator.initialize] Configuration validated, paths verified.")


def simulate_dataset(config: SimulatorConfig) -> SimulationOutput:
    """
    Main entry point for generating a simulated dataset.

    Parameters
    ----------
    config : SimulatorConfig
        Complete simulator configuration (must pass initialize validation first).

    Returns
    -------
    SimulationOutput
        Contains all output file paths and summary statistics.
    """
    initialize(config)

    generator = DatasetGenerator(config)
    output = generator.generate()

    # Automatic splitting (in genome_isolation mode, per-split directories are already written at generation time)
    if config.do_split and not config.genome_isolation:
        splitter = DatasetSplitter(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            seed=config.seed,
        )
        split_dir = os.path.join(config.output_dir, 'split')
        splitter.split(config.output_dir, split_dir)
        output.split_dir = split_dir

    return output


def _check_toplevel(output: SimulationOutput, checks: list):
    """Check the top-level output files in non-isolation mode."""
    bed_ok = os.path.isfile(output.breakpoints_bed) and \
             os.path.getsize(output.breakpoints_bed) > 0
    checks.append(("breakpoints.bed", bed_ok, output.breakpoints_bed))

    prov_ok = os.path.isfile(output.provenance_tsv) and \
              os.path.getsize(output.provenance_tsv) > 0
    checks.append(("provenance.tsv", prov_ok, output.provenance_tsv))

    meta_ok = os.path.isfile(output.metadata_json) and \
              os.path.getsize(output.metadata_json) > 0
    checks.append(("metadata.json", meta_ok, output.metadata_json))


def validate_output(output: SimulationOutput) -> bool:
    """
    Validate the completeness of the output files.

    Returns
    -------
    bool
        True when all files exist and are non-empty.
    """
    checks = []

    # FASTA files
    for name, path in output.fasta_files.items():
        exists = os.path.isfile(path) and os.path.getsize(path) > 0
        checks.append((f"FASTA[{name}]", exists, path))

    # genome_isolation mode: check the completeness of each split subdirectory
    if output.split_dir and os.path.isdir(output.split_dir):
        # Distinguish genome_isolation (train/val/test as direct subdirectories) from a plain split (split/ subdirectory)
        has_train_dir = os.path.isdir(
            os.path.join(output.split_dir, 'train'))
        has_prov_in_train = has_train_dir and os.path.isfile(
            os.path.join(output.split_dir, 'train', 'provenance.tsv'))
        if has_prov_in_train:
            # genome_isolation mode: every split has the complete file set
            for split_name in ['train', 'val', 'test']:
                split_path = os.path.join(output.split_dir, split_name)
                if not os.path.isdir(split_path):
                    continue
                for fname in ['breakpoints.bed', 'provenance.tsv', 'metadata.json']:
                    fpath = os.path.join(split_path, fname)
                    ok = os.path.isfile(fpath) and os.path.getsize(fpath) > 0
                    checks.append((f"{split_name}/{fname}", ok, fpath))
        else:
            # Plain split mode (DatasetSplitter): only check the top-level files
            _check_toplevel(output, checks)
    else:
        _check_toplevel(output, checks)

    all_ok = all(ok for _, ok, _ in checks)
    for name, ok, path in checks:
        status = "OK" if ok else "MISSING"
        print(f" [{status}] {name}: {path}")

    return all_ok


def test() -> bool:
    """
    Run the module self-tests (does not require real genome files).
    Returns True when all tests pass.
    """
    from .tests.test_contracts import run as test_contracts
    from .tests.test_sequence_generator import run as test_seqgen
    from .tests.test_pe_classification import run as test_pe
    from .tests.test_negative import run as test_neg
    from .tests.test_provenance import run as test_prov

    results = {
        "contracts": test_contracts(),
        "sequence_generator": test_seqgen(),
        "pe_classification": test_pe(),
        "negative_samples": test_neg(),
        "provenance": test_prov(),
    }

    print("\n" + "=" * 50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f" [{status}] {name}")
    all_ok = all(results.values())
    print(f"\nResult: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    return all_ok
