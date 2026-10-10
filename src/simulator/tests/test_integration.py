# simulator/tests/test_integration.py
"""
End-to-end integration test — requires real genome files.
Run only when invoked externally (not part of the api.test() self-check).
"""

import os
import tempfile

from ..contracts import SimulatorConfig
from ..api import simulate_dataset, validate_output


def run_integration_test(virus_fasta: str, microbe_dir: str):
    """
    Small-scale end-to-end test (50 pos + 50 neg).

    Parameters
    ----------
    virus_fasta : virus genome FASTA path
    microbe_dir : microbe genome directory path
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, 'sim_output')

        # -- PE mode --
        config_pe = SimulatorConfig(
            mode='paired_end',
            virus_fasta=virus_fasta,
            microbe_dir=microbe_dir,
            output_dir=os.path.join(out_dir, 'pe'),
            n_pos=50,
            n_neg=50,
            do_split=True,
            processes=1,
        )
        output_pe = simulate_dataset(config_pe)
        ok_pe = validate_output(output_pe)
        print(f" [{'PASS' if ok_pe else 'FAIL'}] PE mode integration")

        # -- Contig mode --
        config_ct = SimulatorConfig(
            mode='contig',
            virus_fasta=virus_fasta,
            microbe_dir=microbe_dir,
            output_dir=os.path.join(out_dir, 'contig'),
            n_pos=50,
            n_neg=50,
            do_split=True,
            processes=1,
        )
        output_ct = simulate_dataset(config_ct)
        ok_ct = validate_output(output_ct)
        print(f" [{'PASS' if ok_ct else 'FAIL'}] Contig mode integration")

        return ok_pe and ok_ct


if __name__ == '__main__':
    import sys
    import argparse

    p = argparse.ArgumentParser(description="Integration test (needs real genomes)")
    p.add_argument('--virus_fasta', required=True)
    p.add_argument('--microbe_dir', required=True)
    args = p.parse_args()

    ok = run_integration_test(args.virus_fasta, args.microbe_dir)
    sys.exit(0 if ok else 1)
