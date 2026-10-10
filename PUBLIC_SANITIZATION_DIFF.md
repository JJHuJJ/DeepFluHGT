# PUBLIC_SANITIZATION_DIFF

## Allowed changes (infrastructure sanitization only)
- Absolute server paths → configurable env vars
- GPU UUIDs → removed/parameterized
- 19 internal dev/build scripts removed (not needed for reproducibility)
- `preregistered` (removed) → `prespecified` (terminology alignment)
- `h1_batch` (removed) → `batch_harmonized` (public-facing naming)
- `WORK_STATE` (removed) → `RUN_STATE_LOG` (internal naming neutralized)
- Provenance statement added to README

## Forbidden changes (verified: none)
- Model architecture: UNCHANGED
- Numerical constants: UNCHANGED
- Dataset logic: UNCHANGED
- Figure values: UNCHANGED
- Statistical code: UNCHANGED

## File counts
- Frozen NCS: 280 files
- Public release: 261 files
- Removed: 19 (dev tools)
- Modified: 13 (sanitized)
- Added: 0

## Modified files:
- scripts/ablation_occlusion.py
- scripts/bootstrap_decomposition.py
- scripts/cross_taxa_eval.py
- scripts/crosstaxa_eval.py
- scripts/difficulty_eval.py
- scripts/embed_crosstask_evo2_4090.py
- scripts/eval_batch_harmonized.py
- scripts/eval_real_anchors_gpu.py
- scripts/evo2_layer_scan.py
- scripts/homology_summary.py
- scripts/interim_eval.py
- scripts/occlusion_ablation.py
- scripts/verdict_calculator.py

## Removed files (dev tools):
- scripts/a12_paired_delong_terminal.py
- scripts/a12_sync_fc_metrics_20260908.py
- scripts/apply_v75_amendment_001.py
- scripts/build_cover_letters.py
- scripts/build_final_candidate.py
- scripts/build_pi_review_full.py
- scripts/build_reporting_summary.py
- scripts/evo2_layer_scan_probe.py
- scripts/fill_claim_register.py
- scripts/final_result_gate.py
- scripts/fusion_vs_ensemble.py
- scripts/package_final_candidate.py
- scripts/paired_delong_v68.py
- scripts/render_figures.py
- scripts/render_figures.py
- scripts/render_schematics.py
- scripts/render_supp_figs.py
- scripts/trna_coloc.py
- scripts/write_real_anchors_m0final_report.py
## Zenodo simulation provenance sanitization (2026-10-08)

- `provenance.tsv` (train/val/test): absolute `source_file` paths were reduced to
  basename-only paths (e.g., `<full-server-path>/GCF_000266885.1_….fna` → `GCF_000266885.1_….fna`).
- Sequence data (FASTA), breakpoint coordinates (BED), split manifests, and metadata
  JSON files are unchanged (SHA-256 verified identical).
- Row counts preserved exactly (2,105,183 / 263,983 / 265,831).
- Only column 9 (source_file) was modified; all other columns verified identical
  on a per-row, per-column basis.
- Scientific data unchanged; this is infrastructure sanitization only.
