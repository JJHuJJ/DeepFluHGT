# Reproducibility Guide

## Full Pipeline Reproduction

```bash
# 1. Setup environments
conda env create -f environments/environment.yml
conda env create -f environments/evo2_environment.yml

# 2. Download pretrained weights (see model_weights/README.md)

# 3. Download simulated benchmark (Zenodo, see model_weights/README.md)

# 4. Generate Evo2 embedding cache
conda activate evo2
python scripts/precompute_evo2_embeddings.py --config configs/contig_500k.yaml

# 5. Generate shuffle cache (M3 control)
conda activate thesis
python scripts/build_shuffle_cache.py

# 6. Train M0, M2, M3
python src/DeepFluHGT.py train --config configs/contig_500k_ntv3only.yaml
python src/DeepFluHGT.py train --config configs/contig_500k.yaml
python src/DeepFluHGT.py train --config configs/contig_500k_m3.yaml

# 7. Run terminal evaluation chain
bash scripts/run/terminal_eval_chain.sh

# 8. Run verification gates
python scripts/final_numeric_consistency_check.py   # must PASS 17/17
python scripts/figure_forensic.py                 # must FAIL=0
python scripts/claim_strength_invariance.py          # must PASS
python scripts/final_result_provenance_gate.py       # must ALLOW

# 9. Render figures
python scripts/render_figures.py
python scripts/render_supp_figs.py
```

## Deterministic Reproduction Notes

- Training: fixed seeds (42) in all configs; GPU nondeterminism may cause ±0.001 AUPRC variation across hardware
- Evaluation: all evaluation scripts are deterministic (CPU aggregation, fixed window order)
- Evo2 embeddings: deterministic given same model weights + same data
- Cross-hardware note: M3 terminal re-eval on RTX 4090 vs A100 produced ΔAUPRC ≈ 0.005 (documented in TERMINAL_CANONICAL.json)

## Freeze Contract Verification

The encoder freeze is enforced by:
1. **Unreachable-code assertion** at startup (`training_engine/contracts.py`)
2. **Bit-identical hash check** across epochs (per-epoch hash records; summary checksums in model_weights/CHECKSUMS_SHA256.txt)
3. **Per-checkpoint metadata** embedding backbone/config/dataset SHA-256

## Expected Wall-Clock Times

See code/README.md § Hardware Requirements.
