# DeepFluHGT — Frozen-Representation Attribution Audit

Code release for: **"Separating prediction from attribution in DNA foundation models for horizontal gene transfer detection"** (Nature Computational Science submission).

## Overview

DeepFluHGT evaluates whether a frozen DNA foundation-model system's predictive performance can be attributed to pretrained representation information versus added model capacity. Two pretrained encoders (NTv3-650M-pre + Evo2 7B) are permanently frozen by code contract; only lightweight adapters (~26–28M parameters, <4.4% of backbone) are trained. A shuffled-embedding control (M3) separates capacity-associated from residual performance components.

## Repository Structure

```
code/
├── src/                       # Core training & model code
│   ├── DeepFluHGT.py          # Main entry point (train/eval CLI)
│   ├── model/                 # Model architecture
│   │   ├── api.py             # build_model(), forward pass
│   │   ├── contracts.py       # HGTConfig dataclass
│   │   ├── impl/
│   │   │   ├── backbone.py    # NTv3-pre encoder (frozen, 653M params)
│   │   │   ├── fusion.py      # EvoEncoder + dual-representation fusion
│   │   │   ├── bio_features.py# Biological feature encoder
│   │   │   ├── heads.py       # Classification & breakpoint heads
│   │   │   └── early_exit.py  # Early-exit threshold logic
│   │   └── tests/             # Unit tests (forward shape, gradient, overfit)
│   ├── data_pipeline/         # Data loading & windowing
│   │   ├── api.py             # load_dataset(), load_splits()
│   │   ├── contracts.py       # HGTSample, HGTBatch dataclasses
│   │   ├── impl/
│   │   │   ├── parser.py      # BED/FASTA parsers
│   │   │   ├── dataset.py     # SlidingWindowContigDataset
│   │   │   ├── collator.py    # HGTCollator (batch assembly)
│   │   │   ├── tokenizer.py   # NTv3 tokenization
│   │   │   └── bio_features.py# GC/length/composition extraction
│   │   └── tests/
│   └── training_engine/       # Training loop
│       ├── api.py             # train(), evaluate()
│       ├── contracts.py       # TrainConfig, freeze contracts
│       └── impl/
│           ├── optimizer.py   # Param-group LR, cosine schedule
│           ├── mixup.py       # Mixup augmentation
│           └── lora.py        # LoRA adapters (unused in final config)
├── scripts/                   # Evaluation & analysis scripts (79 .py)
│   ├── eval_real_anchors_gpu.py   # Four external anchors (Benbow/IV4/ICE/Philympics)
│   ├── cross_taxa_eval.py         # OOD per-genome/per-phylum evaluation
│   ├── difficulty_eval.py         # Difficulty ladder D1–D3
│   ├── interim_eval.py            # Terminal evaluation driver
│   ├── residual_corr.py           # Decision-layer residual correlation audit
│   ├── rap_probes.py              # Representation accessibility profiling
│   ├── shortcut_audit.py          # Two-layer containment/exploitation audit
│   ├── verdict_calculator.py      # §12 prespecified rule machine computation
│   ├── ...                           # Full pipeline (see scripts/run/ for launchers)
│   └── run/                          # GPU launcher scripts (38 .sh)
├── configs/                   # Training & evaluation configurations
│   ├── contig_500k.yaml    # M2 training config (Evo2 channel)
│   ├── contig_500k_m3.yaml # M3 shuffled-embedding control config
│   └── ...
├── environments/
│   ├── environment.yml        # Main training env (Python 3.10 + PyTorch 2.5.1 + CUDA 12.1)
│   └── evo2_environment.yml   # Evo2 offline embedding env (Python 3.11 + PyTorch 2.8.0)
└── docs/                      # Additional documentation
```

## Environment Note

DeepFluHGT was evaluated using **separate environments** for NTv3 and Evo2 dependencies. Users should avoid combining incompatible dependency stacks into a single environment.

## Prerequisites

### Main training environment (Python 3.10)

```bash
conda env create -f environments/environment.yml
conda activate thesis
```

Key dependencies:
| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.5.1 (CUDA 12.1) |
| transformers | 4.44.2 |
| scikit-learn | 1.7.2 |
| NumPy | 1.26.4 |
| matplotlib | 3.10.9 |

### Evo2 offline embedding environment (Python 3.11)

```bash
conda env create -f environments/evo2_environment.yml
conda activate evo2
```

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| flash-attn | 2.8.3 |
| evo2 | 0.6.0 |

## Pretrained Weights

DeepFluHGT does not redistribute pretrained model weights. Users should obtain NTv3-650M-pre from InstaDeep and Evo2 from Arc Institute according to their respective upstream licenses.

Both encoders are **publicly available** and must be downloaded separately (not redistributed in this repo):

| Model | Source | Size | License |
|-------|--------|------|---------|
| NTv3-650M-pre | [HuggingFace InstaDeepAI/NTv3_650M_pre](https://huggingface.co/InstaDeepAI/NTv3_650M_pre) | ~2.5 GB | See HF model card |
| Evo2 7B | [Arc Institute evo2](https://github.com/arcinstitute/evo2) | ~14 GB | See Arc Institute |

Place under `model/`:
```
model/InstaDeepAI/NTv3_650M_pre/
model/arcinstitute/evo2_7b/
```

### Evo2 embedding cache

Generated cache artifacts are not included in the initial release. They can be regenerated following the provided workflow (~13 GPU-days on 1x A100-80GB):
```bash
python scripts/precompute_evo2_embeddings.py --model evo2_7b --layer blocks.28.mlp.l3 \
    --data simulated_data/ --output evo_cache/
```

## Training

### M0 (NTv3-only baseline)
```bash
python src/DeepFluHGT.py train --config configs/contig_500k_ntv3only.yaml
```

### M2 (augmented representation: NTv3 + Evo2)
```bash
# Prerequisite: Evo2 embedding cache must exist
python scripts/precompute_evo2_embeddings.py --config configs/contig_500k.yaml
python src/DeepFluHGT.py train --config configs/contig_500k.yaml
```

### M3 (shuffled-embedding capacity control)
```bash
python scripts/build_shuffle_cache.py   # seed-42 key permutation
python src/DeepFluHGT.py train --config configs/contig_500k_m3.yaml
```

**Freeze contract**: Encoder weights are permanently frozen by unreachable-code assertion at startup and verified bit-identical across all epochs. Every checkpoint embeds contract metadata (backbone/config/dataset-manifest SHA-256 hashes, parameter counts, freeze status).

## Evaluation

```bash
# Four external anchors (zero-shot, Benbow/IV4/ICEberg3/Philympics)
python scripts/eval_real_anchors_gpu.py --anchor benbow --gpu <UUID>

# OOD cross-taxa (per-genome/per-phylum)
python scripts/cross_taxa_eval.py

# Difficulty ladder D1-D3
python scripts/difficulty_eval.py --tier D1

# Decision-layer residual audit (containment vs exploitation)
python scripts/residual_corr.py

# Full terminal evaluation chain
bash scripts/run/terminal_eval_chain.sh
```

## Verification Gates

The codebase includes machine-verified audit gates:

```bash
# Quick verification (compile + imports + 35 simulator tests)
bash quick_verify.sh

# §12 prespecified decision rules (machine-computed verdicts)
python scripts/verdict_calculator.py
```

## Hardware Requirements

| Task | GPU | VRAM | Time |
|------|-----|------|------|
| M0 training (12 epochs) | 1× RTX 4090 | ~20 GB | ~5 days |
| M2/M3 training (12 epochs) | 1× A100-80GB | ~60 GB | ~9 days each |
| Evo2 cache generation | 1× A100-80GB | ~70 GB | ~13 days |
| Terminal evaluation (4 anchors + test + OOD + D1-D3) | 1× A100-80GB | ~30 GB | ~12 h |
| Residual audit | CPU only | — | ~4 min |

## Simulated Benchmark Data

The genome-isolated simulated benchmark (Contig_500k: 400k/50k/50k train/val/test contigs, 9,242 training genomes) is archived separately: `[DOI: pending — Zenodo record will be created upon formal publication]`.

## License

The DeepFluHGT source code is released under the [MIT License](LICENSE).

The accompanying simulated benchmark datasets are released under the
Creative Commons Attribution 4.0 International License (CC BY 4.0).

Third-party pretrained models are subject to their respective upstream
licenses. DeepFluHGT does not redistribute pretrained model weights; users
must obtain and use pretrained models under their respective upstream
licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

## Citation

If you use this code, please cite:
```
Hu, J., Hu, S., Lu, S., Zhuang, S., Xu, W., Xiong, J., Feng, J., Chen, Z.,
Zhang, Y., Xiong, C. & Wang, W. Separating prediction from attribution in
DNA foundation models for horizontal gene transfer detection.
Nature Computational Science (2026).
```
[DOI to be added upon publication]

## Contact

Weibing Wang (wwb@fudan.edu.cn)
School of Public Health, Fudan University, Shanghai, China


## Reproducibility entry points
- `scripts/paired_bootstrap.py` — reruns the published Philympics paired-event CI from the two shipped inputs (`docs/evidence/real_anchor_philympics_m0final.npz`, `benchmark/philympics/anchor_data/matching_stats.json`).
- `scripts/homology_repertoire_analyze.py` — recomputes the junction-perturbation statistics from per-arm scores (evidence deposit).
