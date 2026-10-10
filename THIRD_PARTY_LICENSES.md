# Third-Party Software and Model Licenses

DeepFluHGT relies on third-party resources distributed under their respective
licenses. This project does not redistribute pretrained model weights; users
must obtain them from the upstream providers under the terms below.

## Pretrained Models

### NTv3-650M-pre
- **Provider:** InstaDeep
- **License:** InstaDeep Open Model License (non-commercial research use)
- **Source:** https://huggingface.co/InstaDeepAI/NTv3_650M_pre
- **Note:** DeepFluHGT does not redistribute these weights. Users must
  download from HuggingFace and comply with the upstream license terms.

### Evo2 7B
- **Provider:** Arc Institute
- **License:** See the upstream Evo2 repository for the applicable license
  terms governing model weights and inference code.
- **Source:** https://github.com/arcinstitute/evo2
- **Note:** DeepFluHGT does not redistribute these weights. Users must
  obtain them from the Arc Institute and comply with the upstream license.

## External Benchmark Data

The following benchmark datasets are used for evaluation only and are not
redistributed in this repository:

| Dataset | Source | License |
|---------|--------|---------|
| Benbow | Banerjee et al. (2024) | See original publication |
| IslandViewer4 | Bertelli et al. (2017) | https://www.pathogenomics.sfu.ca/islandviewer/ |
| ICEberg3 | Wang et al. (2024) | http://202.119.249.45/ICEberg3/ |
| Philympics 2021 | Roach et al. (2022) | https://github.com/linsalrob/ProphagePredictionComparisons |
| Genomic Benchmarks | Gresova et al. (2023) | https://github.com/instadeepai/genomic-benchmarks |
