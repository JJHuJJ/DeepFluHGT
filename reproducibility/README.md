# DeepFluHGT — Reproducibility Package

This folder contains the evaluation-governance artifacts referenced by the manuscript:

| File | Contents |
|---|---|
| `CLAIM_REGISTER.yaml` | All 19 row-level evidence claims (18 SUPPORTED, 1 PARTIAL): text, manuscript location, adjudicated status, evidence type, artifact pointer, specification status |
| `EXPERIMENT_CONTRACTS/REGISTERED_ROBUSTNESS_PROTOCOL.md` | Robustness experiments (X1/X2/X3) registered before execution: designs, success criteria, no-cherry-pick rules |
| `EXPERIMENT_CONTRACTS/MODEL_IDENTITY_CONTRACT.md` | Frozen-backbone identity contract: what the code checks, runtime enforcement points |
| `EXPERIMENT_CONTRACTS/HASH_SPEC_FINAL.md` | Tensor-hash specification (scope census 321/322/326) used for bit-identity verification |
| `HASH_MANIFEST.txt` | Canonical identity values: backbone tensor-hash prefixes, parameter counts, checkpoint inventory |
| `DATA_PROVENANCE.md` | Dataset construction and split provenance (simulation benchmark + external anchors) |
| `VERSION_INFO.txt` | Software/hardware environment summary |

Verification entry points in the parent code tree:
- `scripts/interim_eval.py` — canonical terminal evaluation wrapper
- freeze/identity audit per MODEL_IDENTITY_CONTRACT (checkpoint-embedded contract metadata)

License: MIT (see ../LICENSE).
