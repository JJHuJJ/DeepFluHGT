# Model Identity Contract (Phase 3.6)

## What the current code checks
- `interim_eval.py` line ~241: GPU assertion (data-center / consumer-grade GPU classes)
- `load_model_gpu()` in `eval_real_anchors_gpu.py`: contract metadata read (encoder_frozen, backbone_sha256)
- Checkpoint `contract` dict: embedded at save time

## What is NOT yet automatic at inference startup
- No runtime `requires_grad == False` assertion on backbone
- No runtime hash comparison against expected 321/322/326 values
- No runtime adapter-namespace validation
- No optimizer-exclusion check

## Required additions (for clean-room Stage A)
1. After model load, assert `all(not p.requires_grad for n,p in model.named_parameters() if 'backbone.' in n)`
2. Recompute 321-scope hash and compare against `3fd27815dc3584f0...` prefix
3. Verify `encoder_frozen == True` from checkpoint contract
4. Verify `trainable_n_params == optimizer_n_params` from contract

## Status
**PARTIAL** — contract metadata is embedded and readable; runtime enforcement is documented but not automated in the evaluation entry point. The freeze audit script (`terminal_freeze_audit.py`) performs these checks when run separately.
