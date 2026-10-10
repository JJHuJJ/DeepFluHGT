# Hash Specification FINAL（Phase 3.6 · 2026-09-23）

## Algorithm
1. **Tensor enumeration**: `sorted(keys)` — lexicographic by tensor name
2. **Per-tensor serialization**:
   - key name → UTF-8 bytes
   - dtype string → UTF-8 bytes
   - tensor → `.detach().cpu().contiguous().numpy().tobytes()`
3. **Concatenation**: sequential SHA-256 update across all tensors in sorted order
4. **Output**: first 16 hex characters of final digest

## Scope Definitions

| Label | Count | Definition | Included | Excluded |
|---|---|---|---|---|
| **321** | 321 | Pretrained NTv3 tensors | All `_nt_model.*` weights and biases | `_proj.*`, `_remap_table`, adapter, heads |
| **322** | 322 | Frozen pretrained scope | 321 + `backbone.global_encoder._remap_table` (code-defined buffer, not pretrained) | `_proj.*`, adapter, heads |
| **326** | 326 | Full frozen backbone namespace | 322 + `backbone.global_encoder._proj.0.{weight,bias}` (LayerNorm, deterministic init) + `backbone.global_encoder._proj.1.{weight,bias}` (Linear, frozen random draw) | adapter, heads, optimizer |

## Expected Hashes (M3 terminal checkpoint)

| Scope | Hash (first 16) |
|---|---|
| 321 (pretrained) | `c8297b1ee6723e55` |
| 322 (frozen pretrained) | (run `terminal_freeze_audit.py`) |
| 326 (full frozen backbone) | (run `terminal_freeze_audit.py`) |

Note: M0's 326-scope hash differs from M2/M3's (projection draw difference — SI S4.4). M0's 321-scope hash is identical.

## Reproducibility
Any reviewer with PyTorch and the checkpoint can recompute these hashes using:
```python
sd = torch.load(path, map_location="cpu")["model"]
keys = sorted(k for k in sd if "_nt_model" in k and "_proj" not in k)
h = hashlib.sha256()
for k in keys:
    h.update(k.encode()); h.update(str(sd[k].dtype).encode())
    h.update(sd[k].detach().numpy().tobytes())
print(h.hexdigest()[:16])
```
