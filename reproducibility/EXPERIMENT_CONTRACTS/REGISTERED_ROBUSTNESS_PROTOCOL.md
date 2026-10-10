# Registered Robustness Protocol (revision-stage experiments)
**Registration date**: 2026-09-21 · **Source**: THREE_SOURCE_VERIFICATION_EXECUTION_PLAN_20260921 (D2 authorized execution) 
**Nature**: fully preregistered protocol — after registration in this file and before the results are known, all design parameters and success criteria are locked. This file must not be modified before the results are produced.

---


| Item | Locked value |
|---|---|
| Permutation set | π₁ = the already-executed seed-42 key permutation (canonical M3); π₂–π₅ = four new fixed permutation seeds 52/53/54/55 |
| Construction | `scripts/build_shuffle_cache.py` method (dataset-level cache swap; the model itself unchanged) |
| Training | one M3 run per permutation: fully identical to the canonical M3 (cap 12 / patience 4 / min_delta 0.001 / frozen contract) |
| Interpretation quantity | M3(πᵢ)−M0 distribution and M2−M3(πᵢ) distribution (validation protocol; per-genome protocol reported in parallel) |
| Success criterion | for every πᵢ both score differences keep the canonical direction (M3(πᵢ)>M0 and M2>M3(πᵢ)); the share band stays above 54.0% |
| Priority | below X2 (M2−M3 already has 9/9 epoch alignment support; less fragile) |


| Item | Locked value |
|---|---|
| **Layer 1 (already started 2026-09-21)** | **2xM0: training.seed 43 (consumer-grade GPU) / 44 (data-center GPU)**; everything else verbatim identical to the production M0 (configs/m0_seed{43,44}_20260921.yaml) |
| Variable isolation | only the adapter-init RNG + data order + RC augmentation sampling vary with the seed; data partition, dataset manifest, π, and the frozen contract all unchanged |
| Interpretation quantity | each replicate's best val AUPRC (min_delta=0.001 selection rule, consistent with production) |
| Success criterion | both replicate bests ∈ [0.82, 0.84] → M0=0.8325 is not peak-value sampling and the 54.0% share is robust (SI adds the replicate records); spread down to ~0.81 → keep the "substantial" wording, front-loaded with the existing 23–98% envelope (wording changes, numbers do not) |
| Layer 2 (review period) | 2 additional seeds x (M2+M3) (~40.4 GPU days), reporting M2−M3 and M3−M0 paired |
| Layer 3 | full n=5 not executed (~100+ GPU days, infeasible) |

## X3 — simulator artifact ablation

| Item | Locked value |
|---|---|
| Arm | A = the original generator (canonical) · B = the generator with terminal transfer removed (microhomology parameters per the §2.6 audit semantic correction) · C = keep the global composition, remove the local insertion-edge signature |
| Evaluation target | M0 / M2 / M3 / composition ceiling (each arm's dataset rebuilt under the same production protocol; genome isolation same rule) |
| Interpretation quantity | whether M2−M0 and M2−M3 persist in every arm |
| Success criterion | M0 retains strong signal and the M2>M3 pattern persists → the result does not depend on a single generator's construction artifact (release condition for the MS §2.6 / SI T11 boundary sentence) |
| Relation to SI T11 | the T11 boundary sentence explicitly says "deferred to a registered artifact-ablation experiment" = this protocol's X3 |


1. **Reporting discipline**: every X-item result is reported regardless of direction (continuing the negative-result governance).
2. **Protocol**: all comparisons follow the engine-mixed protocol + genome-clustered bootstrap CI; no new protocol is introduced.
3. **Gating**: any X-item result entering the manuscript must pass the single-injection window + the full gating re-run (the two-before/one-after architecture unchanged).
4. **No cherry-picking**: the replicate/permutation/arm choices are locked as above; additions or deletions based on results are not allowed.

- X2 Layer 1: **RUNNING** (seed43 PID <PID-anonymized> @consumer-grade GPU · seed44 PID <PID-anonymized> @data-center GPU · start 2026-09-21 20:31 · log logs/M0_seed4{3,4}_replicate.log) 
- X1 / X3 / X2 Layer 2: pending the review period
