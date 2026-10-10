# schema.py
"""
DeepFluHGT  —  global data-structure contracts
=====================================
Enum mappings and constant definitions shared by all modules.
Holds only cross-module shared type definitions; module-internal data structures live in each contracts.py.
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════ #
#  PE Case mapping (simulator generate → model/evaluation makeuse)
# ═══════════════════════════════════════════════════════════════════════════════ #

# int → str mapping (model output → human-readable name)
PE_CASE_INV: dict[int, str] = {
    0: "R1_HOST_R2_VIRUS",
    1: "R1_HOST_R2_MIXED",
    2: "R1_VIRUS_R2_HOST",
    3: "R1_VIRUS_R2_MIXED",
    4: "R1_MIXED_R2_HOST",
    5: "R1_MIXED_R2_VIRUS",
    6: "R1_MIXED_R2_MIXED",
    7: "INTERNAL",        # virus in the gap between R1/R2, should be skipped
}

# str → int reverse mapping (for label encoding)
PE_CASE_MAP: dict[str, int] = {v: k for k, v in PE_CASE_INV.items()}

# 7 kinds of valid positive-sample PE cases (excluding INTERNAL)
VALID_PE_CASES: list[str] = [v for k, v in PE_CASE_INV.items() if k < 7]

# ═══════════════════════════════════════════════════════════════════════════════ #
#  HGT Mode mapping
# ═══════════════════════════════════════════════════════════════════════════════ #

HGT_MODE_INV: dict[int, str] = {
    0: "insertion",
    1: "replacement",
    2: "deletion_associated",
}

HGT_MODE_MAP: dict[str, int] = {v: k for k, v in HGT_MODE_INV.items()}

# ═══════════════════════════════════════════════════════════════════════════════ #
#  negative samplestype
# ═══════════════════════════════════════════════════════════════════════════════ #

NEG_TYPES: list[str] = ["NEG", "NEG_VIRUS", "NEG_HARD", "NEG_PCR"]

# ═══════════════════════════════════════════════════════════════════════════════ #
#  module list (used by the DeepFluHGT.py entry point)
# ═══════════════════════════════════════════════════════════════════════════════ #

MODULES: list[str] = [
    "simulator",
    "data_pipeline",
    "model",
    "loss",
    "training_engine",
    "evaluation",
    "utils",
]
