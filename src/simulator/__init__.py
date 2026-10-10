# simulator/__init__.py
"""DeepFluHGT  —  Simulator module"""

from .api import simulate_dataset, initialize, validate_output, test
from .contracts import (
    SimulatorConfig,
    ProvenanceSegment,
    HGTEvent,
    SampleMetadata,
    SimulationOutput,
)

__all__ = [
    "simulate_dataset",
    "initialize",
    "validate_output",
    "test",
    "SimulatorConfig",
    "ProvenanceSegment",
    "HGTEvent",
    "SampleMetadata",
    "SimulationOutput",
]
