"""Contracts prepared for the next mechanistic stage; no biological result implied."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LocalTransition:
    features: tuple[float, ...]
    action: int
    reward: float
    next_features: tuple[float, ...]
    terminated: bool
    elapsed_ms: float


class EligibilityReadout(Protocol):
    """External actor-critic readout; never label its updates connectome plasticity."""

    def probabilities(self, features: tuple[float, ...]) -> tuple[float, ...]: ...
    def update(self, transition: LocalTransition) -> None: ...
    def clear_eligibility(self) -> None: ...
    def reset_weights(self) -> None: ...


@dataclass(frozen=True)
class PlasticityMask:
    """Original edge IDs, frozen dataset hash and independently owned deltas required."""

    dataset_sha256: str
    original_edge_ids: tuple[int, ...]
    selection_rationale: str
    preserve_sign: bool = True

    def __post_init__(self):
        if (
            len(self.dataset_sha256) != 64
            or not self.selection_rationale
            or any(i < 0 for i in self.original_edge_ids)
            or len(set(self.original_edge_ids)) != len(self.original_edge_ids)
        ):
            raise ValueError(
                "mask requires dataset identity, unique nonnegative edges and rationale"
            )
