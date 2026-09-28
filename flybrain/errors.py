"""Exception hierarchy for FlyBrain.

The distinction that matters for v0: a :class:`ResourceLimitError` is raised when
the requested simulation cannot be safely executed on this machine. The CLI turns
it into a clean stop with an estimated-requirements report, and never degrades a
requested whole-brain run into a subset run.
"""

from __future__ import annotations


class FlyBrainError(Exception):
    """Base class for all FlyBrain errors."""


class DatasetError(FlyBrainError):
    """Dataset missing, unreadable, or failing validation."""


class DatasetValidationError(DatasetError):
    """The dataset loaded but violates an invariant we refuse to simulate on."""


class ResourceLimitError(FlyBrainError):
    """The requested simulation cannot be run safely on this machine.

    Carries a structured estimate so callers can report *why* rather than just
    failing.
    """

    def __init__(self, message: str, estimate: dict | None = None):
        super().__init__(message)
        self.estimate = estimate or {}


class UnknownNeuronError(FlyBrainError):
    """A FlyWire ID was requested that is not present in the loaded dataset."""


class UnknownPopulationError(FlyBrainError):
    """A population name was requested that is not in the population registry."""
