"""FlyBrain v0 - a scientifically honest, reproducible, executable Drosophila
whole-brain connectome simulation.

FlyBrain loads the FlyWire-derived preprocessed connectivity shipped with the
Shiu et al. (2024) Drosophila brain model, runs a leaky integrate-and-fire
network over it, accepts controlled external stimulation, supports silencing,
and records the resulting activity.

Scope boundaries for v0 (enforced by absence, not by documentation alone):
no LLM, no language, no reinforcement learning, no decoder, no consciousness claim,
no claim that simulated activity reproduces biological activity.
"""

from __future__ import annotations

from .config import PROJECT_ROOT, RunConfig, ensure_dirs

__all__ = ["PROJECT_ROOT", "RunConfig", "ensure_dirs", "__version__"]

__version__ = "0.0.1"
