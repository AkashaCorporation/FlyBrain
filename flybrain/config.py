"""Project paths and run configuration for FlyBrain v0.

Nothing in this module touches the filesystem at import time beyond resolving
paths; directory creation is explicit via :func:`ensure_dirs`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
THIRD_PARTY_DIR: Path = PROJECT_ROOT / "third_party"
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
METADATA_DIR: Path = DATA_DIR / "metadata"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
RUNS_DIR: Path = OUTPUTS_DIR / "runs"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

#: Environment overrides. `FLYBRAIN_DATA_DIR` exists so that a dataset staged on a
#: different volume can be used without editing code or copying 180 MB around.
if os.environ.get("FLYBRAIN_DATA_DIR"):
    DATA_DIR = Path(os.environ["FLYBRAIN_DATA_DIR"]).expanduser().resolve()
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    METADATA_DIR = DATA_DIR / "metadata"
if os.environ.get("FLYBRAIN_OUTPUTS_DIR"):
    OUTPUTS_DIR = Path(os.environ["FLYBRAIN_OUTPUTS_DIR"]).expanduser().resolve()
    RUNS_DIR = OUTPUTS_DIR / "runs"

_ALL_DIRS = (DATA_DIR, RAW_DIR, PROCESSED_DIR, METADATA_DIR, OUTPUTS_DIR, RUNS_DIR)


def ensure_dirs() -> None:
    """Create the data/output directory tree if it does not exist."""
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# Simulation defaults (verified against upstream - see docs/MODEL_ASSUMPTIONS.md)
# --------------------------------------------------------------------------------------

#: Integration step used by the upstream protocol. Brian 2's default `defaultclock.dt`
#: is 0.1 ms and the upstream model never overrides it, so 0.1 ms is the reference dt.
DEFAULT_DT_MS: float = 0.1

#: Default trial duration (upstream `t_run`).
DEFAULT_DURATION_MS: float = 1000.0

#: The model's free parameters are calibrated per-trial; upstream runs 30 stochastic
#: trials and analyses the mean. FlyBrain runs a single trial by default (determinism
#: and CPU cost) and exposes `--trials` for multi-trial means.
DEFAULT_TRIALS: int = 1


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to reproduce one experimental run.

    This object is serialised verbatim into ``outputs/runs/<run-id>/config.json``.
    """

    # -- what to simulate ---------------------------------------------------------
    dataset_id: str = "flywire_630"
    mode: str = "subset"                 # "subset" | "whole-brain"
    duration_ms: float = DEFAULT_DURATION_MS
    dt_ms: float = DEFAULT_DT_MS
    trials: int = DEFAULT_TRIALS

    # -- determinism --------------------------------------------------------------
    seed: int = 0

    # -- computation --------------------------------------------------------------
    backend: str = "auto"                # "auto" | "numpy" | "jax"
    dtype: str = "float32"
    allow_fallback: bool = False         # whole-brain refusal -> subset? default: refuse
    max_estimated_minutes: Optional[float] = 60.0   # None disables the runtime gate

    # -- network perturbations ----------------------------------------------------
    silence: List[int] = field(default_factory=list)     # FlyWire IDs to silence
    silence_populations: List[str] = field(default_factory=list)
    disable_recurrence: bool = False      # break all recurrent input (baseline control)

    # -- subset construction ------------------------------------------------------
    subset_seed_ids: List[int] = field(default_factory=list)
    subset_hops: int = 2
    subset_max_neurons: int = 5000

    # -- recording ----------------------------------------------------------------
    record_population: bool = True
    record_population_interval_ms: float = 10.0
    record_spikes: bool = True
    record_spike_neurons: List[int] = field(default_factory=list)   # empty = all spiking
    record_voltage_neurons: List[int] = field(default_factory=list)
    record_voltage_interval_ms: float = 1.0

    # -- output -------------------------------------------------------------------
    run_id: Optional[str] = None
    notes: str = ""
    quiet_log: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown RunConfig keys: {sorted(unknown)}")
        return cls(**d)

    # -- derived ------------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return int(round(self.duration_ms / self.dt_ms))

    def validate(self) -> None:
        if self.mode not in ("subset", "whole-brain"):
            raise ValueError(f"mode must be 'subset' or 'whole-brain', got {self.mode!r}")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be > 0")
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be > 0")
        if self.trials < 1:
            raise ValueError("trials must be >= 1")
        if self.subset_hops < 0:
            raise ValueError("subset_hops must be >= 0")
        if self.subset_max_neurons < 1:
            raise ValueError("subset_max_neurons must be >= 1")
        if self.record_population_interval_ms < self.dt_ms:
            raise ValueError("record_population_interval_ms must be >= dt_ms")
        if self.max_estimated_minutes is not None and self.max_estimated_minutes <= 0:
            raise ValueError("max_estimated_minutes must be > 0, or None to disable")
        if self.allow_fallback and self.mode != "whole-brain":
            raise ValueError("allow_fallback only applies to whole-brain mode")
