"""Runtime layer: hardware detection, backend selection, execution, recording."""

from __future__ import annotations

from .backend import (
    Backend,
    HardwareInfo,
    JaxBackend,
    NumpyBackend,
    available_backends,
    available_ram_gb,
    detect_hardware,
    get_backend,
    query_nvidia_gpus,
)
from .recorder import Recorder, RecordingConfig
from .runner import (
    RunLogger,
    RunPlan,
    RunResult,
    check_feasibility,
    estimate_requirements,
    format_runtime_banner,
    make_run_id,
    plan_run,
    run_experiment,
    spike_digest,
)

__all__ = [
    "Backend",
    "NumpyBackend",
    "JaxBackend",
    "HardwareInfo",
    "detect_hardware",
    "get_backend",
    "available_backends",
    "available_ram_gb",
    "query_nvidia_gpus",
    "Recorder",
    "RecordingConfig",
    "RunLogger",
    "RunPlan",
    "RunResult",
    "plan_run",
    "run_experiment",
    "estimate_requirements",
    "check_feasibility",
    "format_runtime_banner",
    "make_run_id",
    "spike_digest",
]
