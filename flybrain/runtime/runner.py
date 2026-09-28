"""Experiment orchestration: feasibility, execution, provenance, output.

Separated from :class:`~flybrain.model.network.LIFNetwork` on purpose. The network
knows about arrays and arithmetic; it does not know about files, wall-clock time,
or whether the machine can afford the run. All of that lives here.

The safety contract for ``whole-brain`` mode, from the objective:

* estimate the memory requirement **before** allocating anything large;
* if it does not fit, raise :class:`~flybrain.errors.ResourceLimitError` with the
  estimate attached - never shrink the run and call it whole-brain;
* report the measured per-step cost so the operator knows what they are asking for.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import config as fb_config
from ..config import RunConfig
from ..data.provenance import environment_snapshot, git_commit, utc_now_iso
from ..errors import FlyBrainError, ResourceLimitError
from ..model.lif import LIFParams
from ..model.network import LIFNetwork
from ..model.populations import PopulationRegistry
from ..model.stimulus import Stimulus
from .backend import HardwareInfo, available_ram_gb, detect_hardware, get_backend
from .recorder import Recorder, RecordingConfig

# --------------------------------------------------------------------------------------
# requirements estimation
# --------------------------------------------------------------------------------------

#: Peak *extra* bytes the kernel allocates during one step, per edge: the gathered
#: contribution vector (float32) plus the float64 accumulator ``np.bincount`` builds
#: internally before it is narrowed back to float32.
_EDGE_TEMP_BYTES_PER_EDGE = 4 + 8


def estimate_requirements(
    n_neurons: int,
    n_edges: int,
    delay_steps: int,
) -> Dict[str, Any]:
    """Estimate the memory a run needs, with each component named.

    Returns the breakdown rather than one number on purpose: when a run is refused,
    the operator needs to see *which* term dominates in order to choose between a
    shorter run, a subset, or more RAM.
    """
    f32, i32, i64 = 4, 4, 8
    parts = {
        "edge_presynaptic_index": n_edges * i32,
        "edge_postsynaptic_index": n_edges * i32,
        "edge_weights": n_edges * f32,
        "delay_ring_buffer": delay_steps * n_neurons * f32,
        "neuron_state_6_arrays": n_neurons * f32 * 6,
        "flywire_id_table": n_neurons * i64,
        "per_step_transient_peak": n_edges * _EDGE_TEMP_BYTES_PER_EDGE,
    }
    resident = sum(v for k, v in parts.items() if k != "per_step_transient_peak")
    return {
        "components_bytes": parts,
        "resident_bytes": resident,
        "peak_bytes": resident + parts["per_step_transient_peak"],
        "resident_gb": round(resident / 1e9, 3),
        "peak_gb": round((resident + parts["per_step_transient_peak"]) / 1e9, 3),
        "dense_matrix_gb_if_materialised": round(n_neurons * n_neurons * f32 / 1e9, 2),
        "largest_component": max(parts, key=parts.get),
        "note": (
            "A dense connectivity matrix is never built; the value is reported only to show "
            "why the sparse edge-list representation is a requirement, not an optimisation."
        ),
    }


def check_feasibility(
    estimate: Dict[str, Any],
    *,
    ram_available_gb: Optional[float],
    safety_fraction: float = 0.6,
    label: str = "whole-brain",
) -> None:
    """Raise :class:`ResourceLimitError` when ``estimate`` does not fit safely."""
    if ram_available_gb is None:
        # Cannot judge, so refuse to proceed: an unchecked multi-GB allocation is
        # exactly how a machine gets taken down.
        raise ResourceLimitError(
            f"cannot determine available RAM, so the {label} run cannot be checked for "
            f"safety. Estimated peak requirement is {estimate['peak_gb']} GB. "
            f"Install psutil for a reliable answer, or use --mode subset.",
            estimate,
        )
    budget = ram_available_gb * safety_fraction
    if estimate["peak_gb"] > budget:
        raise ResourceLimitError(
            f"{label} run estimated to need {estimate['peak_gb']} GB peak "
            f"({estimate['resident_gb']} GB resident) but only {ram_available_gb} GB RAM is "
            f"available (budget = {safety_fraction:.0%} of it = {budget:.2f} GB). "
            f"Largest component: {estimate['largest_component']} = "
            f"{estimate['components_bytes'][estimate['largest_component']] / 1e9:.3f} GB. "
            f"Use --mode subset, or a shorter run.",
            estimate,
        )


# --------------------------------------------------------------------------------------
# banner
# --------------------------------------------------------------------------------------


def format_runtime_banner(
    hw: HardwareInfo,
    *,
    backend_name: str,
    backend_is_gpu: bool,
    dataset_id: str,
    n_neurons: int,
    n_edges: int,
    dtype: str,
    dt_ms: float,
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """The runtime report the objective asks to print at startup.

    Reports hardware and *capability* separately. A machine with a GPU that the
    active backend cannot address says so, rather than printing "CUDA".
    """
    if hw.nvidia_gpus:
        g = hw.nvidia_gpus[0]
        more = f" (+{len(hw.nvidia_gpus) - 1} more)" if len(hw.nvidia_gpus) > 1 else ""
        gpu_line = f"{g['name']}{more}"
    else:
        gpu_line = "none detected"
    vram_line = (
        f"{hw.vram_total_mb} MB total, {hw.vram_free_mb} MB free" if hw.vram_total_mb else "n/a"
    )

    if backend_is_gpu:
        usable = f"yes - {backend_name} is running on a GPU device"
    elif hw.nvidia_gpus:
        reason = (
            f"JAX reports backend={hw.jax_backend!r} devices={hw.jax_devices}"
            if hw.jax_installed
            else "JAX is not installed, so no GPU backend is available"
        )
        usable = f"no - {reason}"
    else:
        usable = "no GPU present"

    rows: List[Tuple[str, str]] = [
        ("Mode", mode),
        ("Backend", f"{backend_name}  [{'GPU' if backend_is_gpu else 'CPU'}]"),
        ("GPU usable", usable),
        ("GPU", gpu_line),
        ("VRAM", vram_line),
        ("RAM total", f"{hw.ram_total_gb if hw.ram_total_gb is not None else 'unknown'} GB"),
        ("RAM available", f"{available_ram_gb()} GB"),
        ("CPU", f"{hw.cpu_name or 'unknown'} ({hw.cpu_cores_logical or '?'} threads)"),
        ("Python", f"{platform.python_version()} ({hw.python_implementation})"),
        ("Dataset", dataset_id),
        ("Neurons", f"{n_neurons:,}"),
        ("Synaptic edges", f"{n_edges:,}"),
        ("Precision", dtype),
        ("dt", f"{dt_ms} ms"),
    ]
    if extra:
        rows.extend((k, str(v)) for k, v in extra.items())

    width = max(len(k) for k, _ in rows)
    body = "\n".join(f"{k:<{width}}  {v}".rstrip() for k, v in rows)
    return f"FlyBrain Runtime\n\n{body}"


# --------------------------------------------------------------------------------------
# run bookkeeping
# --------------------------------------------------------------------------------------


class RunLogger:
    """Timestamped run log.

    Starts detached (``path=None``): it buffers and echoes, but writes nothing. The
    run directory is only created once the run has passed every gate, so a refused
    run leaves no directory behind for a script to mistake for a result.
    """

    def __init__(self, path: Optional[Path] = None, echo: Optional[Callable[[str], None]] = None):
        self.path = Path(path) if path else None
        self.echo = echo
        self.lines: List[str] = []

    def __call__(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        self.lines.append(line)
        self._append(line)
        if self.echo is not None:
            self.echo(line)

    def _append(self, line: str) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def attach(self, path: Path) -> None:
        """Bind the log to a file and flush everything buffered so far."""
        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()
        for line in self.lines:
            self._append(line)


def make_run_id(mode: str, dataset_id: str, duration_ms: float, label: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() else "-" for c in (label or dataset_id)).strip("-")
    return f"{stamp}-{mode}-{slug}-{duration_ms:g}ms"


@dataclass
class RunResult:
    """Everything a run produced, plus where it was written."""

    run_id: str
    run_dir: Path
    mode: str
    is_subset: bool
    n_neurons: int
    n_edges: int
    n_steps: int
    duration_ms: float
    trials: int
    dt_ms: float
    backend: str
    backend_is_gpu: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    spikes_path: Optional[Path] = None
    population_path: Optional[Path] = None
    voltage_path: Optional[Path] = None

    @property
    def population_rates(self) -> Dict[str, float]:
        return self.summary.get("population_rates_hz", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "mode": self.mode,
            "is_subset": self.is_subset,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "n_steps": self.n_steps,
            "duration_ms": self.duration_ms,
            "trials": self.trials,
            "dt_ms": self.dt_ms,
            "backend": self.backend,
            "backend_is_gpu": self.backend_is_gpu,
            "summary": self.summary,
            "spikes_path": str(self.spikes_path) if self.spikes_path else None,
            "population_path": str(self.population_path) if self.population_path else None,
            "voltage_path": str(self.voltage_path) if self.voltage_path else None,
        }


def spike_digest(spikes_frame, trials: int) -> str:
    """Stable digest of the recorded spike train, for reproducibility checks.

    Hashes the ordered ``(t_ms, flywire_id)`` pairs. Two identical deterministic
    runs must produce the same digest; the tests assert that rather than assume it.
    """
    h = hashlib.sha256()
    if spikes_frame is None or len(spikes_frame) == 0:
        h.update(b"empty")
        return h.hexdigest()
    ordered = spikes_frame.sort_values(["t_ms", "flywire_id"], kind="stable")
    h.update(np.ascontiguousarray(ordered["t_ms"].to_numpy(np.float64)).tobytes())
    h.update(np.ascontiguousarray(ordered["flywire_id"].to_numpy(np.int64)).tobytes())
    h.update(f"|trials={trials}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# planning: subset vs whole-brain, and the refusal that must never become a fallback
# --------------------------------------------------------------------------------------


@dataclass
class RunPlan:
    """The decision of *what* to simulate, and the evidence behind it."""

    connectome: Any
    requested_mode: str
    effective_mode: str
    notes: List[str] = field(default_factory=list)
    estimate: Dict[str, Any] = field(default_factory=dict)
    subset_info: Optional[Dict[str, Any]] = None
    refused: Optional[str] = None

    @property
    def is_subset(self) -> bool:
        return self.effective_mode == "subset"

    @property
    def mode_changed(self) -> bool:
        return self.requested_mode != self.effective_mode

    def label(self) -> str:
        base = f"{self.effective_mode} ({self.connectome.n_neurons:,} neurons)"
        if self.mode_changed:
            base += f" [requested {self.requested_mode}]"
        return base


def plan_run(
    full_connectome,
    cfg: RunConfig,
    stimuli: Sequence[Stimulus] = (),
    *,
    hardware: Optional[HardwareInfo] = None,
    enforce_feasibility: bool = True,
) -> RunPlan:
    """Decide between whole-brain and subset, refusing rather than silently shrinking.

    ``whole-brain`` + infeasible + ``allow_fallback=False`` (the default) raises.
    ``allow_fallback=True`` produces a subset plan with ``mode_changed`` set and an
    explicit note, so every downstream artefact records that the run is a subset
    and that a whole-brain run was requested and not delivered. Nothing here will
    ever describe a subset as whole-brain.
    """
    delay_steps = LIFParams().delay_steps_for(cfg.dt_ms)

    if cfg.mode == "whole-brain":
        est = estimate_requirements(
            full_connectome.n_neurons, full_connectome.n_edges, delay_steps
        )
        ram_avail = available_ram_gb()
        try:
            if enforce_feasibility:
                check_feasibility(est, ram_available_gb=ram_avail, label="whole-brain")
            return RunPlan(
                connectome=full_connectome,
                requested_mode="whole-brain",
                effective_mode="whole-brain",
                estimate=est,
                notes=[f"whole-brain feasibility check passed against {ram_avail} GB available RAM"],
            )
        except ResourceLimitError as exc:
            if not cfg.allow_fallback:
                raise
            seeds = list(cfg.subset_seed_ids) or _stimulus_seeds(stimuli)
            if not seeds:
                raise ResourceLimitError(
                    f"whole-brain run is infeasible and no subset seeds were supplied, so no "
                    f"fallback is possible. Seed the subset with --subset-seed or a --stimulus. "
                    f"Original refusal: {exc}",
                    est,
                ) from exc
            sub = full_connectome.subset(
                seeds, hops=cfg.subset_hops, max_neurons=cfg.subset_max_neurons
            )
            return RunPlan(
                connectome=sub,
                requested_mode="whole-brain",
                effective_mode="subset",
                estimate=est,
                subset_info=sub.subset_info.to_dict() if sub.subset_info else None,
                refused=str(exc),
                notes=[
                    "WHOLE-BRAIN RUN WAS REQUESTED AND WAS NOT DELIVERED.",
                    f"Reason: {exc}",
                    f"An explicit fallback (--allow-fallback) produced a {sub.n_neurons:,}-neuron "
                    f"subset instead. This is NOT a whole-brain run.",
                ],
            )

    # subset mode
    # Seeds are the UNION of the explicit seeds and the stimulus targets. It is never
    # correct to build a subset that excludes the neurons being stimulated: the run
    # would silently measure nothing. `--subset-seed` therefore *adds* a readout
    # population to the working set rather than replacing the stimulus neurons.
    seeds = list(dict.fromkeys(list(cfg.subset_seed_ids) + _stimulus_seeds(stimuli)))
    if not seeds:
        raise FlyBrainError(
            "subset mode needs seeds: pass --subset-seed <flywire_id> (repeatable) or a "
            "--stimulus whose target population provides them"
        )
    sub = full_connectome.subset(
        seeds, hops=cfg.subset_hops, max_neurons=cfg.subset_max_neurons
    )
    est = estimate_requirements(sub.n_neurons, sub.n_edges, delay_steps)
    info = sub.subset_info.to_dict() if sub.subset_info else None
    notes = [
        f"subset of {sub.n_neurons:,} neurons / {sub.n_edges:,} edges from "
        f"{full_connectome.n_neurons:,} / {full_connectome.n_edges:,} in {full_connectome.dataset_id}"
    ]
    if info and info.get("truncated"):
        notes.append(
            f"subset expansion was TRUNCATED at max_neurons={cfg.subset_max_neurons}; "
            f"the neighbourhood is incomplete"
        )
    if info and info.get("missing_seed_ids"):
        notes.append(
            f"{len(info['missing_seed_ids'])} requested seed ID(s) are absent from the dataset"
        )
    return RunPlan(
        connectome=sub,
        requested_mode="subset",
        effective_mode="subset",
        estimate=est,
        subset_info=info,
        notes=notes,
    )


def _stimulus_seeds(stimuli: Sequence[Stimulus]) -> List[int]:
    """Neuron IDs to seed a subset with when nothing else is given."""
    out: List[int] = []
    seen = set()
    for s in stimuli:
        for i in s.neuron_ids:
            if i not in seen:
                seen.add(i)
                out.append(int(i))
    return out


# --------------------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------------------


def _probe_step_seconds(net: LIFNetwork, n_probe: int = 3) -> float:
    """Measure the real per-step wall time of *this* network on *this* machine.

    A measurement, not a model: it exercises the actual arrays, the actual backend
    and the actual stimulus set, so the runtime estimate the operator sees is
    grounded in observed behaviour rather than a constant copied from elsewhere.
    """
    n_probe = max(1, int(n_probe))
    t0 = time.perf_counter()
    for _ in range(n_probe):
        net.step()
    return (time.perf_counter() - t0) / n_probe


def run_experiment(
    plan: RunPlan,
    cfg: RunConfig,
    stimuli: Sequence[Stimulus] = (),
    *,
    registry: Optional[PopulationRegistry] = None,
    backend=None,
    recording: Optional[RecordingConfig] = None,
    run_dir: Optional[Path] = None,
    echo: Optional[Callable[[str], None]] = None,
    hardware: Optional[HardwareInfo] = None,
    params: Optional[LIFParams] = None,
    write_outputs: bool = True,
    progress_every: int = 0,
) -> RunResult:
    """Execute one run described by ``plan`` + ``cfg`` and write its artefacts."""
    connectome = plan.connectome
    registry = registry or PopulationRegistry.builtin()
    hw = hardware or detect_hardware()
    params = params or LIFParams()
    backend = backend if backend is not None else get_backend(cfg.backend, cfg.dtype)

    run_id = cfg.run_id or make_run_id(
        plan.effective_mode, connectome.dataset_id, cfg.duration_ms, label=_label_of(stimuli)
    )
    run_dir = Path(run_dir) if run_dir else (fb_config.RUNS_DIR / run_id)
    # Detached: nothing is written until the run has passed the runtime gate below.
    log = RunLogger(None, echo=echo)

    log(f"run_id={run_id} mode={plan.effective_mode} dataset={connectome.dataset_id}")
    log(
        f"network: {connectome.n_neurons:,} neurons, {connectome.n_edges:,} edges, "
        f"subset={connectome.is_subset}"
    )
    for note in plan.notes:
        log(f"plan: {note}")

    net = LIFNetwork(
        connectome, params, dt_ms=cfg.dt_ms, backend=backend, seed=cfg.seed, registry=registry
    )
    # Validate the stimuli against the connectome *before* anything else, with a
    # message that names the real cause. A raw "IDs not in dataset" here almost
    # always means the subset was built without the stimulus neurons in it.
    for s in stimuli:
        _present, missing = connectome.filter_present(list(s.neuron_ids))
        if missing.size:
            raise FlyBrainError(
                f"stimulus {s.label or s!r} targets {missing.size} neuron(s) that are not in "
                f"the loaded connectome ({connectome.n_neurons:,} neurons, "
                f"subset={connectome.is_subset}); first missing: {missing[:5].tolist()}. "
                f"For a subset run, seed the working set with these neurons "
                f"(--subset-seed), or raise --subset-max-neurons so the neighbourhood is "
                f"not truncated."
            )
    if stimuli:
        net.apply_stimulus(*stimuli)
    silenced_ids: List[int] = []
    for name in cfg.silence_populations:
        silenced_ids.extend(net.silence(name))
    if cfg.silence:
        silenced_ids.extend(net.silence(flywire_ids=cfg.silence))
    if silenced_ids:
        log(f"silenced {len(set(silenced_ids))} neuron(s)")
    net.disable_recurrence = bool(cfg.disable_recurrence)
    for s in stimuli:
        log(f"stimulus: {s.describe()}")

    rec_cfg = recording or RecordingConfig(
        record_spikes=cfg.record_spikes,
        spike_neuron_ids=tuple(cfg.record_spike_neurons),
        record_population=cfg.record_population,
        population_interval_ms=cfg.record_population_interval_ms,
        record_voltage_ids=tuple(cfg.record_voltage_neurons),
        voltage_interval_ms=cfg.record_voltage_interval_ms,
    )
    recorder = Recorder(connectome, rec_cfg, registry=registry, dt_ms=cfg.dt_ms)

    # -- measured cost probe, then a clean reset so the probe does not perturb state
    probe_seconds = _probe_step_seconds(net, 3) if cfg.n_steps > 3 else float("nan")
    net.reset(seed=cfg.seed)
    if np.isfinite(probe_seconds):
        log(
            f"measured step cost: {probe_seconds * 1e3:.1f} ms/step "
            f"({cfg.n_steps} steps x {cfg.trials} trial(s) -> "
            f"~{probe_seconds * cfg.n_steps * cfg.trials / 60:.1f} min estimated)"
        )

    if cfg.max_estimated_minutes is not None and np.isfinite(probe_seconds):
        est_min = probe_seconds * cfg.n_steps * cfg.trials / 60.0
        if est_min > cfg.max_estimated_minutes:
            # Raised before the run directory exists: a refused run must leave nothing
            # behind that a script could mistake for a result.
            raise ResourceLimitError(
                f"estimated runtime {est_min:.1f} min exceeds the configured limit of "
                f"{cfg.max_estimated_minutes:.1f} min "
                f"({probe_seconds * 1e3:.1f} ms/step measured, {cfg.n_steps} steps x "
                f"{cfg.trials} trial(s)). Raise --max-minutes or shorten --duration-ms.",
                {"estimated_minutes": est_min, "seconds_per_step": probe_seconds},
            )

    # Gates passed: now it is safe to create the run directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    log.attach(run_dir / "run.log")

    # -- trials ---------------------------------------------------------------------
    spike_frames: List[Any] = []
    pop_frames: List[Any] = []
    volt_frames: List[Any] = []
    trial_counts: List[np.ndarray] = []
    step_seconds: List[float] = []

    # A whole-brain trial takes tens of minutes; logging only at the end makes the
    # run look hung. Default to reporting roughly every 10% of a long run.
    if progress_every <= 0 and cfg.n_steps >= 2000:
        progress_every = max(cfg.n_steps // 10, 1)

    for trial in range(cfg.trials):
        net.reset(seed=cfg.seed + trial)
        recorder.begin(net)
        t_trial = time.perf_counter()
        for step in range(cfg.n_steps):
            spk = net.step()
            recorder.on_step(step, spk, net)
            if progress_every and (step + 1) % progress_every == 0:
                log(
                    f"trial {trial + 1}/{cfg.trials} step {step + 1}/{cfg.n_steps} "
                    f"({time.perf_counter() - t_trial:.1f}s)"
                )
        step_seconds.append((time.perf_counter() - t_trial) / max(cfg.n_steps, 1))

        sf = recorder.spikes_frame()
        if sf is not None:
            sf = sf.assign(trial=trial)
            spike_frames.append(sf)
        pf = recorder.population_frame()
        if pf is not None:
            pf = pf.assign(trial=trial)
            pop_frames.append(pf)
        vf = recorder.voltage_frame()
        if vf is not None:
            vf = vf.assign(trial=trial)
            volt_frames.append(vf)
        trial_counts.append(np.asarray(net.spike_count, dtype=np.float64).copy())
        log(
            f"trial {trial + 1}/{cfg.trials}: {int(trial_counts[-1].sum())} spikes, "
            f"{int((trial_counts[-1] > 0).sum())} active neurons, "
            f"{step_seconds[-1] * 1e3:.1f} ms/step"
        )

    summary = _summarise(
        net=net,
        connectome=connectome,
        cfg=cfg,
        plan=plan,
        params=params,
        registry=registry,
        trial_counts=np.asarray(trial_counts),
        step_seconds=step_seconds,
        probe_seconds=probe_seconds,
        hardware=hw,
        backend=backend,
        recorder_summary=recorder.summary(),
        silenced_ids=sorted(set(silenced_ids)),
    )

    # -- materialise ----------------------------------------------------------------
    import pandas as pd

    spikes_df = pd.concat(spike_frames, ignore_index=True) if spike_frames else None
    pop_df = pd.concat(pop_frames, ignore_index=True) if pop_frames else None
    volt_df = pd.concat(volt_frames, ignore_index=True) if volt_frames else None

    summary["spike_digest_sha256"] = spike_digest(spikes_df, cfg.trials)
    summary["recorded_spike_rows"] = int(0 if spikes_df is None else len(spikes_df))
    if spikes_df is not None:
        summary["active_flywire_ids"] = sorted(int(x) for x in spikes_df["flywire_id"].unique())

    result = RunResult(
        run_id=run_id,
        run_dir=run_dir,
        mode=plan.effective_mode,
        is_subset=connectome.is_subset,
        n_neurons=connectome.n_neurons,
        n_edges=connectome.n_edges,
        n_steps=cfg.n_steps,
        duration_ms=cfg.duration_ms,
        trials=cfg.trials,
        dt_ms=cfg.dt_ms,
        backend=backend.name,
        backend_is_gpu=bool(getattr(backend, "is_gpu", False)),
        summary=summary,
    )

    if write_outputs:
        # Everything except summary.json, which must be written *after* plotting so
        # that it can list the figures actually produced.
        _write_outputs(
            result, cfg, plan, connectome, hw, backend, params,
            spikes_df, pop_df, volt_df,
        )
        result.spikes_path = run_dir / "spikes.parquet" if spikes_df is not None else None
        result.population_path = run_dir / "population_activity.parquet" if pop_df is not None else None
        result.voltage_path = run_dir / "voltage_samples.parquet" if volt_df is not None else None

        # Figures are produced here rather than by each caller, so that every path
        # into a run - CLI, packaged experiment, Python API - gets them. They are
        # built from the in-memory frames, so plotting costs no re-read.
        summary["plots"] = []
        if cfg.make_plots:
            try:
                written = _write_plots(result, stimuli, connectome, spikes_df, pop_df, volt_df)
                summary["plots"] = [p.name for p in written]
                if written:
                    log(f"wrote {len(written)} figure(s) to {run_dir / 'plots'}")
            except Exception as exc:  # missing matplotlib, or an unlucky figure
                summary["plots_error"] = f"{type(exc).__name__}: {exc}"
                log(f"plots skipped: {type(exc).__name__}: {exc}")

        result.summary = summary
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        log(f"wrote artefacts to {run_dir}")

    return result


def _write_plots(result, stimuli, connectome, spikes_df, pop_df, volt_df) -> List[Path]:
    """Render the standard figure set for a run into ``<run_dir>/plots``."""
    from ..analysis import plot_summary_dashboard, plot_voltage_trace

    if spikes_df is None and pop_df is None:
        return []

    plot_dir = result.run_dir / "plots"
    stimulus_populations = [s.label for s in stimuli if s.label]

    # Highlight one representative neuron per stimulated population in the raster, so
    # the figure shows the driven neurons separately from the responders.
    highlight: Dict[str, int] = {}
    for s in stimuli:
        if not s.label:
            continue
        present, _missing = connectome.filter_present(list(s.neuron_ids))
        if present.size:
            highlight[f"stim {s.label}"] = int(present[0])

    written = plot_summary_dashboard(
        result.summary, spikes_df, pop_df, plot_dir,
        stimulus_populations=stimulus_populations,
        highlight=highlight or None,
    )
    if volt_df is not None:
        p = plot_voltage_trace(volt_df, plot_dir / "membrane_potential.png")
        if p is not None:
            written.append(p)
    return written


def plot_condition_comparison(conditions: Dict[str, Dict[str, float]], out_path: Path,
                              *, populations: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Input-vs-downstream comparison across named experimental conditions.

    Used by the silencing experiment, where the two conditions are *the same stimulus*
    with and without a perturbation - the figure that makes the causal comparison
    readable at a glance.
    """
    from ..analysis import plot_input_vs_downstream

    return plot_input_vs_downstream(conditions, out_path, populations=populations)


def _label_of(stimuli: Sequence[Stimulus]) -> str:
    labels = [s.label for s in stimuli if s.label]
    return "+".join(labels[:3]) if labels else ""


def _summarise(
    *,
    net: LIFNetwork,
    connectome,
    cfg: RunConfig,
    plan: RunPlan,
    params: LIFParams,
    registry: PopulationRegistry,
    trial_counts: np.ndarray,
    step_seconds: List[float],
    probe_seconds: float,
    hardware: HardwareInfo,
    backend,
    recorder_summary: Dict[str, Any],
    silenced_ids: List[int],
) -> Dict[str, Any]:
    """Build ``summary.json``'s payload. All numbers here come from measurements."""

    duration_s = max(cfg.duration_ms / 1000.0, 1e-12)
    # rates per trial, then mean/std across trials
    rates = trial_counts / duration_s                       # (trials, n_neurons) Hz
    mean_rate = rates.mean(axis=0)
    std_rate = rates.std(axis=0) if cfg.trials > 1 else np.zeros_like(mean_rate)
    active_mask = (trial_counts > 0).any(axis=0)

    pop_rates: Dict[str, float] = {}
    pop_rates_std: Dict[str, float] = {}
    pop_counts: Dict[str, int] = {}
    # Build the dataset ID set once. Resolving 114 populations independently would
    # rebuild a 127 400-element set 114 times, costing seconds per run for nothing.
    dataset_ids = set(int(x) for x in connectome.flywire_ids)
    for pop in registry:
        present = [i for i in pop.ids if i in dataset_ids]
        pop_counts[pop.name] = len(present)
        if not present:
            continue
        idx = connectome.indices_of(present)
        r = rates[:, idx].mean(axis=1)                      # per trial mean rate
        pop_rates[pop.name] = float(r.mean())
        pop_rates_std[pop.name] = float(r.std())

    # top active neurons and populations, which the REPL and the plots need
    order = np.argsort(-mean_rate)[:50]
    top_neurons = [
        {
            "flywire_id": int(connectome.flywire_ids[i]),
            "mean_rate_hz": float(mean_rate[i]),
            "std_rate_hz": float(std_rate[i]),
            "index": int(i),
        }
        for i in order
        if mean_rate[i] > 0
    ]
    top_pops = sorted(
        ({"population": k, "mean_rate_hz": v} for k, v in pop_rates.items()),
        key=lambda d: -d["mean_rate_hz"],
    )[:25]

    step_times = [s for s in step_seconds if np.isfinite(s)]
    return {
        "requested_mode": plan.requested_mode,
        "effective_mode": plan.effective_mode,
        "mode_changed": plan.mode_changed,
        "plan_notes": plan.notes,
        "refusal_reason": plan.refused,
        "is_subset": bool(connectome.is_subset),
        "subset_info": plan.subset_info,
        "dataset_id": connectome.dataset_id,
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "n_synapses_total": connectome.synapse_count,
        "dt_ms": cfg.dt_ms,
        "duration_ms": cfg.duration_ms,
        "n_steps": cfg.n_steps,
        "trials": cfg.trials,
        "seed": cfg.seed,
        "backend": backend.name,
        "backend_is_gpu": bool(getattr(backend, "is_gpu", False)),
        "dtype": str(net.dtype),
        "model_params": params.to_dict(),
        "stimuli": [s.stimulus.to_dict() for s in net._samplers],
        "silenced_flywire_ids": silenced_ids,
        "disable_recurrence": bool(cfg.disable_recurrence),
        # ---------------- activity, from measurement ----------------
        "total_spikes": int(trial_counts.sum()),
        "spikes_per_trial_mean": float(trial_counts.sum(axis=1).mean()),
        "active_neurons_any_spike": int(active_mask.sum()),
        "active_neurons_mean_per_trial": float((trial_counts > 0).sum(axis=1).mean()),
        "mean_rate_hz_all_neurons": float(mean_rate.mean()),
        "mean_rate_hz_active_neurons": float(mean_rate[active_mask].mean()) if active_mask.any() else 0.0,
        "max_rate_hz": float(mean_rate.max()),
        "population_rates_hz": pop_rates,
        "population_rates_std_hz": pop_rates_std,
        "population_sizes_present": pop_counts,
        "top_active_neurons": top_neurons,
        "top_active_populations": top_pops,
        "recording": recorder_summary,
        # ---------------- cost, from measurement ----------------
        "timing": {
            "measured_step_ms_probe": (probe_seconds * 1e3) if np.isfinite(probe_seconds) else None,
            "measured_step_ms_mean": (float(np.mean(step_times)) * 1e3) if step_times else None,
            "measured_steps_per_second": (1.0 / float(np.mean(step_times))) if step_times else None,
            "wall_seconds_total": float(sum(step_times) * cfg.n_steps),
            "estimated_minutes_for_this_run": (
                float(np.mean(step_times) * cfg.n_steps * cfg.trials / 60.0) if step_times else None
            ),
            "note": "measured on this machine for this configuration; not a model",
        },
        "requirements_estimate": plan.estimate,
        "hardware": hardware.to_dict(),
    }


def _write_outputs(
    result: RunResult,
    cfg: RunConfig,
    plan: RunPlan,
    connectome,
    hw: HardwareInfo,
    backend,
    params: LIFParams,
    spikes_df,
    pop_df,
    volt_df,
) -> None:
    """Write config.json, environment.json, dataset.json, summary.json and the tables."""
    rd = result.run_dir
    rd.mkdir(parents=True, exist_ok=True)

    config_blob = {
        **cfg.to_dict(),
        "run_id": result.run_id,
        "effective_mode": plan.effective_mode,
        "requested_mode": plan.requested_mode,
        "backend_resolved": backend.name,
        "backend_is_gpu": bool(getattr(backend, "is_gpu", False)),
        "stimuli": [s.to_dict() for s in plan_stimuli(result.summary)],
        "model_params": params.to_dict(),
        "written_at": utc_now_iso(),
    }
    (rd / "config.json").write_text(json.dumps(config_blob, indent=2) + "\n", encoding="utf-8")

    env_blob = {
        "captured_at": utc_now_iso(),
        "code_revision": git_commit(),
        "python": environment_snapshot(),
        "hardware": hw.to_dict(),
        "backend": {
            "name": backend.name,
            "is_gpu": bool(getattr(backend, "is_gpu", False)),
            "dtype": str(getattr(backend, "dtype", None)),
        },
        "gpu_usable_note": (
            "hardware.nvidia_gpus lists GPUs physically present; backend.is_gpu states whether "
            "this process can use one. They are reported separately and must not be conflated."
        ),
    }
    (rd / "environment.json").write_text(json.dumps(env_blob, indent=2) + "\n", encoding="utf-8")

    ds_blob = {
        "dataset_id": connectome.dataset_id,
        "is_subset": connectome.is_subset,
        "n_neurons": connectome.n_neurons,
        "n_edges": connectome.n_edges,
        "n_synapses_total": connectome.synapse_count,
        "card": connectome.card.to_dict() if connectome.card is not None else None,
        "subset": plan.subset_info,
        "population_sizes_present": result.summary.get("population_sizes_present", {}),
        "note": (
            "The dataset card below is the provenance record for the bytes on disk "
            "(SHA-256 per file). See outputs/dataset_report.json for content validation."
        ),
    }
    (rd / "dataset.json").write_text(json.dumps(ds_blob, indent=2) + "\n", encoding="utf-8")

    # summary.json is intentionally NOT written here: the caller writes it after
    # plotting, so that it can list the figures that were actually produced.

    if spikes_df is not None:
        spikes_df.to_parquet(rd / "spikes.parquet", compression="brotli", index=False)
    if pop_df is not None:
        pop_df.to_parquet(rd / "population_activity.parquet", compression="brotli", index=False)
    if volt_df is not None:
        volt_df.to_parquet(rd / "voltage_samples.parquet", compression="brotli", index=False)


def plan_stimuli(summary: Dict[str, Any]) -> List[Stimulus]:
    """Reconstruct the declared stimuli from a summary payload."""
    out = []
    for d in summary.get("stimuli", []) or []:
        try:
            out.append(Stimulus.from_dict(d))
        except Exception:
            continue
    return out
