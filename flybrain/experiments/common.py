"""Shared plumbing for the packaged experiments.

Everything here is about *running and recording* an experiment, not about what the
experiment means. Each experiment module owns its own hypothesis and its own
success criteria.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .. import config as fb_config
from ..config import RunConfig
from ..data import DATASET_SOURCES, load_connectome
from ..model.populations import PopulationRegistry
from ..model.stimulus import Stimulus
from ..runtime import (
    RecordingConfig,
    format_runtime_banner,
    run_experiment,
)
from ..runtime.runner import RunPlan, RunResult

#: Upstream repository that ships both dataset versions and the published results.
UPSTREAM_ORIGINAL = fb_config.THIRD_PARTY_DIR / "Drosophila_brain_model"
UPSTREAM_JAX = fb_config.THIRD_PARTY_DIR / "drosophila_whole_brain_snn_simulation"
PUBLISHED_RESULTS_DIR = UPSTREAM_ORIGINAL / "results" / "example"

#: MN9 motor neurons: the paper's primary sensorimotor readout.
MN9_LEFT = 720575940660219265
MN9_RIGHT = 720575940645521262


# --------------------------------------------------------------------------------------
# argument parsing shared by the experiments
# --------------------------------------------------------------------------------------


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--mode", default="subset", choices=["subset", "whole-brain"])
    p.add_argument("--dataset", default="flywire_630", choices=sorted(DATASET_SOURCES))
    p.add_argument("--duration-ms", type=float, default=1000.0)
    p.add_argument("--dt-ms", type=float, default=fb_config.DEFAULT_DT_MS)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="auto", choices=["auto", "numpy", "jax"])
    p.add_argument("--subset-hops", type=int, default=2)
    p.add_argument("--subset-max-neurons", type=int, default=5000)
    p.add_argument("--subset-seed", action="append", type=int, default=None,
                   help="FlyWire ID to include in the subset (repeatable). Add the readout "
                        "population so the comparison target is inside the working set.")
    p.add_argument("--allow-fallback", action="store_true")
    p.add_argument("--max-minutes", type=float, default=60.0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out-dir", default=None)
    return p


def build_run_config(args, **overrides) -> RunConfig:
    kwargs: Dict[str, Any] = dict(
        dataset_id=args.dataset,
        mode=getattr(args, "mode", "subset"),
        duration_ms=getattr(args, "duration_ms", 1000.0),
        dt_ms=getattr(args, "dt_ms", fb_config.DEFAULT_DT_MS),
        trials=getattr(args, "trials", 1),
        seed=getattr(args, "seed", 0),
        backend=getattr(args, "backend", "auto"),
        allow_fallback=bool(getattr(args, "allow_fallback", False)),
        max_estimated_minutes=(
            None if getattr(args, "max_minutes", 60.0) < 0 else getattr(args, "max_minutes", 60.0)
        ),
        subset_hops=getattr(args, "subset_hops", 2),
        subset_max_neurons=getattr(args, "subset_max_neurons", 5000),
        subset_seed_ids=list(getattr(args, "subset_seed", None) or []),
        quiet_log=bool(getattr(args, "quiet", False)),
    )
    kwargs.update(overrides)
    cfg = RunConfig(**kwargs)
    cfg.validate()
    return cfg


# --------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------


def load_dataset(dataset_id: str, *, quiet: bool = False):
    conn = load_connectome(
        dataset_id, fb_config.RAW_DIR, fb_config.PROCESSED_DIR, fb_config.METADATA_DIR
    )
    if not quiet:
        print(f"dataset {dataset_id}: {conn.n_neurons:,} neurons, {conn.n_edges:,} edges")
    return conn


def announce(conn, hw, backend, cfg: RunConfig, extra: Optional[Dict[str, Any]] = None) -> None:
    print()
    print(
        format_runtime_banner(
            hw,
            backend_name=backend.name,
            backend_is_gpu=bool(getattr(backend, "is_gpu", False)),
            dataset_id=conn.dataset_id,
            n_neurons=conn.n_neurons,
            n_edges=conn.n_edges,
            dtype=str(getattr(backend, "dtype", cfg.dtype)),
            dt_ms=cfg.dt_ms,
            mode=cfg.mode,
            extra=extra,
        )
    )
    print()


def experiment_dir(name: str, override: Optional[str] = None) -> Path:
    d = Path(override) if override else (fb_config.OUTPUTS_DIR / "experiments" / name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def execute(
    plan: RunPlan,
    cfg: RunConfig,
    stimuli: Sequence[Stimulus],
    *,
    registry: PopulationRegistry,
    backend,
    hardware,
    recording: RecordingConfig,
    run_dir: Path,
    echo=None,
    progress_every: int = 0,
) -> RunResult:
    return run_experiment(
        plan, cfg, stimuli,
        registry=registry, backend=backend, hardware=hardware,
        recording=recording, run_dir=run_dir, echo=echo, progress_every=progress_every,
    )


def render_plan(plan: RunPlan) -> None:
    print(f"plan: {plan.label()}")
    for note in plan.notes:
        print(f"  - {note}")
    print()


# --------------------------------------------------------------------------------------
# published reference numbers
# --------------------------------------------------------------------------------------


def load_published_reference() -> Dict[str, Any]:
    """Read the upstream repository's published spike output as a comparison target.

    Returns an empty dict when the upstream results are not present, so the
    experiments degrade to "no reference available" instead of inventing one. Every
    number here is measured from the upstream parquet files
    (``third_party/Drosophila_brain_model/results/example/*.parquet``), which are the
    authors' own published output.
    """
    if not PUBLISHED_RESULTS_DIR.is_dir():
        return {}
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return {}

    sugar_ids = set(PopulationRegistry.builtin().get("sugar_grn").ids)
    out: Dict[str, Any] = {"source": str(PUBLISHED_RESULTS_DIR), "files": {}}
    for f in sorted(PUBLISHED_RESULTS_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if not {"t", "flywire_id", "trial"}.issubset(df.columns):
            continue
        trials = int(df["trial"].nunique())
        seconds = float(df["t"].max()) if len(df) else 1.0
        cnt = df.groupby("flywire_id").size()
        sugar_present = [i for i in sugar_ids if i in cnt.index]
        out["files"][f.name] = {
            "trials": trials,
            "spikes": int(len(df)),
            "active_neurons": int(df["flywire_id"].nunique()),
            "mn9_left_rate_hz": float(cnt.get(MN9_LEFT, 0) / trials),
            "mn9_right_rate_hz": float(cnt.get(MN9_RIGHT, 0) / trials),
            "median_sugar_grn_rate_hz": (
                float(np.median([cnt[i] / trials for i in sugar_present])) if sugar_present else None
            ),
            "sugar_grns_present": len(sugar_present),
            "max_t_ms_observed": seconds,
        }
    return out


def published_sugar_reference(rate_hz: float = 100.0) -> Optional[Dict[str, Any]]:
    """Pick the published ``sugarR`` run closest to the requested stimulation rate.

    The published files are ``sugarR`` (the committed tutorial run, which
    ``docs/UPSTREAM_AUDIT.md`` establishes was produced at 200 Hz) and
    ``sugarR_100Hz``. Matching on the measured sugar-GRN rate rather than on the
    filename avoids depending on upstream naming.
    """
    ref = load_published_reference().get("files", {})
    candidates = []
    for name, d in ref.items():
        measured = d.get("median_sugar_grn_rate_hz")
        if measured is None or not name.startswith("sugarR"):
            continue
        if "-" in name:  # silencing runs
            continue
        candidates.append((abs(measured - rate_hz), name, d))
    if not candidates:
        return None
    _, name, d = min(candidates, key=lambda t: t[0])
    return {"file": name, **d}


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------


def write_report(out_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return p


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def check(passed: bool, description: str, detail: str = "") -> Dict[str, Any]:
    """One named assertion in an experiment's report, with its evidence."""
    return {"passed": bool(passed), "check": description, "detail": detail}


def print_checks(checks: List[Dict[str, Any]]) -> bool:
    ok = True
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        if not c["passed"]:
            ok = False
        line = f"  [{mark}] {c['check']}"
        if c["detail"]:
            line += f"\n         {c['detail']}"
        print(line)
    return ok
