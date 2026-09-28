"""FlyBrain command line interface.

Exit codes are distinct so that a refusal is machine-visible and cannot be mistaken
for success by a script:

===  ==============================================================
0    success
2    usage / configuration error
3    refused: the requested run cannot run safely on this machine
4    dataset error (missing, corrupted, or failing validation)
5    unknown population or FlyWire ID
===  ==============================================================
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__, config as fb_config
from .config import RunConfig
from .data import (
    DATASET_SOURCES,
    check_id_index_consistency,
    load_connectome,
    load_dataset_card,
    list_dataset_cards,
    stage_dataset,
    update_dataset_card,
    validate_connectome,
)
from .data.provenance import git_commit
from .errors import (
    DatasetError,
    DatasetValidationError,
    FlyBrainError,
    ResourceLimitError,
    UnknownPopulationError,
)
from .model.populations import PopulationRegistry
from .model.stimulus import Stimulus
from .runtime import (
    RecordingConfig,
    detect_hardware,
    format_runtime_banner,
    get_backend,
    plan_run,
    run_experiment,
)

DEFAULT_UPSTREAM = fb_config.THIRD_PARTY_DIR


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------


def _registry() -> PopulationRegistry:
    return PopulationRegistry.builtin()


def _load_connectome(dataset_id: str, *, quiet: bool = False):
    if not quiet:
        card = load_dataset_card(dataset_id, fb_config.METADATA_DIR)
        if card is None:
            print(
                f"dataset {dataset_id!r} is not staged. Run:\n"
                f"  flybrain datasets --stage {dataset_id}",
                file=sys.stderr,
            )
    return load_connectome(
        dataset_id, fb_config.RAW_DIR, fb_config.PROCESSED_DIR, fb_config.METADATA_DIR
    )


def _load_connectome_deep(dataset_id: str):
    """Load a dataset reading *every* column, for verification only.

    Attaches the raw columns to the returned object so the ID<->index check can run
    without re-reading the parquet. Memory cost is deliberately paid only here.
    """
    from .data.loader import build_connectome, raw_paths

    neuron_p, conn_p = raw_paths(dataset_id, fb_config.RAW_DIR)
    conn = build_connectome(dataset_id, neuron_p, conn_p, deep_columns=True)
    from .data.loader import read_connectivity_table

    conn.raw_columns = read_connectivity_table(conn_p, deep=True)  # type: ignore[attr-defined]
    return conn


def parse_stimulus_spec(spec: str, registry: PopulationRegistry) -> Stimulus:
    """Parse a ``--stimulus`` value.

    Accepted forms::

        sugar_grn                       population, default rate
        sugar_grn:150                   population, 150 Hz
        sugar_grn:150:100:500           population, 150 Hz, 100-500 ms
        ids:7205759...,7205759...:150   explicit FlyWire IDs, 150 Hz
        ids:7205759...:150:100:500      explicit IDs with a window
    """
    parts = spec.split(":")
    try:
        if parts[0] == "ids":
            if len(parts) < 3:
                raise ValueError("need ids:<id[,id...]>:<rate_hz>[:start_ms[:end_ms]]")
            ids = [int(x) for x in parts[1].split(",") if x.strip()]
            rate = float(parts[2])
            start = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
            end = float(parts[4]) if len(parts) > 4 and parts[4] else math.inf
            return Stimulus(neuron_ids=tuple(ids), rate_hz=rate, start_ms=start, end_ms=end,
                            label="explicit_ids")
        name = parts[0]
        if not name:
            raise ValueError("population name is empty")
        rate = float(parts[1]) if len(parts) > 1 and parts[1] else None
        start = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        end = float(parts[3]) if len(parts) > 3 and parts[3] else math.inf
        pop = registry.get(name)
        return Stimulus(
            neuron_ids=pop.ids,
            rate_hz=rate if rate is not None else 150.0,
            start_ms=start,
            end_ms=end,
            label=name,
        )
    except UnknownPopulationError:
        raise
    except (ValueError, IndexError) as exc:
        raise FlyBrainError(
            f"could not parse --stimulus {spec!r}: {exc}. "
            f"Expected sugar_grn[:rate[:start[:end]]] or ids:<id,id,...>:rate[:start[:end]]"
        ) from exc


def parse_silence_spec(spec: str, registry: PopulationRegistry) -> Dict[str, Any]:
    """``--silence`` accepts a population name or a bare FlyWire ID."""
    s = spec.strip()
    if s.isdigit() and len(s) > 10:
        return {"flywire_id": int(s)}
    return {"population": registry.get(s).name}


def _apply_record_flags(args, mode: str) -> RecordingConfig:
    """Translate ``--record`` values into a recording plan.

    Default is population rates only. Spike tables are produced **only** when
    requested, because a whole-brain spike dump is unbounded output.
    """
    values = [v.strip().lower() for v in (args.record or [])]
    if not values:
        values = ["population"]
    if "all" in values:
        values = ["population", "spikes", "voltage"]
    if "none" in values:
        values = []

    record_spikes = "spikes" in values
    if args.record_spikes is not None:
        record_spikes = bool(args.record_spikes)

    return RecordingConfig(
        record_spikes=record_spikes,
        spike_neuron_ids=tuple(args.record_spikes_neurons or ()),
        record_population="population" in values,
        population_names=tuple(args.record_populations or ()) or _default_panel(args),
        population_interval_ms=args.population_interval_ms,
        record_voltage_ids=tuple(args.record_voltage_neurons or ()),
        voltage_interval_ms=args.voltage_interval_ms,
    )


def _default_panel(args) -> tuple:
    """Which populations to record by default: the stimulated ones plus known readouts."""
    names: List[str] = []
    for spec in args.stimulus or []:
        head = spec.split(":")[0]
        if head and head != "ids":
            names.append(head)
    for extra in ("mn9",):
        if extra not in names:
            names.append(extra)
    return tuple(names) if names else ()


def _print_banner(conn, hw, backend, cfg: RunConfig, *, extra: Optional[Dict[str, Any]] = None) -> None:
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


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------


def cmd_info(args) -> int:
    from .runtime.backend import available_ram_gb

    hw = detect_hardware()
    backend = get_backend(args.backend)
    print(f"FlyBrain v{__version__}")
    print()
    print("Hardware and capability")
    for line in hw.summary_lines():
        print(f"  {line}")
    print(f"  available RAM now: {available_ram_gb()} GB")
    print(f"  backends available: {'numpy' + (', jax' if hw.jax_installed else '')}")
    print()
    print(f"Selected backend: {backend.name}  [{'GPU' if getattr(backend, 'is_gpu', False) else 'CPU'}]")
    print()
    print("Paths")
    for label, p in (
        ("project root", fb_config.PROJECT_ROOT),
        ("third_party (upstream clones)", fb_config.THIRD_PARTY_DIR),
        ("data/raw", fb_config.RAW_DIR),
        ("data/processed", fb_config.PROCESSED_DIR),
        ("data/metadata", fb_config.METADATA_DIR),
        ("outputs/runs", fb_config.RUNS_DIR),
    ):
        print(f"  {label:<30} {p}")
    print()
    print("Code revision")
    rev = git_commit()
    if rev.get("commit"):
        print(f"  {rev['commit'][:12]} on {rev.get('branch')} (dirty={rev.get('dirty')})")
    else:
        print(f"  not under version control: {rev.get('error')}")
    print()
    print("Dataset registry")
    cards = {c.dataset_id: c for c in list_dataset_cards(fb_config.METADATA_DIR)}
    for ds in DATASET_SOURCES:
        card = cards.get(ds)
        status = "staged" if card else "not staged"
        extra = f" sha256={card.sha256[:12]}" if card else ""
        print(f"  {ds:<14} {status}{extra}")
    return 0


def cmd_datasets(args) -> int:
    fb_config.ensure_dirs()

    if args.stage:
        targets = [args.stage] if args.stage != "all" else list(DATASET_SOURCES)
        src = Path(args.from_dir) if args.from_dir else DEFAULT_UPSTREAM / "Drosophila_brain_model"
        # version 630 is also vendored by the chaobrain reproduction; prefer the
        # original repository for both, since it ships both versions.
        for ds in targets:
            spec = DATASET_SOURCES[ds]
            source_dir = src
            if not (source_dir / spec["neuron_table"]).is_file():
                alt = DEFAULT_UPSTREAM / "drosophila_whole_brain_snn_simulation"
                if (alt / spec["neuron_table"]).is_file():
                    source_dir = alt
            print(f"staging {ds} from {source_dir} ...")
            card = stage_dataset(ds, source_dir, fb_config.RAW_DIR, fb_config.METADATA_DIR,
                                 force=args.force)
            print(f"  card sha256 = {card.sha256}")
            for f in card.files:
                print(f"  {f.role:<20} {f.name}  {f.size_bytes / 1e6:.2f} MB")
        print()
        print("Now validating content ...")
        for ds in targets:
            conn = load_connectome(
                ds, fb_config.RAW_DIR, fb_config.PROCESSED_DIR, fb_config.METADATA_DIR
            )
            rep = validate_connectome(conn)
            rep.raise_if_failed()
            # Staging vouched for bytes; only reading the files can fill in the counts,
            # so the card is completed here rather than left at zero.
            update_dataset_card(
                ds, fb_config.METADATA_DIR,
                neuron_count=conn.n_neurons,
                edge_count=conn.n_edges,
                extra={
                    "synapses_total": conn.synapse_count,
                    "excitatory_neurons": rep.statistics.get("excitatory_neurons"),
                    "inhibitory_neurons": rep.statistics.get("inhibitory_neurons"),
                    "neurons_unknown_sign": rep.statistics.get("neurons_unknown_sign"),
                    "validation_failures": len(rep.failures),
                    "validation_warnings": len(rep.warnings),
                },
            )
            print(f"  {ds}: {conn.n_neurons:,} neurons, {conn.n_edges:,} edges, "
                  f"{len(rep.failures)} failures, {len(rep.warnings)} warnings")
        return 0

    cards = list_dataset_cards(fb_config.METADATA_DIR)
    if not cards:
        print("no datasets staged. Run: flybrain datasets --stage all")
        return 0

    for card in cards:
        print(f"{card.dataset_id}")
        print(f"  source        {card.source}")
        print(f"  release       {card.release}")
        print(f"  retrieved_at  {card.retrieved_at}")
        print(f"  sha256        {card.sha256}")
        for f in card.files:
            present = (fb_config.RAW_DIR / card.dataset_id / f.name).is_file()
            print(f"  file          {f.name}  {f.size_bytes / 1e6:.2f} MB  present={present}")
        if args.verify:
            print("  verifying content ...")
            # The verification path reads every column and checks the ID<->index
            # mapping. That is the expensive, thorough check; the load path stays lean.
            conn = _load_connectome_deep(card.dataset_id)
            rep = validate_connectome(conn)
            idcheck = check_id_index_consistency(conn, conn.raw_columns)
            rep.add(idcheck.name, idcheck.status, idcheck.detail, idcheck.value)
            update_dataset_card(
                card.dataset_id, fb_config.METADATA_DIR,
                neuron_count=conn.n_neurons,
                edge_count=conn.n_edges,
                extra={
                    "synapses_total": conn.synapse_count,
                    "excitatory_neurons": rep.statistics.get("excitatory_neurons"),
                    "inhibitory_neurons": rep.statistics.get("inhibitory_neurons"),
                    "neurons_unknown_sign": rep.statistics.get("neurons_unknown_sign"),
                    "validation_failures": len(rep.failures),
                    "validation_warnings": len(rep.warnings),
                    "id_index_consistency": idcheck.status,
                },
            )
            d = rep.to_dict()
            print(f"    ID<->index consistency: [{idcheck.status}] {idcheck.detail}")
            update_dataset_card(
                card.dataset_id, fb_config.METADATA_DIR,
                neuron_count=conn.n_neurons,
                edge_count=conn.n_edges,
                extra={
                    "synapses_total": conn.synapse_count,
                    "excitatory_neurons": rep.statistics.get("excitatory_neurons"),
                    "inhibitory_neurons": rep.statistics.get("inhibitory_neurons"),
                    "neurons_unknown_sign": rep.statistics.get("neurons_unknown_sign"),
                    "validation_failures": len(rep.failures),
                    "validation_warnings": len(rep.warnings),
                },
            )
            d = rep.to_dict()
            print(f"    neurons={d['neurons']} connections={d['connections']} "
                  f"excitatory_neurons={d['excitatory_neurons']} "
                  f"inhibitory_neurons={d['inhibitory_neurons']} "
                  f"unknown_sign={d['unknown_sign']} invalid_edges={d['invalid_edges']} "
                  f"status={d['status']}")
            for w in rep.warnings:
                print(f"    [WARN] {w.name}: {w.detail}")
        print()

    if args.report:
        conn = _load_connectome_deep(args.report)
        rep = validate_connectome(conn)
        idcheck = check_id_index_consistency(conn, conn.raw_columns)
        rep.add(idcheck.name, idcheck.status, idcheck.detail, idcheck.value)
        out = fb_config.OUTPUTS_DIR / "dataset_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        print(f"ID<->index consistency: [{idcheck.status}] {idcheck.detail}")
        update_dataset_card(
            args.report, fb_config.METADATA_DIR,
            neuron_count=conn.n_neurons,
            edge_count=conn.n_edges,
            extra={
                "synapses_total": conn.synapse_count,
                "excitatory_neurons": rep.statistics.get("excitatory_neurons"),
                "inhibitory_neurons": rep.statistics.get("inhibitory_neurons"),
                "neurons_unknown_sign": rep.statistics.get("neurons_unknown_sign"),
                "validation_failures": len(rep.failures),
                "validation_warnings": len(rep.warnings),
                "id_index_consistency": idcheck.status,
            },
        )
        rep.raise_if_failed()
    return 0


def cmd_populations(args) -> int:
    reg = _registry()
    if args.search:
        found = reg.find(args.search)
        if not found:
            print(f"no population matches {args.search!r}")
            return 0
        for p in found:
            print(f"{p.name:<20} n={len(p):<5} {p.description}")
        return 0

    if args.show:
        p = reg.get(args.show)
        conn = _load_connectome(args.dataset, quiet=True)
        present, missing = reg.resolve(p.name, conn.flywire_ids)
        print(f"{p.name}")
        print(f"  description  {p.description}")
        print(f"  upstream     {p.upstream_symbol}")
        print(f"  declared     {len(p)} neurons")
        print(f"  present      {present.size} in {conn.dataset_id}")
        if missing.size:
            print(f"  missing      {missing.size}: {missing[:10].tolist()}"
                  + (" ..." if missing.size > 10 else ""))
        print(f"  ids          {list(p.ids)[:8]}{' ...' if len(p) > 8 else ''}")
        return 0

    print(f"{len(reg)} curated populations (source: upstream notebooks, see flybrain/model/populations.json)")
    print()
    for p in reg:
        print(f"  {p.name:<20} n={len(p):<5} {p.description}")
    return 0


def cmd_neuron(args) -> int:
    from .analysis import neuron_profile
    from .analysis.connectivity import neighbours

    conn = _load_connectome(args.dataset, quiet=True)
    prof = neuron_profile(conn, args.flywire_id, registry=_registry())
    print(f"FlyWire ID: {prof['flywire_id']}")
    print(f"index in {conn.dataset_id}: {prof['index']}")
    print(f"neuron type: {prof['neuron_type'] if prof['neuron_type'] else 'unavailable'}")
    print(f"  ({prof['neuron_type_note']})")
    print(f"neurotransmitter sign: {prof['neurotransmitter_sign']}")
    print(f"incoming: {prof['in_degree']} connections, {prof['in_synapse_count']} synapses")
    print(f"outgoing: {prof['out_degree']} connections, {prof['out_synapse_count']} synapses")
    if prof["populations"]:
        print(f"curated populations: {', '.join(prof['populations'])}")
    if args.limit:
        for direction in ("out", "in"):
            nb = neighbours(conn, args.flywire_id, direction=direction, limit=args.limit)
            print()
            print(f"strongest {direction}going connections:")
            for c in nb["connections"]:
                print(f"  {c['direction']:<8} {c['flywire_id']:<20} {c['synapses']:>5} synapses  {c['sign']}")
    return 0


def cmd_run(args) -> int:
    import dataclasses

    fb_config.ensure_dirs()
    reg = _registry()
    cfg = _run_config_from_args(args)

    stimuli: List[Stimulus] = [parse_stimulus_spec(s, reg) for s in (args.stimulus or [])]
    silence_pops: List[str] = []
    silence_ids: List[int] = []
    for spec in args.silence or []:
        parsed = parse_silence_spec(spec, reg)
        if "population" in parsed:
            silence_pops.append(parsed["population"])
        else:
            silence_ids.append(parsed["flywire_id"])
    # RunConfig is frozen, so interventions are applied by replacement rather than
    # by mutation: a configuration object must not change identity under a caller.
    cfg = dataclasses.replace(
        cfg, silence_populations=silence_pops, silence=silence_ids
    )
    cfg.validate()

    conn = _load_connectome(args.dataset)
    backend = get_backend(cfg.backend, cfg.dtype)
    hw = detect_hardware()

    _print_banner(conn, hw, backend, cfg)

    plan = plan_run(conn, cfg, stimuli, hardware=hw)
    print(f"Plan: {plan.label()}")
    for note in plan.notes:
        print(f"  - {note}")
    print()

    if plan.mode_changed:
        print(
            "!! NOTE: a whole-brain run was requested and NOT delivered. The run below is a "
            "SUBSET and must not be reported as whole-brain.",
            file=sys.stderr,
        )

    if args.estimate_only:
        print(json.dumps(plan.estimate, indent=2))
        return 0

    recording = _apply_record_flags(args, plan.effective_mode)
    run_dir = (Path(args.out_dir) / args.run_id) if (args.out_dir and args.run_id) else None

    result = run_experiment(
        plan, cfg, stimuli,
        registry=reg, backend=backend, hardware=hw,
        recording=recording, run_dir=run_dir,
        echo=None if args.quiet else (lambda line: print(line)),
        progress_every=args.progress_every,
    )

    _print_run_summary(result, args)
    plots = result.summary.get("plots") or []
    for name in plots:
        print(f"  plot              {result.run_dir / 'plots' / name}")
    if result.summary.get("plots_error"):
        print(f"  (plots skipped: {result.summary['plots_error']})", file=sys.stderr)
    return 0


def _run_config_from_args(args) -> RunConfig:
    return RunConfig(
        dataset_id=args.dataset,
        mode=args.mode,
        duration_ms=args.duration_ms,
        dt_ms=args.dt_ms,
        trials=args.trials,
        seed=args.seed,
        backend=args.backend,
        dtype=args.dtype,
        allow_fallback=bool(args.allow_fallback),
        max_estimated_minutes=None if args.max_minutes < 0 else args.max_minutes,
        subset_seed_ids=list(args.subset_seed or []),
        subset_hops=args.subset_hops,
        subset_max_neurons=args.subset_max_neurons,
        record_population_interval_ms=args.population_interval_ms,
        record_voltage_interval_ms=args.voltage_interval_ms,
        disable_recurrence=bool(args.disable_recurrence),
        run_id=args.run_id,
        quiet_log=bool(args.quiet),
        make_plots=not bool(args.no_plots),
    )


def _print_run_summary(result, args) -> None:
    s = result.summary
    print()
    print("=" * 72)
    print(f"Run {result.run_id}")
    print(f"  mode              {result.mode}{' (SUBSET - NOT whole-brain)' if result.is_subset else ''}")
    print(f"  neurons / edges   {result.n_neurons:,} / {result.n_edges:,}")
    print(f"  duration          {result.duration_ms:g} ms x {result.trials} trial(s) @ dt={result.dt_ms} ms")
    print(f"  backend           {result.backend} ({'GPU' if result.backend_is_gpu else 'CPU'})")
    print(f"  stimulus          {', '.join(x.describe() for x in _stimuli_of(s)) or 'none'}")
    print(f"  silenced          {len(s.get('silenced_flywire_ids', []))} neuron(s)")
    print()
    print(f"  total spikes      {s['total_spikes']:,}")
    print(f"  active neurons    {s['active_neurons_any_spike']:,} "
          f"(mean {s['active_neurons_mean_per_trial']:.1f} per trial)")
    print(f"  mean rate (all)   {s['mean_rate_hz_all_neurons']:.4f} Hz")
    print(f"  mean rate (active) {s['mean_rate_hz_active_neurons']:.2f} Hz")
    print(f"  max rate          {s['max_rate_hz']:.1f} Hz")
    if s["timing"]["measured_step_ms_mean"]:
        print(f"  measured cost     {s['timing']['measured_step_ms_mean']:.2f} ms/step, "
              f"{s['timing']['measured_steps_per_second']:.1f} steps/s")
    print(f"  spike digest      {s['spike_digest_sha256'][:16]}")
    print()
    nz = {k: v for k, v in s["population_rates_hz"].items() if v > 0}
    if nz:
        print("  population rates [Hz] (nonzero):")
        for k, v in sorted(nz.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {k:<22} {v:8.2f}")
    print()
    print(f"  outputs           {result.run_dir}")
    print("=" * 72)


def _stimuli_of(summary: Dict[str, Any]) -> List[Stimulus]:
    from .runtime.runner import plan_stimuli

    return plan_stimuli(summary)


def cmd_experiments(args) -> int:
    from .experiments import EXPERIMENTS, run_named

    if args.list or not args.name:
        print("available experiments:")
        for name, (fn, doc) in sorted(EXPERIMENTS.items()):
            print(f"  {name:<22} {doc}")
        return 0
    if args.name not in EXPERIMENTS:
        print(f"unknown experiment {args.name!r}; try `flybrain experiments --list`", file=sys.stderr)
        return 2
    return run_named(args.name, argv=args.rest)


def cmd_interactive(args) -> int:
    from .repl import run_repl

    return run_repl(
        dataset_id=args.dataset,
        backend=args.backend,
        duration_ms=args.duration_ms,
        mode=args.mode,
        seed=args.seed,
    )


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="flybrain",
        description=(
            "FlyBrain v0 - executable Drosophila whole-brain connectome simulation on the "
            "FlyWire-derived leaky integrate-and-fire model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  flybrain datasets --stage all\n"
            "  flybrain run --mode subset --duration-ms 100 --stimulus sugar_grn\n"
            "  flybrain run --mode whole-brain --duration-ms 1000 --stimulus sugar_grn\n"
            "  flybrain run --mode subset --stimulus sugar_grn --duration-ms 1000 --record population\n"
            "  flybrain run --mode subset --stimulus sugar_grn --silence mn9 --duration-ms 1000\n"
            "  flybrain experiments sugar_stimulation\n"
            "  flybrain interactive\n"
        ),
    )
    p.add_argument("--version", action="version", version=f"FlyBrain {__version__}")
    sub = p.add_subparsers(dest="command")

    # -- info ---------------------------------------------------------------------
    s = sub.add_parser("info", help="print hardware, backend and path report")
    s.add_argument("--backend", default="auto", choices=["auto", "numpy", "jax"])
    s.set_defaults(func=cmd_info)

    # -- datasets -----------------------------------------------------------------
    s = sub.add_parser("datasets", help="list, stage and validate datasets")
    s.add_argument("--stage", nargs="?", const="all", default=None,
                   help="stage a dataset id (or all) into data/raw")
    s.add_argument("--from-dir", default=None, help="upstream clone to stage from")
    s.add_argument("--force", action="store_true", help="re-copy files even if present")
    s.add_argument("--verify", action="store_true", help="re-read and validate content")
    s.add_argument("--report", metavar="DATASET_ID", default=None,
                   help="validate and write outputs/dataset_report.json")
    s.set_defaults(func=cmd_datasets)

    # -- populations --------------------------------------------------------------
    s = sub.add_parser("populations", help="list or inspect curated neuron populations")
    s.add_argument("--search", default=None, help="substring search")
    s.add_argument("--show", default=None, help="show one population in detail")
    s.add_argument("--dataset", default="flywire_630")
    s.set_defaults(func=cmd_populations)

    # -- neuron -------------------------------------------------------------------
    s = sub.add_parser("neuron", help="inspect one neuron by FlyWire ID")
    s.add_argument("flywire_id", type=int)
    s.add_argument("--dataset", default="flywire_630")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_neuron)

    # -- run ----------------------------------------------------------------------
    s = sub.add_parser("run", help="run a simulation")
    s.add_argument("--mode", default="subset", choices=["subset", "whole-brain"])
    s.add_argument("--dataset", default="flywire_630", choices=sorted(DATASET_SOURCES))
    s.add_argument("--duration-ms", type=float, default=fb_config.DEFAULT_DURATION_MS)
    s.add_argument("--dt-ms", type=float, default=fb_config.DEFAULT_DT_MS)
    s.add_argument("--trials", type=int, default=fb_config.DEFAULT_TRIALS)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--backend", default="auto", choices=["auto", "numpy", "jax"])
    s.add_argument("--dtype", default="float32", choices=["float32"])
    s.add_argument("--stimulus", action="append", default=None, metavar="SPEC",
                   help="sugar_grn[:rate[:start_ms[:end_ms]]] or ids:<id,id,...>:rate[:start[:end]]")
    s.add_argument("--silence", action="append", default=None, metavar="POP_OR_ID")
    s.add_argument("--disable-recurrence", action="store_true",
                   help="break all recurrent input (silent-network control)")
    s.add_argument("--subset-seed", action="append", type=int, default=None,
                   help="FlyWire ID to seed the subset with (repeatable)")
    s.add_argument("--subset-hops", type=int, default=2)
    s.add_argument("--subset-max-neurons", type=int, default=5000)
    s.add_argument("--allow-fallback", action="store_true",
                   help="if whole-brain is infeasible, run a labelled subset instead of refusing")
    s.add_argument("--max-minutes", type=float, default=60.0,
                   help="refuse if the estimated runtime exceeds this; negative disables")
    s.add_argument("--record", action="append", default=None,
                   choices=["population", "spikes", "voltage", "all", "none"],
                   help="recording channels (default: population)")
    s.add_argument("--record-spikes", dest="record_spikes", action="store_true", default=None)
    s.add_argument("--record-spikes-neurons", action="append", type=int, default=None,
                   help="restrict spike recording to these FlyWire IDs")
    s.add_argument("--record-populations", action="append", default=None)
    s.add_argument("--record-voltage-neurons", action="append", type=int, default=None)
    s.add_argument("--population-interval-ms", type=float, default=10.0)
    s.add_argument("--voltage-interval-ms", type=float, default=1.0)
    s.add_argument("--run-id", default=None)
    s.add_argument("--out-dir", default=None, help="parent directory for the run (with --run-id)")
    s.add_argument("--estimate-only", action="store_true",
                   help="print the requirements estimate and exit without simulating")
    s.add_argument("--no-plots", action="store_true")
    s.add_argument("--progress-every", type=int, default=0,
                   help="log every N steps (0 = only per trial)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_run)

    # -- experiments ---------------------------------------------------------------
    s = sub.add_parser("experiments", help="run a packaged experiment")
    s.add_argument("name", nargs="?", default=None)
    s.add_argument("--list", action="store_true")
    s.add_argument("rest", nargs=argparse.REMAINDER,
                   help="arguments forwarded to the experiment")
    s.set_defaults(func=cmd_experiments)

    # -- interactive ---------------------------------------------------------------
    s = sub.add_parser("interactive", help="interactive experiment console")
    s.add_argument("--dataset", default="flywire_630")
    s.add_argument("--backend", default="auto", choices=["auto", "numpy", "jax"])
    s.add_argument("--mode", default="subset", choices=["subset", "whole-brain"])
    s.add_argument("--duration-ms", type=float, default=200.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_interactive)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args) or 0)
    except ResourceLimitError as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        if exc.estimate:
            print("\nEstimated requirements:", file=sys.stderr)
            print(json.dumps(exc.estimate, indent=2), file=sys.stderr)
        print(
            "\nNo simulation was run. This is a refusal, not a silent fallback. "
            "Use --mode subset, or --allow-fallback to get an explicitly labelled subset run.",
            file=sys.stderr,
        )
        return 3
    except (DatasetError, DatasetValidationError) as exc:
        print(f"DATASET ERROR: {exc}", file=sys.stderr)
        return 4
    except UnknownPopulationError as exc:
        print(f"UNKNOWN POPULATION: {exc}", file=sys.stderr)
        return 5
    except FlyBrainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
