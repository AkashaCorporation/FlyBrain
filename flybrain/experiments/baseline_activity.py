"""Experiment B - baseline activity with no stimulation.

Purpose: measure the model's spontaneous activity. This matters more than it
sounds, because the upstream model has no noise source at all: with no stimulus
the network is *exactly* at its fixed point and produces zero spikes. Establishing
that quantitatively is what makes every later claim of the form "this population
responded *because* we stimulated X" meaningful - if the network were buzzing on
its own, the downstream responses would be uninterpretable.

A second condition is measured for the same reason: ``--disable-recurrence`` breaks
all recurrent input, so comparing it against the intact network quantifies how much
of the activity is driven by the stimulus directly versus by network reverberation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


from ..model.populations import PopulationRegistry
from ..runtime import RecordingConfig, detect_hardware, get_backend, plan_run
from .common import (
    announce,
    base_parser,
    build_run_config,
    check,
    default_seed_ids,
    execute,
    experiment_dir,
    load_dataset,
    print_checks,
    write_report,
)

PANEL = ("sugar_grn", "mn9", "bitter_grn", "water_grn")


def run(argv: Optional[Sequence[str]] = None) -> int:
    p = base_parser("FlyBrain baseline: spontaneous activity with no stimulation")
    p.add_argument("--with-recurrence-control", action="store_true",
                   help="also run with all recurrent input disabled, for comparison")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = experiment_dir("baseline_activity", args.out_dir)
    registry = PopulationRegistry.builtin()
    cfg = build_run_config(args)
    conn = load_dataset(args.dataset)
    hw = detect_hardware()
    backend = get_backend(cfg.backend, cfg.dtype)

    announce(conn, hw, backend, cfg, {"Stimulus": "none (baseline)"})

    conditions: Dict[str, Dict[str, float]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    checks: List[Dict[str, Any]] = []

    def measure(name: str, *, disable_recurrence: bool) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {"disable_recurrence": disable_recurrence}
        if args.mode == "subset" and not (args.subset_seed or []):
            # "no stimulus" still has to name a network. Seeding from the sugar
            # circuit and the MN9 readout makes this condition comparable to the
            # stimulation experiments: same working set, only the stimulus differs.
            overrides["subset_seed_ids"] = default_seed_ids(conn, registry)
        c = build_run_config(args, **overrides)
        plan = plan_run(conn, c, [], hardware=hw)
        print(f"--- condition {name!r}: {plan.label()}")
        for note in plan.notes:
            print(f"    {note}")
        res = execute(
            plan, c, [], registry=registry, backend=backend, hardware=hw,
            recording=RecordingConfig(
                record_spikes=True, record_population=True, population_names=PANEL,
                population_interval_ms=cfg.record_population_interval_ms,
            ),
            run_dir=out_dir / f"run-{name}",
        )
        conditions[name] = dict(res.population_rates)
        summaries[name] = res.summary
        return res.summary

    intact = measure("baseline", disable_recurrence=False)
    checks.append(check(
        intact["total_spikes"] == 0,
        "intact network is silent with no stimulus",
        f"{intact['total_spikes']} spikes over {intact['duration_ms']} ms, "
        f"{intact['n_neurons']:,} neurons. The model has no intrinsic noise term, so this "
        f"is the expected outcome, not a bug.",
    ))
    checks.append(check(
        intact["active_neurons_any_spike"] == 0,
        "no neuron is active without stimulation",
        f"{intact['active_neurons_any_spike']} active neurons",
    ))

    if args.with_recurrence_control:
        broken = measure("no-recurrence", disable_recurrence=True)
        checks.append(check(
            broken["total_spikes"] == 0,
            "disabling recurrence does not create activity",
            f"{broken['total_spikes']} spikes (silencing recurrent input can only remove "
            f"activity, never add it)",
        ))
        checks.append(check(
            intact["total_spikes"] >= broken["total_spikes"],
            "intact network has at least as much activity as the recurrence-free control",
            f"intact={intact['total_spikes']} vs no-recurrence={broken['total_spikes']}",
        ))

    ok = print_checks(checks)
    write_report(out_dir, "baseline_activity", {
        "experiment": "baseline_activity",
        "interpretation": (
            "With no stimulus the model is exactly quiescent: it has no stochastic drive "
            "and no intrinsic noise. Any activity observed in other experiments is therefore "
            "attributable to the declared stimulus, which is what makes the causal claims "
            "in the silencing experiment meaningful."
        ),
        "conditions": {k: {"total_spikes": v["total_spikes"],
                          "active_neurons": v["active_neurons_any_spike"]}
                       for k, v in summaries.items()},
        "model_params": intact["model_params"],
        "checks": checks,
        "passed": ok,
        "summaries": summaries,
    })
    print(f"\nreport: {out_dir / 'baseline_activity.json'}")
    print("BASELINE " + ("QUIESCENT AS EXPECTED" if ok else "UNEXPECTED"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
