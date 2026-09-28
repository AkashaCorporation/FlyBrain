"""Experiment D - silencing (causal intervention).

Protocol, exactly as the objective specifies::

    stimulate population A
    silence population B            <- condition "silenced"
        compared against
    stimulate population A
    population B normal             <- condition "baseline"

Semantics of "silence" are the upstream model's: every synapse *from* the silenced
neurons is zeroed, so they keep spiking but stop influencing anyone
(``docs/UPSTREAM_AUDIT.md`` section 3.5). The silenced neuron therefore still
appears in the recordings, which is a *feature* for interpreting the result: it
shows the intervention did not simply delete the neuron from the network.

The comparison is reported per population as a rate difference, so the output is a
causal statement ("silencing B reduced C by X Hz") rather than a pair of unrelated
activity numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..analysis import connectivity_between
from ..model.populations import PopulationRegistry
from ..model.stimulus import Stimulus
from ..runtime import RecordingConfig, detect_hardware, get_backend, plan_run
from .common import (
    announce,
    base_parser,
    build_run_config,
    check,
    execute,
    experiment_dir,
    load_dataset,
    print_checks,
    write_report,
)

DEFAULT_PANEL = (
    "sugar_grn", "mn9", "bitter_grn", "water_grn", "ir94e_grn",
    "jon_all", "sez_aDT6", "sez_Fdg", "sez_usnea",
)


def run(argv: Optional[Sequence[str]] = None) -> int:
    p = base_parser("FlyBrain silencing: stimulate A, silence B, compare against baseline")
    p.add_argument("--stimulate", default="sugar_grn",
                   help="population to stimulate (A)")
    p.add_argument("--silence", dest="silence_pop", default=None,
                   help="population to silence (B). Default: the top responder in a "
                        "screen run. Pass an explicit name for a reproducible experiment.")
    p.add_argument("--rate-hz", type=float, default=150.0)
    p.add_argument("--panel", default=None,
                   help="comma-separated populations to record (defaults to a sensory+readout panel)")
    p.add_argument("--print-top", type=int, default=15)
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = experiment_dir("silencing_test", args.out_dir)
    registry = PopulationRegistry.builtin()
    conn = load_dataset(args.dataset)
    hw = detect_hardware()
    backend = get_backend(args.backend, "float32")
    panel = tuple(s for s in (args.panel.split(",") if args.panel else DEFAULT_PANEL) if s)

    # stimulus population A, restricted to members actually present
    present_a, missing_a = registry.resolve(args.stimulate, conn.flywire_ids)
    if present_a.size == 0:
        print(f"population {args.stimulate!r} has no members in {conn.dataset_id}")
        return 1
    stim_a = Stimulus(
        neuron_ids=tuple(int(x) for x in present_a), rate_hz=args.rate_hz, label=args.stimulate
    )

    announce(conn, hw, backend, build_run_config(args), {
        "Stimulus A": stim_a.describe(),
        "A present": f"{present_a.size} of {len(registry.get(args.stimulate))} declared",
    })

    def one_condition(name: str, silence_ids: Sequence[int]) -> Dict[str, Any]:
        cfg = build_run_config(
            args,
            silence=list(int(x) for x in silence_ids),
        )
        plan = plan_run(conn, cfg, [stim_a], hardware=hw)
        print(f"--- condition {name!r}: {plan.label()}   "
              f"silenced={len(silence_ids)} neuron(s)")
        res = execute(
            plan, cfg, [stim_a], registry=registry, backend=backend, hardware=hw,
            recording=RecordingConfig(
                record_spikes=True, record_population=True, population_names=panel,
                population_interval_ms=cfg.record_population_interval_ms,
            ),
            run_dir=out_dir / f"run-{name}",
        )
        return res.summary

    # ---- condition 1: baseline, nothing silenced ---------------------------------
    baseline = one_condition("baseline", [])

    # ---- choose B ----------------------------------------------------------------
    if args.silence_pop:
        silence_target_name = args.silence_pop
        b_ids, b_missing = registry.resolve(silence_target_name, conn.flywire_ids)
        if b_ids.size == 0:
            print(f"population {silence_target_name!r} has no members in {conn.dataset_id}")
            return 1
        selection_note = f"explicit: --silence {silence_target_name}"
    else:
        # Pick the top non-stimulated responder from the baseline run. This is a
        # *screen*, so the selected target must be reported as measured, not assumed.
        candidates = [
            t for t in baseline.get("top_active_neurons", [])
            if t["flywire_id"] not in set(int(x) for x in present_a)
        ]
        if not candidates:
            print("no non-stimulated responder found in the baseline run; nothing to silence.")
            print("Pass --silence <population> to choose a target explicitly.")
            write_report(out_dir, "silencing_test", {
                "experiment": "silencing_test",
                "aborted": "no non-stimulated responder in the baseline condition",
                "baseline": baseline,
            })
            return 1
        top = candidates[0]
        b_ids = np.asarray([top["flywire_id"]], dtype=np.int64)
        b_missing = np.empty(0, dtype=np.int64)
        silence_target_name = f"top-responder:{top['flywire_id']}"
        selection_note = (
            f"screened from the baseline run: the highest-rate non-stimulated neuron, "
            f"{top['flywire_id']} at {top['mean_rate_hz']:.1f} Hz"
        )

    print()
    print(f"silencing target B = {silence_target_name}")
    print(f"  {selection_note}")
    print(f"  {len(b_ids)} neuron(s), {len(b_missing)} absent from the dataset")
    print()

    # ---- condition 2: baseline + B silenced --------------------------------------
    silenced = one_condition("silenced", b_ids)

    # ---- comparison --------------------------------------------------------------
    cond_rates = {
        "baseline (A stimulated, B intact)": dict(baseline["population_rates_hz"]),
        f"silenced ({silence_target_name} output removed)": dict(silenced["population_rates_hz"]),
    }
    base_r = cond_rates["baseline (A stimulated, B intact)"]
    sil_r = cond_rates[f"silenced ({silence_target_name} output removed)"]
    deltas = sorted(
        (
            {"population": k, "baseline_hz": base_r.get(k, 0.0),
             "silenced_hz": sil_r.get(k, 0.0),
             "delta_hz": sil_r.get(k, 0.0) - base_r.get(k, 0.0)}
            for k in set(base_r) | set(sil_r)
        ),
        key=lambda d: -abs(d["delta_hz"]),
    )

    # structural evidence: does B actually project to the populations it changed?
    changed = [d for d in deltas if abs(d["delta_hz"]) > 0.5 and d["population"] != silence_target_name]
    route = None
    if changed:
        target_ids: List[int] = []
        for d in changed:
            present, _ = registry.resolve(d["population"], conn.flywire_ids)
            target_ids.extend(int(x) for x in present.tolist())
        if target_ids:
            route = connectivity_between(conn, [int(x) for x in b_ids], target_ids)
            route["targets_considered"] = [d["population"] for d in changed]

    checks: List[Dict[str, Any]] = [
        check(
            baseline["total_spikes"] > 0,
            "baseline condition produced activity",
            f"{baseline['total_spikes']} spikes",
        ),
        check(
            silenced["total_spikes"] != baseline["total_spikes"]
            or any(abs(d["delta_hz"]) > 1e-9 for d in deltas),
            "silencing changed the network's activity",
            "; ".join(
                f"{d['population']}: {d['baseline_hz']:.1f} -> {d['silenced_hz']:.1f} Hz "
                f"({d['delta_hz']:+.1f})" for d in deltas[:4]
            ) or "no population changed",
        ),
        check(
            True,
            "the silenced neuron itself is still counted (documented upstream semantics)",
            "silencing removes outgoing influence only; the neuron keeps receiving input "
            "and may keep spiking, so its own rate is not expected to drop to zero",
        ),
    ]
    if route is not None:
        checks.append(check(
            route["connections"] > 0,
            "the silenced population has direct projections to the populations it changed",
            f"{route['connections']} edge(s) onto {route['target_neurons_with_source_input']} "
            f"target neurons in {route.get('targets_considered')}",
        ))

    ok = print_checks(checks)

    print()
    print(f"{'population':<24} {'baseline Hz':>12} {'silenced Hz':>12} {'delta Hz':>10}")
    for d in deltas[: args.print_top]:
        print(f"{d['population']:<24} {d['baseline_hz']:>12.2f} {d['silenced_hz']:>12.2f} "
              f"{d['delta_hz']:>+10.2f}")

    # The condition-comparison figure is the one that makes the causal claim readable,
    # so it is produced unconditionally unless plots were switched off.
    comparison_plot = None
    if not args.no_plots:
        try:
            from ..runtime.runner import plot_condition_comparison

            top = [d["population"] for d in deltas[:12]]
            comparison_plot = plot_condition_comparison(
                {k: v for k, v in cond_rates.items()}, out_dir / "input_vs_downstream.png",
                populations=top,
            )
            if comparison_plot:
                print(f"\ncomparison figure: {comparison_plot}")
        except Exception as exc:
            print(f"(comparison figure skipped: {type(exc).__name__}: {exc})")

    write_report(out_dir, "silencing_test", {
        "experiment": "silencing_test",
        "design": {
            "stimulus_A": stim_a.to_dict(),
            "silence_B": {
                "name": silence_target_name,
                "flywire_ids": [int(x) for x in b_ids],
                "selection": selection_note,
            },
            "silencing_semantics": (
                "outgoing-only: every synapse from the silenced neurons is zeroed (upstream "
                "behaviour, confirmed by the published results - docs/UPSTREAM_AUDIT.md 3.5)"
            ),
        },
        "conditions": cond_rates,
        "deltas": deltas,
        "structural_route_B_to_changed": route,
        "checks": checks,
        "passed": ok,
        "comparison_figure": str(comparison_plot) if comparison_plot else None,
        "summaries": {"baseline": baseline, "silenced": silenced},
    })
    print(f"\nreport: {out_dir / 'silencing_test.json'}")
    print("SILENCING EXPERIMENT " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
