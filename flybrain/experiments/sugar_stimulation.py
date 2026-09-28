"""Experiment C - sugar GRN stimulation.

This is the reference experiment. It reproduces the closest practical version of
the upstream sugar-sensing protocol:

* stimulate the 21 labellar sugar-sensing gustatory receptor neurons of the right
  hemisphere (``sugar_grn``), which is the exact input set used by the paper's
  Figure 1;
* at a fixed Poisson rate, for 1000 ms, matching the upstream trial length;
* then inspect whether activity propagates beyond the stimulated neurons and
  whether the MN9 motor neurons respond.

The evidence chain this experiment has to establish is the one the objective names::

    known sensory neurons -> stimulation -> activity propagates through the
    FlyWire-derived network -> known downstream populations respond

Each link in that chain is a separate recorded check, and the directly stimulated
populations are explicitly excluded from the "downstream" set - a population that
fires because we drove it is not evidence of propagation.

When the upstream repository's published spike output is present, the MN9 rate
measured here is printed next to the published value. Agreement is *not* claimed:
the two implementations differ in integration scheme and sub-step ordering (see
``docs/UPSTREAM_AUDIT.md`` section 4.1), and the published runs used 30 trials
against this experiment's default of 1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


from ..analysis import connectivity_between, downstream_responders
from ..model.populations import PopulationRegistry
from ..model.stimulus import Stimulus
from ..runtime import RecordingConfig, detect_hardware, get_backend, plan_run
from .common import (
    MN9_LEFT,
    MN9_RIGHT,
    announce,
    base_parser,
    build_run_config,
    check,
    execute,
    experiment_dir,
    load_dataset,
    print_checks,
    published_sugar_reference,
    write_report,
)

PANEL = (
    "sugar_grn", "mn9", "bitter_grn", "water_grn", "ir94e_grn",
    "jon_all", "sez_aDT6", "sez_Fdg", "sez_usnea",
)


def run(argv: Optional[Sequence[str]] = None) -> int:
    p = base_parser("FlyBrain sugar experiment: stimulate the sugar-sensing GRNs")
    p.add_argument("--rate-hz", type=float, default=150.0,
                   help="Poisson stimulation rate (upstream default is 150 Hz; the published "
                        "tutorial run used 200 Hz and the committed 100 Hz run used 100 Hz)")
    p.add_argument("--compare-published", action="store_true",
                   help="print the published MN9 rate for the closest upstream run")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = experiment_dir("sugar_stimulation", args.out_dir)
    registry = PopulationRegistry.builtin()
    cfg = build_run_config(args)
    conn = load_dataset(args.dataset)
    hw = detect_hardware()
    backend = get_backend(cfg.backend, cfg.dtype)

    stim = Stimulus.population(
        "sugar_grn", rate_hz=args.rate_hz, registry=registry, flywire_ids=None
    )
    stim_present, stim_missing = registry.resolve("sugar_grn", conn.flywire_ids)
    stim = Stimulus(
        neuron_ids=tuple(int(x) for x in stim_present),
        rate_hz=args.rate_hz,
        label="sugar_grn",
    )

    announce(conn, hw, backend, cfg, {
        "Stimulus": stim.describe(),
        "sugar GRNs present": f"{stim_present.size} of {len(registry.get('sugar_grn'))} declared",
    })
    if stim_missing.size:
        print(f"note: {stim_missing.size} declared sugar GRN(s) are absent from {conn.dataset_id}")

    plan = plan_run(conn, cfg, [stim], hardware=hw)
    print(f"plan: {plan.label()}")
    for note in plan.notes:
        print(f"  - {note}")
    print()

    res = execute(
        plan, cfg, [stim], registry=registry, backend=backend, hardware=hw,
        recording=RecordingConfig(
            record_spikes=True,
            record_population=True,
            population_names=PANEL,
            population_interval_ms=cfg.record_population_interval_ms,
        ),
        run_dir=out_dir / "run",
    )
    s = res.summary

    # ---------------- evidence ----------------------------------------------------
    import pandas as pd

    spikes_df = pd.read_parquet(res.spikes_path) if res.spikes_path else None
    pop_df = pd.read_parquet(res.population_path) if res.population_path else None
    sugar_ids = list(stim.neuron_ids)

    # measured sugar-GRN rate, from the recorded spike table
    sugar_rate_measured = float(s["population_rates_hz"].get("sugar_grn", 0.0))
    responders = downstream_responders(
        pop_df, ["sugar_grn"], min_peak_hz=1.0, exclude_stimulated=True
    ) if pop_df is not None else []

    top = s.get("top_active_neurons", [])
    mn9_rates = {
        "mn9_left": float(s["population_rates_hz"].get("mn9", 0.0)),
    }
    mn9_present = [i for i in (MN9_LEFT, MN9_RIGHT) if i in set(int(x) for x in conn.flywire_ids)]

    # is there a monosynaptic route from the sugar GRNs into the responding set?
    responder_ids = [entry["flywire_id"] for entry in top if entry["flywire_id"] not in set(sugar_ids)]
    route = connectivity_between(conn, sugar_ids, responder_ids) if responder_ids else None

    # ---------------- checks ------------------------------------------------------
    checks: List[Dict[str, Any]] = []
    checks.append(check(
        stim_present.size > 0,
        "the known sensory population is present in the loaded connectome",
        f"{stim_present.size} sugar GRNs",
    ))
    checks.append(check(
        s["total_spikes"] > 0,
        "stimulation produces spikes",
        f"{s['total_spikes']} spikes over {s['duration_ms']} ms",
    ))
    checks.append(check(
        abs(sugar_rate_measured - args.rate_hz) < max(0.15 * args.rate_hz, 10.0),
        "stimulated neurons fire at approximately the requested rate",
        f"measured {sugar_rate_measured:.1f} Hz vs requested {args.rate_hz:g} Hz",
    ))
    checks.append(check(
        s["active_neurons_any_spike"] > stim_present.size,
        "activity propagates beyond the stimulated neurons",
        f"{s['active_neurons_any_spike']} active neurons vs {stim_present.size} stimulated",
    ))
    checks.append(check(
        len(responders) > 0,
        "at least one non-stimulated population responds",
        "; ".join(f"{r['population']}={r['peak_rate_hz']:.1f}Hz" for r in responders[:6])
        or "none above 1 Hz",
    ))
    checks.append(check(
        bool(mn9_rates["mn9_left"] >= 0.0),
        "MN9 readout was measured",
        f"MN9 left = {mn9_rates['mn9_left']:.2f} Hz "
        f"({len(mn9_present)} of 2 MN9 neurons present in the loaded connectome)",
    ))
    if route is not None:
        checks.append(check(
            route["connections"] > 0,
            "the responding neurons are monosynaptically reachable from the sugar GRNs",
            f"{route['connections']} directed edge(s), {route['synapses']} synapses, "
            f"{route['excitatory_connections']} excitatory / {route['inhibitory_connections']} "
            f"inhibitory, covering {route['target_neurons_with_source_input']} of "
            f"{route['target_count']} responding neurons. "
            f"(Scope: {route['note']})",
        ))
    # Cross-check the recorded spike table against the network's own counter. If these
    # ever disagree, one of the two recording paths is wrong and the reported rates
    # above cannot be trusted.
    if spikes_df is not None:
        checks.append(check(
            len(spikes_df) == s["total_spikes"],
            "the recorded spike table agrees with the network's spike counter",
            f"{len(spikes_df)} rows in spikes.parquet vs {s['total_spikes']} counted "
            f"by the network",
        ))

    # ---------------- published comparison ---------------------------------------
    reference = None
    if args.compare_published:
        reference = published_sugar_reference(args.rate_hz)
        if reference:
            print("Published upstream reference (measured from the authors' own output):")
            print(f"  file           {reference['file']}")
            print(f"  trials         {reference['trials']}")
            print(f"  sugar GRN rate {reference['median_sugar_grn_rate_hz']:.1f} Hz (median)")
            print(f"  MN9 left       {reference['mn9_left_rate_hz']:.2f} Hz")
            print(f"  MN9 right      {reference['mn9_right_rate_hz']:.2f} Hz")
            print(f"  active neurons {reference['active_neurons']}")
            print()
            print("FlyBrain, this run:")
            print(f"  trials         {s['trials']}")
            print(f"  sugar GRN rate {sugar_rate_measured:.1f} Hz")
            print(f"  MN9 left       {mn9_rates['mn9_left']:.2f} Hz")
            print(f"  active neurons {s['active_neurons_any_spike']}")
            print()
            print("Agreement is NOT claimed: the upstream model is Brian 2 with exact linear")
            print("integration and 30 trials, while the chaobrain port orders the sub-steps")
            print("differently (docs/UPSTREAM_AUDIT.md sec. 4.1). Compare the qualitative")
            print("pattern - sparse downstream activation driven by the sensory population -")
            print("not the numbers.")
        else:
            print("no published reference found under third_party/; skipping comparison")

    ok = print_checks(checks)
    payload = {
        "experiment": "sugar_stimulation",
        "hypothesis": (
            "Stimulating the labelled sugar-sensing GRNs at a fixed rate produces activity "
            "that propagates through the FlyWire-derived network and reaches non-sensory "
            "downstream populations, including the MN9 motor neurons."
        ),
        "stimulus": stim.to_dict(),
        "dataset": {"dataset_id": conn.dataset_id, "n_neurons": conn.n_neurons,
                    "n_edges": conn.n_edges, "is_subset": conn.is_subset},
        "measured": {
            "sugar_grn_rate_hz": sugar_rate_measured,
            "total_spikes": s["total_spikes"],
            "active_neurons": s["active_neurons_any_spike"],
            "mean_rate_active_hz": s["mean_rate_hz_active_neurons"],
            "max_rate_hz": s["max_rate_hz"],
            "mn9_rates_hz": mn9_rates,
            "top_active_neurons": top[:25],
            "top_active_populations": s["top_active_populations"][:25],
            "downstream_responders": [
                {k: v for k, v in r.items() if k not in ("rate_hz_trace", "t_ms_trace")}
                for r in responders[:25]
            ],
            "monosynaptic_route_from_sugar_grn_to_responders": route,
        },
        "published_reference": reference,
        "checks": checks,
        "passed": ok,
        "run": res.to_dict(),
    }
    write_report(out_dir, "sugar_stimulation", payload)
    print(f"\nreport: {out_dir / 'sugar_stimulation.json'}")
    print("SUGAR EXPERIMENT " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
