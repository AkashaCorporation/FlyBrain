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
from ..analysis.firing_rates import rate_table_from_spikes
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
    # Per-neuron MN9 rates. The published reference reports the LEFT neuron
    # specifically, so the comparison must be neuron-to-neuron. Comparing the
    # two-neuron population mean against the published single-neuron value would be a
    # like-for-unlike comparison and would overstate the agreement.
    mn9_by_id = (
        rate_table_from_spikes(spikes_df, [MN9_LEFT, MN9_RIGHT], cfg.duration_ms, cfg.trials)
        if spikes_df is not None
        else {}
    )
    mn9_left = mn9_by_id.get(MN9_LEFT, {}).get("mean_rate_hz")
    mn9_right = mn9_by_id.get(MN9_RIGHT, {}).get("mean_rate_hz")
    mn9_pop_mean = s["population_rates_hz"].get("mn9")
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
        mn9_left is not None,
        "MN9 left readout was measured from the recorded spikes",
        (f"MN9 left ({MN9_LEFT}) = {mn9_left:.2f} Hz; "
         f"MN9 right ({MN9_RIGHT}) = {mn9_right:.2f} Hz; "
         f"two-neuron population mean = {mn9_pop_mean:.2f} Hz. "
         f"({len(mn9_present)} of 2 MN9 neurons present)")
        if mn9_left is not None
        else f"MN9 left ({MN9_LEFT}) did not fire in this run "
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
            print(f"  MN9 left       {reference['mn9_left_rate_hz']:.2f} Hz  (single neuron)")
            print(f"  MN9 right      {reference['mn9_right_rate_hz']:.2f} Hz  (single neuron)")
            print(f"  active neurons {reference['active_neurons']}")
            print()
            print("FlyBrain, this run:")
            print(f"  trials         {s['trials']}")
            print(f"  sugar GRN rate {sugar_rate_measured:.1f} Hz")
            if mn9_left is not None:
                print(f"  MN9 left       {mn9_left:.2f} Hz  (same single neuron)")
            print(f"  active neurons {s['active_neurons_any_spike']}")
            print()
            print("The comparison is neuron-to-neuron: the published figure is for the LEFT")
            print(f"MN9 neuron alone ({MN9_LEFT}), and so is the FlyBrain figure above.")
            print("Compare the qualitative pattern - sparse downstream activation driven by")
            print("the sensory population, with MN9 among the driven - not the numbers:")
            print("different RNG, 1 trial here versus a 30-trial mean upstream, and a")
            print("different sub-step ordering in the chaobrain port")
            print("(docs/UPSTREAM_AUDIT.md sec. 4.1).")
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
        # Describes the *working set this run simulated*, taken from the run summary.
        # An earlier version described the parent dataset instead, so a subset run
        # reported `is_subset: False` and `n_neurons: 127400` - exactly the confusion
        # between subset and whole-brain that the project is meant to prevent.
        "run_scope": {
            "effective_mode": s["effective_mode"],
            "is_subset": s["is_subset"],
            "n_neurons": s["n_neurons"],
            "n_edges": s["n_edges"],
            "subset_info": s.get("subset_info"),
        },
        "parent_dataset": {
            "dataset_id": conn.dataset_id,
            "n_neurons": conn.n_neurons,
            "n_edges": conn.n_edges,
        },
        "measured": {
            "sugar_grn_rate_hz": sugar_rate_measured,
            "total_spikes": s["total_spikes"],
            "active_neurons": s["active_neurons_any_spike"],
            "mean_rate_active_hz": s["mean_rate_hz_active_neurons"],
            "max_rate_hz": s["max_rate_hz"],
            "mn9_rates_hz": {
                # per-neuron, so the comparison against the published single-neuron
                # figure is like-for-like; the population mean is reported separately
                f"mn9_left_{MN9_LEFT}": mn9_left,
                f"mn9_right_{MN9_RIGHT}": mn9_right,
                "mn9_population_mean": mn9_pop_mean,
            },
            "top_active_neurons": top[:25],
            "top_active_populations": s["top_active_populations"][:25],
            "downstream_responders": [
                {k: v for k, v in r.items() if k not in ("rate_hz_trace", "t_ms_trace")}
                for r in responders[:25]
            ],
            "monosynaptic_route_from_sugar_grn_to_responders": route,
        },
        "comparison_basis": (
            "MN9 figures are per-neuron: mn9_left_<id> and mn9_right_<id> are the two "
            "individual motor neurons, matching the published per-neuron reference. "
            "mn9_population_mean is the mean over whichever MN9 neurons are present and "
            "must not be compared against the published single-neuron value."
        ),
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
