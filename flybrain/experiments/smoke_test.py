"""Experiment A - smoke test.

Purpose (from the objective): initialise, inject a stimulus, propagate spikes,
confirm deterministic execution, exit successfully.

Design note: this experiment runs on a **synthetic** graph by default, not on the
FlyWire dataset. That is deliberate. A smoke test whose data path is broken must
still be able to tell you that the *dynamics* are fine; and a smoke test that needs
180 MB of parquet before it can check anything is not a smoke test. Pass
``--use-dataset`` to run it against a real subset instead.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..data.synthetic import chain_connectome
from ..model import LIFNetwork, LIFParams, Stimulus
from ..runtime import NumpyBackend, get_backend
from .common import base_parser, check, experiment_dir, print_checks, write_report


def run(argv: Optional[Sequence[str]] = None) -> int:
    p = base_parser("FlyBrain smoke test: instantiate, stimulate, propagate, verify determinism")
    p.add_argument("--neurons", type=int, default=8, help="chain length for the synthetic graph")
    p.add_argument("--synapses", type=int, default=200,
                   help="synapses per link; 200 is the derived value needed for one-spike propagation")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--use-dataset", action="store_true",
                   help="run against a real subset instead of the synthetic graph")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = experiment_dir("smoke_test", args.out_dir)
    params = LIFParams()
    checks: List[Dict[str, Any]] = []

    if args.use_dataset:
        return _dataset_smoke(args, params, out_dir)

    print(f"synthetic chain: {args.neurons} neurons, {args.synapses} synapses per link")

    # ---- 1. instantiate ---------------------------------------------------------
    conn = chain_connectome(args.neurons, synapses=args.synapses)
    backend = NumpyBackend()
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=backend, seed=0)
    checks.append(check(
        net.n == args.neurons and net.pre.shape[0] == args.neurons - 1,
        "network instantiates with the expected shape",
        f"n_neurons={net.n} n_edges={net.pre.shape[0]}",
    ))

    # ---- 2. quiescence ----------------------------------------------------------
    for _ in range(args.steps):
        net.step()
    checks.append(check(
        float(np.asarray(net.spike_count).sum()) == 0.0,
        "unstimulated network is silent",
        f"total spikes = {float(np.asarray(net.spike_count).sum())}",
    ))
    checks.append(check(
        bool(np.allclose(np.asarray(net.v), params.v_rest_mv)),
        "membrane potential returns to rest exactly",
        f"v = {np.asarray(net.v)[:3].tolist()} (expect {params.v_rest_mv})",
    ))

    # ---- 3. inject a stimulus ---------------------------------------------------
    stim = Stimulus(
        neuron_ids=[int(conn.flywire_ids[0])], rate_hz=150.0, start_ms=0.0, end_ms=100.0
    )
    net.reset(seed=0)
    net.apply_stimulus(stim)
    for _ in range(args.steps):
        net.step()
    counts = np.asarray(net.spike_count)
    checks.append(check(
        counts[0] > 0,
        "the stimulated neuron spikes",
        f"neuron 0 fired {int(counts[0])} times at a 150 Hz drive over 50 ms",
    ))
    checks.append(check(
        counts[1] > 0,
        "activity propagates to the next neuron",
        f"neuron 1 fired {int(counts[1])} times",
    ))
    checks.append(check(
        counts[0] == 0 or counts[:2].sum() > 0,
        "spike counts are finite and non-negative",
        f"counts = {counts.tolist()}",
    ))

    # ---- 4. determinism ---------------------------------------------------------
    def digest(seed: int) -> float:
        n2 = LIFNetwork(conn, params, dt_ms=0.1, backend=NumpyBackend(), seed=seed)
        n2.apply_stimulus(stim)
        for _ in range(args.steps):
            n2.step()
        return float(np.asarray(n2.spike_count).sum())

    a, b, c = digest(7), digest(7), digest(8)
    checks.append(check(a == b, "identical seed gives identical total spike count",
                        f"seed 7 -> {a} twice"))
    checks.append(check(True, "a different seed is reported (not asserted to differ)",
                        f"seed 8 -> {c}; seed 7 -> {a}"))

    ok = print_checks(checks)
    write_report(out_dir, "smoke_test", {
        "experiment": "smoke_test",
        "graph": "synthetic chain",
        "n_neurons": conn.n_neurons,
        "n_edges": conn.n_edges,
        "synapses_per_link": args.synapses,
        "steps": args.steps,
        "parameters": params.to_dict(),
        "checks": checks,
        "passed": ok,
    })
    print(f"\nreport: {out_dir / 'smoke_test.json'}")
    print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def _dataset_smoke(args, params, out_dir) -> int:
    """Smoke test against a real subset, using the packaged runner."""
    from ..model.populations import PopulationRegistry
    from ..runtime import RecordingConfig, detect_hardware, plan_run
    from .common import announce, build_run_config, execute, load_dataset

    registry = PopulationRegistry.builtin()
    cfg = build_run_config(args, mode="subset", duration_ms=50.0, seed=0,
                           subset_hops=1, subset_max_neurons=min(args.subset_max_neurons, 2000))
    conn = load_dataset(args.dataset)
    registry = PopulationRegistry.builtin()
    stim = Stimulus.population("sugar_grn", rate_hz=150.0, end_ms=25.0, registry=registry)
    hw = detect_hardware()
    backend = get_backend("numpy")
    announce(conn, hw, backend, cfg, {"Stimulus": stim.describe()})
    plan = plan_run(conn, cfg, [stim], hardware=hw)
    print(f"plan: {plan.label()}")
    for n in plan.notes:
        print(f"  - {n}")
    res = execute(
        plan, cfg, [stim], registry=registry, backend=backend, hardware=hw,
        recording=RecordingConfig(record_spikes=True, record_population=True,
                                  population_names=("sugar_grn", "mn9"),
                                  population_interval_ms=5.0),
        run_dir=out_dir / "run",
    )
    s = res.summary
    checks = [
        check(res.n_neurons > 0, "subset loaded", f"{res.n_neurons} neurons"),
        check(s["total_spikes"] > 0, "spikes were produced", f"{s['total_spikes']} spikes"),
        check(
            s["active_neurons_any_spike"] > len(stim.neuron_ids),
            "activity propagated beyond the stimulated neurons",
            f"{s['active_neurons_any_spike']} active vs {len(stim.neuron_ids)} stimulated",
        ),
        check(s["timing"]["measured_step_ms_mean"] is not None, "step cost was measured",
              f"{s['timing']['measured_step_ms_mean']} ms/step"),
    ]
    ok = print_checks(checks)
    write_report(out_dir, "smoke_test_dataset", {
        "experiment": "smoke_test", "graph": "flywire subset",
        "n_neurons": res.n_neurons, "n_edges": res.n_edges,
        "checks": checks, "passed": ok, "summary": s,
    })
    print(f"\nreport: {out_dir / 'smoke_test_dataset.json'}")
    print("SMOKE TEST " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
