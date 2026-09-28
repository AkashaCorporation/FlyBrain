#!/usr/bin/env python
"""Profile the parts of FlyBrain the objective names, and only those.

    python scripts/profile_pipeline.py --mode subset --neurons 8000
    python scripts/profile_pipeline.py --mode whole-brain --load-only

Measured stages, matching the objective's list:

    dataset loading
    graph construction (CSR)
    simulation step
    memory use
    recording overhead

The results are written as JSON so they can be quoted rather than remembered.
Nothing here optimises anything; it measures first, which is what the objective asks
for ("Do not optimize prematurely. First make it correct. After a working
implementation: profile ... Only optimize demonstrated bottlenecks").
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import config as fb_config  # noqa: E402
from flybrain.data import load_connectome  # noqa: E402
from flybrain.data.loader import build_connectome, raw_paths  # noqa: E402
from flybrain.model import LIFNetwork, LIFParams, Stimulus  # noqa: E402
from flybrain.model.populations import PopulationRegistry  # noqa: E402
from flybrain.runtime import NumpyBackend, RecordingConfig, Recorder, available_ram_gb  # noqa: E402


def rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return None


def timeit(label, fn, results, notes=None):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    results[label] = {"seconds": round(dt, 4), "ms": round(dt * 1e3, 2)}
    if notes:
        results[label].update(notes)
    print(f"{label:42s} {dt:8.3f} s" + (f"   {notes}" if notes else ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="flywire_630")
    ap.add_argument("--mode", default="subset", choices=["subset", "whole-brain"])
    ap.add_argument("--neurons", type=int, default=8000, help="subset size")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "jax"])
    ap.add_argument("--load-only", action="store_true", help="stop after loading the dataset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fb_config.ensure_dirs()
    results: dict = {
        "dataset": args.dataset,
        "mode": args.mode,
        "backend": args.backend,
        "steps": args.steps,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ram_available_gb_before": available_ram_gb(),
        "rss_mb_before": rss_mb(),
    }
    print(f"FlyBrain profiling - {args.dataset}, mode={args.mode}, backend={args.backend}")
    print(f"RAM available before: {results['ram_available_gb_before']} GB, RSS {results['rss_mb_before']} MB")
    print()

    # ---- 1. dataset loading (parquet decode path, cache bypassed) ----------------
    neuron_p, conn_p = raw_paths(args.dataset, fb_config.RAW_DIR)

    def load_uncached():
        return build_connectome(args.dataset, neuron_p, conn_p)

    conn = timeit("dataset.load.parquet_decode", load_uncached, results,
                  {"edges": None}) if not args.load_only else None
    if conn is not None:
        results["dataset.load.parquet_decode"]["edges"] = conn.n_edges
        results["dataset.load.parquet_decode"]["neurons"] = conn.n_neurons
        results["rss_mb_after_load"] = rss_mb()

    # ---- 2. processed-cache load (the path a real run takes) ---------------------
    conn = timeit("dataset.load.processed_cache", lambda: load_connectome(
        args.dataset, fb_config.RAW_DIR, fb_config.PROCESSED_DIR, fb_config.METADATA_DIR
    ), results, {"neurons": None, "edges": None})
    results["dataset.load.processed_cache"]["neurons"] = conn.n_neurons
    results["dataset.load.processed_cache"]["edges"] = conn.n_edges
    results["rss_mb_after_cache_load"] = rss_mb()

    # ---- 3. graph construction (CSR) ---------------------------------------------
    timeit("graph.build.csr", lambda: (conn.csr(), conn.out_degree.shape)[1], results,
           {"rows": conn.n_neurons + 1, "nnz": conn.n_edges})

    # ---- 4. subset extraction ----------------------------------------------------
    reg = PopulationRegistry.builtin()
    present, _ = reg.resolve("sugar_grn", conn.flywire_ids)
    if args.mode == "subset":
        sub = timeit("graph.build.subset", lambda: conn.subset(
            [int(x) for x in present], hops=args.hops, max_neurons=args.neurons
        ), results, {"hops": args.hops, "cap": args.neurons})
        results["graph.build.subset"]["neurons"] = sub.n_neurons
        results["graph.build.subset"]["edges"] = sub.n_edges
        target = sub
    else:
        target = conn
        results["graph.build.subset"] = {"seconds": 0.0, "ms": 0.0,
                                        "neurons": conn.n_neurons, "edges": conn.n_edges,
                                        "note": "whole-brain: no subset built"}

    # ---- 5. network construction -------------------------------------------------
    bk = NumpyBackend("float32")
    params = LIFParams()
    net = timeit("network.construct", lambda: LIFNetwork(
        target, params, dt_ms=0.1, backend=bk, seed=0, registry=reg
    ), results, {"neurons": target.n_neurons, "edges": target.n_edges})
    results["rss_mb_after_network"] = rss_mb()

    stim = Stimulus(neuron_ids=tuple(int(x) for x in present if int(x) in target.id_to_index),
                    rate_hz=100.0, label="sugar_grn")
    net.apply_stimulus(stim)

    # ---- 6. simulation step ------------------------------------------------------
    n_warm = 5
    for _ in range(n_warm):
        net.step()
    net.reset(seed=0)
    net.apply_stimulus(stim)

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        net.step()
    step_s = (time.perf_counter() - t0) / args.steps
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["simulation.step"] = {
        "seconds_per_step": round(step_s, 6),
        "ms_per_step": round(step_s * 1e3, 3),
        "steps_per_second": round(1.0 / step_s, 1),
        "python_alloc_peak_mb": round(peak / 1e6, 2),
        "note": "python allocation peak is the transient per step, not resident state",
    }
    print(f"{'simulation.step':42s} {step_s * 1e3:8.3f} ms/step "
          f"({1.0 / step_s:,.0f} steps/s), python alloc peak {peak / 1e6:.1f} MB")
    results["rss_mb_after_steps"] = rss_mb()

    # ---- 7. recording overhead ---------------------------------------------------
    def measure_recording(cfg: RecordingConfig, label: str) -> None:
        rec = Recorder(target, cfg, registry=reg, dt_ms=0.1)
        net.reset(seed=0)
        net.apply_stimulus(stim)
        rec.begin(net)
        t0 = time.perf_counter()
        for k in range(args.steps):
            rec.on_step(k, net.step(), net)
        dt = (time.perf_counter() - t0) / args.steps
        rows = rec.summary()["recorded_spike_rows"]
        results[label] = {
            "ms_per_step": round(dt * 1e3, 3),
            "overhead_vs_bare_ms": round((dt - step_s) * 1e3, 3),
            "overhead_ratio": round(dt / step_s, 3) if step_s else None,
            "spike_rows": rows,
        }
        print(f"{label:42s} {dt * 1e3:8.3f} ms/step  "
              f"(+{(dt - step_s) * 1e3:.3f} ms, x{dt / step_s:.2f})")

    measure_recording(RecordingConfig(record_spikes=False, record_population=False),
                      "recording.none")
    measure_recording(RecordingConfig(record_spikes=False, record_population=True,
                                      population_names=("sugar_grn", "mn9"),
                                      population_interval_ms=10.0),
                      "recording.population_rates")
    measure_recording(RecordingConfig(record_spikes=True, record_population=False),
                      "recording.spikes_all")

    # ---- 8. memory model vs reality ---------------------------------------------
    from flybrain.runtime.runner import estimate_requirements

    est = estimate_requirements(target.n_neurons, target.n_edges,
                               params.delay_steps_for(0.1))
    results["memory"] = {
        "estimate": est,
        "rss_mb_after_network": results.get("rss_mb_after_network"),
        "rss_mb_after_steps": results.get("rss_mb_after_steps"),
    }
    print()
    print(f"estimate: resident {est['resident_gb']} GB, peak {est['peak_gb']} GB")
    if results.get("rss_mb_after_network") and results.get("rss_mb_before"):
        measured_delta = results["rss_mb_after_network"] - results["rss_mb_before"]
        results["memory"]["rss_mb_delta_for_network"] = round(measured_delta, 1)
        print(f"measured RSS delta for the network: {measured_delta:.1f} MB "
              f"(estimate {est['resident_gb'] * 1000:.0f} MB resident)")

    out = Path(args.out) if args.out else (
        fb_config.OUTPUTS_DIR / f"profile_{args.mode}_{args.backend}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
