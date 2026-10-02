"""Bounded synthetic NumPy/Rust block benchmark; run under cycle_evidence.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flybrain.data.synthetic import make_synthetic_connectome  # noqa: E402
from flybrain.model import LIFNetwork  # noqa: E402
from flybrain.model.rust import RecordingOptions, RustBrain  # noqa: E402
from flybrain.runtime.backend import NumpyBackend  # noqa: E402
from cycle_evidence import identity, save  # noqa: E402


class Recorded:
    def __init__(self, table):
        self.table = table
        self.indices = np.arange(table.shape[1], dtype=np.int32)

    def counts(self, t):
        return self.table[round(t / 0.1)]


def main():
    rng = np.random.default_rng(711)
    n, steps, repeats = 128, 2000, 5
    graph = make_synthetic_connectome(
        [
            (i, int(j), 8 if i % 4 else -8)
            for i in range(n)
            for j in rng.choice(n, 8, replace=False)
        ],
        n,
    )
    result = identity()
    result.update(
        requested_mode="synthetic_numpy_vs_rust",
        effective_mode="synthetic_numpy_vs_rust",
        dataset="synthetic_128_1024",
        mask=None,
        seeds=[711],
        completed=False,
        stop_reason="running",
        evidence_level="synthetic_kernel_benchmark",
        dt_ms=0.1,
        precision="mixed_f32_f64_original_edge_order",
        threads=1,
        limits={"wall_seconds": 120, "memory_bytes": 256_000_000, "spike_recording": 0},
        graph_sha256=hashlib.sha256(
            graph.pre.tobytes() + graph.post.tobytes() + graph.signed_count.tobytes()
        ).hexdigest(),
        cases=[],
    )
    started = time.perf_counter()
    for name, probability in [("weak", 0.001), ("moderate", 0.03), ("strong", 0.3)]:
        table = (rng.random((steps, n)) < probability).astype(np.float32)
        events = [(int(t), int(i), 68.75, 0.0) for t, i in np.argwhere(table)]
        ref = LIFNetwork(graph, backend=NumpyBackend())
        rust = RustBrain(graph)
        ref._samplers = [Recorded(table)]
        warm = {}
        begin = time.perf_counter()
        ref.run_steps(100)
        warm["numpy_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        rust.advance(100, [e for e in events if e[0] < 100])
        warm["rust_seconds"] = time.perf_counter() - begin
        samples = {"numpy": [], "rust": []}
        metrics = []
        for repeat in range(repeats):
            if time.perf_counter() - started > 120:
                raise RuntimeError("benchmark wall budget exceeded")
            # Alternate timing order to reduce a fixed-order warm-cache bias.
            counts = {}
            for backend in ["numpy", "rust"] if repeat % 2 == 0 else ["rust", "numpy"]:
                if backend == "numpy":
                    ref.reset()
                    ref._samplers = [Recorded(table)]
                    begin = time.perf_counter()
                    ref.run_steps(steps)
                    elapsed = time.perf_counter() - begin
                    counts["numpy_spikes"] = int(ref.spike_count.sum())
                else:
                    rust.reset()
                    begin = time.perf_counter()
                    s = rust.advance(steps, events, RecordingOptions(max_seconds=10))
                    elapsed = time.perf_counter() - begin
                    if not s["completed"]:
                        raise RuntimeError(s)
                    counts.update(
                        rust_spikes=s["spikes"], active_edges=s["active_edges"]
                    )
                samples[backend].append(elapsed)
            for field in ("v", "g", "t_last", "tau_ref"):
                np.testing.assert_array_equal(
                    rust.state()[field].view(np.uint32),
                    np.asarray(getattr(ref, field)).view(np.uint32),
                )
            assert counts["numpy_spikes"] == counts["rust_spikes"]
            metrics.append(counts)
        case = {
            "name": name,
            "external_event_probability_per_tick": probability,
            "n": n,
            "edges": len(graph.pre),
            "steps_per_block": steps,
            "repeats": repeats,
            "input_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
            "warmup": warm,
            "block_seconds": samples,
            "metrics": metrics,
            "native_estimated_bytes": rust.estimated_native_bytes,
            "process_rss_bytes_after_case": psutil.Process().memory_info().rss,
        }
        for backend in samples:
            median = float(np.median(samples[backend]))
            case[backend] = {
                "median_block_seconds": median,
                "p95_block_seconds": float(np.percentile(samples[backend], 95)),
                "median_ms_per_step": median / steps * 1000,
                "steps_per_second": steps / median,
                "simulated_seconds_per_wall_second": steps * 0.0001 / median,
            }
        result["cases"].append(case)
        print(name, json.dumps({b: case[b] for b in samples}), flush=True)
    result.update(
        completed=True,
        stop_reason="finished",
        elapsed_seconds=time.perf_counter() - started,
    )
    save(ROOT / "outputs/first_cycle/synthetic_benchmark.json", result)


if __name__ == "__main__":
    main()
