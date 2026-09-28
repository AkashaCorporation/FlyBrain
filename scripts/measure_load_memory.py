#!/usr/bin/env python
"""Measure the cost of reading all connectivity columns vs the three that are used.

This exists so the claim "reading every column costs about twice the memory" is
reproducible rather than a number remembered from one profiling session. It measures
the *same* thing both ways, in one process, back to back.

    python scripts/measure_load_memory.py
    python scripts/measure_load_memory.py --dataset flywire_783
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain import config as fb_config  # noqa: E402
from flybrain.data import raw_paths  # noqa: E402
from flybrain.data.loader import build_connectome, read_connectivity_table  # noqa: E402


def rss_mb() -> float:
    import psutil

    gc.collect()
    return round(psutil.Process().memory_info().rss / 1e6, 1)


def peak_rss_mb(fn):
    """Peak RSS observed while ``fn`` runs, sampled from a child-free perspective.

    Uses psutil's own peak-resident tracking where available, which avoids depending
    on sampling frequency.
    """
    import psutil

    proc = psutil.Process()
    gc.collect()
    before = proc.memory_info().rss
    peak_before = getattr(proc.memory_info(), "peak_wset", None)
    out = fn()
    after = proc.memory_info().rss
    peak_after = getattr(proc.memory_info(), "peak_wset", None)
    return {
        "rss_delta_mb": round((after - before) / 1e6, 1),
        "peak_wset_delta_mb": (
            round((peak_after - peak_before) / 1e6, 1)
            if peak_after is not None and peak_before is not None
            else None
        ),
        "result": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="flywire_630")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    neuron_p, conn_p = raw_paths(args.dataset, fb_config.RAW_DIR)
    print(f"dataset     : {args.dataset}")
    print(f"file size   : {conn_p.stat().st_size / 1e6:.1f} MB ({conn_p.name})")
    print()

    # --- deep: every column, which is what the load path used to do ---------------
    result: dict = {"dataset": args.dataset, "file_mb": round(conn_p.stat().st_size / 1e6, 1)}

    deep = peak_rss_mb(lambda: read_connectivity_table(conn_p, deep=True))
    result["deep_all_columns"] = {
        "columns": 7,
        "rss_delta_mb": deep["rss_delta_mb"],
        "peak_wset_delta_mb": deep["peak_wset_delta_mb"],
        "n_edges": int(deep["result"]["pre"].size),
    }
    print(f"all 7 columns      : RSS delta {deep['rss_delta_mb']:9.1f} MB "
          f"(peak {deep['peak_wset_delta_mb']} MB)")
    del deep
    gc.collect()

    minimal = peak_rss_mb(lambda: read_connectivity_table(conn_p))
    result["minimal_columns"] = {
        "columns": 3,
        "rss_delta_mb": minimal["rss_delta_mb"],
        "peak_wset_delta_mb": minimal["peak_wset_delta_mb"],
        "n_edges": int(minimal["result"]["pre"].size),
    }
    print(f"3 required columns : RSS delta {minimal['rss_delta_mb']:9.1f} MB "
          f"(peak {minimal['peak_wset_delta_mb']} MB)")
    del minimal
    gc.collect()

    # --- full network construction, the path a run actually takes -----------------
    net = peak_rss_mb(lambda: build_connectome(args.dataset, neuron_p, conn_p))
    result["network_construction_minimal"] = {
        "rss_delta_mb": net["rss_delta_mb"],
        "peak_wset_delta_mb": net["peak_wset_delta_mb"],
        "n_neurons": int(net["result"].n_neurons),
        "n_edges": int(net["result"].n_edges),
    }
    print(f"build_connectome   : RSS delta {net['rss_delta_mb']:9.1f} MB "
          f"(peak {net['peak_wset_delta_mb']} MB)")

    if result["deep_all_columns"]["rss_delta_mb"] and result["minimal_columns"]["rss_delta_mb"]:
        d = result["deep_all_columns"]["rss_delta_mb"]
        m = result["minimal_columns"]["rss_delta_mb"]
        result["reduction_pct"] = round(100.0 * (1.0 - m / d), 1)
        print(f"\nreduction: {result['reduction_pct']}% less RSS on the decode step")
    if (
        result["deep_all_columns"]["peak_wset_delta_mb"]
        and result["minimal_columns"]["peak_wset_delta_mb"]
    ):
        d = result["deep_all_columns"]["peak_wset_delta_mb"]
        m = result["minimal_columns"]["peak_wset_delta_mb"]
        result["peak_reduction_pct"] = round(100.0 * (1.0 - m / d), 1)
        print(f"reduction: {result['peak_reduction_pct']}% less peak resident memory")

    out = Path(args.out) if args.out else (fb_config.OUTPUTS_DIR / f"load_memory_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
