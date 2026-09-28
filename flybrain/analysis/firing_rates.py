"""Firing-rate analysis.

All functions here take recorded artefacts (spike tables, rate arrays) and return
derived quantities. Nothing re-runs a simulation, and nothing invents a rate: a
"rate" is always spikes divided by the window it was counted over, and the window
is an explicit argument rather than an implicit convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def rates_from_counts(counts: np.ndarray, duration_ms: float) -> np.ndarray:
    """Spike counts -> Hz, for the window the counts were accumulated over."""
    seconds = max(float(duration_ms) / 1000.0, 1e-12)
    return np.asarray(counts, dtype=np.float64) / seconds


def rate_table_from_spikes(
    spikes_df,
    flywire_ids: Sequence[int],
    duration_ms: float,
    trials: int = 1,
) -> Dict[int, Dict[str, float]]:
    """Per-neuron mean and std firing rate from a recorded spike table."""
    if spikes_df is None or len(spikes_df) == 0:
        return {int(i): {"mean_rate_hz": 0.0, "std_rate_hz": 0.0, "n_spikes": 0} for i in flywire_ids}
    seconds = max(float(duration_ms) / 1000.0, 1e-12)
    grouped = spikes_df.groupby(["flywire_id", "trial"]).size()
    out: Dict[int, Dict[str, float]] = {}
    for fid in flywire_ids:
        fid = int(fid)
        try:
            per_trial = grouped.loc[fid]
        except KeyError:
            out[fid] = {"mean_rate_hz": 0.0, "std_rate_hz": 0.0, "n_spikes": 0}
            continue
        per_trial = np.asarray(per_trial.reindex(range(trials), fill_value=0), dtype=np.float64)
        r = per_trial / seconds
        out[fid] = {
            "mean_rate_hz": float(r.mean()),
            "std_rate_hz": float(r.std()) if trials > 1 else 0.0,
            "n_spikes": int(per_trial.sum()),
        }
    return out


def top_neurons(
    rates: np.ndarray,
    flywire_ids: np.ndarray,
    n: int = 20,
    *,
    min_rate_hz: float = 0.0,
) -> List[Dict[str, Any]]:
    """The ``n`` highest-rate neurons, descending. Ties break on FlyWire ID for
    a stable ordering that does not depend on NumPy's sort implementation."""
    rates = np.asarray(rates, dtype=np.float64)
    order = np.lexsort((np.asarray(flywire_ids), -rates))
    out = []
    for i in order[: max(int(n), 0)]:
        if rates[i] <= min_rate_hz:
            break
        out.append({"flywire_id": int(flywire_ids[i]), "rate_hz": float(rates[i]), "index": int(i)})
    return out


def active_neuron_count(spikes_df, trials: int = 1) -> int:
    if spikes_df is None or len(spikes_df) == 0:
        return 0
    return int(spikes_df["flywire_id"].nunique())


def spike_count_summary(spikes_df, duration_ms: float, trials: int = 1) -> Dict[str, Any]:
    if spikes_df is None or len(spikes_df) == 0:
        return {"n_spikes": 0, "n_active_neurons": 0, "mean_rate_hz": 0.0, "max_rate_hz": 0.0}
    counts = spikes_df.groupby("flywire_id").size()
    seconds = max(float(duration_ms) / 1000.0, 1e-12)
    per_neuron_rate = counts.to_numpy(dtype=np.float64) / (seconds * max(trials, 1))
    return {
        "n_spikes": int(len(spikes_df)),
        "n_active_neurons": int(counts.size),
        "mean_rate_hz": float(per_neuron_rate.mean()),
        "max_rate_hz": float(per_neuron_rate.max()),
    }


def rate_histogram(rates: np.ndarray, edges: Sequence[float] = (0, 0.1, 1, 5, 10, 25, 50, 100, np.inf)) -> Dict[str, int]:
    """Counts of neurons whose rate falls in each half-open bin. Useful for asking
    "how many neurons actually did anything", which a mean rate hides."""
    rates = np.asarray(rates, dtype=np.float64)
    edges = list(edges)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        label = f"[{lo:g}, {'inf' if np.isinf(hi) else format(hi, 'g')})"
        out[label] = int(((rates >= lo) & (rates < hi)).sum())
    return out
