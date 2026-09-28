"""Population-level activity analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def population_rate_matrix(pop_df) -> Dict[str, Dict[str, np.ndarray]]:
    """Reshape a population-activity table into ``{population: {t_ms, rate_hz, trial}}``."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    if pop_df is None or len(pop_df) == 0:
        return out
    for name, grp in pop_df.groupby("population", sort=True):
        grp = grp.sort_values(["trial", "t_ms"] if "trial" in grp.columns else ["t_ms"])
        out[str(name)] = {
            "t_ms": grp["t_ms"].to_numpy(np.float64),
            "rate_hz": grp["rate_hz"].to_numpy(np.float64),
            "trial": grp["trial"].to_numpy(np.int64) if "trial" in grp.columns else np.zeros(len(grp), np.int64),
        }
    return out


def population_summary(pop_df) -> List[Dict[str, Any]]:
    """Per-population mean/peak rate and the time of the peak, sorted by peak rate.

    ``time_to_peak_ms`` is reported alongside the peak because a population that
    peaks immediately is probably a directly stimulated one, while a late peak
    indicates a downstream responder - which is what the sugar experiment is
    actually trying to show.
    """
    out: List[Dict[str, Any]] = []
    for name, series in population_rate_matrix(pop_df).items():
        r = series["rate_hz"]
        t = series["t_ms"]
        if r.size == 0:
            continue
        k = int(np.argmax(r))
        out.append(
            {
                "population": name,
                "mean_rate_hz": float(r.mean()),
                "peak_rate_hz": float(r.max()),
                "time_to_peak_ms": float(t[k]),
                "samples": int(r.size),
                "rate_hz_trace": r,
                "t_ms_trace": t,
            }
        )
    out.sort(key=lambda d: (-d["peak_rate_hz"], d["population"]))
    return out


def downstream_responders(
    pop_df,
    stimulus_populations: Sequence[str],
    *,
    min_peak_hz: float = 1.0,
    exclude_stimulated: bool = True,
) -> List[Dict[str, Any]]:
    """Populations that responded, with the directly stimulated ones separable.

    This is the evidence for the objective's claim chain
    ``known sensory neurons -> stimulation -> propagation -> downstream population
    response``. Excluding the stimulated populations by default keeps that claim
    honest: a population firing because we drove it is not evidence of propagation.
    """
    stim = set(stimulus_populations)
    out = []
    for entry in population_summary(pop_df):
        if exclude_stimulated and entry["population"] in stim:
            continue
        if entry["peak_rate_hz"] < min_peak_hz:
            continue
        out.append(entry)
    return out


def response_latency_ms(pop_df, population: str, threshold_hz: float = 1.0) -> Optional[float]:
    """First time a population's rate reaches ``threshold_hz``, or None."""
    series = population_rate_matrix(pop_df).get(population)
    if series is None:
        return None
    idx = np.flatnonzero(series["rate_hz"] >= threshold_hz)
    return float(series["t_ms"][idx[0]]) if idx.size else None
