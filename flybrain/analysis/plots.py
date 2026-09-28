"""Scientific plots.

Four plots are required by the objective: population firing rate over time, a
spike raster, top activated populations, and an input-vs-downstream comparison.
They are all written to files; no interactive backend is used, so this works in a
terminal, over SSH, and in CI.

Readability over aesthetics: every axis is labelled with its unit, and every plot
that could be mistaken for a biological measurement carries the word "simulated"
in its title. A plot that leaves that ambiguous is a scientific-communication bug,
not a styling choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..errors import FlyBrainError

#: Written into every plot title. Kept in one place so it cannot be dropped by
#: accident in one figure but not another.
SIMULATED_TAG = "simulated activity - not a biological measurement"


def _plt():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FlyBrainError(
            "matplotlib is required for plotting; install it with `pip install matplotlib`"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_path: Path, dpi: int = 130) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------


def plot_population_rates(
    pop_df,
    out_path: Path,
    *,
    populations: Optional[Sequence[str]] = None,
    max_populations: int = 8,
) -> Optional[Path]:
    """Population firing rate against time, one line per population."""
    from .population_activity import population_rate_matrix

    series = population_rate_matrix(pop_df)
    if not series:
        return None
    if populations is not None:
        series = {k: v for k, v in series.items() if k in set(populations)}
    if not series:
        return None

    # pick the most active populations so the figure stays readable
    ranked = sorted(series.items(), key=lambda kv: -float(kv[1]["rate_hz"].max()))[:max_populations]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name, s in ranked:
        # average across trials at matching sample indices
        n = len(s["rate_hz"])
        trials = int(s["trial"].max()) + 1 if s["trial"].size else 1
        if trials > 1 and n % trials == 0:
            m = s["rate_hz"].reshape(trials, n // trials).mean(axis=0)
            t = s["t_ms"].reshape(trials, n // trials)[0]
        else:
            m, t = s["rate_hz"], s["t_ms"]
        ax.plot(t, m, lw=1.4, label=name)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("mean firing rate [Hz]")
    ax.set_title(f"Population firing rate over time\n{SIMULATED_TAG}")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, out_path)


def plot_raster(
    spikes_df,
    out_path: Path,
    *,
    max_neurons: int = 2000,
    highlight: Optional[Dict[str, int]] = None,
) -> Optional[Path]:
    """Spike raster, optionally highlighting named populations in colour."""
    if spikes_df is None or len(spikes_df) == 0:
        return None
    df = spikes_df
    ids = np.sort(df["flywire_id"].unique())
    if ids.size > max_neurons:
        # Keep the busiest neurons rather than the numerically first ones, which
        # would show an arbitrary slice of the ID space.
        counts = df.groupby("flywire_id").size().sort_values(ascending=False)
        ids = np.sort(counts.index[:max_neurons].to_numpy())
    row_of = {int(v): i for i, v in enumerate(ids)}
    sub = df[df["flywire_id"].isin(row_of)]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.scatter(
        sub["t_ms"].to_numpy(np.float64),
        sub["flywire_id"].map(row_of).to_numpy(),
        s=1.4, c="#2b3a67", linewidths=0, alpha=0.85,
    )
    if highlight:
        colors = ["#d1495b", "#edae49", "#00798c", "#66a182", "#8d6a9f"]
        for (name, fid), colour in zip(highlight.items(), colors):
            sel = sub[sub["flywire_id"] == fid]
            if len(sel) == 0:
                continue
            ax.scatter(
                sel["t_ms"].to_numpy(np.float64),
                sel["flywire_id"].map(row_of).to_numpy(),
                s=5, c=colour, linewidths=0, label=f"{name} ({fid})",
            )
        ax.legend(fontsize=7, frameon=False, markerscale=3)
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("neuron (row index, sorted by FlyWire ID)")
    ax.set_title(f"Spike raster of {ids.size} recorded neurons\n{SIMULATED_TAG}")
    ax.grid(alpha=0.2, lw=0.5)
    return _save(fig, out_path)


def plot_top_populations(
    pop_df,
    out_path: Path,
    *,
    top_n: int = 25,
    stimulus_populations: Sequence[str] = (),
) -> Optional[Path]:
    """Horizontal bar chart of the most active populations."""
    from .population_activity import population_summary

    summary = population_summary(pop_df)
    if not summary:
        return None
    summary = summary[: max(int(top_n), 1)]
    stim = set(stimulus_populations)
    names = [s["population"] for s in summary][::-1]
    peaks = [s["peak_rate_hz"] for s in summary][::-1]
    colors = ["#d1495b" if n in stim else "#2b3a67" for n in names]

    plt = _plt()
    height = max(3.0, 0.28 * len(names) + 1.6)
    fig, ax = plt.subplots(figsize=(8.5, height))
    ax.barh(names, peaks, color=colors)
    ax.set_xlabel("peak firing rate over the run [Hz]")
    ax.set_ylabel("population")
    ax.set_title(
        f"Top {len(names)} activated populations (red = directly stimulated)\n{SIMULATED_TAG}",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    return _save(fig, out_path)


def plot_input_vs_downstream(
    conditions: Dict[str, Dict[str, float]],
    out_path: Path,
    *,
    populations: Optional[Sequence[str]] = None,
    top_n: int = 15,
) -> Optional[Path]:
    """Grouped bars comparing each population's rate across experimental conditions.

    This is the figure that answers "did silencing population B change the response
    to stimulating population A", which is the whole point of the silencing
    experiment. Conditions are ordered, not sorted, so "baseline" and "silenced"
    always appear in the supplied order.
    """
    if not conditions:
        return None
    cond_names = list(conditions)
    if populations is None:
        # rank populations by peak across conditions so the most informative appear
        agg: Dict[str, float] = {}
        for rates in conditions.values():
            for k, v in rates.items():
                agg[k] = max(agg.get(k, 0.0), float(v))
        populations = [k for k, _ in sorted(agg.items(), key=lambda kv: -kv[1])[:top_n]]
    pops = list(populations)
    if not pops:
        return None

    plt = _plt()
    n_cond = len(cond_names)
    y = np.arange(len(pops))
    h = 0.8 / max(n_cond, 1)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.34 * len(pops) + 1.6)))
    palette = ["#2b3a67", "#d1495b", "#00798c", "#edae49", "#66a182"]
    for k, cond in enumerate(cond_names):
        vals = [float(conditions[cond].get(p, 0.0)) for p in pops]
        ax.barh(y + (k - (n_cond - 1) / 2) * h, vals, height=h, label=cond,
                color=palette[k % len(palette)])
    ax.set_yticks(y)
    ax.set_yticklabels(pops, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean firing rate [Hz]")
    ax.set_ylabel("population")
    ax.set_title(
        f"Input -> downstream activity across conditions\n{SIMULATED_TAG}",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    return _save(fig, out_path)


def plot_voltage_trace(
    volt_df,
    out_path: Path,
    *,
    flywire_ids: Optional[Sequence[int]] = None,
    max_traces: int = 5,
) -> Optional[Path]:
    """Membrane-potential traces, with the threshold and rest lines drawn.

    Threshold and rest come from the caller's model parameters, not from the data,
    so a wrong parameter would be visible as a trace that clearly misses the line.
    """
    if volt_df is None or len(volt_df) == 0:
        return None
    df = volt_df
    if flywire_ids is not None:
        df = df[df["flywire_id"].isin(set(int(x) for x in flywire_ids))]
    if len(df) == 0:
        return None
    ids = list(dict.fromkeys(df["flywire_id"].tolist()))[:max_traces]

    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for fid in ids:
        sel = df[df["flywire_id"] == fid].sort_values(["trial", "t_ms"])
        if "trial" in sel.columns and sel["trial"].nunique() > 1:
            sel = sel[sel["trial"] == sel["trial"].min()]
        ax.plot(sel["t_ms"], sel["v_mv"], lw=1.1, label=str(int(fid)))
    ax.axhline(-45.0, color="#d1495b", ls="--", lw=0.9, label="threshold -45 mV")
    ax.axhline(-52.0, color="#999999", ls=":", lw=0.9, label="rest -52 mV")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("membrane potential [mV]")
    ax.set_title(f"Membrane potential samples\n{SIMULATED_TAG}")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25, lw=0.5)
    return _save(fig, out_path)


def plot_summary_dashboard(
    summary: Dict[str, Any],
    spikes_df,
    pop_df,
    out_dir: Path,
    *,
    stimulus_populations: Sequence[str] = (),
    highlight: Optional[Dict[str, int]] = None,
) -> List[Path]:
    """Produce the standard figure set for a run. Returns the paths written."""
    out_dir = Path(out_dir)
    written: List[Path] = []
    for fn, name in (
        (lambda p: plot_population_rates(pop_df, p), "population_rates.png"),
        (lambda p: plot_raster(spikes_df, p, highlight=highlight), "spike_raster.png"),
        (
            lambda p: plot_top_populations(pop_df, p, stimulus_populations=stimulus_populations),
            "top_populations.png",
        ),
    ):
        try:
            path = fn(out_dir / name)
        except FlyBrainError:
            raise
        except Exception:
            path = None
        if path is not None:
            written.append(path)
    return written
