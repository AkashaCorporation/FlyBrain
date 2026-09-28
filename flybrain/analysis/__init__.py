"""Analysis layer: firing rates, population activity, connectivity, plots."""

from __future__ import annotations

from .connectivity import connectivity_between, degree_stats, neighbours, neuron_profile
from .firing_rates import (
    active_neuron_count,
    rate_histogram,
    rate_table_from_spikes,
    rates_from_counts,
    spike_count_summary,
    top_neurons,
)
from .plots import (
    SIMULATED_TAG,
    plot_input_vs_downstream,
    plot_population_rates,
    plot_raster,
    plot_summary_dashboard,
    plot_top_populations,
    plot_voltage_trace,
)
from .population_activity import (
    downstream_responders,
    population_rate_matrix,
    population_summary,
    response_latency_ms,
)

__all__ = [
    "rates_from_counts",
    "rate_table_from_spikes",
    "top_neurons",
    "active_neuron_count",
    "spike_count_summary",
    "rate_histogram",
    "population_rate_matrix",
    "population_summary",
    "downstream_responders",
    "response_latency_ms",
    "degree_stats",
    "neighbours",
    "neuron_profile",
    "connectivity_between",
    "SIMULATED_TAG",
    "plot_population_rates",
    "plot_raster",
    "plot_top_populations",
    "plot_input_vs_downstream",
    "plot_voltage_trace",
    "plot_summary_dashboard",
]
