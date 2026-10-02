"""The per-neuron readout contract, pinned so it cannot be misread again.

flybrain-core builds `population_counts` as `vec![0; populations.len()]`: it is
ONE COUNTER PER POPULATION, not one counter per neuron. The reference test in
tests/test_rust_differential.py already pins it with populations=[[0], [1, 2]]
returning [1, 0], which means "population [1,2] fired zero times between them" and
NOT "neuron 1 fired once and neuron 2 fired none".

A first version of the C2a runner passed the whole 655-neuron output set as a
single population, got back one scalar per trial, and measured a
"655-dimensional representation" that was in fact one number per trial. The run
produced plausible-looking accuracies and they were meaningless. These tests
state the contract in both directions so the mistake has a name and a failure
mode instead of being rediscovered by accident.
"""
from __future__ import annotations

import numpy as np
import pytest

from flybrain.data.synthetic import make_synthetic_connectome

pytest.importorskip("flybrain_native", reason="optional Rust extension not installed")

from flybrain.model.rust import RecordingOptions, RustBrain  # noqa: E402

# The reference protocol from tests/test_rust_differential.py: one driven neuron,
# 20 steps, so the spike lands inside the block and is counted.
DRIVE = [(0, 0, 68.75, 0)]
STEPS = 20


def test_one_population_of_many_neurons_gives_one_count_not_many():
    """The trap, stated as a test: N neurons in one population is ONE number."""
    edges = [(0, i, 200) for i in range(1, 5)]
    core = RustBrain(make_synthetic_connectome(edges, 5))
    result = core.advance(STEPS, DRIVE, RecordingOptions(max_spikes=64),
                          populations=[[0, 1, 2, 3, 4]])
    assert len(result["population_counts"]) == 1, (
        "one population must give exactly one count, whatever its size"
    )
    assert result["population_counts"] == [1]


def test_singleton_populations_give_one_count_per_neuron():
    """The correct form: one singleton population per neuron you want to read."""
    edges = [(0, i, 200) for i in range(1, 5)]
    core = RustBrain(make_synthetic_connectome(edges, 5))
    result = core.advance(STEPS, DRIVE, RecordingOptions(max_spikes=64),
                          populations=[[0], [1], [2], [3], [4]])
    counts = result["population_counts"]
    assert len(counts) == 5, "five singleton populations must give five counts"
    assert counts[0] == 1, "the driven neuron must be counted in its own slot"
    assert all(c >= 0 for c in counts)


def test_per_neuron_counts_sum_to_the_single_population_total():
    """Consistency: splitting the same neurons across populations changes nothing."""
    edges = [(0, i, 200) for i in range(1, 5)] + [(1, 3, 200)]
    core = RustBrain(make_synthetic_connectome(edges, 5))
    single = core.advance(STEPS, DRIVE, RecordingOptions(max_spikes=64),
                          populations=[[0, 1, 2, 3, 4]])["population_counts"]
    core.reset()
    per_neuron = np.asarray(
        core.advance(STEPS, DRIVE, RecordingOptions(max_spikes=64),
                     populations=[[0], [1], [2], [3], [4]])["population_counts"],
        dtype=np.int64,
    )
    assert per_neuron.sum() == single[0], (
        "splitting the same neurons into singleton populations changed the total; "
        "the readout is not measuring what the single population measured"
    )


def test_a_readout_of_many_neurons_must_return_that_many_counts():
    """The guard the C2a runner now raises on, as a test."""
    edges = [(0, i, 200) for i in range(1, 5)]
    core = RustBrain(make_synthetic_connectome(edges, 5))
    neurons = [0, 1, 2, 3, 4]
    result = core.advance(STEPS, DRIVE, RecordingOptions(max_spikes=64),
                          populations=[[i] for i in neurons])
    assert len(result["population_counts"]) == len(neurons)
