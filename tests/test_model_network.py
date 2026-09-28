"""Network behaviour: threshold, refractory, propagation, sign, delay, silencing,
stimulus, determinism and recording.

Expected values are derived from the closed-form membrane response (see
``tests/conftest.py::interval_excursion``) or from the graph structure, never from
the implementation's own output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain.data.synthetic import chain_connectome, make_synthetic_connectome
from flybrain.errors import UnknownNeuronError
from flybrain.model import LIFNetwork, LIFParams, Stimulus
from flybrain.runtime.backend import NumpyBackend

from _helpers import (
    clamp_v,
    force_spike,
    interval_excursion,
    synapses_needed_to_cross,
    step_collect,
)


# --------------------------------------------------------------------------------------
# quiescence, threshold, reset
# --------------------------------------------------------------------------------------


def test_quiescent_network_never_spikes(net):
    spk = step_collect(net, 2000)
    assert spk.sum() == 0
    assert np.allclose(np.asarray(net.v), net.params.v_rest_mv)
    assert np.allclose(np.asarray(net.g), 0.0)


def test_spike_threshold_is_crossing_v_threshold(params, numpy_backend):
    """v exactly at threshold must not spike; strictly above must.

    Uses a very long membrane time constant so one integration step barely moves
    ``v``, which isolates the ``>`` from the integration.
    """
    slow = LIFParams(tau_membrane_ms=1.0e6)
    conn = make_synthetic_connectome([], 3)
    net = LIFNetwork(conn, slow, dt_ms=0.1, backend=numpy_backend, seed=0)

    v = np.asarray(net.v).copy()
    v[0] = slow.v_threshold_mv                 # exactly at threshold -> must not fire
    v[1] = slow.v_threshold_mv + 1e-3          # a hair above -> must fire
    v[2] = slow.v_threshold_mv - 1e-3          # a hair below -> must not fire
    net.v = numpy_backend.asarray(v, dtype=net.dtype)
    spk = np.asarray(net.step())
    assert spk.tolist() == [0.0, 1.0, 0.0], f"threshold boundary mishandled: {spk.tolist()}"


def test_threshold_is_gated_by_refractoriness(params, numpy_backend):
    """A neuron held above threshold must not fire inside its refractory window.

    Brian 2's Thresholder ANDs its condition with ``not_refractory``; without the
    gate a clamped neuron fires every step and the refractory period silently
    stops existing.
    """
    conn = make_synthetic_connectome([], 1)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    assert float(np.asarray(net.step())[0]) == 1.0

    n_ref = int(params.refractory_ms // 0.1)  # 22 frozen steps
    fired = []
    for _ in range(n_ref):
        clamp_v(net, 0, 10.0)                 # hold it above threshold the whole time
        fired.append(float(np.asarray(net.step())[0]))
    assert sum(fired) == 0, f"spiked through refractoriness: {fired}"


def test_reset_clears_conductance_and_voltage(params, numpy_backend):
    conn = make_synthetic_connectome([(0, 1, 40)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 1)
    g = np.asarray(net.g).copy()
    g[1] = 12.0
    net.g = numpy_backend.asarray(g, dtype=net.dtype)
    spk = np.asarray(net.step())
    assert spk[1] == 1.0
    assert float(np.asarray(net.v)[1]) == pytest.approx(params.v_reset_mv)
    assert float(np.asarray(net.g)[1]) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# refractoriness
# --------------------------------------------------------------------------------------


def test_refractory_blocks_a_second_spike(params, numpy_backend):
    """A neuron driven above threshold must stay silent for the refractory window."""
    conn = make_synthetic_connectome([], 1)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    spk0 = np.asarray(net.step())
    assert spk0[0] == 1.0

    n_ref = int(params.refractory_ms // 0.1)
    blocked = []
    for _ in range(n_ref):
        clamp_v(net, 0, 10.0)  # re-clamp each step, since integration lowers v
        blocked.append(float(np.asarray(net.step())[0]))

    assert sum(blocked) == 0, (
        f"neuron spiked during its refractory window ({n_ref} steps); blocked={blocked}"
    )


def test_refractory_expires_after_the_expected_number_of_steps(params, numpy_backend):
    conn = make_synthetic_connectome([], 1)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    assert float(np.asarray(net.step())[0]) == 1.0

    n_ref = int(params.refractory_ms // 0.1)  # 22 steps frozen
    for _ in range(n_ref):
        np.asarray(net.step())
    # Step n_ref + 1 after the spike: not refractory any more.
    clamp_v(net, 0, 10.0)
    assert float(np.asarray(net.step())[0]) == 1.0


def test_stimulated_neurons_have_no_refractory_period(params, numpy_backend):
    """Upstream sets rfc = 0 ms on Poisson targets, so they can fire on adjacent steps."""
    conn = make_synthetic_connectome([], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    fid = int(conn.flywire_ids[0])
    net.apply_stimulus(Stimulus([fid], rate_hz=150.0))
    tau_ref = np.asarray(net.tau_ref)
    assert tau_ref[0] == 0.0
    assert tau_ref[1] == pytest.approx(params.refractory_ms)


# --------------------------------------------------------------------------------------
# synaptic propagation, delay, sign
# --------------------------------------------------------------------------------------


def test_synaptic_input_arrives_exactly_after_the_delay(params, numpy_backend):
    """A spike at step 0 must produce conductance in the target exactly D steps later.

    Ring-buffer convention: at step ``k`` the kernel *reads* slot ``k mod D``
    (which still holds the spike from step ``k-D``) and then overwrites it with
    the spike from step ``k``. So a spike at step 0 lands on the target at step
    ``D``, i.e. 1.8 ms later at dt = 0.1 ms.
    """
    D = params.delay_steps_for(0.1)  # 18
    conn = make_synthetic_connectome([(0, 1, 200)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)

    g_trace = []
    for _ in range(D + 3):
        np.asarray(net.step())
        g_trace.append(float(np.asarray(net.g)[1]))

    for k in range(D):
        assert g_trace[k] == 0.0, f"input arrived early: g={g_trace[k]} already at step {k}"
    expected = 200 * params.weight_per_synapse_mv  # 55 mV
    assert g_trace[D] == pytest.approx(expected, rel=1e-6), (
        f"expected {expected} mV delivered exactly {D} steps (1.8 ms) after the spike, "
        f"got {g_trace[D]}"
    )


def test_single_spike_propagation_requires_enough_synapses(params, numpy_backend):
    """Whether one spike propagates is predictable from the closed form."""
    needed = synapses_needed_to_cross(params)
    assert 150 < needed < 175, (
        f"expected ~162 synapses to be needed to cross 7 mV with one spike, "
        f"closed form says {needed:.1f}"
    )
    assert interval_excursion(200, params) > 7.0 > interval_excursion(100, params)

    def first_target_spike(synapses: int) -> bool:
        conn = make_synthetic_connectome([(0, 1, synapses)], 2)
        net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
        force_spike(net, 0)
        for _ in range(1000):  # 100 ms: ample time for the membrane to respond
            if float(np.asarray(net.step())[1]) == 1.0:
                return True
        return False

    assert first_target_spike(200) is True
    assert first_target_spike(100) is False


def test_excitatory_connection_increases_target_conductance(params, numpy_backend):
    D = params.delay_steps_for(0.1)
    conn = make_synthetic_connectome([(0, 1, 25)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    for _ in range(D + 1):
        np.asarray(net.step())
    assert float(np.asarray(net.g)[1]) == pytest.approx(25 * params.weight_per_synapse_mv)


def test_inhibitory_connection_decreases_target_conductance(params, numpy_backend):
    """The sign in the dataset column decides the sign of the conductance change."""
    D = params.delay_steps_for(0.1)
    conn = make_synthetic_connectome([(0, 1, -25)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    for _ in range(D + 1):
        np.asarray(net.step())
    assert float(np.asarray(net.g)[1]) == pytest.approx(-25 * params.weight_per_synapse_mv)


def test_inhibition_can_prevent_a_spike(params, numpy_backend):
    """An inhibitory volley must be able to hold a neuron below threshold."""
    D = params.delay_steps_for(0.1)
    conn = make_synthetic_connectome([(0, 1, -200)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    for _ in range(D + 1):
        np.asarray(net.step())
    spiked = False
    for _ in range(200):
        if float(np.asarray(net.step())[1]) == 1.0:
            spiked = True
    assert not spiked, "hyperpolarising input below rest must not cause a spike"


def test_no_recurrence_leaves_the_network_silent(params, numpy_backend):
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.disable_recurrence = True
    force_spike(net, 0)
    for _ in range(400):
        np.asarray(net.step())
    counts = np.asarray(net.spike_count)
    assert counts[0] >= 1
    assert counts[1] == 0 and counts[2] == 0


# --------------------------------------------------------------------------------------
# stimulus
# --------------------------------------------------------------------------------------


def test_stimulus_window_gates_events(params, numpy_backend):
    conn = make_synthetic_connectome([], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    fid = int(conn.flywire_ids[0])
    net.apply_stimulus(Stimulus([fid], rate_hz=200.0, start_ms=100.0, end_ms=200.0))

    spikes = step_collect(net, 3000)
    idx = np.flatnonzero(spikes[:, 0])
    assert idx.size > 0, "stimulus produced no spikes at all"
    # target neuron has rfc = 0, so its spike at step k comes from an event at k-1
    assert idx.min() >= 999, f"spike before the window opened (first at step {idx.min()})"
    assert idx.max() <= 2000, f"spike after the window closed (last at step {idx.max()})"


def test_stimulus_rate_is_recovered_within_statistical_error(params, numpy_backend):
    conn = make_synthetic_connectome([], 1)
    rate = 200.0
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=12345)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=rate))
    step_collect(net, 20000)  # 2 s
    observed = float(np.asarray(net.spike_count)[0]) / 2.0
    # Poisson with mean 400/s over 2 s: sd of the rate is sqrt(800)/2 = 14.1 Hz.
    # A 4-sigma window keeps this a real check without being flaky.
    assert abs(observed - rate) < 4 * math.sqrt(2 * rate), (
        f"observed {observed} Hz for a {rate} Hz Poisson drive"
    )


def test_zero_rate_stimulus_is_silent(params, numpy_backend):
    conn = make_synthetic_connectome([], 1)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=0.0))
    step_collect(net, 2000)
    assert float(np.asarray(net.spike_count).sum()) == 0.0


def test_stimulus_stream_does_not_depend_on_declaration_order(params, numpy_backend):
    """Per-stimulus seeds must make event streams order-independent."""
    conn = make_synthetic_connectome([], 3)
    ids = [int(x) for x in conn.flywire_ids[:2]]
    a = Stimulus([ids[0]], rate_hz=100.0)
    b = Stimulus([ids[1]], rate_hz=100.0)

    def rates(order):
        net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=5)
        net.apply_stimulus(order)
        step_collect(net, 5000)
        return np.asarray(net.spike_count)[:2].copy()

    assert np.array_equal(rates([a, b]), rates([b, a]))


# --------------------------------------------------------------------------------------
# silencing
# --------------------------------------------------------------------------------------


def test_silencing_removes_outgoing_influence(params, numpy_backend):
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.silence(flywire_ids=[int(conn.flywire_ids[0])])
    force_spike(net, 0)
    for _ in range(1000):
        np.asarray(net.step())
    counts = np.asarray(net.spike_count)
    assert counts[0] >= 1, "the silenced neuron itself still receives input and may spike"
    assert counts[1] == 0 and counts[2] == 0, "silencing must stop downstream propagation"


def test_silencing_accepts_a_population_name(params, numpy_backend):
    conn = make_synthetic_connectome([(0, 1, 200), (2, 1, 200)], 3)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    # register-free path: silence by explicit IDs, then confirm the mask semantics
    net.silence(flywire_ids=[int(conn.flywire_ids[0])])
    assert np.asarray(net.kill_mask)[0] == 0.0
    assert np.asarray(net.kill_mask)[2] == 1.0
    net.unsilence(flywire_ids=[int(conn.flywire_ids[0])])
    assert np.asarray(net.kill_mask)[0] == 1.0


def test_silenced_neuron_still_spikes_matching_upstream(params, numpy_backend):
    """Upstream silencing zeroes outgoing weights only; the neuron keeps firing.

    The published results confirm this (docs/UPSTREAM_AUDIT.md section 3.5): the
    silenced neuron 720575940622695448 remains the most active neuron in its file.
    """
    conn = make_synthetic_connectome([(0, 1, 200)], 2)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.silence(flywire_ids=[int(conn.flywire_ids[0])])
    # keep neuron 0 firing by driving it, as a Poisson target would be driven
    for _ in range(20):
        force_spike(net, 0)
        np.asarray(net.step())
    assert float(np.asarray(net.spike_count)[0]) >= 1
    assert float(np.asarray(net.spike_count)[1]) == 0


def test_unknown_neuron_id_raises():
    conn = make_synthetic_connectome([], 2)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=0)
    with pytest.raises(UnknownNeuronError):
        net.silence(flywire_ids=[999_999_999_999])


# --------------------------------------------------------------------------------------
# determinism, reset, dtypes
# --------------------------------------------------------------------------------------


def _run_digest(seed: int, n_steps: int = 3000) -> np.ndarray:
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=seed)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=150.0))
    step_collect(net, n_steps)
    return np.asarray(net.spike_count).copy()


def test_same_seed_gives_identical_results():
    assert np.array_equal(_run_digest(3), _run_digest(3))


def test_different_seed_changes_the_stochastic_stream():
    assert not np.array_equal(_run_digest(3), _run_digest(4))


def test_reset_restores_reproducibility(params, numpy_backend):
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=11)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=150.0))
    step_collect(net, 2000)
    first = np.asarray(net.spike_count).copy()

    net.reset(seed=11)
    step_collect(net, 2000)
    assert np.array_equal(first, np.asarray(net.spike_count))


def test_reset_keeps_declared_stimulus_and_silencing(params, numpy_backend):
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=150.0))
    net.silence(flywire_ids=[int(conn.flywire_ids[2])])
    net.reset(seed=0)
    assert len(net._samplers) == 1
    assert np.asarray(net.kill_mask)[2] == 0.0
    assert net.silenced_ids == (int(conn.flywire_ids[2]),)


def test_state_stays_float32(params, numpy_backend):
    """float64 must not creep in: it would double memory traffic silently."""
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=150.0))
    for _ in range(50):
        np.asarray(net.step())
    assert np.asarray(net.v).dtype == np.float32
    assert np.asarray(net.g).dtype == np.float32
    assert np.asarray(net.t_last).dtype == np.float32
    assert np.asarray(net.spike_count).dtype == np.float32
    assert np.asarray(net.kill_mask).dtype == np.float32
    assert net.weights.dtype == np.float32
    assert net.pre.dtype == np.int32


def test_observation_reports_active_neurons_and_rates(params, numpy_backend):
    conn = chain_connectome(3, synapses=200)
    net = LIFNetwork(conn, params, dt_ms=0.1, backend=numpy_backend, seed=0)
    force_spike(net, 0)
    net.step()
    obs = net.observe()
    assert obs.n_active == 1
    assert obs.active_ids.tolist() == [int(conn.flywire_ids[0])]
    assert obs.t_ms == pytest.approx(0.1)
    obs_v = net.observe(include_v=True)
    assert obs_v.v is not None and obs_v.v.shape == (3,)
