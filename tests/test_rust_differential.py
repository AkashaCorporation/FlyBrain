"""E0 on synthetic fixtures with prerecorded external events, no Poisson conflation."""

import numpy as np
import pytest

pytest.importorskip("flybrain_native", reason="optional Rust extension not installed")

from flybrain.data.synthetic import make_synthetic_connectome  # noqa: E402
from flybrain.errors import FlyBrainError  # noqa: E402
from flybrain.model import LIFNetwork, LIFParams  # noqa: E402
from flybrain.model.rust import RecordingOptions, RustBrain  # noqa: E402
from flybrain.runtime.backend import NumpyBackend  # noqa: E402


class RecordedVoltage:
    def __init__(self, table, dt):
        self.table, self.dt = table, dt
        self.indices = np.arange(table.shape[1], dtype=np.int32)

    def counts(self, t):
        return self.table[round(t / self.dt)]


def equal_state(reference, rust, tick):
    actual = rust.state()
    for field in ("v", "g", "t_last", "tau_ref"):
        wanted = np.asarray(getattr(reference, field), dtype=np.float32)
        a = actual[field].view(np.uint32)
        b = wanted.view(np.uint32)
        bad = np.flatnonzero(a != b)
        assert not len(bad), (
            f"first divergence step={tick} field={field} index={bad[0]} "
            f"rust={actual[field][bad[0]]} numpy={wanted[bad[0]]}"
            if len(bad)
            else ""
        )
    np.testing.assert_array_equal(
        actual["last_spike"], reference.last_spike, err_msg=f"step {tick}"
    )
    np.testing.assert_array_equal(
        actual["spike_count"], reference.spike_count, err_msg=f"step {tick}"
    )
    assert actual["step_index"] == reference.step_index


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("silenced", [False, True])
@pytest.mark.parametrize("reference_order", [True, False])
def test_differential_every_state_and_event(dt, silenced, reference_order):
    rng = np.random.default_rng(903)
    n, steps = 16, 300
    edges = [
        (a, b, int((1 if a % 3 else -1) * rng.integers(1, 100)))
        for a in range(n)
        for b in range(n)
        if rng.random() < 0.25
    ]
    conn = make_synthetic_connectome(edges, n)
    # Destroy presynaptic order to exercise CSR versus original edge order.
    perm = rng.permutation(len(conn.pre))
    conn.pre, conn.post, conn.signed_count = (
        conn.pre[perm],
        conn.post[perm],
        conn.signed_count[perm],
    )
    ref = LIFNetwork(conn, backend=NumpyBackend(), dt_ms=dt)
    rust = RustBrain(conn, dt_ms=dt, reference_order=reference_order)
    ref.v[:] = rng.uniform(-53, -44, n)
    ref.g[:] = rng.uniform(-10, 20, n)
    ref.tau_ref[::4] = 0
    ref.t_last[1::5] = 0
    rust.set_state(ref.v, ref.g, ref.t_last, ref.tau_ref)
    if silenced:
        ref.silence_indices([0, 3, 7])
        rust.silence_indices([0, 3, 7])
    voltage = (rng.random((steps, n)) < 0.06).astype(np.float32) * np.float32(68.75)
    conductance = rng.integers(-2, 3, (steps, n)).astype(np.float32) * np.float32(0.275)
    ref._samplers = [RecordedVoltage(voltage, dt)]
    ref.stim_weight = np.array(1, np.float32)
    for tick in range(steps):
        ref.step()
        ref.g += conductance[tick]
        events = [
            (0, i, float(voltage[tick, i]), float(conductance[tick, i]))
            for i in range(n)
        ]
        rust.advance(1, events)
        equal_state(ref, rust, tick)


def test_block_counts_delay_input_and_recording_bounds():
    conn = make_synthetic_connectome([(0, 1, 60)], 3)
    core = RustBrain(conn)
    result = core.advance(
        20,
        [(0, 0, 68.75, 0)],
        RecordingOptions(max_spikes=2),
        populations=[[0], [1, 2]],
    )
    assert result["completed"] and result["completed_steps"] == 20
    assert result["recorded_spikes"] == [(1, 0)]
    assert result["active_edges"] == 1
    assert result["population_counts"] == [1, 0]
    assert core.state()["g"][1] == np.float32(60 * 0.275)
    assert "v" not in result
    core.reset()
    core.silence_indices([0])
    result = core.advance(
        60, [(0, 0, 68.75, 0), (30, 0, 68.75, 0)], RecordingOptions(max_spikes=1)
    )
    assert result["spikes"] == 2 and result["active_edges"] == 0
    assert result["recording_truncated"]
    core.reset()
    assert not core.state()["spike_count"].any()
    assert core.advance(60, [(0, 0, 68.75, 0)])["active_edges"] == 0


def test_invalid_blocks_are_rejected_before_mutation():
    core = RustBrain(make_synthetic_connectome([], 2))
    for events in [
        [(2, 0, 1, 0)],
        [(0, 2, 1, 0)],
        [(1, 0, 1, 0), (0, 0, 1, 0)],
        [(0, 0, float("nan"), 0)],
    ]:
        with pytest.raises(ValueError):
            core.advance(2, events)
        assert core.state()["step_index"] == 0
    with pytest.raises(ValueError):
        core.advance(1, populations=[[0, 0]])
    with pytest.raises(FlyBrainError, match="memory budget"):
        core.advance(1, recording_options=RecordingOptions(max_spikes=10**9))
    with pytest.raises(FlyBrainError, match="not reduced"):
        RustBrain(make_synthetic_connectome([], 2000), max_memory_bytes=100)


def test_exact_threshold_frozen_g_and_disabled_recurrence():
    params = LIFParams(tau_membrane_ms=1e6)
    conn = make_synthetic_connectome([(1, 0, 60)], 3)
    core = RustBrain(conn, params)
    core.set_state([-45, -44, 10], [0, 0, 3], [-1e7, -1e7, 0], [2.2] * 3)
    core.set_recurrence_disabled(True)
    s = core.advance(1)
    assert s["spikes"] == 1
    np.testing.assert_array_equal(core.state()["last_spike"], [0, 1, 0])
    assert core.state()["v"][2] == 10 and core.state()["g"][2] == 3
    assert core.advance(18)["active_edges"] == 0
