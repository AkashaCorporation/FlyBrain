"""Versioned stimulus contracts; all fixtures are synthetic."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from flybrain.data.synthetic import chain_connectome
from flybrain.errors import FlyBrainError
from flybrain.model import LIFNetwork, LIFParams, Stimulus
from flybrain.runtime.backend import NumpyBackend


def brain(version="v1"):
    return LIFNetwork(
        chain_connectome(4), backend=NumpyBackend(), seed=47, contract_version=version
    )


def stim(net, *indices):
    return Stimulus(tuple(int(net.connectome.flywire_ids[i]) for i in indices), 200)


@pytest.mark.parametrize(
    "sequence",
    [
        [(0,), (1,)],
        [(0,), (0, 1)],
        [(0, 1), (1,)],
        [(0,), ()],
        [(0,), (1,), (0,), (2,), (1,)],
    ],
)
def test_v1_reconstructs_refractory_defaults(sequence):
    net = brain()
    for targets in sequence:
        net.apply_stimulus(stim(net, *targets)) if targets else net.clear_stimuli()
        expected = np.full(4, 2.2, dtype=np.float32)
        expected[list(targets)] = 0
        np.testing.assert_array_equal(net.tau_ref, expected)
    net.reset(47)
    np.testing.assert_array_equal(net.tau_ref, expected)


def test_legacy_is_default_and_retains_historical_transition():
    net = brain("legacy_v0")
    default = LIFNetwork(chain_connectome(4), backend=NumpyBackend())
    assert default.contract_version == "legacy_v0"
    net.apply_stimulus(stim(net, 0))
    net.apply_stimulus(stim(net, 1))
    np.testing.assert_array_equal(net.tau_ref[:2], [0, 0])
    net.reset()
    np.testing.assert_array_equal(net.tau_ref, np.array([2.2, 0, 2.2, 2.2], np.float32))
    net.clear_stimuli()
    np.testing.assert_array_equal(net.tau_ref, np.full(4, 2.2, np.float32))


def test_legacy_seed_depends_on_window_rate_and_label():
    s = Stimulus((1000000,), 150, end_ms=20, label="a")
    for changed in [
        replace(s, start_ms=1),
        replace(s, end_ms=21),
        replace(s, rate_hz=151),
        replace(s, label="b"),
    ]:
        assert s.spec_hash() != changed.spec_hash()
        assert s.stream_seed(47) != changed.stream_seed(47)


def declare(net, channel="sensor", agent="individual-a", **kwargs):
    return net.declare_input_channel(
        channel,
        [1000000, 1000001],
        experiment_id="fixture",
        agent_id=agent,
        rate_hz=5000,
        gain_mv=3.0,
        **kwargs,
    )


def test_rate_update_keeps_sampler_and_rng_state():
    net = brain()
    channel = declare(net)
    channel.counts(0)
    before = copy.deepcopy(channel.rng.bit_generator.state)
    net.set_input_rates("sensor", [100, 800])
    assert channel is net._input_channels["sensor"]
    assert channel.rng.bit_generator.state == before
    assert channel.rates_hz.tolist() == [100, 800]
    # New parameter takes effect on the existing stream, no draw during setter.
    reference = np.random.Generator(np.random.PCG64())
    reference.bit_generator.state = before
    np.testing.assert_array_equal(channel.counts(0.1), reference.poisson([0.01, 0.08]))


def test_channel_window_and_label_do_not_reseed():
    a = declare(brain(), label="old", start_ms=0, end_ms=10)
    b = declare(brain(), label="new", start_ms=1, end_ms=5)
    for tick in range(60):
        x, y = a.counts(tick / 10), b.counts(tick / 10)
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
        if 10 <= tick < 50:
            np.testing.assert_array_equal(x, y)
        else:
            assert not y.any()


def test_channel_order_and_individual_streams_are_independent():
    a, b = brain(), brain()
    aa = declare(a)
    declare(a, "unrelated")
    declare(b, "unrelated")
    bb = declare(b)
    other = declare(brain(), agent="individual-b")
    assert aa.stream_seed != other.stream_seed
    assert aa.rng is not bb.rng
    for tick in range(50):
        np.testing.assert_array_equal(aa.counts(tick / 10), bb.counts(tick / 10))


def test_explicit_rng_reset_and_physiological_reset():
    net = brain()
    channel = declare(net)
    first = np.array([channel.counts(0) for _ in range(12)])
    net.v[2] = -48
    net.reset_input_rng("sensor")
    assert net.v[2] == -48
    np.testing.assert_array_equal(first, [channel.counts(0) for _ in range(12)])
    net.set_input_rates("sensor", 700)
    net.reset(13)
    fresh = brain()
    fresh.reset(13)
    fresh_channel = declare(fresh)
    fresh.set_input_rates("sensor", 700)
    assert net.describe()["contract_version"] == "v1"
    assert net.describe()["input_channels"][0]["rate_hz"] == [700, 700]
    np.testing.assert_array_equal(net.v, np.full(4, -52, np.float32))
    np.testing.assert_array_equal(channel.counts(0), fresh_channel.counts(0))


def test_sensory_channel_preserves_refractory_and_applies_after_threshold():
    net = brain()
    declare(net)
    expected = np.full(4, 2.2, np.float32)
    np.testing.assert_array_equal(net.tau_ref, expected)
    assert net.step().sum() == 0
    np.testing.assert_array_equal(net.v, np.full(4, -52, np.float32))
    assert net.g[:2].sum() > 0
    np.testing.assert_array_equal(net.tau_ref, expected)


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), [1, 2, 3]])
def test_invalid_rate_update_is_atomic(bad):
    net = brain()
    ch = declare(net)
    before = copy.deepcopy(ch.rng.bit_generator.state)
    with pytest.raises(FlyBrainError):
        net.set_input_rates("sensor", bad)
    assert ch.rng.bit_generator.state == before
    np.testing.assert_array_equal(ch.rates_hz, [5000, 5000])


def test_explicit_contract_and_duplicate_channels():
    with pytest.raises(FlyBrainError):
        brain("made_up")
    with pytest.raises(FlyBrainError):
        declare(brain("legacy_v0"))
    net = brain()
    declare(net)
    with pytest.raises(FlyBrainError):
        declare(net)


def test_reference_scatter_accumulates_in_f64_then_narrows():
    # f32 running sum loses the unit; mixed reference arithmetic retains it.
    values = np.array([2**24, 1, -(2**24)], dtype=np.float32)
    got = NumpyBackend().scatter_add(1, np.zeros(3, np.int32), values)
    assert got.dtype == np.float32
    assert got.tolist() == [1.0]


@pytest.mark.parametrize("dt", [0.5, 1.0])
def test_delay_incompatible_dt_is_rejected(dt):
    with pytest.raises(FlyBrainError, match="integer multiple"):
        LIFParams().validate(dt)
