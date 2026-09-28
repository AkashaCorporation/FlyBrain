"""LIF dynamics: the integrator, threshold, reset and refractory period.

Every expected value here is derived from the closed-form solution of the
upstream equations, not from the implementation's own output.
"""

from __future__ import annotations

import math

import pytest

from flybrain.model import (
    ExactLinearCoefficients,
    LIFParams,
    integrate_exact_scalar,
    refractory_step_count,
)


# --------------------------------------------------------------------------------------
# the integrator itself
# --------------------------------------------------------------------------------------


def test_rest_is_an_exact_fixed_point(params):
    """With g = 0 and v = v_rest the exact solution must not move at all.

    This is the strongest single check on the coupled integrator: an incorrect
    coupling coefficient K, or a sign error in the (v - v_rest) term, breaks it.
    """
    coeff = ExactLinearCoefficients.build(params, 0.1)
    v, g = params.v_rest_mv, 0.0
    for _ in range(1000):
        v, g = integrate_exact_scalar(v, g, coeff)
    assert v == pytest.approx(params.v_rest_mv, abs=1e-12)
    assert g == 0.0


def test_zero_conductance_decays_to_rest(params):
    coeff = ExactLinearCoefficients.build(params, 0.1)
    v = params.v_rest_mv + 10.0
    for _ in range(5000):
        v, _ = integrate_exact_scalar(v, 0.0, coeff)
    assert v == pytest.approx(params.v_rest_mv, abs=1e-6)


def test_conductance_decays_with_tau_synapse(params):
    coeff = ExactLinearCoefficients.build(params, 0.1)
    g0 = 11.0
    g = g0
    n = 50  # 5 ms
    for _ in range(n):
        _, g = integrate_exact_scalar(params.v_rest_mv, g, coeff)
    assert g == pytest.approx(g0 * math.exp(-5.0 / params.tau_synapse_ms), rel=1e-6)


def test_integrator_matches_closed_form_impulse_response(params):
    """A single conductance impulse must give the textbook exponential response.

    Integrating factor ``e^{t/tm}`` on ``dv/dt = (v_rest - v + g)/tm`` with
    ``g(t) = g0 e^{-t/ts}``::

        v(t) - v_rest = g0 * ts/(tm - ts) * (e^{-t/tm} - e^{-t/ts})

    The coefficient is ``ts/(tm - ts)``. The commonly written ``tm/(tm - ts)`` is
    wrong by a factor of ``tm/ts`` (= 4 here); the sanity anchor is the initial
    slope, which must be ``g0/tm``.
    """
    coeff = ExactLinearCoefficients.build(params, 0.1)
    g0 = 20.0
    tm, ts = params.tau_membrane_ms, params.tau_synapse_ms

    v, g = params.v_rest_mv, g0
    v_peak = -1e9
    for k in range(1, 1201):  # 120 ms
        v, g = integrate_exact_scalar(v, g, coeff)
        t = k * 0.1
        expected = params.v_rest_mv + g0 * ts / (tm - ts) * (
            math.exp(-t / tm) - math.exp(-t / ts)
        )
        assert v == pytest.approx(expected, abs=5e-3)
        v_peak = max(v_peak, v)

    # initial slope check, independent of the peak formula
    v1, _ = integrate_exact_scalar(params.v_rest_mv, g0, ExactLinearCoefficients.build(params, 1e-4))
    assert (v1 - params.v_rest_mv) / 1e-4 == pytest.approx(g0 / tm, rel=1e-3)

    t_pk = tm * ts / (tm - ts) * math.log(tm / ts)
    assert t_pk == pytest.approx(9.242, abs=0.01)
    assert v_peak == pytest.approx(
        params.v_rest_mv + g0 * ts / (tm - ts) * (math.exp(-t_pk / tm) - math.exp(-t_pk / ts)),
        abs=3e-3,
    )


def test_integrator_first_order_agreement_with_euler(params):
    """For dt -> small, the exact solution must agree with explicit Euler."""
    dt = 1e-6
    coeff = ExactLinearCoefficients.build(params, dt)
    v0, g0 = params.v_rest_mv + 3.0, 9.0
    v, g = integrate_exact_scalar(v0, g0, coeff)
    dv = dt * (params.v_rest_mv - v0 + g0) / params.tau_membrane_ms
    dg = dt * (-g0) / params.tau_synapse_ms
    assert v == pytest.approx(v0 + dv, abs=1e-9)
    assert g == pytest.approx(g0 + dg, abs=1e-9)


def test_degenerate_equal_time_constants_is_finite():
    """tau_m == tau_syn would divide by zero in K; the limit must be used instead."""
    p = LIFParams(tau_membrane_ms=5.0, tau_synapse_ms=5.0)
    coeff = ExactLinearCoefficients.build(p, 0.1)
    assert math.isfinite(coeff.k)
    v, g = integrate_exact_scalar(p.v_rest_mv + 1.0, 5.0, coeff)
    assert math.isfinite(v) and math.isfinite(g)


# --------------------------------------------------------------------------------------
# parameter contract
# --------------------------------------------------------------------------------------


def test_verified_upstream_parameters():
    """Pin the values read out of the upstream source in the audit.

    If someone changes a default, this test fails and forces the change to be
    deliberate and documented in docs/MODEL_ASSUMPTIONS.md.
    """
    p = LIFParams()
    assert p.v_rest_mv == -52.0
    assert p.v_reset_mv == -52.0
    assert p.v_threshold_mv == -45.0
    assert p.tau_membrane_ms == 20.0
    assert p.tau_synapse_ms == 5.0
    assert p.refractory_ms == 2.2
    assert p.synaptic_delay_ms == 1.8
    assert p.weight_per_synapse_mv == 0.275
    assert p.stimulus_rate_hz == 150.0
    assert p.stimulus_weight_scale == 250.0
    assert p.stimulus_weight_mv == pytest.approx(68.75)


def test_delay_and_refractory_step_counts(params):
    assert params.delay_steps_for(0.1) == 18
    assert params.delay_steps_for(0.2) == 9
    assert params.delay_steps_for(0.6) == 3
    # refractory condition `(t - t_last) <= 2.2` is true for k-j in 1..22 at dt=0.1
    assert refractory_step_count(2.2, 0.1) == 22
    assert refractory_step_count(0.0, 0.1) == 0


def test_dt_must_divide_the_synaptic_delay(params):
    """A dt that does not land the 1.8 ms delay on an integer step is rejected."""
    params.validate(0.1)
    params.validate(0.2)
    with pytest.raises(Exception):
        params.validate(0.7)


def test_threshold_must_exceed_reset():
    with pytest.raises(Exception):
        LIFParams(v_threshold_mv=-60.0, v_reset_mv=-52.0).validate(0.1)
