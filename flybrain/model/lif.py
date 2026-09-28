"""Leaky integrate-and-fire dynamics and the per-timestep kernel.

The equations are taken verbatim from the upstream original
(``third_party/Drosophila_brain_model/model.py``, ``default_params['eqs']``)::

    dv/dt = (v_0 - v + g) / t_mbr      : mV, unless refractory
    dg/dt = -g / tau                   : mV, unless refractory
    threshold: v > v_th
    reset:     v = v_rst ; g = 0

Integration
-----------
Upstream runs Brian 2 with ``method='linear'``. The system is linear in
``(v, g)`` with constant coefficients, so ``linear`` means "solve it exactly over
the step" rather than "Euler". FlyBrain reproduces that exactly.

Writing the state as ``x = (v, g)``, the system is ``dx/dt = A x + b`` with

    A = [[-1/tau_m,  1/tau_m],
         [        0, -1/tau_s]]        b = [v_rest/tau_m, 0]

The unique fixed point is ``x* = -A^-1 b = (v_rest, 0)``. Hence

    x(t+dt) - x* = expm(A dt) (x(t) - x*)

and for this upper-triangular ``A`` with distinct eigenvalues ``a = -1/tau_m``,
``d = -1/tau_syn``::

    expm(A dt) = [[ e^{a dt},  K ],
                  [       0, e^{d dt}]]         K = c (e^{a dt} - e^{d dt}) / (a - d)

with ``c = 1/tau_m``. Therefore::

    v(t+dt) = e^{a dt} (v - v_rest) + K g + v_rest
    g(t+dt) = e^{d dt} g

This is the scheme that produced the published results.

Ordering within one timestep
----------------------------
Brian 2's object schedule is state-updater, then thresholder, then resetter, then
synapses. ``PoissonInput`` defaults to ``when='synapses'``. FlyBrain therefore runs::

    1. integrate (exactly, unless refractory)
    2. threshold  -> spikes
    3. reset      -> v = v_reset, g = 0 where a spike occurred
    4. synaptic events   -> g += w   for spikes that occurred `delay` ago
    5. stimulus events   -> v += k*w for Poisson events drawn this step

Steps 4 and 5 touch different variables, so their relative order is immaterial.
A consequence worth stating: a stimulus event delivered at step ``k`` is first
able to cause a spike at step ``k+1``. The upstream model has the same one-step
latency, so the emitted rate is unaffected.

Refractoriness
--------------
``refractory`` is evaluated as ``(t - t_last_spike) <= refractory`` (Brian 2
spells the same condition ``not_refractory = (t - t_last_spike) > refractory``).
While refractory, **both** ``v`` and ``g`` are frozen, because upstream marks both
equations ``(unless refractory)``. Incoming synaptic events are still applied to
``g`` during refractoriness, because ``on_pre`` code is not gated by the
refractory flag in Brian 2.

This is the original model's dynamics and the user is explicitly told not to
treat the task prompt as scientific authority; these constants were read out of
the upstream source. See ``docs/MODEL_ASSUMPTIONS.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from ..errors import FlyBrainError

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LIFParams:
    """Model constants, in mV and ms.

    Names are FlyBrain's; the upstream ``default_params`` key is recorded in
    :data:`PARAMETER_PROVENANCE` so the mapping is never in doubt.
    """

    # membrane
    v_rest_mv: float = -52.0          # upstream v_0
    v_reset_mv: float = -52.0         # upstream v_rst
    v_threshold_mv: float = -45.0     # upstream v_th
    tau_membrane_ms: float = 20.0     # upstream t_mbr
    tau_synapse_ms: float = 5.0       # upstream tau
    refractory_ms: float = 2.2        # upstream t_rfc
    synaptic_delay_ms: float = 1.8    # upstream t_dly

    # synapses and stimulation
    weight_per_synapse_mv: float = 0.275   # upstream w_syn (free parameter)
    stimulus_rate_hz: float = 150.0        # upstream r_poi
    stimulus_weight_scale: float = 250.0   # upstream f_poi

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self, dt_ms: float) -> None:
        if self.tau_membrane_ms <= 0 or self.tau_synapse_ms <= 0:
            raise FlyBrainError("time constants must be positive")
        if self.v_threshold_mv <= self.v_reset_mv:
            raise FlyBrainError(
                "v_threshold_mv must be above v_reset_mv, otherwise the model is "
                "trivially unstable (reset would immediately re-threshold)"
            )
        if dt_ms <= 0:
            raise FlyBrainError("dt_ms must be positive")
        # The synaptic delay must land on an integer number of steps, otherwise the
        # implemented delay silently differs from the requested one.
        steps = self.synaptic_delay_ms / dt_ms
        if abs(steps - round(steps)) > 1e-9 or round(steps) < 1:
            raise FlyBrainError(
                f"synaptic_delay_ms={self.synaptic_delay_ms} is not an integer multiple of "
                f"dt_ms={dt_ms} ({steps} steps). Choose a dt that divides the delay "
                f"(0.1 ms and 0.2 ms both work for the default 1.8 ms delay)."
            )

    @property
    def stimulus_weight_mv(self) -> float:
        """mV added to ``v`` per Poisson event (upstream ``w_syn * f_poi``)."""
        return self.weight_per_synapse_mv * self.stimulus_weight_scale

    def delay_steps_for(self, dt_ms: float) -> int:
        """Synaptic delay expressed in timesteps. ``dt_ms`` must divide it exactly.

        ``validate`` enforces the exact-divisibility requirement, so this method is
        only reached with a dt for which the result is exact.
        """
        return int(round(self.synaptic_delay_ms / dt_ms))

    def refractory_steps_for(self, dt_ms: float) -> int:
        """Number of steps a neuron is frozen after spiking.

        ``(t - t_last) <= refractory`` first becomes false at
        ``ceil(refractory/dt)`` steps after the spike.
        """
        return int(refractory_step_count(self.refractory_ms, dt_ms))


def refractory_step_count(refractory_ms: float, dt_ms: float) -> int:
    """Steps ``k-j`` for which a neuron that spiked at step ``j`` is still frozen.

    The neuron is frozen while ``(k - j) * dt_ms <= refractory_ms``; the condition
    first fails at ``k - j = floor(refractory_ms / dt_ms) + 1``.
    """
    if refractory_ms <= 0:
        return 0
    return int(refractory_ms // dt_ms)


#: ``parameter -> (value, source, reason, confidence)``.
#: ``confidence`` uses three honest levels:
#:   * ``cited``     - value taken from a cited measurement/paper by upstream
#:   * ``free``      - upstream explicitly labels it a free parameter
#:   * ``derived``   - arithmetic on other quantities
#: A test asserts that ``docs/MODEL_ASSUMPTIONS.md`` and this table agree, so the
#: documentation cannot silently drift from the code.
PARAMETER_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "v_rest_mv": {
        "upstream_key": "v_0",
        "value": -52.0,
        "unit": "mV",
        "source": "upstream model.py default_params; comment cites Kakaria & de Bivort 2017, "
                  "https://doi.org/10.3389/fnbeh.2017.00008",
        "reason": "resting potential of the model neuron",
        "confidence": "cited",
    },
    "v_reset_mv": {
        "upstream_key": "v_rst",
        "value": -52.0,
        "unit": "mV",
        "source": "upstream model.py default_params (`eq_rst`)",
        "reason": "post-spike reset; upstream sets it equal to rest (no hyperpolarising reset)",
        "confidence": "free",
    },
    "v_threshold_mv": {
        "upstream_key": "v_th",
        "value": -45.0,
        "unit": "mV",
        "source": "upstream model.py default_params (`eq_th`)",
        "reason": "spike threshold; 7 mV above rest",
        "confidence": "free",
    },
    "tau_membrane_ms": {
        "upstream_key": "t_mbr",
        "value": 20.0,
        "unit": "ms",
        "source": "upstream model.py default_params; inline comment "
                  "'capacitance * resistance = .002 * uF * 10. * Mohm'",
        "reason": "membrane time constant of the model neuron",
        "confidence": "cited",
    },
    "tau_synapse_ms": {
        "upstream_key": "tau",
        "value": 5.0,
        "unit": "ms",
        "source": "upstream model.py default_params; comment cites Jurgensen et al., "
                  "https://doi.org/10.1088/2634-4386/ac3ba6",
        "reason": "post-synaptic conductance decay constant",
        "confidence": "cited",
    },
    "refractory_ms": {
        "upstream_key": "t_rfc",
        "value": 2.2,
        "unit": "ms",
        "source": "upstream model.py default_params; comment cites Lazar et al., "
                  "https://doi.org/10.7554/eLife.62362",
        "reason": "absolute refractory period",
        "confidence": "cited",
    },
    "synaptic_delay_ms": {
        "upstream_key": "t_dly",
        "value": 1.8,
        "unit": "ms",
        "source": "upstream model.py default_params; comment cites Paul et al. 2015, "
                  "https://doi.org/10.3389/fncel.2015.00029",
        "reason": "uniform axonal + synaptic transmission delay applied to every connection",
        "confidence": "cited",
    },
    "weight_per_synapse_mv": {
        "upstream_key": "w_syn",
        "value": 0.275,
        "unit": "mV",
        "source": "upstream model.py default_params, inline comment literally '# Free parameter'",
        "reason": "per-synapse weight; modulated by exponential decay. NOT derived from "
                  "measurement - it is the single calibrated knob that sets network excitability",
        "confidence": "free",
    },
    "stimulus_rate_hz": {
        "upstream_key": "r_poi",
        "value": 150.0,
        "unit": "Hz",
        "source": "upstream model.py default_params; the published tutorial runs used 200 Hz "
                  "(see docs/UPSTREAM_AUDIT.md section 6)",
        "reason": "default rate of the Poisson stimulation applied to target neurons",
        "confidence": "free",
    },
    "stimulus_weight_scale": {
        "upstream_key": "f_poi",
        "value": 250.0,
        "unit": "dimensionless",
        "source": "upstream model.py default_params, inline comment '250 is sufficient to "
                  "cause spiking'",
        "reason": "Poisson events are delivered to v, not g, so they need a larger amplitude "
                  "than a single synapse. 0.275 mV * 250 = 68.75 mV, which always crosses the "
                  "7 mV rest-to-threshold gap in one event",
        "confidence": "free",
    },
    "dt_ms": {
        "upstream_key": "(brian2 defaultclock.dt)",
        "value": 0.1,
        "unit": "ms",
        "source": "Brian 2's default clock; the upstream model never overrides it",
        "reason": "integration timestep used for the published results",
        "confidence": "cited",
    },
}

#: Literature that the numeric values rest on, recorded so a reader can check.
LITERATURE = {
    "v_rest": "Kakaria & de Bivort 2017, Front. Behav. Neurosci. 11:8, doi:10.3389/fnbeh.2017.00008",
    "tau_synapse": "Jurgensen et al., Neuromorphic Computing and Engineering, doi:10.1088/2634-4386/ac3ba6",
    "refractory": "Lazar et al., eLife 2021;10:e62362, doi:10.7554/eLife.62362",
    "synaptic_delay": "Paul et al. 2015, Front. Cell. Neurosci. 9:29, doi:10.3389/fncel.2015.00029",
    "model": "Shiu et al., Nature 634, 210-219 (2024), doi:10.1038/s41586-024-07763-9",
}


# --------------------------------------------------------------------------------------
# Exact linear integrator
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExactLinearCoefficients:
    """Precomputed constants for the exact integration of the coupled (v, g) system."""

    ea: float   # exp(-dt / tau_membrane)
    ed: float   # exp(-dt / tau_synapse)
    k: float    # coupling coefficient K
    v_rest: float
    dt_ms: float

    @staticmethod
    def build(params: LIFParams, dt_ms: float) -> "ExactLinearCoefficients":
        import math

        a = -1.0 / params.tau_membrane_ms
        c = 1.0 / params.tau_membrane_ms
        d = -1.0 / params.tau_synapse_ms
        ea = math.exp(a * dt_ms)
        ed = math.exp(d * dt_ms)
        if abs(a - d) < 1e-12:
            # Degenerate case tau_membrane == tau_synapse: K is the limit of
            # c (e^{a dt} - e^{d dt}) / (a - d) as d -> a, equal to c dt e^{a dt}.
            k = c * dt_ms * ea
        else:
            k = c * (ea - ed) / (a - d)
        return ExactLinearCoefficients(ea=ea, ed=ed, k=k, v_rest=params.v_rest_mv, dt_ms=dt_ms)


def integrate_exact(v, g, coeff: ExactLinearCoefficients, xp):
    """One exact integration step of the unforced system.

    ``v`` and ``g`` may be scalars or arrays. Returns the integrated ``(v, g)``.
    Does not apply refractoriness, thresholding, reset or input; the kernel does
    that around this call.
    """
    v_new = coeff.ea * (v - coeff.v_rest) + coeff.k * g + coeff.v_rest
    g_new = coeff.ed * g
    return v_new, g_new


def integrate_exact_scalar(v: float, g: float, coeff: ExactLinearCoefficients):
    """Pure-Python version, used by unit tests and the interactive console."""
    v_new = coeff.ea * (v - coeff.v_rest) + coeff.k * g + coeff.v_rest
    g_new = coeff.ed * g
    return v_new, g_new
