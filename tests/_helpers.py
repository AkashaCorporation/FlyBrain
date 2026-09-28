"""Test helpers, importable by any test module in this directory.

Kept out of ``conftest.py`` so that tests can import them directly (``conftest``
is a pytest hook module, not a library). All helpers are on the *fixture*
side of the test/implementation boundary: they construct controlled inputs or
compute expectations in closed form, and never re-derive an expected value from
the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flybrain import config as fb_config  # noqa: E402
from flybrain.model import LIFNetwork, LIFParams  # noqa: E402


def dataset_available(dataset_id: str = "flywire_630") -> bool:
    """True when the staged dataset files and their provenance card are all present."""
    from flybrain.data import load_dataset_card, raw_paths

    try:
        csv_p, parquet_p = raw_paths(dataset_id, fb_config.RAW_DIR)
    except Exception:
        return False
    return (
        csv_p.is_file()
        and parquet_p.is_file()
        and load_dataset_card(dataset_id, fb_config.METADATA_DIR) is not None
    )


def force_spike(net: LIFNetwork, indices) -> None:
    """Push neurons above threshold so they spike on the next step.

    Used to test the delay/propagation machinery with a fully deterministic
    source instead of depending on when a Poisson draw happens to land. Setting
    ``v`` to +10 mV survives one exact-integration step comfortably above the
    -45 mV threshold, so exactly one spike is produced.
    """
    idx = np.atleast_1d(np.asarray(indices, dtype=np.int64))
    v = np.asarray(net.v).copy()
    v[idx] = 10.0
    net.v = net.backend.asarray(v, dtype=net.dtype)


def clamp_v(net: LIFNetwork, index: int, value: float) -> None:
    """Hold one neuron's membrane potential at ``value`` (used for refractory tests)."""
    v = np.asarray(net.v).copy()
    v[index] = value
    net.v = net.backend.asarray(v, dtype=net.dtype)


def step_collect(net: LIFNetwork, n_steps: int) -> np.ndarray:
    """Run ``n_steps`` and return an ``(n_steps, n_neurons)`` spike matrix."""
    out = np.zeros((n_steps, net.n), dtype=np.float32)
    for k in range(n_steps):
        out[k] = np.asarray(net.step())
    return out


def first_spike_step(net: LIFNetwork, index: int, max_steps: int) -> int | None:
    for k in range(max_steps):
        if float(np.asarray(net.step())[index]) > 0:
            return k
    return None


def interval_excursion(synapses: float, params: LIFParams) -> float:
    """Peak mV excursion of ``v`` after one presynaptic spike of ``synapses`` synapses.

    For a conductance impulse ``g(t) = g0 e^{-t/ts}`` and ``v(0) = v_rest`` the
    exact solution of ``dv/dt = (v_rest - v + g)/tm`` is

        v(t) - v_rest = g0 * ts/(tm - ts) * (e^{-t/tm} - e^{-t/ts}),   g0 = synapses*w

    (integrating factor ``e^{t/tm}``; the small-``t`` limit is ``g0 t/tm``, which
    is the required initial slope). The bracket peaks at
    ``t* = tm*ts/(tm-ts) * ln(tm/ts)`` where it equals 0.4725 for the verified
    constants, so the peak excursion is ``g0 * 0.1575``.

    Note the coefficient is ``ts/(tm-ts)``, **not** ``tm/(tm-ts)``: the latter is a
    common slip and overstates the excursion fourfold for tm=20, ts=5.

    Used as the independent expectation in the propagation tests.
    """
    g0 = synapses * params.weight_per_synapse_mv
    tm, ts = params.tau_membrane_ms, params.tau_synapse_ms
    t_peak = tm * ts / (tm - ts) * np.log(tm / ts)
    return float(g0 * ts / (tm - ts) * (np.exp(-t_peak / tm) - np.exp(-t_peak / ts)))


def synapses_needed_to_cross(params: LIFParams, gap_mv: float | None = None) -> float:
    """How many synapses on one neuron a *single* presynaptic spike needs to cross threshold.

    Derived, never hard-coded, so a change to the membrane constants or to
    ``w_syn`` automatically moves the expected value.
    """
    gap = gap_mv if gap_mv is not None else (params.v_threshold_mv - params.v_rest_mv)
    per_synapse_excursion = interval_excursion(1.0, params)
    return gap / per_synapse_excursion
