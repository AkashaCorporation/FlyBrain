"""Continuous sensory input v1: explicit identity, persistent Poisson RNG.

This is a declared conductance-input model, not naturalistic transduction.
It never changes refractory periods. Scheduled legacy stimulation stays separate.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ..errors import FlyBrainError


class InputChannel:
    def __init__(
        self,
        channel_id,
        neuron_ids,
        indices,
        *,
        experiment_id,
        agent_id,
        base_seed,
        dt_ms,
        rate_hz,
        gain_mv,
        start_ms=0.0,
        end_ms=math.inf,
        label="",
    ):
        if any(
            not isinstance(x, str) or not x
            for x in (channel_id, experiment_id, agent_id)
        ):
            raise FlyBrainError(
                "experiment, agent and channel identities must be nonempty strings"
            )
        if not math.isfinite(gain_mv):
            raise FlyBrainError("gain_mv must be finite")
        if not math.isfinite(dt_ms) or dt_ms <= 0:
            raise FlyBrainError("dt_ms must be finite and positive")
        if not math.isfinite(start_ms) or math.isnan(end_ms) or end_ms <= start_ms:
            raise FlyBrainError(
                "input window must be finite-start, ordered [start, end)"
            )
        self.channel_id = channel_id
        self.experiment_id = experiment_id
        self.agent_id = agent_id
        self.neuron_ids = tuple(neuron_ids)
        self.indices = np.array(indices, dtype=np.int32, copy=True)
        self.indices.flags.writeable = False
        self.dt_ms = float(dt_ms)
        self.gain_mv = float(gain_mv)
        self.start_ms, self.end_ms = float(start_ms), float(end_ms)
        self.label = label
        self.set_rates(rate_hz)
        self.reset_rng(base_seed)

    def set_rates(self, rates_hz):
        try:
            rates = np.asarray(rates_hz, dtype=np.float64)
            if rates.ndim == 0:
                rates = np.full(self.indices.size, rates, dtype=np.float64)
            if (
                rates.shape != self.indices.shape
                or not np.all(np.isfinite(rates))
                or np.any(rates < 0)
            ):
                raise ValueError("rates must be finite, nonnegative and match targets")
            # NumPy's Poisson sampler rejects means close to int64 overflow.
            if np.any(rates * (self.dt_ms / 1000) > 1e12):
                raise ValueError(
                    "Poisson mean exceeds v1 input limit of 1e12 events/step"
                )
        except (ValueError, TypeError, OverflowError) as exc:
            raise FlyBrainError(str(exc)) from exc
        self._rates = rates.copy()

    @property
    def rates_hz(self):
        return self._rates.copy()

    def reset_rng(self, base_seed):
        # Unambiguous serialization: labels, windows, gains and rates are not identity.
        payload = json.dumps(
            [
                "flybrain.input.v1",
                int(base_seed),
                self.experiment_id,
                self.agent_id,
                self.channel_id,
            ],
            separators=(",", ":"),
        )
        self.stream_seed = int.from_bytes(
            hashlib.sha256(payload.encode()).digest()[:16], "big"
        )
        self.rng = np.random.Generator(np.random.PCG64(self.stream_seed))

    def counts(self, t_ms):
        draws = self.rng.poisson(self._rates * (self.dt_ms / 1000)).astype(np.float32)
        if not self.start_ms <= t_ms < self.end_ms:
            draws[:] = 0
        return draws

    def to_dict(self):
        return {
            "contract": "sensory_conductance_v1",
            "channel_id": self.channel_id,
            "experiment_id": self.experiment_id,
            "agent_id": self.agent_id,
            "neuron_ids": list(self.neuron_ids),
            "rate_hz": self._rates.tolist(),
            "gain_mv": self.gain_mv,
            "start_ms": self.start_ms,
            "end_ms": None if math.isinf(self.end_ms) else self.end_ms,
            "label": self.label,
            "stream_seed": self.stream_seed,
            "rng": "numpy.PCG64/poisson",
            "dt_ms": self.dt_ms,
        }
