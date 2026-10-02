"""Explicit optional Rust kernel. No NumPy fallback and no implicit graph subset.

Inputs are prerecorded (relative_step, neuron_index, delta_v_mv, delta_g_mv).
The Python network remains the stochastic-input oracle, not the inner loop.
"""

from dataclasses import dataclass
import math

import numpy as np

from ..errors import FlyBrainError
from .lif import ExactLinearCoefficients, LIFParams


@dataclass(frozen=True)
class RecordingOptions:
    max_spikes: int = 0  # zero disables recording; observations still contain totals
    max_seconds: float = 30.0


class RustBrain:
    def __init__(
        self,
        connectome,
        params=None,
        *,
        dt_ms=0.1,
        reference_order=True,
        max_memory_bytes=256_000_000,
    ):
        try:
            from flybrain_native import NativeCore
        except ImportError as exc:
            raise FlyBrainError(
                "Rust requested but flybrain_native is not built/installed"
            ) from exc
        self.params = params or LIFParams()
        self.params.validate(dt_ms)
        self.dt_ms = float(dt_ms)
        self.connectome = connectome
        self.reference_order = bool(reference_order)
        self.max_memory_bytes = int(max_memory_bytes)
        self.n = connectome.n_neurons
        # Python tuples and integers cost much more than native arrays; refuse
        # before constructing that bridge representation. No whole-brain claim.
        edge_count = len(connectome.pre)
        self.bridge_estimate = edge_count * 256 + self.n * (
            160 + 8 * self.params.delay_steps_for(dt_ms)
        )
        if self.bridge_estimate > self.max_memory_bytes:
            raise FlyBrainError(
                "Rust graph/bridge memory budget exceeded; graph was not reduced"
            )
        c = ExactLinearCoefficients.build(self.params, dt_ms)
        weights = self.params.weight_per_synapse_mv * connectome.signed_count.astype(
            np.float32
        )
        edges = [
            (int(a), int(b), float(w))
            for a, b, w in zip(connectome.pre, connectome.post, weights)
        ]
        self._core = NativeCore(
            self.n,
            edges,
            dt_ms,
            self.params.delay_steps_for(dt_ms),
            (c.ea, c.ed, c.k),
            self.params.v_rest_mv,
            self.params.v_reset_mv,
            self.params.v_threshold_mv,
            self.params.refractory_ms,
            self.reference_order,
            self.max_memory_bytes,
        )

    def advance(self, n_steps, inputs=(), recording_options=None, *, populations=()):
        options = recording_options or RecordingOptions()
        if not math.isfinite(options.max_seconds) or options.max_seconds <= 0:
            raise FlyBrainError("max_seconds must be finite and positive")
        # Require sized sequences to budget conversion before consuming iterators.
        estimate = (
            self.bridge_estimate
            + len(inputs) * 256
            + options.max_spikes * 128
            + sum(len(p) * 48 + 64 for p in populations)
        )
        if estimate > self.max_memory_bytes:
            raise FlyBrainError("Rust block/bridge memory budget exceeded")
        result = self._core.advance(
            n_steps,
            list(inputs),
            [list(p) for p in populations],
            options.max_spikes,
            options.max_seconds,
        )
        result.update(
            requested_mode="rust_cpu",
            effective_mode="rust_cpu",
            precision="mixed_f32_f64",
            reference_order=self.reference_order,
            recording_enabled=options.max_spikes > 0,
            dataset=self.connectome.dataset_id,
            is_subset=self.connectome.is_subset,
        )
        return result

    def state(self):
        """Explicit full-state diagnostic copy, never part of default advance."""
        v, g, last, tau, spikes, counts, tick = self._core.state()
        return {
            "v": np.array(v, np.float32),
            "g": np.array(g, np.float32),
            "t_last": np.array(last, np.float32),
            "tau_ref": np.array(tau, np.float32),
            "last_spike": np.frombuffer(spikes, dtype=np.uint8).copy(),
            "spike_count": np.array(counts, np.uint64),
            "step_index": tick,
        }

    def set_state(self, v, g, t_last, tau_ref):
        self._core.set_state(list(v), list(g), list(t_last), list(tau_ref))

    def silence_indices(self, indices):
        """Replace outgoing silencing declaration; empty sequence clears it."""
        self._core.silence(list(indices))

    def set_recurrence_disabled(self, disabled):
        self._core.set_recurrence_disabled(disabled)

    def reset(self):
        self._core.reset()

    @property
    def estimated_native_bytes(self):
        return self._core.estimated_bytes
