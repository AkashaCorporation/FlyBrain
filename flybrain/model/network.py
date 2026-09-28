"""The LIF network: exact-integration dynamics over a FlyWire-derived connectome.

This class is the "brain" object the objective asks to expose. Its five public
verbs are the whole boundary between the simulation and anything built on top of
it later (an environment, a decoder, a language layer):

.. code-block:: python

    brain.apply_stimulus(stimulus)   # perturb
    brain.step()                     # advance one dt
    brain.observe()                  # read state
    brain.silence(population)        # causal intervention
    brain.reset(seed)                # return to a known state

Nothing here knows about experiments, files, plotting, or language. It owns
arrays and arithmetic.

Sub-step order per timestep (matching Brian 2; see ``model/lif.py`` for why)::

    1. integrate (exact, skipped while refractory)
    2. threshold        v > v_threshold
    3. reset            v := v_reset, g := 0 where spiked
    4. stimulus events  v += k * w_stim
    5. synaptic events  g += W @ spikes(t - delay)

Quietly falling back to a subset is impossible here: the network simulates
exactly the connectome it was handed, and the connectome knows whether it is a
subset. Refusing an over-large whole-brain run is the runner's job, and the
runner refuses rather than shrinks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import FlyBrainError
from .lif import ExactLinearCoefficients, LIFParams
from .populations import Population, PopulationRegistry
from .stimulus import Stimulus, StimulusSampler, build_samplers

__all__ = ["LIFNetwork", "Observation", "Brain"]


def _resolve_target(target, flywire_ids, registry) -> Sequence[int]:
    """Normalise the many ways a caller can name a set of neurons.

    Accepts a population name, a :class:`Population`, an iterable of FlyWire IDs,
    or an explicit ``flywire_ids=`` keyword. Ambiguity is rejected rather than
    guessed: passing both a positional target and ``flywire_ids`` is an error.
    """
    if target is not None and flywire_ids is not None:
        raise FlyBrainError("pass either `target` or `flywire_ids`, not both")
    if flywire_ids is not None:
        ids: Sequence[int] = list(flywire_ids)
    elif target is None:
        raise FlyBrainError(
            "no target given: pass a population name, a Population, or flywire_ids=[...]"
        )
    elif isinstance(target, str):
        ids = registry.get(target).ids
    elif isinstance(target, Population):
        ids = target.ids
    else:
        ids = list(target)
    if not ids:
        raise FlyBrainError("target resolved to an empty neuron set")
    return ids


@dataclass
class Observation:
    """A read-only view of the network at one instant.

    Heavy fields (``v``, ``g``) are populated only when explicitly requested, so
    that "observe" never silently allocates a 127 400-element array per step.
    """

    step: int
    t_ms: float
    n_active: int
    active_indices: np.ndarray
    active_ids: np.ndarray
    n_spikes_total: int
    population_rates_hz: Dict[str, float] = field(default_factory=dict)
    v: Optional[np.ndarray] = None
    g: Optional[np.ndarray] = None

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "t_ms": self.t_ms,
            "n_active": self.n_active,
            "n_spikes_total": self.n_spikes_total,
            "active_flywire_ids": [int(x) for x in self.active_ids.tolist()],
            "population_rates_hz": self.population_rates_hz,
        }
        if include_arrays and self.v is not None:
            d["v_min_mv"] = float(np.min(self.v))
            d["v_max_mv"] = float(np.max(self.v))
        return d


class LIFNetwork:
    """A leaky integrate-and-fire network over a directed, signed connectome."""

    def __init__(
        self,
        connectome,
        params: Optional[LIFParams] = None,
        *,
        dt_ms: float = 0.1,
        backend=None,
        seed: int = 0,
        registry: Optional[PopulationRegistry] = None,
    ):
        from ..runtime.backend import get_backend  # local import avoids a cycle

        self.connectome = connectome
        self.params = params or LIFParams()
        self.dt_ms = float(dt_ms)
        self.seed = int(seed)
        self.registry = registry or PopulationRegistry.builtin()

        self.params.validate(self.dt_ms)
        self.backend = backend if backend is not None else get_backend("auto")
        self.xp = self.backend.xp
        self.dtype = self.backend.float_dtype
        self.n = connectome.n_neurons

        # -- static, backend-resident graph ---------------------------------------
        self.pre = self.backend.asarray(connectome.pre.astype(np.int32))
        self.post = self.backend.asarray(connectome.post.astype(np.int32))
        self.weights = self.backend.asarray(self.params.weight_per_synapse_mv * connectome.signed_count.astype(np.float32))
        # CSR row pointers are kept for analysis/reporting; the kernel itself uses
        # edge lists because a gather+scatter is faster than a segmented reduction
        # at this sparsity on CPU.
        self.indptr, self.csr_indices = connectome.csr()

        # -- integration coefficients (as 0-d arrays, so precision is unambiguous)
        self.coeff = ExactLinearCoefficients.build(self.params, self.dt_ms)
        self._ea = self.backend.asarray(self.coeff.ea, dtype=self.dtype)
        self._ed = self.backend.asarray(self.coeff.ed, dtype=self.dtype)
        self._k = self.backend.asarray(self.coeff.k, dtype=self.dtype)
        self._v_rest = self.backend.asarray(self.params.v_rest_mv, dtype=self.dtype)
        self._v_reset = self.backend.asarray(self.params.v_reset_mv, dtype=self.dtype)
        self._v_th = self.backend.asarray(self.params.v_threshold_mv, dtype=self.dtype)
        self._zero = self.backend.asarray(0.0, dtype=self.dtype)

        self.delay_steps = self.params.delay_steps_for(self.dt_ms)
        self.stim_weight = self.backend.asarray(self.params.stimulus_weight_mv, dtype=self.dtype)

        # -- mutable state --------------------------------------------------------
        self.v = None
        self.g = None
        self.t_last = None
        self.tau_ref = None
        self.ring = None
        self.cursor = 0
        self.step_index = 0
        self.spike_count = None
        self.kill_mask = None
        self._silenced_idx: set[int] = set()
        self._samplers: List[StimulusSampler] = []
        self._stim_targets = np.empty(0, dtype=np.int32)
        self._declared_stimuli: Tuple[Stimulus, ...] = ()
        self.last_spike = None
        self.disable_recurrence = False
        self._id_set: Optional[set] = None

        self.reset(seed=self.seed)

    # ==================================================================================
    # lifecycle
    # ==================================================================================
    def reset(self, seed: Optional[int] = None) -> "LIFNetwork":
        """Return every state variable to its initial value.

        Reproducibility contract: after ``reset(seed=s)`` the network produces the
        same spike trains as a freshly constructed network with ``seed=s``. The
        unit tests assert this directly rather than trusting it.
        """
        if seed is not None:
            self.seed = int(seed)
        bk = self.backend
        n = self.n
        f = self.dtype

        self.v = bk.asarray(np.full(n, self.params.v_rest_mv, dtype=f), dtype=f)
        self.g = bk.zeros(n, dtype=f)
        # -1e7 ms = "never spiked", matching the upstream initialisation.
        self.t_last = bk.asarray(np.full(n, -1e7, dtype=f), dtype=f)
        self.tau_ref = bk.asarray(np.full(n, self.params.refractory_ms, dtype=f), dtype=f)
        self.ring = bk.zeros((self.delay_steps, n), dtype=f)
        self.cursor = 0
        self.step_index = 0
        self.spike_count = bk.zeros(n, dtype=f)
        self.last_spike = bk.zeros(n, dtype=f)

        # A reset restores the *declared* intervention state rather than discarding
        # it: which stimuli are configured and which neurons are silenced are part of
        # the experiment, not part of the transient state. Use `clear_silencing()` /
        # `apply_stimulus()` with no arguments to drop an intervention explicitly.
        self._rebuild_samplers()
        self._rebuild_kill_mask()
        self._apply_stimulus_refractory()
        return self

    def _rebuild_kill_mask(self) -> None:
        mask = np.ones(self.n, dtype=np.float32)
        if self._silenced_idx:
            mask[np.fromiter(sorted(self._silenced_idx), dtype=np.int32,
                             count=len(self._silenced_idx))] = 0.0
        self.kill_mask = self.backend.asarray(mask, dtype=self.dtype)

    # ==================================================================================
    # perturbations
    # ==================================================================================
    def apply_stimulus(self, *stimuli: Stimulus | Iterable[Stimulus]) -> "LIFNetwork":
        """Set the active stimuli, replacing any previously declared set.

        Accepts one or more :class:`Stimulus` objects, or a single iterable of them.
        Stimulated neurons get their refractory period set to ``0 ms`` (upstream
        behaviour). Replacing the stimulus set restores the refractory period of
        neurons that are no longer targets.
        """
        flat: List[Stimulus] = []
        for s in stimuli:
            if isinstance(s, Stimulus):
                flat.append(s)
            else:
                flat.extend(list(s))
        self._declared_stimuli = tuple(flat)
        self._rebuild_samplers()
        self._apply_stimulus_refractory()
        return self
    def _rebuild_samplers(self) -> None:
        self._samplers = build_samplers(
            self._declared_stimuli, self.connectome, self.dt_ms, self.seed
        )
        if self._samplers:
            targets = np.unique(np.concatenate([s.indices for s in self._samplers]))
            self._stim_targets = targets.astype(np.int32)
        else:
            self._stim_targets = np.empty(0, dtype=np.int32)

    def _apply_stimulus_refractory(self) -> None:
        """Set the refractory period of stimulated neurons to 0 ms (upstream behaviour)."""
        base = np.asarray(self.tau_ref, dtype=np.float32).copy()
        if self._stim_targets.size:
            base[self._stim_targets] = 0.0
        else:
            base[:] = self.params.refractory_ms
        self.tau_ref = self.backend.asarray(base, dtype=self.dtype)

    def silence(
        self,
        target: str | Population | Sequence[int] | None = None,
        *,
        flywire_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Silence the outgoing influence of neurons.

        Semantics are taken from the upstream original (``model.py::silence``),
        which zeroes the synaptic weights of every connection *from* the silenced
        neurons. Upstream's docstring claims "to and from"; its code, and the
        published results, say outgoing only - see ``docs/UPSTREAM_AUDIT.md``
        section 3.5.

        A silenced neuron therefore keeps receiving input and may keep spiking; it
        simply stops affecting other neurons. That is what the published
        ``sugarR-*`` files show.

        ``target`` may be a population name, a :class:`Population`, or an iterable
        of FlyWire IDs. Returns the FlyWire IDs that were silenced.
        """
        ids = _resolve_target(target, flywire_ids, self.registry)
        idx = self.connectome.indices_of(list(ids))
        self._silenced_idx.update(int(i) for i in idx)
        self._rebuild_kill_mask()
        return [int(i) for i in ids]

    def silence_indices(self, indices: Sequence[int]) -> np.ndarray:
        """Silence by *dataset index* rather than FlyWire ID. Diagnostics only."""
        idx = np.asarray(list(indices), dtype=np.int32)
        self._silenced_idx.update(int(i) for i in idx)
        self._rebuild_kill_mask()
        return idx

    def unsilence(
        self,
        target: str | Population | Sequence[int] | None = None,
        *,
        flywire_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """Undo :meth:`silence` for the given target."""
        ids = _resolve_target(target, flywire_ids, self.registry)
        idx = self.connectome.indices_of(list(ids))
        self._silenced_idx.difference_update(int(i) for i in idx)
        self._rebuild_kill_mask()

    def clear_silencing(self) -> None:
        """Remove every silencing intervention."""
        self._silenced_idx.clear()
        self._rebuild_kill_mask()

    def clear_stimuli(self) -> None:
        """Remove every declared stimulus."""
        self.apply_stimulus()

    @property
    def silenced_ids(self) -> Tuple[int, ...]:
        if not self._silenced_idx:
            return ()
        return tuple(sorted(int(self.connectome.flywire_ids[i]) for i in self._silenced_idx))

    # ==================================================================================
    # stepping
    # ==================================================================================
    def step(self) -> Any:
        """Advance the network by one ``dt_ms`` and return the spike vector."""
        xp, bk = self.xp, self.backend
        t_ms = self.step_index * self.dt_ms

        # --- 1-2. integrate (exactly) unless refractory, then threshold ------------
        # The threshold is gated on `not_refractory`. This is not cosmetic: Brian 2's
        # Thresholder ANDs its condition with `not_refractory`, so a neuron held above
        # threshold still cannot fire inside its refractory window. Without the gate a
        # suprathreshold neuron fires every step, which is a silent qualitative change.
        refractory = (t_ms - self.t_last) <= self.tau_ref
        v_int = self._ea * (self.v - self._v_rest) + self._k * self.g + self._v_rest
        g_int = self._ed * self.g
        v_use = xp.where(refractory, self.v, v_int)
        g_use = xp.where(refractory, self.g, g_int)
        spike_bool = (v_use > self._v_th) & ~refractory

        # --- 3. reset --------------------------------------------------------------
        self.v = xp.where(spike_bool, self._v_reset, v_use)
        self.g = xp.where(spike_bool, self._zero, g_use)
        self.t_last = xp.where(spike_bool, t_ms, self.t_last)

        spike_f = spike_bool.astype(self.dtype)

        # --- 4. stimulus events -> v ----------------------------------------------
        for sampler in self._samplers:
            counts = sampler.counts(t_ms)
            if counts.any():
                contrib = self.stim_weight * bk.asarray(counts, dtype=self.dtype)
                self.v = bk.add_at(self.v, bk.asarray(sampler.indices), contrib)

        # --- 5. synaptic events -> g (delayed by `delay_steps`) -------------------
        delayed = self.ring[self.cursor]
        delayed = delayed.copy()
        self.ring = bk.set_row(self.ring, self.cursor, spike_f)
        self.cursor = (self.cursor + 1) % self.delay_steps

        if not self.disable_recurrence:
            pre_spk = delayed * self.kill_mask
            contrib = self.weights * pre_spk[self.pre]
            self.g = self.g + bk.scatter_add(self.n, self.post, contrib, dtype=self.dtype)

        self.spike_count = self.spike_count + spike_f
        self.step_index += 1
        self.last_spike = spike_f
        return spike_f

    def run_steps(self, n_steps: int, on_step=None) -> None:
        """Step ``n_steps`` times, optionally calling ``on_step(step, spikes)``."""
        for _ in range(int(n_steps)):
            spk = self.step()
            if on_step is not None:
                on_step(self.step_index - 1, spk)

    # ==================================================================================
    # observation
    # ==================================================================================
    def observe(
        self,
        *,
        include_v: bool = False,
        include_g: bool = False,
        populations: Optional[Sequence[str]] = None,
        spikes: Any = None,
    ) -> Observation:
        """Read the current state.

        ``spikes`` defaults to the vector produced by the most recent :meth:`step`,
        which is what a caller almost always wants. Passing an explicit vector is
        supported for callers that pipelined the step themselves.
        """
        xp = self.xp
        spk = self.last_spike if spikes is None else spikes
        if spk is None:
            active_idx = np.empty(0, dtype=np.int64)
        else:
            active_idx = np.asarray(xp.flatnonzero(spk), dtype=np.int64)

        t_ms = self.step_index * self.dt_ms
        rates: Dict[str, float] = {}
        for name in populations or ():
            rates[name] = self.population_rate_hz(name)

        return Observation(
            step=self.step_index,
            t_ms=t_ms,
            n_active=int(active_idx.size),
            active_indices=active_idx,
            active_ids=self.connectome.flywire_ids[active_idx] if active_idx.size else np.empty(0, np.int64),
            n_spikes_total=int(np.asarray(self.spike_count).sum()),
            population_rates_hz=rates,
            v=np.asarray(self.v) if include_v else None,
            g=np.asarray(self.g) if include_g else None,
        )

    # ==================================================================================
    # convenience
    # ==================================================================================
    def rates_hz(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Mean firing rate since ``reset``, per neuron (or for ``indices``)."""
        counts = np.asarray(self.spike_count)
        seconds = max(self.step_index * self.dt_ms / 1000.0, 1e-12)
        rates = counts / seconds
        return rates if indices is None else rates[indices]

    def population_rate_hz(self, name: str) -> float:
        present, _ = self.registry.resolve(name, self.connectome.flywire_ids)
        if present.size == 0:
            return 0.0
        idx = self.connectome.indices_of(present.tolist())
        seconds = max(self.step_index * self.dt_ms / 1000.0, 1e-12)
        return float(np.asarray(self.spike_count)[idx].sum() / (present.size * seconds))

    @property
    def dataset_id_set(self) -> set:
        """The dataset's FlyWire IDs as a set, built once.

        Population lookups are frequent (114 populations x every report), and
        rebuilding a 127 400-element set per lookup is pure waste.
        """
        if getattr(self, "_id_set", None) is None:
            self._id_set = set(int(x) for x in self.connectome.flywire_ids)
        return self._id_set

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend.name,
            "is_gpu": self.backend.is_gpu,
            "dtype": str(self.dtype),
            "dt_ms": self.dt_ms,
            "delay_steps": self.delay_steps,
            "n_neurons": self.n,
            "n_edges": int(self.pre.shape[0]),
            "dataset_id": self.connectome.dataset_id,
            "is_subset": self.connectome.is_subset,
            "stimuli": [s.stimulus.to_dict() for s in self._samplers],
            "silenced_flywire_ids": list(self.silenced_ids),
            "disable_recurrence": self.disable_recurrence,
            "params": self.params.to_dict(),
        }


#: The objective's long-term boundary names the object ``brain``. It is this class.
Brain = LIFNetwork
