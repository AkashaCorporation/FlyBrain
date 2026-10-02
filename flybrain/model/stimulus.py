"""Controlled external stimulation.

Deliberately decoupled from any experiment script: a :class:`Stimulus` is a plain
description of "make these neurons spike at roughly this rate, between these two
times", plus the sampling logic that turns it into per-timestep events. Event
generation lives in :class:`StimulusSampler`, so an environment or a future
decoder can drive the network without touching this module.

Upstream semantics being reproduced (``third_party/Drosophila_brain_model/model.py``,
function ``poi``):

* one ``PoissonInput(target=neu[i], target_var='v', N=1, rate=r_poi,
  weight=w_syn * f_poi)`` **per stimulated neuron**;
* the target neuron's refractory period is set to ``0 ms``;
* the event is added to the membrane potential ``v``, *not* to the synaptic
  conductance ``g``.

Because ``w_syn * f_poi = 68.75 mV`` far exceeds the 7 mV rest-to-threshold gap,
each event is effectively a guaranteed spike, so a stimulated neuron emits an
approximately rate-``r_poi`` Poisson spike train.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..errors import FlyBrainError, UnknownPopulationError
from .populations import PopulationRegistry


@dataclass(frozen=True)
class Stimulus:
    """A rate-coded stimulation applied to a fixed set of neurons.

    Parameters
    ----------
    neuron_ids:
        FlyWire root IDs of the stimulated neurons. A population name is not
        accepted here on purpose - resolution needs a connectome to check
        membership, and that belongs at the call site (:meth:`population`).
    rate_hz:
        Mean spike rate of the Poisson drive (upstream ``r_poi``).
    start_ms, end_ms:
        Half-open window ``[start_ms, end_ms)``. ``end_ms`` defaults to +inf,
        meaning "for the rest of the run".
    label:
        Free-text provenance for the run record, e.g. the population name.
    """

    neuron_ids: Tuple[int, ...]
    rate_hz: float
    start_ms: float = 0.0
    end_ms: float = math.inf
    label: str = ""

    def __post_init__(self) -> None:
        # Accept any iterable of ints, normalise to a sorted unique tuple so that the
        # same logical stimulus always produces the same RNG stream and the same hash.
        ids = tuple(sorted({int(x) for x in self.neuron_ids}))
        object.__setattr__(self, "neuron_ids", ids)
        object.__setattr__(self, "rate_hz", float(self.rate_hz))
        object.__setattr__(self, "start_ms", float(self.start_ms))
        object.__setattr__(self, "end_ms", float(self.end_ms))
        if not ids:
            raise FlyBrainError("Stimulus has no neuron_ids")
        if self.rate_hz < 0:
            raise FlyBrainError(f"rate_hz must be >= 0, got {self.rate_hz}")
        if self.end_ms <= self.start_ms:
            raise FlyBrainError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )

    # -- alternative constructors --------------------------------------------------
    @classmethod
    def population(
        cls,
        name: str,
        rate_hz: float = 150.0,
        start_ms: float = 0.0,
        end_ms: float = math.inf,
        *,
        registry: Optional[PopulationRegistry] = None,
        flywire_ids: Optional[Sequence[int]] = None,
    ) -> "Stimulus":
        """Build a stimulus from a named population in :class:`PopulationRegistry`.

        ``flywire_ids`` optionally restricts the population to the IDs present in
        the loaded dataset; without it, the caller must ensure every member exists
        (the connectome resolver raises with the full missing list otherwise).
        """
        reg = registry or PopulationRegistry.builtin()
        pop = reg.get(name)
        ids = tuple(pop.ids) if flywire_ids is None else tuple(int(x) for x in flywire_ids)
        if not ids:
            raise UnknownPopulationError(
                f"population {name!r} has {len(pop)} declared members but none are present "
                f"in the loaded dataset"
            )
        return cls(
            neuron_ids=ids,
            rate_hz=rate_hz,
            start_ms=start_ms,
            end_ms=end_ms,
            label=name,
        )

    # -- description --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "neuron_ids": list(self.neuron_ids),
            "n_neurons": len(self.neuron_ids),
            "rate_hz": self.rate_hz,
            "start_ms": self.start_ms,
            "end_ms": None if math.isinf(self.end_ms) else self.end_ms,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Stimulus":
        end = d.get("end_ms")
        return cls(
            neuron_ids=tuple(d["neuron_ids"]),
            rate_hz=float(d["rate_hz"]),
            start_ms=float(d.get("start_ms", 0.0)),
            end_ms=math.inf if end is None else float(end),
            label=d.get("label", ""),
        )

    def describe(self) -> str:
        ids = f"{len(self.neuron_ids)} neurons"
        window = (
            f"[{self.start_ms:g}, inf) ms"
            if math.isinf(self.end_ms)
            else f"[{self.start_ms:g}, {self.end_ms:g}) ms"
        )
        label = f" ({self.label})" if self.label else ""
        return f"{self.rate_hz:g} Hz -> {ids}{label} over {window}"

    # -- determinism ---------------------------------------------------------------
    def spec_hash(self) -> str:
        """Stable digest of the stimulus definition.

        Used to derive a per-stimulus RNG seed, so the events a given stimulus
        produces do not depend on how many *other* stimuli were declared, or in
        what order.
        """
        h = hashlib.sha256()
        h.update(b"|".join(str(i).encode() for i in self.neuron_ids))
        h.update(f"|{self.rate_hz!r}|{self.start_ms!r}|{self.end_ms!r}|{self.label}".encode())
        return h.hexdigest()

    def stream_seed(self, base_seed: int) -> int:
        """Derive a 32-bit seed for this stimulus from a run-level base seed."""
        digest = self.spec_hash()
        return int(hashlib.sha256(f"{base_seed}:{digest}".encode()).hexdigest()[:8], 16)

    def is_active(self, t_ms: float) -> bool:
        return self.start_ms <= t_ms < self.end_ms

    def active_step_range(self, dt_ms: float, n_steps: int) -> Tuple[int, int]:
        """Half-open ``[first, last)`` step indices during which the stimulus is on."""
        first = int(max(0, math.ceil(self.start_ms / dt_ms - 1e-9)))
        if math.isinf(self.end_ms):
            last = n_steps
        else:
            last = int(min(n_steps, math.ceil(self.end_ms / dt_ms - 1e-9)))
        return first, max(first, last)


class StimulusSampler:
    """Turns a :class:`Stimulus` into per-timestep Poisson event counts.

    The RNG is drawn **every** step, including steps outside the stimulus window,
    and the window is applied as a mask afterwards. In the historical contract,
    however, spec_hash includes the window, rate and label: changing those fields
    changes the RNG seed. This class preserves that behavior for v0 replay.
    Use InputChannel for explicitly identified, window-independent sensory streams.
    """

    def __init__(
        self,
        stimulus: Stimulus,
        indices: np.ndarray,
        dt_ms: float,
        base_seed: int = 0,
    ):
        self.stimulus = stimulus
        self.indices = np.asarray(indices, dtype=np.int32)
        self.dt_ms = float(dt_ms)
        self.rng = np.random.Generator(np.random.PCG64(stimulus.stream_seed(base_seed)))
        # Poisson mean event count per step per target neuron.
        self.lam = stimulus.rate_hz * (dt_ms / 1000.0)

    def counts(self, t_ms: float) -> np.ndarray:
        """Poisson event count per target neuron for the step starting at ``t_ms``.

        Returns a zero array when inactive, after drawing. This does not undo the
        legacy seed's dependence on the configured window.
        """
        draws = self.rng.poisson(self.lam, size=self.indices.size).astype(np.float32)
        if not self.stimulus.is_active(t_ms):
            draws[:] = 0.0
        return draws

    def __len__(self) -> int:
        return int(self.indices.size)


def build_samplers(
    stimuli: Iterable[Stimulus],
    connectome,
    dt_ms: float,
    base_seed: int = 0,
) -> list:
    """Resolve a list of stimuli against a connectome and build their samplers.

    Stimuli are sorted by ``(start_ms, end_ms, rate_hz, label)`` so that the
    resulting order - and therefore the recorded configuration - is stable
    regardless of how the caller supplied them.
    """
    ordered = sorted(
        stimuli,
        key=lambda s: (s.start_ms, s.end_ms, s.rate_hz, s.label),
    )
    samplers = []
    for s in ordered:
        idx = connectome.indices_of(s.neuron_ids)
        samplers.append(StimulusSampler(s, idx, dt_ms=dt_ms, base_seed=base_seed))
    return samplers
