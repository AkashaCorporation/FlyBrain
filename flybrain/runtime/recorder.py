"""Selective recording of simulation output.

The objective is explicit that recording must not be unconditional: dumping every
timestep of every neuron for a 127 400-neuron run is an unbounded output. So
recording is opt-in per channel, and each channel has a declared cost.

Channels
--------
``spikes``               individual spike times for a selected neuron set (or all
                         spiking neurons, capped)
``population_activity``  per-population firing rate on a coarse interval
``voltage``              membrane-potential samples for a selected neuron set
``summary``              network-level scalars at the end of the run

Callers learn about truncation: when the spike cap is hit the recorder stops
appending, sets ``truncated=True``, and that flag is written into the run summary.
A silently truncated recording would be worse than no recording.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..errors import FlyBrainError


@dataclass
class RecordingConfig:
    """What to record. Every field defaults to "cheap and bounded"."""

    record_spikes: bool = True
    spike_neuron_ids: Tuple[int, ...] = ()          # empty = every spiking neuron
    spike_populations: Tuple[str, ...] = ()
    max_spike_rows: int = 20_000_000                # ~1 GB of parquet at 3 cols

    record_population: bool = True
    population_names: Tuple[str, ...] = ()
    population_interval_ms: float = 10.0

    record_voltage_ids: Tuple[int, ...] = ()
    voltage_interval_ms: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_spikes": self.record_spikes,
            "spike_neuron_ids": list(self.spike_neuron_ids),
            "spike_populations": list(self.spike_populations),
            "max_spike_rows": self.max_spike_rows,
            "record_population": self.record_population,
            "population_names": list(self.population_names),
            "population_interval_ms": self.population_interval_ms,
            "record_voltage_ids": list(self.record_voltage_ids),
            "voltage_interval_ms": self.voltage_interval_ms,
        }


class Recorder:
    """Accumulates the selected channels while the network steps."""

    def __init__(
        self,
        connectome,
        cfg: RecordingConfig,
        *,
        registry=None,
        dt_ms: float = 0.1,
    ):
        from ..model.populations import PopulationRegistry

        self.connectome = connectome
        self.cfg = cfg
        self.registry = registry or PopulationRegistry.builtin()
        self.dt_ms = float(dt_ms)
        self.n = connectome.n_neurons

        # -- spike channel ---------------------------------------------------------
        self._spike_steps: List[int] = []
        self._spike_idx: List[np.ndarray] = []
        self.spike_rows = 0
        self.truncated = False

        self._watch_idx: Optional[np.ndarray] = None
        self.watch_ids: Tuple[int, ...] = ()
        if cfg.record_spikes and cfg.spike_neuron_ids:
            ids = list(cfg.spike_neuron_ids)
            present, missing = connectome.filter_present(ids)
            if missing.size:
                raise FlyBrainError(
                    f"{missing.size} spike-recording neuron ID(s) are not in the dataset; "
                    f"first: {missing[:5].tolist()}"
                )
            self._watch_idx = connectome.indices_of(present.tolist())
            self.watch_ids = tuple(int(x) for x in present.tolist())

        # -- population channel ----------------------------------------------------
        self._pop_names: List[str] = []
        self._pop_idx: List[np.ndarray] = []
        self._pop_rates: List[Dict[str, Any]] = []
        self._prev_counts: Optional[np.ndarray] = None
        self._next_sample_step = 0
        self.population_missing: Dict[str, List[int]] = {}

        if cfg.record_population:
            names = list(cfg.population_names) or _default_populations()
            for name in names:
                present, missing = self.registry.resolve(name, connectome.flywire_ids)
                self.population_missing[name] = [int(x) for x in missing.tolist()]
                if present.size == 0:
                    continue
                self._pop_names.append(name)
                self._pop_idx.append(connectome.indices_of(present.tolist()))
            self._sample_every = max(
                1, int(round(cfg.population_interval_ms / self.dt_ms))
            )
        else:
            self._sample_every = 0

        # -- voltage channel -------------------------------------------------------
        self._volt_idx: Optional[np.ndarray] = None
        self._volt_rows: List[Tuple[int, np.ndarray]] = []
        self.volt_interval = 0
        if cfg.record_voltage_ids:
            present, missing = connectome.filter_present(list(cfg.record_voltage_ids))
            if missing.size:
                raise FlyBrainError(
                    f"{missing.size} voltage-recording neuron ID(s) are not in the dataset"
                )
            self._volt_idx = connectome.indices_of(present.tolist())
            self.volt_interval = max(
                1, int(round(cfg.voltage_interval_ms / self.dt_ms))
            )

        # -- summary channel -------------------------------------------------------
        self.active_per_step: List[int] = []

    # ------------------------------------------------------------------------------
    def begin(self, network) -> None:
        """Prepare per-run accumulators. Called once per trial, after reset."""
        self._spike_steps.clear()
        self._spike_idx.clear()
        self.spike_rows = 0
        self.truncated = False
        self._pop_rates.clear()
        self._volt_rows.clear()
        self.active_per_step.clear()
        self._next_sample_step = self._sample_every
        self._prev_counts = np.asarray(network.spike_count).copy() * 0.0

    def on_step(self, step: int, spikes, network) -> None:
        """Consume one step's spike vector."""
        spk = np.asarray(spikes)

        # -- spikes ---------------------------------------------------------------
        if self.cfg.record_spikes:
            idx = self._watch_idx if self._watch_idx is not None else np.flatnonzero(spk)
            if idx.size:
                hit = idx[spk[idx] > 0] if self._watch_idx is not None else idx
                if hit.size:
                    if self.spike_rows + hit.size > self.cfg.max_spike_rows:
                        room = max(self.cfg.max_spike_rows - self.spike_rows, 0)
                        hit = hit[:room]
                        self.truncated = True
                    if hit.size:
                        self._spike_steps.append(step)
                        self._spike_idx.append(hit.astype(np.int32))
                        self.spike_rows += int(hit.size)

        # -- population rates -----------------------------------------------------
        if self.cfg.record_population and self._sample_every and (step + 1) >= self._next_sample_step:
            counts = np.asarray(network.spike_count)
            delta = counts - self._prev_counts
            self._prev_counts = counts.copy()
            window_s = (self._sample_every * self.dt_ms) / 1000.0
            t_ms = (step + 1) * self.dt_ms
            for name, idx in zip(self._pop_names, self._pop_idx):
                n_members = idx.size
                rate = float(delta[idx].sum() / (n_members * window_s))
                self._pop_rates.append(
                    {"t_ms": t_ms, "population": name, "n_neurons": n_members, "rate_hz": rate}
                )
            self._next_sample_step += self._sample_every

        # -- membrane potential ---------------------------------------------------
        if self._volt_idx is not None and self.volt_interval and (step + 1) % self.volt_interval == 0:
            v = np.asarray(network.v)[self._volt_idx]
            self._volt_rows.append((step + 1, v.astype(np.float32).copy()))

        # -- network summary -----------------------------------------------------
        self.active_per_step.append(int((spk > 0).sum()))

    # ------------------------------------------------------------------------------
    # materialisation
    # ------------------------------------------------------------------------------
    def spikes_frame(self):
        """Spike table as a DataFrame, or None when there is nothing to write."""
        import pandas as pd

        if not self.cfg.record_spikes or not self._spike_steps:
            return None
        steps = np.concatenate(
            [np.full(idx.size, s, dtype=np.int64) for s, idx in zip(self._spike_steps, self._spike_idx)]
        )
        indices = np.concatenate(self._spike_idx).astype(np.int64)
        ids = self.connectome.flywire_ids[indices]
        return pd.DataFrame(
            {
                "t_ms": steps.astype(np.float64) * self.dt_ms,
                "step": steps,
                "flywire_id": ids,
                "index": indices.astype(np.int32),
            }
        )

    def population_frame(self):
        import pandas as pd

        if not self._pop_rates:
            return None
        return pd.DataFrame(self._pop_rates, columns=["t_ms", "population", "n_neurons", "rate_hz"])

    def voltage_frame(self):
        import pandas as pd

        if not self._volt_rows or self._volt_idx is None:
            return None
        ids = self.connectome.flywire_ids[self._volt_idx]
        rows = []
        for step, v in self._volt_rows:
            rows.append(
                pd.DataFrame(
                    {
                        "t_ms": np.full(ids.size, step * self.dt_ms),
                        "flywire_id": ids,
                        "v_mv": v,
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def summary(self) -> Dict[str, Any]:
        aps = np.asarray(self.active_per_step, dtype=np.int64) if self.active_per_step else np.zeros(1, np.int64)
        out = {
            "recorded_spike_rows": int(self.spike_rows),
            "spike_recording_truncated": bool(self.truncated),
            "spike_recording_scope": (
                "all spiking neurons" if not self.cfg.spike_neuron_ids else f"{len(self.watch_ids)} selected neurons"
            ),
            "n_steps": int(len(self.active_per_step)),
            "active_neurons_per_step": {
                "max": int(aps.max()),
                "mean": float(aps.mean()),
                "min": int(aps.min()),
                "total_neuron_steps_active": int(aps.sum()),
            },
            "population_interval_ms": self.cfg.population_interval_ms if self.cfg.record_population else None,
            "recorded_populations": list(self._pop_names),
            "populations_with_missing_members": {
                k: len(v) for k, v in self.population_missing.items() if v
            },
            "voltage_samples": len(self._volt_rows),
        }
        return out


def _default_populations() -> List[str]:
    """A small default panel: the stimulus-relevant sensory inputs and the readout.

    Deliberately short. Recording all 114 curated populations by default would be
    the kind of unbounded output the objective warns against.
    """
    return ["sugar_grn", "bitter_grn", "water_grn", "ir94e_grn", "mn9"]
