"""Interactive experiment console.

This is **not** conversation with a simulated organism, and it is not a language
interface. It is a command console for us: every command is a named operation on
the network with explicit arguments, and the help text says so. The distinction
matters because the objective explicitly puts natural-language interaction out of
scope, and a console that blurred it would be the first step toward doing it by
accident.

Commands mirror the five operations the network exposes
(``apply_stimulus`` / ``step`` / ``observe`` / ``silence`` / ``reset``).
"""

from __future__ import annotations

import shlex
import time
from typing import List, Sequence, Tuple

import numpy as np

from . import config as fb_config
from .analysis import neuron_profile
from .data import load_connectome
from .errors import FlyBrainError, UnknownNeuronError, UnknownPopulationError
from .model import LIFNetwork, LIFParams, Stimulus
from .model.populations import PopulationRegistry
from .runtime import detect_hardware, get_backend, plan_run

BANNER = """FlyBrain v0 - experiment console

This is a command console for running experiments on the connectome model.
It is NOT a language interface to the simulated organism, and the model has no
language, no learning, and no decision layer. Type `help` for the command list.
"""

HELP = """
Commands
  stimulate <population> [rate_hz] [duration_ms]   drive a population and run
  run [duration_ms]                               advance the simulation
  silence <population|flywire_id>                 remove a neuron group's output
  unsilence <population|flywire_id>               undo the above
  reset                                           zero state, keep interventions
  clear                                           drop all interventions
  top-active [n]                                  highest-rate neurons since reset
  populations [search]                            list curated populations
  neuron <flywire_id>                             inspect one neuron
  status                                          current network and run state
  save [dir]                                      write the current state to JSON
  seed <n>                                        set the RNG seed (implies reset)
  help / quit
"""


class Console:
    """Stateful command processor over one :class:`LIFNetwork`."""

    def __init__(
        self,
        *,
        dataset_id: str = "flywire_630",
        backend_name: str = "auto",
        mode: str = "subset",
        duration_ms: float = 200.0,
        seed: int = 0,
        subset_hops: int = 2,
        subset_max_neurons: int = 5000,
        echo=print,
    ):
        self.echo = echo
        self.registry = PopulationRegistry.builtin()
        self.dataset_id = dataset_id
        self.mode = mode
        self.seed = int(seed)
        self.default_duration_ms = float(duration_ms)
        self.subset_hops = subset_hops
        self.subset_max_neurons = subset_max_neurons

        self.connectome = load_connectome(
            dataset_id, fb_config.RAW_DIR, fb_config.PROCESSED_DIR, fb_config.METADATA_DIR
        )
        self.full_connectome = self.connectome
        self.hardware = detect_hardware()
        self.backend = get_backend(backend_name, "float32")
        self._build()

    # ------------------------------------------------------------------------------
    def _build(self) -> None:
        import dataclasses

        from .config import RunConfig

        cfg = RunConfig(
            dataset_id=self.dataset_id,
            mode=self.mode,
            duration_ms=max(self.default_duration_ms, 1.0),
            dt_ms=fb_config.DEFAULT_DT_MS,
            seed=self.seed,
            subset_hops=self.subset_hops,
            subset_max_neurons=self.subset_max_neurons,
        )
        # The console must open with a usable working set before the operator has
        # declared anything, so it seeds from the sensory circuit the packaged
        # experiments use. Any later command that needs a population outside this
        # set grows it explicitly (`_ensure_targets`) rather than failing.
        if cfg.mode == "subset" and not cfg.subset_seed_ids:
            cfg = dataclasses.replace(cfg, subset_seed_ids=self._default_seed_ids())
        cfg.validate()
        self.cfg = cfg
        # The console always starts from an explicit plan, so a "whole-brain" console
        # that silently became a subset cannot happen here either.
        self.plan = plan_run(self.full_connectome, cfg, [], hardware=self.hardware)
        self.connectome = self.plan.connectome
        self.params = LIFParams()
        self.net = LIFNetwork(
            self.connectome, self.params, dt_ms=cfg.dt_ms,
            backend=self.backend, seed=self.seed, registry=self.registry,
        )
        self.steps_run = 0
        self.total_seconds = 0.0

    def _default_seed_ids(self) -> List[int]:
        """Working-set seeds for a fresh console: the sugar circuit plus the readout."""
        seeds: List[int] = []
        for name in ("sugar_grn", "mn9"):
            try:
                present, _missing = present_in(self.registry.get(name), self.full_connectome)
            except (UnknownPopulationError, FlyBrainError):
                continue
            seeds.extend(int(x) for x in present)
        if not seeds:
            seeds = [int(self.full_connectome.flywire_ids[0])]
        return sorted(set(seeds))

    def _rebuild_net(self) -> None:
        """Rebuild the network if a stimulus target is outside the current subset."""
        needed: List[int] = []
        for s in getattr(self.net, "_declared_stimuli", ()):
            needed.extend(s.neuron_ids)
        needed.extend(self.net.silenced_ids)
        if not needed:
            return
        present, missing = self.connectome.filter_present(needed)
        if missing.size == 0:
            return
        # Grow the subset deterministically around everything the console needs,
        # instead of refusing or silently dropping members.
        sub = self.full_connectome.subset(
            [int(x) for x in needed], hops=max(self.subset_hops, 2),
            max_neurons=max(self.subset_max_neurons, len(needed) + 1000),
        )
        self.echo(
            f"  (growing working set to {sub.n_neurons:,} neurons to cover "
            f"{missing.size} neuron(s) outside the previous subset)"
        )
        declared = tuple(getattr(self.net, "_declared_stimuli", ()))
        silenced = list(self.net.silenced_ids)
        self.connectome = sub
        self.net = LIFNetwork(
            sub, self.params, dt_ms=self.cfg.dt_ms,
            backend=self.backend, seed=self.seed, registry=self.registry,
        )
        if declared:
            self.net.apply_stimulus(*declared)
        if silenced:
            self.net.silence(flywire_ids=silenced)

    # ------------------------------------------------------------------------------
    def prompt(self) -> str:
        return "flybrain> "

    def banner(self) -> str:
        n = self.connectome
        lines = [
            BANNER,
            f"dataset: {n.dataset_id}   neurons: {n.n_neurons:,}   edges: {n.n_edges:,}",
            f"mode: {self.plan.effective_mode}   backend: {self.backend.name}"
            f" ({'GPU' if getattr(self.backend, 'is_gpu', False) else 'CPU'})   dt: {self.cfg.dt_ms} ms",
        ]
        if self.plan.mode_changed:
            lines.append(
                "WARNING: a whole-brain console was requested but a SUBSET is loaded. "
                "Results are not whole-brain."
            )
        for note in self.plan.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------------------
    def dispatch(self, line: str) -> bool:
        """Execute one command line. Returns False to end the session."""
        # Strip a UTF-8 BOM as well as ordinary whitespace: a command file produced on
        # Windows (or piped through PowerShell) commonly begins with one, and without
        # this the very first command would silently be reported as unknown.
        line = line.lstrip("\ufeff").strip()
        if not line:
            return True
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            self.echo(f"could not parse command: {exc}")
            return True
        cmd, rest = parts[0].lower(), parts[1:]
        handler = getattr(self, f"cmd_{cmd}", None)
        if handler is None:
            self.echo(f"unknown command {cmd!r}. Type `help`.")
            return True
        try:
            handler(rest)
        except (FlyBrainError, UnknownPopulationError, UnknownNeuronError, ValueError) as exc:
            self.echo(f"error: {exc}")
        except KeyboardInterrupt:
            self.echo("interrupted")
        return True

    # -- commands ------------------------------------------------------------------
    def cmd_help(self, rest: List[str]) -> None:
        self.echo(HELP.strip())

    def cmd_quit(self, rest: List[str]) -> None:
        raise SystemExit(0)

    cmd_exit = cmd_quit

    def cmd_stimulate(self, rest: List[str]) -> None:
        if not rest:
            raise FlyBrainError("usage: stimulate <population> [rate_hz] [duration_ms]")
        name = rest[0]
        rate = float(rest[1]) if len(rest) > 1 else 150.0
        duration = float(rest[2]) if len(rest) > 2 else self.default_duration_ms
        pop = self.registry.get(name)
        present, missing = present_in(pop, self.full_connectome)
        if present.size == 0:
            raise FlyBrainError(f"population {name!r} has no members in {self.dataset_id}")
        if missing.size:
            self.echo(f"  ({missing.size} of {len(pop)} {name} neurons are absent from {self.dataset_id})")
        self._ensure_targets([int(x) for x in present])
        stim = Stimulus(neuron_ids=tuple(int(x) for x in present), rate_hz=rate, label=name)
        self.net.apply_stimulus(stim)
        self.echo(f"stimulating {name}: {stim.describe()}")
        self._run(duration)
        self._report()

    def cmd_run(self, rest: List[str]) -> None:
        duration = float(rest[0]) if rest else self.default_duration_ms
        self._run(duration)
        self._report()

    def cmd_silence(self, rest: List[str]) -> None:
        if not rest:
            raise FlyBrainError("usage: silence <population|flywire_id>")
        target = rest[0]
        if target.isdigit() and len(target) > 10:
            ids = [int(target)]
        else:
            present, missing = present_in(self.registry.get(target), self.full_connectome)
            if present.size == 0:
                raise FlyBrainError(f"population {target!r} has no members in {self.dataset_id}")
            ids = [int(x) for x in present]
            if missing.size:
                self._ensure_targets(ids)
        silenced = self.net.silence(flywire_ids=ids)
        self.echo(f"silenced {len(silenced)} neuron(s) (outgoing influence only)")

    def cmd_unsilence(self, rest: List[str]) -> None:
        if not rest:
            raise FlyBrainError("usage: unsilence <population|flywire_id>")
        target = rest[0]
        if target.isdigit() and len(target) > 10:
            self.net.unsilence(flywire_ids=[int(target)])
        else:
            self.net.unsilence(target)
        self.echo(f"un-silenced {target}; {len(self.net.silenced_ids)} still silenced")

    def cmd_reset(self, rest: List[str]) -> None:
        self.net.reset(seed=self.seed)
        self.steps_run = 0
        self.total_seconds = 0.0
        self.echo("state reset (stimuli and silencing kept)")

    def cmd_clear(self, rest: List[str]) -> None:
        self.net.clear_stimuli()
        self.net.clear_silencing()
        self.net.reset(seed=self.seed)
        self.steps_run = 0
        self.total_seconds = 0.0
        self.echo("all stimuli and silencing cleared; state reset")

    def cmd_top_active(self, rest: List[str]) -> None:
        n = int(rest[0]) if rest else 20
        rates = self.net.rates_hz()
        ids = self.connectome.flywire_ids
        order = np.lexsort((ids, -rates))[:n]
        if self.steps_run == 0:
            self.echo("nothing has run yet; use `run <ms>` or `stimulate ...`")
            return
        self.echo(f"top {n} neurons by mean rate since reset "
                  f"({self.steps_run * self.cfg.dt_ms:.0f} ms elapsed):")
        for i in order:
            if rates[i] <= 0:
                break
            prof = neuron_profile(self.connectome, int(ids[i]), registry=self.registry)
            tags = ",".join(prof["populations"]) or "-"
            self.echo(f"  {int(ids[i]):<20} {rates[i]:7.2f} Hz  {prof['neurotransmitter_sign']:<24} {tags}")

    def cmd_populations(self, rest: List[str]) -> None:
        if rest:
            for p in self.registry.find(rest[0]):
                present, missing = present_in(p, self.full_connectome)
                self.echo(f"  {p.name:<20} {len(p):>4} declared, {present.size:>4} present   {p.description}")
            return
        names = ", ".join(self.registry.names())
        self.echo(f"{len(self.registry)} populations: {names}")

    def cmd_neuron(self, rest: List[str]) -> None:
        if not rest:
            raise FlyBrainError("usage: neuron <flywire_id>")
        fid = int(rest[0])
        prof = neuron_profile(self.connectome, fid, registry=self.registry)
        rate = None
        if self.steps_run:
            idx = self.connectome.index_of(fid)
            rate = float(self.net.rates_hz()[idx])
        self.echo(f"FlyWire ID: {prof['flywire_id']}  (index {prof['index']} in {self.connectome.dataset_id})")
        self.echo(f"type: {prof['neuron_type'] or 'unavailable from dataset'}")
        self.echo(f"neurotransmitter: {prof['neurotransmitter_sign']}")
        if prof["populations"]:
            self.echo(f"curated populations: {', '.join(prof['populations'])}")
        self.echo(f"incoming: {prof['in_degree']} connections / {prof['in_synapse_count']} synapses")
        self.echo(f"outgoing: {prof['out_degree']} connections / {prof['out_synapse_count']} synapses")
        if rate is not None:
            self.echo(f"current firing rate: {rate:.2f} Hz")

    def cmd_status(self, rest: List[str]) -> None:
        ns = len(getattr(self.net, "_declared_stimuli", ()))
        self.echo(f"dataset          {self.connectome.dataset_id} ({self.connectome.n_neurons:,} neurons, "
                  f"{self.connectome.n_edges:,} edges)"
                  f"{' - SUBSET' if self.connectome.is_subset else ''}")
        self.echo(f"backend          {self.backend.name} "
                  f"({'GPU' if getattr(self.backend, 'is_gpu', False) else 'CPU'}), dt={self.cfg.dt_ms} ms")
        self.echo(f"elapsed          {self.steps_run} steps = {self.steps_run * self.cfg.dt_ms:.1f} ms "
                  f"({self.total_seconds:.2f} s wall)")
        self.echo(f"stimuli          {ns}")
        for s in getattr(self.net, "_declared_stimuli", ()):
            self.echo(f"  - {s.describe()}")
        self.echo(f"silenced         {len(self.net.silenced_ids)} neuron(s)")
        self.echo(f"spikes since reset {int(np.asarray(self.net.spike_count).sum())}")

    def cmd_seed(self, rest: List[str]) -> None:
        if not rest:
            raise FlyBrainError("usage: seed <n>")
        self.seed = int(rest[0])
        self.net.reset(seed=self.seed)
        self.steps_run = 0
        self.total_seconds = 0.0
        self.echo(f"seed set to {self.seed} and state reset")

    def cmd_save(self, rest: List[str]) -> None:
        import json
        from pathlib import Path

        out = Path(rest[0]) if rest else (fb_config.OUTPUTS_DIR / "interactive")
        out.mkdir(parents=True, exist_ok=True)
        rates = self.net.rates_hz()
        ids = self.connectome.flywire_ids
        order = np.lexsort((ids, -rates))[:200]
        payload = {
            "dataset": {
                "dataset_id": self.connectome.dataset_id,
                "n_neurons": self.connectome.n_neurons,
                "n_edges": self.connectome.n_edges,
                "is_subset": self.connectome.is_subset,
            },
            "backend": self.backend.name,
            "dt_ms": self.cfg.dt_ms,
            "steps_run": self.steps_run,
            "elapsed_ms": self.steps_run * self.cfg.dt_ms,
            "seed": self.seed,
            "model_params": self.params.to_dict(),
            "stimuli": [s.to_dict() for s in getattr(self.net, "_declared_stimuli", ())],
            "silenced_flywire_ids": list(self.net.silenced_ids),
            "total_spikes": int(np.asarray(self.net.spike_count).sum()),
            "top_neurons": [
                {"flywire_id": int(ids[i]), "rate_hz": float(rates[i])}
                for i in order if rates[i] > 0
            ],
        }
        p = out / "console_state.json"
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.echo(f"wrote {p}")

    # -- internals ------------------------------------------------------------------
    def _ensure_targets(self, ids: Sequence[int]) -> None:
        """Grow the working set so every id in ``ids`` is present, preserving interventions.

        The console runs on a subset for speed. Rather than silently ignoring a
        target that falls outside it, or refusing the command, the working set is
        rebuilt to include it and the operator is told that happened - and that the
        clock restarted, so the activity numbers that follow are from a fresh run.
        """
        if not ids:
            return
        _present, missing = self.connectome.filter_present(ids)
        if missing.size == 0:
            return

        declared = tuple(getattr(self.net, "_declared_stimuli", ()))
        silenced = list(self.net.silenced_ids)
        needed = {int(x) for x in ids} | {int(x) for x in silenced}
        for s in declared:
            needed.update(int(x) for x in s.neuron_ids)

        sub = self.full_connectome.subset(
            sorted(needed),
            hops=max(self.subset_hops, 2),
            max_neurons=max(self.subset_max_neurons, len(needed) + 2000),
        )
        self.connectome = sub
        self.net = LIFNetwork(
            sub, self.params, dt_ms=self.cfg.dt_ms,
            backend=self.backend, seed=self.seed, registry=self.registry,
        )
        if declared:
            self.net.apply_stimulus(*declared)
        if silenced:
            self.net.silence(flywire_ids=silenced)
        self.steps_run = 0
        self.total_seconds = 0.0
        self.echo(
            f"  (working set grown to {sub.n_neurons:,} neurons to include "
            f"{missing.size} neuron(s); simulation clock restarted)"
        )

    def _run(self, duration_ms: float) -> None:
        n_steps = int(round(duration_ms / self.cfg.dt_ms))
        if n_steps <= 0:
            self.echo("nothing to do (duration rounds to 0 steps)")
            return
        self.echo(f"Running {n_steps} steps ({duration_ms:g} ms) ...")
        t0 = time.perf_counter()
        chunk = max(n_steps // 4, 1)
        for k in range(0, n_steps, chunk):
            m = min(chunk, n_steps - k)
            for _ in range(m):
                self.net.step()
            self.steps_run += m
            # report progress within *this* call, not against the cumulative counter
            if n_steps >= 2000 and (k + m) < n_steps:
                self.echo(f"  {k + m}/{n_steps} steps "
                          f"({time.perf_counter() - t0:.1f}s)")
        self.total_seconds += time.perf_counter() - t0

    def _report(self) -> None:
        total = int(np.asarray(self.net.spike_count).sum())
        self.echo(
            f"Spikes: {total} in {self.steps_run * self.cfg.dt_ms:.0f} ms "
            f"({self.total_seconds:.2f} s wall)"
        )
        rates = self.net.rates_hz()
        active = int((rates > 0).sum())
        self.echo(f"Active neurons: {active} of {self.connectome.n_neurons:,} "
                  f"(max rate {rates.max():.1f} Hz)")
        pops = []
        for name in self._report_panel():
            try:
                r = self.net.population_rate_hz(name)
            except UnknownPopulationError:
                continue
            if r > 0:
                pops.append((name, r))
        pops.sort(key=lambda kv: -kv[1])
        if pops:
            self.echo("Active populations: " + ", ".join(f"{n}={r:.1f}Hz" for n, r in pops[:10]))

    def _report_panel(self) -> List[str]:
        """Which populations to summarise after a run.

        The stimulated populations plus a fixed sensory-and-readout panel, rather
        than all 114 curated populations: a 114-line summary hides the answer.
        """
        names: List[str] = []
        for s in getattr(self.net, "_declared_stimuli", ()):
            if s.label and s.label in self.registry:
                names.append(s.label)
        for extra in ("mn9", "bitter_grn", "water_grn", "ir94e_grn", "jon_all"):
            if extra in self.registry and extra not in names:
                names.append(extra)
        return names


def present_in(pop, connectome) -> Tuple[np.ndarray, np.ndarray]:
    ids = set(int(x) for x in connectome.flywire_ids)
    present = np.fromiter((i for i in pop.ids if i in ids), dtype=np.int64)
    missing = np.fromiter((i for i in pop.ids if i not in ids), dtype=np.int64)
    return present, missing


def run_repl(
    *,
    dataset_id: str = "flywire_630",
    backend: str = "auto",
    mode: str = "subset",
    duration_ms: float = 200.0,
    seed: int = 0,
    stdin=None,
    stdout=None,
) -> int:
    import sys

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    console = Console(
        dataset_id=dataset_id, backend_name=backend, mode=mode,
        duration_ms=duration_ms, seed=seed, echo=lambda s: print(s, file=stdout, flush=True),
    )
    print(console.banner(), file=stdout, flush=True)

    interactive = stdin.isatty() if hasattr(stdin, "isatty") else False
    while True:
        try:
            if interactive:
                line = input(console.prompt())
            else:
                line = stdin.readline()
                if not line:
                    break
                print(f"{console.prompt()}{line.rstrip()}", file=stdout, flush=True)
        except (EOFError, KeyboardInterrupt):
            print(file=stdout)
            break
        if not console.dispatch(line):
            break
    return 0
