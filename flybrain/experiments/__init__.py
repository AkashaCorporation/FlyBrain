"""Packaged experiments.

Each module exposes ``run(argv) -> int`` returning a process exit code, so the
experiments can be invoked three ways:

* ``flybrain experiments <name> [args]``
* ``python -m flybrain.experiments.<name> [args]``
* ``python experiments/<name>.py [args]``  (thin wrappers at the project root)

Each experiment writes a JSON report containing its hypothesis, its measured
numbers, and the individual checks with their evidence.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import baseline_activity, silencing_test, smoke_test, sugar_stimulation

EXPERIMENTS: Dict[str, Tuple[Callable[..., int], str]] = {
    "smoke_test": (
        smoke_test.main,
        "initialise, stimulate, propagate, verify determinism (no dataset needed)",
    ),
    "baseline_activity": (
        baseline_activity.main,
        "measure spontaneous activity with no stimulation",
    ),
    "sugar_stimulation": (
        sugar_stimulation.main,
        "stimulate the sugar-sensing GRNs and check downstream propagation",
    ),
    "silencing_test": (
        silencing_test.main,
        "stimulate A, silence B, compare against the un-silenced baseline",
    ),
}


def run_named(name: str, argv: Optional[Sequence[str]] = None) -> int:
    fn, _doc = EXPERIMENTS[name]
    return int(fn(list(argv) if argv is not None else None))


def names() -> List[str]:
    return sorted(EXPERIMENTS)


__all__ = ["EXPERIMENTS", "run_named", "names"]
