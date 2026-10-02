"""C1: one private cue, an arbitrary 3-symbol channel, one binary choice.

Actors only receive frozen observations. Evaluator access is intentionally separate;
this is an information-flow contract for trusted code, not a Python security sandbox.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    phase: str
    role: str
    options: tuple[int, int]
    private_cue: int | None = None
    message: int | None = None


class HiddenChoice:
    """World reward depends only on chosen object, never on signal identity.

    Messages 0 and 1 are detectable emissions; 2 is declared silence. None means
    no reception/before reception. Both silence and a blocked channel contain no
    private cue by themselves. Public order and roles are independent of target.
    """

    agents = ("individual-a", "individual-b")

    def __init__(self, seed):
        hidden, public = np.random.SeedSequence(seed).spawn(2)
        self._hidden_rng = np.random.default_rng(hidden)
        self._public_rng = np.random.default_rng(public)
        self.phase = "uninitialized"

    def reset(self):
        return self.reset_for_evaluation(
            int(self._hidden_rng.integers(2)),
            int(self._public_rng.integers(2)),
            tuple(int(x) for x in self._public_rng.permutation(2)),
        )

    def reset_for_evaluation(self, target, sender_index, options):
        """Evaluator-only controlled world construction; never passed to actors."""
        if (
            target not in (0, 1)
            or sender_index not in (0, 1)
            or sorted(options) != [0, 1]
        ):
            raise ValueError("invalid controlled world")
        self._target = int(target)
        self.sender = self.agents[sender_index]
        self.receiver = self.agents[1 - sender_index]
        self.options = tuple(options)
        self.phase = "signal"
        self._message = None
        return self.observations()

    def observations(self):
        return {
            agent: Observation(
                self.phase,
                "sender" if agent == self.sender else "receiver",
                self.options,
                self._target
                if agent == self.sender and self.phase == "signal"
                else None,
                self._message
                if agent == self.receiver and self.phase == "choice"
                else None,
            )
            for agent in self.agents
        }

    def step(self, actions):
        if self.phase == "signal":
            if (
                set(actions) != {self.sender}
                or type(actions[self.sender]) is not int
                or actions[self.sender] not in (0, 1, 2)
            ):
                raise ValueError("signal phase requires only sender action in {0,1,2}")
            self._message = actions[self.sender]
            self.phase = "choice"
            return self.observations(), {a: 0.0 for a in self.agents}, False
        if self.phase == "choice":
            if (
                set(actions) != {self.receiver}
                or type(actions[self.receiver]) is not int
                or actions[self.receiver] not in (0, 1)
            ):
                raise ValueError("choice phase requires only receiver action in {0,1}")
            reward = float(self.options[actions[self.receiver]] == self._target)
            self.phase = "terminal"
            self._message = None
            return self.observations(), {a: reward for a in self.agents}, True
        raise ValueError("reset required before another action")

    def snapshot(self):
        """Full world/RNG snapshot; used only by the evaluator."""
        return copy.deepcopy(self)

    def intervene_reception(self, message):
        if self.phase != "choice" or (message is not None and message not in (0, 1, 2)):
            raise ValueError(
                "intervene only after emission and before recipient observation"
            )
        self._message = message

    def expected_reward(self, action_probabilities):
        """Evaluator-only scoring of a frozen receiver distribution."""
        return float(action_probabilities[self.options.index(self._target)])
