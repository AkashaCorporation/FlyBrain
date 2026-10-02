"""Controls are explicit environment checks, never training targets."""

import numpy as np


class Policy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        probabilities = np.asarray(self.distribution(observation), dtype=float)
        expected = 3 if observation.role == "sender" else 2
        if (
            probabilities.shape != (expected,)
            or np.any(probabilities < 0)
            or not np.all(np.isfinite(probabilities))
            or not np.isclose(probabilities.sum(), 1)
        ):
            raise ValueError("invalid action distribution")
        return int(self.rng.choice(expected, p=probabilities)), probabilities


class RandomPolicy(Policy):
    def distribution(self, obs):
        return np.full(3, 1 / 3) if obs.role == "sender" else np.full(2, 0.5)


class NoInformationPolicy(Policy):
    def distribution(self, obs):
        # Fixed action indices deliberately expose any bias in world balancing.
        return (
            np.array([0.0, 0.0, 1.0]) if obs.role == "sender" else np.array([1.0, 0.0])
        )


class ManualChannelControl(Policy):
    """Handcoded codebook ONLY for testing the causal evaluator (not learned)."""

    def distribution(self, obs):
        if obs.role == "sender":
            p = np.zeros(3)
            p[obs.private_cue] = 1
            return p
        if obs.message in (0, 1):
            p = np.zeros(2)
            p[obs.options.index(obs.message)] = 1
            return p
        return np.full(2, 0.5)
