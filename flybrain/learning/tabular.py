"""Independent tabular score-function learning with local outcome baselines."""

from __future__ import annotations

import time

import numpy as np

from ..social.choice import HiddenChoice
from ..social.policies import Policy


class TabularPolicy(Policy):
    def __init__(
        self, init_seed, action_seed, *, learning_rate=0.06, baseline_rate=0.05
    ):
        super().__init__(action_seed)
        init = np.random.default_rng(init_seed)
        self.sender_logits = init.normal(0, 0.01, (2, 3))
        # option order x received symbol (including blocked/absent) x action index
        self.receiver_logits = init.normal(0, 0.01, (2, 4, 2))
        self.sender_baseline = np.zeros(2)
        self.receiver_baseline = np.zeros((2, 4))
        self.learning_rate = learning_rate
        self.baseline_rate = baseline_rate

    def _row(self, obs):
        if obs.role == "sender":
            if obs.phase != "signal" or obs.private_cue not in (0, 1):
                raise ValueError("sender requires local cue during signal phase")
            return self.sender_logits, self.sender_baseline, obs.private_cue
        if (
            obs.role != "receiver"
            or obs.phase != "choice"
            or obs.private_cue is not None
        ):
            raise ValueError("receiver requires local choice observation only")
        key = (obs.options[0], 3 if obs.message is None else obs.message)
        return self.receiver_logits, self.receiver_baseline, key

    def distribution(self, obs):
        table, _, key = self._row(obs)
        row = table[key]
        p = np.exp(row - row.max())
        return p / p.sum()

    def learn(self, observation, action, reward):
        """Only own observation/action and cooperative environmental outcome."""
        if reward not in (0.0, 1.0):
            raise ValueError("C1 outcome must be binary")
        table, baseline, key = self._row(observation)
        gradient = -self.distribution(observation)
        gradient[action] += 1
        advantage = reward - baseline[key]
        table[key] += self.learning_rate * advantage * gradient
        baseline[key] += self.baseline_rate * advantage


def make_individuals(seed):
    roots = np.random.SeedSequence(seed).spawn(5)
    values = [int(s.generate_state(1)[0]) for s in roots]
    agents = {
        a: TabularPolicy(values[1 + 2 * i], values[2 + 2 * i])
        for i, a in enumerate(HiddenChoice.agents)
    }
    return HiddenChoice(values[0]), agents, values


def train(env, policies, *, episodes=12000, max_seconds=30, curve_every=1000):
    if not 0 < episodes <= 50000 or not 0 < max_seconds <= 600 or curve_every <= 0:
        raise ValueError("invalid pilot budget")
    start = time.perf_counter()
    curve = []
    window = []
    completed = 0
    for episode in range(episodes):
        if time.perf_counter() - start >= max_seconds:
            break
        obs = env.reset()
        sender = policies[env.sender]
        receiver = policies[env.receiver]
        sender_obs = obs[env.sender]
        message, _ = sender.act(sender_obs)
        obs, _, _ = env.step({env.sender: message})
        receiver_obs = obs[env.receiver]
        action, _ = receiver.act(receiver_obs)
        _, reward, done = env.step({env.receiver: action})
        assert done
        sender.learn(sender_obs, message, reward[env.sender])
        receiver.learn(receiver_obs, action, reward[env.receiver])
        window.append(reward[env.receiver])
        completed = episode + 1
        if completed % curve_every == 0 or completed == episodes:
            curve.append(
                {
                    "episodes": completed,
                    "mean_training_return": float(np.mean(window)),
                    "window_episodes": len(window),
                }
            )
            window.clear()
    if window:
        curve.append(
            {
                "episodes": completed,
                "mean_training_return": float(np.mean(window)),
                "window_episodes": len(window),
            }
        )
    return {
        "completed": completed == episodes,
        "stop_reason": "finished" if completed == episodes else "wall_limit",
        "episodes_requested": episodes,
        "episodes_completed": completed,
        "elapsed_seconds": time.perf_counter() - start,
        "curve": curve,
        "algorithm": "independent_tabular_REINFORCE_with_local_EMA_baseline",
        "learning_rate": 0.06,
        "baseline_rate": 0.05,
        "entropy_bonus": 0.0,
        "reward": "cooperative_environment_outcome_only",
        "parameters_shared": False,
        "parameters_per_individual": 22,
        "baseline_values_per_individual": 10,
    }
