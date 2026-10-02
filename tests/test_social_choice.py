"""Causal controls and side-channel regressions for the single-choice task."""

import copy
import pickle
from dataclasses import FrozenInstanceError, replace
from itertools import permutations

import numpy as np
import pytest

from flybrain.social.choice import HiddenChoice
from flybrain.social.evaluation import control_gate, evaluate
from flybrain.social.policies import ManualChannelControl, RandomPolicy


def test_exhaustive_hidden_state_observation_reward_and_phase_isolation():
    for sender in (0, 1):
        for options in ((0, 1), (1, 0)):
            histories = []
            for target in (0, 1):
                env = HiddenChoice(7)
                obs = env.reset_for_evaluation(target, sender, options)
                receiver = env.receiver
                before = obs[receiver]
                assert set(before.__dict__) == {
                    "phase",
                    "role",
                    "options",
                    "private_cue",
                    "message",
                }
                assert before.private_cue is None and before.message is None
                with pytest.raises(FrozenInstanceError):
                    before.private_cue = target
                emitted, reward, done = env.step({env.sender: 1})
                assert reward == dict.fromkeys(env.agents, 0.0) and not done
                assert emitted[receiver].private_cue is None
                histories.append((before, emitted[receiver], reward))
                terminal, _, done = env.step({receiver: 0})
                assert done and all(
                    o.message is None and o.private_cue is None
                    for o in terminal.values()
                )
            assert histories[0] == histories[1]


def test_no_message_semantics_in_world_reward():
    for target in (0, 1):
        for message in (0, 1, 2):
            for action in (0, 1):
                env = HiddenChoice(1)
                env.reset_for_evaluation(target, 0, (1, 0))
                env.step({env.sender: message})
                _, rewards, _ = env.step({env.receiver: action})
                assert list(rewards.values()) == [float((1, 0)[action] == target)] * 2


def test_phases_cannot_be_shortcut_by_agent_order():
    env = HiddenChoice(3)
    env.reset()
    for actions in (
        {env.receiver: 0},
        {a: 0 for a in reversed(env.agents)},
        {env.sender: 3},
    ):
        with pytest.raises(ValueError):
            env.step(actions)
        assert env.phase == "signal"
    env.step({env.sender: 0})
    with pytest.raises(ValueError):
        env.step({env.sender: 0})
    assert env.phase == "choice"
    env.step({env.receiver: 0})
    with pytest.raises(ValueError):
        env.step({env.receiver: 0})


def test_reset_rng_independence_from_public_rng_actions_and_rewards():
    a, b = HiddenChoice(811), HiddenChoice(811)
    for k in range(100):
        # Consume extra public draws only in b: hidden target sequence is unchanged.
        b._public_rng.integers(2, size=k % 7)
        a.reset()
        b.reset()
        assert a._target == b._target
        a.step({a.sender: 0})
        b.step({b.sender: 2})
        a.step({a.receiver: 0})
        b.step({b.receiver: 1})
    a.reset()
    assert a.phase == "signal" and all(
        o.message is None for o in a.observations().values()
    )


def test_seed_roles_order_and_option_indices_do_not_predict_target():
    for seed in (13, 71, 207, 991):
        env = HiddenChoice(seed)
        bins = {k: [] for k in [(s, o) for s in env.agents for o in ((0, 1), (1, 0))]}
        for _ in range(1000):
            env.reset()
            bins[env.sender, env.options].append(env._target)
        assert all(0.35 < np.mean(v) < 0.65 for v in bins.values())


def test_pre_message_snapshot_copies_world_and_action_rng():
    env = HiddenChoice(17)
    env.reset()
    snap = env.snapshot()
    agent = RandomPolicy(53)
    saved = copy.deepcopy(agent)
    a, b = snap.snapshot(), snap.snapshot()
    a.step({a.sender: 1})
    b.step({b.sender: 1})
    assert (
        agent.act(a.observations()[a.receiver])[0]
        == saved.act(b.observations()[b.receiver])[0]
    )
    a.step({a.receiver: 0})
    assert snap.phase == "signal" and b.phase == "choice"
    for _ in range(10):
        snap.reset()
        env.reset()
        assert snap.observations() == env.observations()


def test_causal_control_gate_and_no_training_claim():
    report = control_gate(repeats=8)
    assert report["passed"] and not report["learned"]
    pos = report["controls"]["ManualChannelControl"]
    assert pos["classification"] == "functional_channel_use"
    assert pos["expected"]["gain_vs_resampled"] == pytest.approx(0.5)
    assert pos["expected"]["tv_uniform_pairs"] == pytest.approx(4 / 9)
    assert not pos["retention_tested"]
    assert pos["resampling_marginal"] == pytest.approx([0.5, 0.5, 0])
    for name in ("RandomPolicy", "NoInformationPolicy"):
        neg = report["controls"][name]
        assert neg["expected"]["intact"] == pytest.approx(0.5)
        assert neg["expected"]["gain_vs_resampled"] == pytest.approx(0)
        assert neg["classification"] == "functional_communication_not_demonstrated"


def test_evaluation_does_not_mutate_individuals_or_share_rng():
    agents = {
        a: ManualChannelControl(seed) for a, seed in zip(HiddenChoice.agents, (5, 9))
    }
    before = pickle.dumps(agents)
    report = evaluate(agents, repeats=4)
    assert pickle.dumps(agents) == before
    assert agents["individual-a"].rng is not agents["individual-b"].rng
    assert report["recording_truncated"] and len(report["recorded_episodes"]) == 16


def test_informative_sender_ignored_by_receiver_is_not_communication():
    class IgnoreMessage(ManualChannelControl):
        def distribution(self, obs):
            return (
                super().distribution(obs)
                if obs.role == "sender"
                else np.array([1.0, 0.0])
            )

    agents = {a: IgnoreMessage(s) for a, s in zip(HiddenChoice.agents, (1, 2))}
    result = evaluate(agents, repeats=4)
    assert result["expected"]["tv_0_1"] == 0
    assert result["expected"]["intact"] == 0.5


def test_functional_classification_is_invariant_to_all_symbol_permutations():
    class PermutedManual(ManualChannelControl):
        def distribution(self, obs):
            if obs.role == "sender":
                original = super().distribution(obs)
                permuted = np.zeros(3)
                for old, new in enumerate(self.code):
                    permuted[new] = original[old]
                return permuted
            message = None if obs.message is None else self.code.index(obs.message)
            return super().distribution(replace(obs, message=message))

    for code in permutations((0, 1, 2)):
        agents = {a: PermutedManual(s) for a, s in zip(HiddenChoice.agents, (1, 2))}
        for agent in agents.values():
            agent.code = code
        result = evaluate(agents, repeats=2)
        assert result["classification"] == "functional_channel_use"
        assert result["expected"]["tv_uniform_pairs"] == pytest.approx(4 / 9)


def test_receiver_sensitivity_without_informative_emission_is_not_functional():
    class UninformativeManual(ManualChannelControl):
        def distribution(self, obs):
            return (
                np.array([0.0, 0.0, 1.0])
                if obs.role == "sender"
                else super().distribution(obs)
            )

    agents = {a: UninformativeManual(s) for a, s in zip(HiddenChoice.agents, (1, 2))}
    result = evaluate(agents, repeats=2)
    assert result["expected"]["tv_uniform_pairs"] > 0.4
    assert result["expected"]["intact"] == 0.5
    assert result["classification"] == "functional_communication_not_demonstrated"
