import copy

import numpy as np

from flybrain.learning.tabular import make_individuals, train


def test_update_is_local_and_weights_rngs_are_individual():
    env, agents, _ = make_individuals(11)
    obs = env.reset()
    sender, receiver = agents[env.sender], agents[env.receiver]
    before = copy.deepcopy(receiver.__dict__)
    old = sender.sender_logits.copy()
    action, _ = sender.act(obs[env.sender])
    sender.learn(obs[env.sender], action, 1.0)
    assert not np.array_equal(old, sender.sender_logits)
    np.testing.assert_array_equal(receiver.sender_logits, before["sender_logits"])
    np.testing.assert_array_equal(receiver.receiver_logits, before["receiver_logits"])
    assert sender.rng is not receiver.rng
    assert not np.shares_memory(sender.sender_logits, receiver.sender_logits)


def test_training_budget_and_reproducibility_not_success_assumption():
    results = []
    for _ in range(2):
        env, agents, _ = make_individuals(23)
        report = train(env, agents, episodes=100, max_seconds=10, curve_every=20)
        assert report["completed"] and report["episodes_completed"] == 100
        results.append((report["curve"], agents))
    assert results[0][0] == results[1][0]
    for name in agents:
        np.testing.assert_array_equal(
            results[0][1][name].sender_logits, results[1][1][name].sender_logits
        )


def test_wall_limit_is_explicit():
    env, agents, _ = make_individuals(37)
    report = train(env, agents, episodes=50000, max_seconds=1e-12)
    assert not report["completed"] and report["stop_reason"] == "wall_limit"
    assert report["episodes_completed"] == 0
