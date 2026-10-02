"""The training channel and the evaluation channel must be the same channel.

The C2b failure this pins: train() accepted a message_fn, so the receiver learned
a message produced by the circuit, but evaluate() had no way to inject one and
re-emitted with the sender's own policy - which is frozen and untrained when the
message comes from a circuit. Training reached 0.990 return while evaluation
reported 0.459. The receiver had learned the right thing and was then graded on a
channel it had never seen.

These tests make the two paths comparable, and they fail if either one drifts.
"""
from __future__ import annotations

import pytest

from flybrain.learning.tabular import make_individuals, train
from flybrain.social.evaluation import evaluate

EPISODES = 12_000
SEED = 20261003


def _identity(cue):
    return int(cue)


def test_receiver_learns_and_evaluation_confirms_it():
    """The end-to-end claim: if training used message_fn, evaluation must too."""
    env, agents, _ = make_individuals(SEED)
    train(env, agents, episodes=EPISODES, max_seconds=60, message_fn=_identity)

    external = evaluate(agents, seed=SEED + 20000, repeats=64, message_fn=_identity)
    without = evaluate(agents, seed=SEED + 20000, repeats=64)

    assert external["expected"]["intact"] > 0.90, (
        "with a perfect external channel the receiver should decode it; "
        f"got intact={external['expected']['intact']:.4f}"
    )
    assert external["expected"]["gain_vs_resampled"] > 0.40
    # the bug being pinned: grading without the message_fn drops the score
    assert without["expected"]["intact"] < 0.70, (
        "evaluation without message_fn should now score near chance, proving the "
        "two paths really differ; if it does not, this test no longer guards "
        "anything"
    )


def test_blocked_and_resampled_stay_at_chance_under_a_perfect_channel():
    env, agents, _ = make_individuals(SEED)
    train(env, agents, episodes=EPISODES, max_seconds=60, message_fn=_identity)
    res = evaluate(agents, seed=SEED + 20000, repeats=64, message_fn=_identity)
    e = res["expected"]
    assert e["blocked"] == pytest.approx(0.5, abs=0.02)
    assert e["resampled"] == pytest.approx(0.5, abs=0.02)
    assert res["classification"] == "functional_channel_use"


def test_evaluation_does_not_mutate_the_policies():
    env, agents, _ = make_individuals(SEED)
    train(env, agents, episodes=2000, max_seconds=30, message_fn=_identity)
    before = {n: p.receiver_logits.copy() for n, p in agents.items()}
    evaluate(agents, seed=7, repeats=16, message_fn=_identity)
    for n, v in before.items():
        import numpy as np
        np.testing.assert_array_equal(agents[n].receiver_logits, v)


def test_bad_message_fn_is_rejected_at_evaluation_time():
    env, agents, _ = make_individuals(SEED)
    with pytest.raises(ValueError, match="not 0, 1 or 2"):
        evaluate(agents, seed=7, repeats=8, message_fn=lambda cue: 9)
    with pytest.raises(TypeError, match="callable"):
        evaluate(agents, seed=7, repeats=8, message_fn="not callable")


def test_default_path_is_unchanged():
    """C1 must still be graded entirely through the sender's own policy."""
    env, agents, _ = make_individuals(SEED)
    train(env, agents, episodes=EPISODES, max_seconds=60)
    e = evaluate(agents, seed=SEED + 20000, repeats=64)["expected"]
    assert e["intact"] > 0.90, f"C1 path regressed: intact={e['intact']:.4f}"
    assert e["gain_vs_resampled"] > 0.40