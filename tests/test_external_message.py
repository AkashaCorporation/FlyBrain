"""The external-message path: when the circuit produces the message, the sender
must not learn.

If the sender's policy were still updated while the message came from somewhere
else, the run would credit it with an action it never chose, and the C2b claim
that only the receiver learns would be false.
"""
from __future__ import annotations

import numpy as np
import pytest

from flybrain.learning.tabular import make_individuals, train
from flybrain.social.choice import HiddenChoice


def _fingerprint(policies):
    return {
        name: {f: getattr(p, f).copy() for f in
               ("sender_logits", "receiver_logits", "sender_baseline", "receiver_baseline")}
        for name, p in policies.items()
    }


def test_external_message_leaves_the_sender_untouched():
    env, agents, _ = make_individuals(4242)
    before = _fingerprint(agents)
    train(env, agents, episodes=400, max_seconds=30, message_fn=lambda cue: cue)
    after = _fingerprint(agents)
    for name, fields in before.items():
        is_sender = name == env.sender or name == env.agents[env.sender == name]
        for f, v in fields.items():
            if f.startswith("sender_"):
                np.testing.assert_array_equal(
                    after[name][f], v,
                    err_msg=f"sender field {f} of {name} changed while the "
                            f"message was external",
                )


def test_external_message_still_learns_the_receiver():
    env, agents, _ = make_individuals(4242)
    before = _fingerprint(agents)
    out = train(env, agents, episodes=400, max_seconds=30, message_fn=lambda cue: cue)
    after = _fingerprint(agents)
    changed = False
    for name, fields in before.items():
        for f, v in fields.items():
            if f.startswith("receiver_") and not np.array_equal(after[name][f], v):
                changed = True
    assert changed, "the receiver must still learn when the message is external"
    assert out["episodes_completed"] == 400


def test_default_path_still_trains_both_sides():
    """The C1 path must be untouched by the new parameter."""
    env, agents, _ = make_individuals(4242)
    before = _fingerprint(agents)
    train(env, agents, episodes=400, max_seconds=30)
    after = _fingerprint(agents)
    sender_changed = any(
        not np.array_equal(after[n][f], v)
        for n, fields in before.items() for f, v in fields.items()
        if f.startswith("sender_")
    )
    receiver_changed = any(
        not np.array_equal(after[n][f], v)
        for n, fields in before.items() for f, v in fields.items()
        if f.startswith("receiver_")
    )
    assert receiver_changed
    assert sender_changed, "without message_fn the sender must still learn"


def test_message_fn_returning_a_bad_symbol_is_rejected():
    env, agents, _ = make_individuals(4242)
    with pytest.raises(ValueError, match="not 0, 1 or 2"):
        train(env, agents, episodes=5, message_fn=lambda cue: 7)
    with pytest.raises(TypeError, match="callable"):
        train(env, agents, episodes=5, message_fn=123)


def test_external_message_reaches_the_world_unchanged():
    """The message the circuit produces must be the message the world receives."""
    env, agents, _ = make_individuals(99)
    seen = []

    def spy(cue):
        m = 1 if cue == 1 else 0
        seen.append((cue, m))
        return m

    train(env, agents, episodes=50, max_seconds=30, message_fn=spy)
    assert seen, "message_fn was never called"
    for cue, m in seen:
        assert m == cue, "message_fn result must reach the world untouched"
    assert all(c in (0, 1) for c, _ in seen), "cue must be binary here"