"""Frozen-policy paired replay from a pre-message snapshot.

Reports exact expected returns for this one-choice task and sampled paired returns.
No subsequent choice exists in C1; delayed-message effects need another protocol.
"""

from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np

from .choice import HiddenChoice


def policy_fingerprint(policies):
    payload = {
        name: (type(p).__module__, type(p).__qualname__, p.__dict__)
        for name, p in sorted(policies.items())
    }
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


def evaluate(policies, *, seed=907, repeats=64):
    if set(policies) != set(HiddenChoice.agents) or not 1 <= repeats <= 4096:
        raise ValueError("two individual policies and 1..4096 repeats required")
    frozen = policy_fingerprint(policies)
    world_rng, intervention_rng = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)
    ]
    # Exact hidden/public cross product; shuffle order, never expose the schedule.
    conditions = [
        (target, sender, options)
        for _ in range(repeats)
        for target in (0, 1)
        for sender in (0, 1)
        for options in ((0, 1), (1, 0))
    ]
    world_rng.shuffle(conditions)
    # Unconditional marginal over target, roles and public context. It explicitly
    # does NOT stratify by hidden label. No receiver or learner receives this table.
    marginal = np.zeros(3)
    env = HiddenChoice(seed)
    for target, sender, options in conditions:
        obs = env.reset_for_evaluation(target, sender, options)
        actor = copy.deepcopy(policies[env.sender])
        _, p = actor.act(obs[env.sender])
        marginal += p / len(conditions)
    marginal /= marginal.sum()
    sums = {
        k: 0.0 for k in ("intact", "blocked", "resampled", "tv_0_1", "tv_uniform_pairs")
    }
    sampled = {k: 0.0 for k in ("intact", "blocked", "resampled")}
    records = []
    evaluation_agents = copy.deepcopy(policies)
    for target, sender, options in conditions:
        obs = env.reset_for_evaluation(target, sender, options)
        before = env.snapshot()
        sender_copy = copy.deepcopy(evaluation_agents[env.sender])
        emitted, _ = sender_copy.act(obs[env.sender])
        # Advance only evaluation copies; training individuals remain untouched.
        evaluation_agents[env.sender] = sender_copy
        receiver_before = copy.deepcopy(evaluation_agents[env.receiver])
        choices = {}
        probs = {}
        returns = {}
        replacement = int(intervention_rng.choice(3, p=marginal))
        for condition, message in [
            ("intact", emitted),
            ("blocked", None),
            ("resampled", replacement),
            ("m0", 0),
            ("m1", 1),
            ("m2", 2),
        ]:
            trial = before.snapshot()
            trial.step({trial.sender: emitted})
            trial.intervene_reception(message)
            actor = copy.deepcopy(receiver_before)
            action, p = actor.act(trial.observations()[trial.receiver])
            probs[condition] = p
            returns[condition] = trial.expected_reward(p)
            _, reward, _ = trial.step({trial.receiver: action})
            choices[condition] = action
            if condition in sampled:
                sampled[condition] += reward[trial.receiver]
            if condition == "intact":
                evaluation_agents[env.receiver] = actor
        # Exact marginal intervention expectation, separate from its sampled replay.
        for key in ("intact", "blocked"):
            sums[key] += returns[key]
        sums["resampled"] += sum(marginal[m] * returns[f"m{m}"] for m in range(3))
        sums["tv_0_1"] += 0.5 * np.abs(probs["m0"] - probs["m1"]).sum()
        sums["tv_uniform_pairs"] += (
            sum(
                0.5 * np.abs(probs[f"m{a}"] - probs[f"m{b}"]).sum()
                for a in range(3)
                for b in range(3)
            )
            / 9
        )
        if len(records) < 16:
            records.append(
                {
                    "target_evaluator_only": target,
                    "sender": before.sender,
                    "options": list(options),
                    "emitted": emitted,
                    "replacement": replacement,
                    "receiver_before_message": obs[before.receiver].__dict__,
                    "paired_actions": choices,
                }
            )
    if policy_fingerprint(policies) != frozen:
        raise RuntimeError("evaluation mutated input policies")
    means = {k: float(v / len(conditions)) for k, v in sums.items()}
    means.update(
        gain_vs_blocked=means["intact"] - means["blocked"],
        gain_vs_resampled=means["intact"] - means["resampled"],
    )
    functional = (
        means["intact"] >= 0.75
        and means["gain_vs_resampled"] >= 0.1
        and means["tv_uniform_pairs"] >= 0.1
    )
    return {
        "completed": True,
        "stop_reason": "finished",
        "seed": seed,
        "episodes": len(conditions),
        "expected": means,
        "sampled_returns": {k: float(v / len(conditions)) for k, v in sampled.items()},
        "resampling_marginal": marginal.tolist(),
        "policy_sha256_before_after": frozen,
        "classification": "functional_channel_use"
        if functional
        else "functional_communication_not_demonstrated",
        "evidence_level": "frozen_policy_C1_causal_evaluation",
        "retention_tested": False,
        "later_decisions": "not_applicable_single_choice_terminal_task",
        "snapshot": "world+world_RNGs+individual_policy_state+action_RNG_before_emission",
        "paired_action_rng": "same receiver RNG state per intervention",
        "recorded_episodes": records,
        "recording_limit": 16,
        "recording_truncated": len(conditions) > 16,
    }


def control_gate(seed=907, repeats=64):
    from .policies import ManualChannelControl, NoInformationPolicy, RandomPolicy

    controls = {}
    for cls in (ManualChannelControl, NoInformationPolicy, RandomPolicy):
        policies = {a: cls(s) for a, s in zip(HiddenChoice.agents, (101, 203))}
        controls[cls.__name__] = evaluate(policies, seed=seed, repeats=repeats)
    p = controls["ManualChannelControl"]["expected"]
    negatives = [
        controls[c]["expected"] for c in ("NoInformationPolicy", "RandomPolicy")
    ]
    passed = (
        np.isclose(p["intact"], 1)
        and np.isclose(p["blocked"], 0.5)
        and np.isclose(p["resampled"], 0.5)
        and np.isclose(p["tv_0_1"], 1)
        and all(
            np.isclose(n["intact"], 0.5) and np.isclose(n["tv_0_1"], 0)
            for n in negatives
        )
    )
    return {
        "passed": bool(passed),
        "controls": controls,
        "learned": False,
        "classification": "environment_controls_passed"
        if passed
        else "control_positive_or_negative_failed",
    }
