"""Three prespecified exploratory seeds; controls gate all training."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cycle_evidence import identity, save  # noqa: E402
from flybrain.social.evaluation import control_gate, evaluate  # noqa: E402
from flybrain.learning.tabular import make_individuals, train  # noqa: E402

out = ROOT / "outputs/first_cycle/tabular_pilot"
out.mkdir(exist_ok=False)
gate = control_gate(repeats=16)
save(out / "control_gate.json", gate)
if not gate["passed"]:
    raise SystemExit("control gate failed; no training performed")
for seed in (11, 23, 37):
    report = identity()
    env, agents, streams = make_individuals(seed)
    report.update(
        seed=seed,
        seeds={"root": seed, "world_init_action_streams": streams},
        requested_mode="C1_independent_tabular_pilot",
        effective_mode="C1_independent_tabular_pilot",
        dataset="synthetic_hidden_choice",
        mask=None,
        evidence_level="exploratory_three_seeds_not_confirmation",
        limits={
            "training_episodes": 12000,
            "training_seconds": 30,
            "evaluation_episodes": 512,
        },
        retention_tested=False,
        neural_substrate_used=False,
        control_gate_passed=True,
    )
    report["before"] = evaluate(agents, seed=seed + 10000, repeats=64)
    report["training"] = train(env, agents)
    report["after"] = evaluate(agents, seed=seed + 20000, repeats=64)
    path = out / f"seed_{seed}_weights.npz"
    np.savez(
        path,
        **{
            f"{name}_{field}": getattr(p, field)
            for name, p in agents.items()
            for field in (
                "sender_logits",
                "receiver_logits",
                "sender_baseline",
                "receiver_baseline",
            )
        },
    )
    report.update(
        completed=report["training"]["completed"],
        stop_reason=report["training"]["stop_reason"],
        weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    report["classification"] = (
        "budget_insufficient"
        if not report["completed"]
        else "learned_functional_communication_pilot"
        if report["after"]["classification"] == "functional_channel_use"
        else "learning_without_demonstrated_communication"
        if report["after"]["expected"]["intact"] > 0.6
        else "task_not_solved_under_pilot_budget"
    )
    save(out / f"seed_{seed}.json", report)
    print(
        seed,
        report["classification"],
        json.dumps(report["after"]["expected"]),
        flush=True,
    )
