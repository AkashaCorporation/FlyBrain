"""Re-evaluate saved pilot weights with current evaluator, without retraining."""

from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cycle_evidence import identity, save, sha  # noqa: E402
from flybrain.learning.tabular import make_individuals  # noqa: E402
from flybrain.social.evaluation import control_gate, evaluate  # noqa: E402

out = ROOT / "outputs/first_cycle/tabular_pilot"
report = identity()
report.update(
    requested_mode="frozen_pilot_reevaluation",
    effective_mode="frozen_pilot_reevaluation",
    seeds=[11, 23, 37],
    dataset="synthetic_hidden_choice",
    mask=None,
    evidence_level="pilot_frozen_checkpoint_causal_evaluation",
    retention_tested=False,
    control_gate=control_gate(repeats=16),
    units=[],
)
assert report["control_gate"]["passed"]
for seed in (11, 23, 37):
    _, agents, _ = make_individuals(seed)
    path = out / f"seed_{seed}_weights.npz"
    previous = json.loads((out / f"seed_{seed}.json").read_text())
    assert sha(path) == previous["weights_sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        for name, agent in agents.items():
            for field in (
                "sender_logits",
                "receiver_logits",
                "sender_baseline",
                "receiver_baseline",
            ):
                setattr(agent, field, arrays[f"{name}_{field}"].copy())
    # Fresh action streams deliberately differ from post-training streams;
    # evaluate expected receiver return and sampled sender under the same world design.
    unit = evaluate(agents, seed=seed + 30000, repeats=64)
    report["units"].append(
        {"seed": seed, "weights_sha256": sha(path), "evaluation": unit}
    )
    print(seed, json.dumps(unit["expected"]), flush=True)
report.update(completed=True, stop_reason="finished")
save(out / "frozen_reevaluation.json", report)
