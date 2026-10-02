"""Execute C1 controls only, never label these measurements as learning."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cycle_evidence import identity, save  # noqa: E402
from flybrain.social.evaluation import control_gate  # noqa: E402

report = identity()
report.update(
    requested_mode="C1_handcoded_and_negative_controls",
    effective_mode="C1_handcoded_and_negative_controls",
    seeds=[907, 101, 203],
    dataset="synthetic_hidden_choice",
    mask=None,
    evidence_level="environment_causal_controls_not_learning",
)
report["evaluation"] = control_gate()
report.update(
    completed=report["evaluation"]["passed"],
    stop_reason="finished" if report["evaluation"]["passed"] else "control_gate_failed",
)
save(ROOT / "outputs/first_cycle/social_controls.json", report)
for name, r in report["evaluation"]["controls"].items():
    print(name, json.dumps(r["expected"]))
if not report["completed"]:
    raise SystemExit(1)
