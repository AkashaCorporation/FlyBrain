"""Preservation, dependency provenance and authoritative artifact inventory."""

import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cycle_evidence import identity, save, sha  # noqa: E402
from flybrain.data.provenance import FileRecord, combined_digest  # noqa: E402
from flybrain.runtime.runner import spike_digest  # noqa: E402


def main():
    out = ROOT / "outputs/first_cycle"
    initial = json.loads((out / "initial_manifest.json").read_text())
    preserved = {}
    for name, record in initial["files"].items():
        normalized = name.replace("\\", "/")
        if (
            normalized.startswith(("data/", "outputs/"))
            or name == "docs/PROPOSTA_PESQUISA.md"
        ):
            actual = sha(ROOT / name)
            preserved[name] = {
                "sha256": actual,
                "matches_initial": actual == record["sha256"],
            }
    if not all(x["matches_initial"] for x in preserved.values()):
        raise RuntimeError("a preserved reference artifact changed")
    card = json.loads((ROOT / "data/metadata/flywire_630.json").read_text())
    source_digest = combined_digest([FileRecord(**r) for r in card["files"]])
    assert source_digest == card["sha256"]
    for row in card["files"]:
        assert sha(ROOT / "data/raw/flywire_630" / row["name"]) == row["sha256"]
    historical = ROOT / "outputs/repro/wholebrain-1000ms/run"
    config = json.loads((historical / "config.json").read_text())
    digest = spike_digest(
        pd.read_parquet(historical / "spikes.parquet"), config["trials"]
    )
    assert digest == initial["historical_spike_digest"]
    meta = subprocess.run(
        [
            str(Path.home() / ".cargo/bin/cargo.exe"),
            "metadata",
            "--locked",
            "--format-version",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    dependencies = []
    for package in json.loads(meta.stdout)["packages"]:
        manifest = Path(package["manifest_path"])
        dependencies.append(
            {
                "name": package["name"],
                "version": package["version"],
                "license": package["license"],
                "manifest_sha256": sha(manifest),
                "source": package["source"],
            }
        )
    maturin = importlib.metadata.distribution("maturin")
    licenses = [str(p) for p in maturin.files if "license" in str(p).lower()]
    mlicense = {p: sha(maturin.locate_file(p)) for p in licenses}
    suite_counts = {}
    for path in out.glob("*.xml"):
        root = ET.parse(path).getroot()
        suites = list(root.iter("testsuite"))
        suite_counts[path.name] = {
            k: sum(int(s.get(k, "0")) for s in suites)
            for k in ("tests", "failures", "errors", "skipped")
        }
    cache = json.loads((ROOT / "data/processed/flywire_630/cache.json").read_text())
    npz_sha = sha(ROOT / "data/processed/flywire_630/connectome.npz")
    assert cache["npz_sha256"] == npz_sha
    report = identity()
    report.update(
        completed=True,
        stop_reason="finished",
        requested_mode="first_cycle_preservation_audit",
        effective_mode="first_cycle_preservation_audit",
        seeds=None,
        dataset="flywire_630",
        mask=None,
        evidence_level="hashes_raw_cards_saved_spikes_and_test_outputs",
        preserved=preserved,
        historical_saved_spikes_digest=digest,
        historical_spikes_file_sha256=sha(historical / "spikes.parquet"),
        wholebrain_simulation_rerun=False,
        source_combined_digest=source_digest,
        processed_npz_sha256=npz_sha,
        historical_document_correction="ESTADO_ATUAL labels combined source digest as processed NPZ SHA; they are distinct",
        dependencies=dependencies,
        maturin_license_files=mlicense,
        cargo_lock_sha256=sha(ROOT / "Cargo.lock"),
        suite_counts=suite_counts,
        reference_protocol_feasibility={
            "n": 127400,
            "edges": 14687178,
            "bridge_estimated_bytes": 14687178 * 256 + 127400 * (160 + 8 * 18),
            "configured_max_memory_bytes": 256000000,
            "available_bytes_at_audit": psutil.virtual_memory().available,
            "decision": "not_run_bridge_budget_exceeded",
            "silent_fallback": False,
        },
        build_artifacts={
            str(p.relative_to(ROOT)): sha(p) for folder in ("wheels", "wheels_final")
            for p in (out / folder).glob("*.whl")
        },
    )
    save(out / "preservation_audit.json", report)
    print(
        json.dumps(
            {
                "preserved_files": len(preserved),
                "source_digest": source_digest,
                "npz_sha256": npz_sha,
                "historical_saved_spikes_digest": digest,
                "tests": suite_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
