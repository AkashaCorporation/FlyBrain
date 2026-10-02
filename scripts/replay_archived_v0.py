"""Run frozen original sources against current legacy and v1 on synthetic inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def worker(source, version):
    sys.path.insert(0, str(source))
    import numpy as np
    from flybrain.data.synthetic import chain_connectome
    from flybrain.model import LIFNetwork, Stimulus
    from flybrain.runtime.backend import NumpyBackend

    result = {}
    for scenario in ("static", "switches"):
        kwargs = {} if version == "frozen" else {"contract_version": version}
        net = LIFNetwork(
            chain_connectome(8), backend=NumpyBackend(), seed=421, **kwargs
        )
        h = hashlib.sha256()
        net.apply_stimulus(Stimulus((1000000,), 900))
        for tick in range(500):
            if scenario == "switches":
                if tick == 120:
                    net.apply_stimulus(Stimulus((1000001,), 600))
                if tick == 230:
                    net.apply_stimulus(Stimulus((1000000, 1000001), 700))
                if tick == 310:
                    net.clear_stimuli()
                if tick == 400:
                    net.reset(421)
            net.step()
            for field in (
                "v",
                "g",
                "t_last",
                "tau_ref",
                "last_spike",
                "spike_count",
                "ring",
            ):
                h.update(np.asarray(getattr(net, field)).tobytes())
        result[scenario] = h.hexdigest()
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--version")
    args = parser.parse_args()
    if args.source:
        worker(args.source, args.version)
        return
    from cycle_evidence import ROOT, OUT, identity, save, sha

    archive = OUT / "v0-30ad237.zip"
    expected = json.loads((OUT / "reference_archive.json").read_text())["sha256"]
    assert sha(archive) == expected
    frozen = OUT / "archived_source"
    frozen.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        z.extractall(frozen)
    runs = {}
    for version, path in [("frozen", frozen), ("legacy_v0", ROOT), ("v1", ROOT)]:
        p = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(path),
                "--version",
                version,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
            cwd=frozen,
        )
        runs[version] = json.loads(p.stdout)
    report = identity()
    report.update(
        requested_mode="archived_v0_synthetic_replay",
        effective_mode="archived_v0_synthetic_replay",
        dataset="synthetic_chain_8",
        mask=None,
        seeds=[421],
        completed=True,
        stop_reason="finished",
        evidence_level="E0_synthetic_state_bytes",
        runs=runs,
        legacy_all_scenarios_match=runs["frozen"] == runs["legacy_v0"],
        v1_static_match=runs["frozen"]["static"] == runs["v1"]["static"],
        v1_transition_intentionally_differs=runs["frozen"]["switches"]
        != runs["v1"]["switches"],
    )
    assert (
        report["legacy_all_scenarios_match"]
        and report["v1_static_match"]
        and report["v1_transition_intentionally_differs"]
    )
    save(OUT / "archived_replay.json", report)
    print(json.dumps(runs, indent=2))


if __name__ == "__main__":
    main()
