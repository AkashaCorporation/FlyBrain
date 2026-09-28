"""CLI behaviour, including the refusal path.

Exit codes are part of the interface: a refusal must be distinguishable from a
success by a script, so it is asserted here rather than left to the operator's eye.
"""

from __future__ import annotations

import json

import pytest

from flybrain.cli import main, parse_silence_spec, parse_stimulus_spec
from flybrain.model.populations import PopulationRegistry

from conftest import needs_dataset


# --------------------------------------------------------------------------------------
# pure parsing
# --------------------------------------------------------------------------------------


def test_stimulus_spec_population_name_only():
    reg = PopulationRegistry.builtin()
    s = parse_stimulus_spec("sugar_grn", reg)
    assert len(s.neuron_ids) == 21
    assert s.rate_hz == 150.0
    assert s.label == "sugar_grn"


def test_stimulus_spec_with_rate_and_window():
    reg = PopulationRegistry.builtin()
    s = parse_stimulus_spec("sugar_grn:200:100:500", reg)
    assert s.rate_hz == 200.0
    assert s.start_ms == 100.0
    assert s.end_ms == 500.0


def test_stimulus_spec_explicit_ids():
    reg = PopulationRegistry.builtin()
    s = parse_stimulus_spec("ids:11,22,33:75", reg)
    assert s.neuron_ids == (11, 22, 33)
    assert s.rate_hz == 75.0
    assert s.label == "explicit_ids"


def test_stimulus_spec_rejects_junk():
    reg = PopulationRegistry.builtin()
    with pytest.raises(Exception):
        parse_stimulus_spec("ids:not_an_int:100", reg)


def test_silence_spec_distinguishes_ids_from_populations():
    reg = PopulationRegistry.builtin()
    assert parse_silence_spec("mn9", reg) == {"population": "mn9"}
    assert parse_silence_spec("720575940660219265", reg) == {"flywire_id": 720575940660219265}


# --------------------------------------------------------------------------------------
# commands that need no dataset
# --------------------------------------------------------------------------------------


def test_version_and_info_exit_cleanly():
    assert main(["info"]) == 0


def test_help_exits_zero():
    assert main([]) == 0


def test_populations_search():
    assert main(["populations", "--search", "sugar"]) == 0


def test_unknown_population_returns_code_5():
    assert main(["populations", "--show", "definitely_not_a_population"]) == 5


def test_smoke_experiment_runs_without_a_dataset():
    """The smoke test must not depend on the 180 MB dataset being staged."""
    assert main(["experiments", "smoke_test"]) == 0


def test_experiments_list():
    assert main(["experiments", "--list"]) == 0


def test_unknown_experiment_returns_usage_error():
    assert main(["experiments", "no_such_experiment"]) == 2


# --------------------------------------------------------------------------------------
# commands that need the dataset
# --------------------------------------------------------------------------------------


@needs_dataset
def test_datasets_command_reports_the_staged_card(capsys):
    assert main(["datasets"]) == 0
    out = capsys.readouterr().out
    assert "flywire_630" in out
    assert "sha256" in out


@needs_dataset
def test_neuron_command(capsys):
    assert main(["neuron", "720575940660219265", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "720575940660219265" in out
    assert "incoming:" in out and "outgoing:" in out
    assert "neurotransmitter sign:" in out


@needs_dataset
def test_neuron_command_with_unknown_id_returns_error():
    assert main(["neuron", "123"]) == 2


@needs_dataset
def test_subset_run_writes_every_required_artifact(tmp_path, capsys):
    run_id = "cli-test-run"
    code = main([
        "run", "--mode", "subset", "--duration-ms", "20", "--dt-ms", "0.1",
        "--stimulus", "sugar_grn:150:0:10",
        "--record", "population", "--record", "spikes",
        "--subset-hops", "1", "--subset-max-neurons", "1500",
        "--no-plots", "--quiet",
        "--out-dir", str(tmp_path), "--run-id", run_id,
    ])
    assert code == 0, capsys.readouterr().err
    rd = tmp_path / run_id
    for name in ("config.json", "environment.json", "dataset.json", "summary.json",
                 "run.log", "spikes.parquet", "population_activity.parquet"):
        assert (rd / name).is_file(), f"missing artifact {name}"

    summary = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
    assert summary["effective_mode"] == "subset"
    assert summary["is_subset"] is True
    # a 1-hop neighbourhood of the 21 sugar GRNs is whatever it is; what must hold
    # is that it covers the stimulus and respects the cap
    assert 21 <= summary["n_neurons"] <= 1500
    sub = json.loads((rd / "dataset.json").read_text(encoding="utf-8"))["subset"]
    assert len(sub["present_seed_ids"]) == 21
    assert sub["truncated"] is False
    assert summary["total_spikes"] > 0
    assert summary["timing"]["measured_step_ms_mean"] is not None
    assert len(summary["spike_digest_sha256"]) == 64

    config = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    assert config["mode"] == "subset"
    assert config["stimuli"][0]["label"] == "sugar_grn"

    env = json.loads((rd / "environment.json").read_text(encoding="utf-8"))
    assert env["backend"]["name"] in ("numpy", "jax")
    assert env["hardware"]["os_name"]
    assert "code_revision" in env


@needs_dataset
def test_run_records_dataset_provenance(tmp_path):
    code = main([
        "run", "--mode", "subset", "--duration-ms", "5",
        "--stimulus", "mn9:100",
        "--record", "none", "--quiet", "--no-plots",
        "--subset-hops", "1", "--subset-max-neurons", "200",
        "--out-dir", str(tmp_path), "--run-id", "prov",
    ])
    assert code == 0
    ds = json.loads((tmp_path / "prov" / "dataset.json").read_text(encoding="utf-8"))
    assert ds["dataset_id"] == "flywire_630"
    assert ds["card"]["sha256"]
    assert all(f["sha256"] for f in ds["card"]["files"])


@needs_dataset
def test_two_identical_runs_produce_the_same_spike_digest(tmp_path):
    args = [
        "run", "--mode", "subset", "--duration-ms", "20",
        "--stimulus", "sugar_grn:150:0:10",
        "--record", "spikes", "--quiet", "--no-plots",
        "--subset-hops", "1", "--subset-max-neurons", "1200",
        "--seed", "5",
    ]
    digests = []
    for rid in ("det-a", "det-b"):
        assert main(args + ["--out-dir", str(tmp_path), "--run-id", rid]) == 0
        s = json.loads((tmp_path / rid / "summary.json").read_text(encoding="utf-8"))
        digests.append(s["spike_digest_sha256"])
    assert digests[0] == digests[1], (
        "two identical deterministic runs must produce the same digest; "
        "if this fails, the reproducibility claim in the docs is false"
    )


@needs_dataset
def test_whole_brain_refusal_exits_with_code_3_and_does_not_fall_back(tmp_path, capsys):
    """The refusal must be a refusal: distinct exit code, no artefacts, no subset."""
    code = main([
        "run", "--mode", "whole-brain", "--duration-ms", "60000",
        "--max-minutes", "1",
        "--stimulus", "sugar_grn",
        "--record", "none", "--quiet", "--no-plots",
        "--out-dir", str(tmp_path), "--run-id", "refused",
    ])
    assert code == 3, capsys.readouterr().err
    assert not (tmp_path / "refused").exists(), "a refused run must not write artefacts"


@needs_dataset
def test_allow_fallback_does_not_override_the_explicit_time_limit(tmp_path, capsys):
    """``--allow-fallback`` covers *memory* infeasibility, not the user's time limit.

    Conflating the two would mean a user who says "do not run longer than a minute"
    silently gets a different experiment instead of a refusal, which is exactly the
    silent-substitution failure the objective warns about. The memory-fallback path
    itself is covered end to end in tests/test_runtime.py.
    """
    code = main([
        "run", "--mode", "whole-brain", "--duration-ms", "60000",
        "--max-minutes", "1", "--allow-fallback",
        "--stimulus", "sugar_grn",
        "--record", "none", "--quiet", "--no-plots",
        "--out-dir", str(tmp_path), "--run-id", "refused-time",
    ])
    assert code == 3
    assert "exceeds the configured limit" in capsys.readouterr().err
    assert not (tmp_path / "refused-time").exists()


@needs_dataset
def test_estimate_only_runs_no_simulation(tmp_path, capsys):
    code = main([
        "run", "--mode", "whole-brain", "--duration-ms", "1000",
        "--estimate-only", "--record", "none", "--quiet",
        "--out-dir", str(tmp_path), "--run-id", "est",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "peak_gb" in out
    assert not (tmp_path / "est").exists()
