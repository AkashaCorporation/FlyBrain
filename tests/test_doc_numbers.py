"""Documented numbers must match the artefacts they are drawn from.

An independent audit found that ``docs/STATUS.md`` and ``README.md`` contained
several numbers that came from an earlier run, from a *different* run, or from
comparing two unlike quantities. Documentation that drifts from the artefacts is a
scientific-accuracy bug, not a typo, so it is checked here rather than trusted.

Every assertion below reads a value from a file in ``outputs/`` and then requires the
document to contain that value. If a run is regenerated with different numbers, these
tests fail and the docs must be updated — which is the point.

Tests skip when the artefacts they need are absent, so the suite still runs on a fresh
checkout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from flybrain import config as fb_config

OUTPUTS = fb_config.OUTPUTS_DIR
DOCS = fb_config.DOCS_DIR
README = fb_config.PROJECT_ROOT / "README.md"

WHOLE_BRAIN_RUN = OUTPUTS / "repro" / "wholebrain-1000ms" / "run"
SUBSET_SUGAR = OUTPUTS / "experiments" / "sugar_stimulation"
PUBLISHED = (
    fb_config.THIRD_PARTY_DIR / "Drosophila_brain_model" / "results" / "example"
    / "sugarR_100Hz.parquet"
)

MN9_LEFT = 720575940660219265
MN9_RIGHT = 720575940645521262


def normalise(text: str) -> str:
    """Collapse thousands separators and whitespace so formats cannot cause a miss."""
    return re.sub(r"[\s\u00a0]+", " ", text.replace("\u202f", " ")).replace(" ", "")


def docs_text() -> str:
    parts = [README.read_text(encoding="utf-8")]
    for p in DOCS.glob("*.md"):
        parts.append(p.read_text(encoding="utf-8"))
    return normalise("\n".join(parts))


def require(path: Path):
    if not path.exists():
        pytest.skip(f"artefact not present: {path}")
    return path


def load_json(path: Path):
    return json.loads(require(path).read_text(encoding="utf-8"))


def number_variants(value) -> list[str]:
    """The ways a number may be written in prose: 127400, 127,400, 127 400."""
    s = f"{value}"
    out = {s}
    if isinstance(value, int):
        out.add(f"{value:,}")
        out.add(f"{value:,}".replace(",", " "))
        out.add(f"{value:,}".replace(",", " "))
    if isinstance(value, float):
        out.add(f"{value:.1f}")
        out.add(f"{value:.2f}")
        out.add(f"{value:g}")
    return [normalise(v) for v in out]


def assert_documented(value, *, what: str, docs: str) -> None:
    variants = number_variants(value)
    if not any(v in docs for v in variants):
        pytest.fail(
            f"{what} = {value!r} does not appear anywhere in the docs "
            f"(looked for {variants}). Update docs/STATUS.md and README.md."
        )


# --------------------------------------------------------------------------------------
# dataset facts
# --------------------------------------------------------------------------------------


def test_documented_dataset_numbers_match_the_cards():
    docs = docs_text()
    for dataset_id, expected_neurons, expected_edges, sizes in (
        ("flywire_630", 127_400, 14_687_178, (3_057_611, 86_630_944)),
        ("flywire_783", 138_639, 15_091_983, (3_465_987, 100_804_642)),
    ):
        card = load_json(fb_config.METADATA_DIR / f"{dataset_id}.json")
        assert card["neuron_count"] == expected_neurons
        assert card["edge_count"] == expected_edges
        assert_documented(card["neuron_count"], what=f"{dataset_id} neuron_count", docs=docs)
        assert_documented(card["edge_count"], what=f"{dataset_id} edge_count", docs=docs)
        by_role = {f["role"]: f for f in card["files"]}
        assert by_role["neuron_table"]["size_bytes"] == sizes[0]
        assert by_role["connectivity_table"]["size_bytes"] == sizes[1]
        for role, size in (("neuron_table", sizes[0]), ("connectivity_table", sizes[1])):
            assert_documented(
                size, what=f"{dataset_id} {role} size_bytes", docs=docs
            )


def test_documented_dataset_report_numbers_match_the_report():
    docs = docs_text()
    rep = load_json(OUTPUTS / "dataset_report.json")
    st = rep["statistics"]
    for key in (
        "neurons", "edges", "synapses", "excitatory_neurons", "inhibitory_neurons",
        "neurons_unknown_sign", "neurons_no_outgoing", "neurons_no_incoming",
        "max_out_degree", "max_in_degree",
    ):
        assert_documented(st[key], what=f"dataset_report {key}", docs=docs)
    assert rep["n_warnings"] == 3, "the warning count is quoted as 3 in the docs"
    assert rep["n_failed_checks"] == 0


def test_the_degree_numbers_in_the_docs_are_degrees_not_synapse_counts():
    """Regression: max degree was once documented as the max synapse count (2358/1801)."""
    rep = load_json(OUTPUTS / "dataset_report.json")
    st = rep["statistics"]
    assert st["max_out_degree"] > st["signed_count_max"], (
        "max out-degree should exceed the largest single-connection synapse count; "
        "if this ever fails the two have been confused again"
    )
    assert st["max_in_degree"] > st["signed_count_max"]


# --------------------------------------------------------------------------------------
# whole-brain run
# --------------------------------------------------------------------------------------


def test_documented_whole_brain_numbers_match_the_run_artefacts():
    docs = docs_text()
    s = load_json(WHOLE_BRAIN_RUN / "summary.json")
    env = load_json(WHOLE_BRAIN_RUN / "environment.json")

    assert s["requested_mode"] == "whole-brain"
    assert s["effective_mode"] == "whole-brain"
    assert s["mode_changed"] is False
    assert s["is_subset"] is False
    assert s["n_neurons"] == 127_400
    assert s["n_edges"] == 14_687_178
    assert env["code_revision"]["commit"], "the run must record a git revision"

    assert_documented(s["total_spikes"], what="whole-brain total_spikes", docs=docs)
    assert_documented(s["active_neurons_any_spike"], what="whole-brain active_neurons", docs=docs)
    assert_documented(
        round(s["recording"]["active_neurons_per_step"]["mean"], 4),
        what="whole-brain mean active neurons per step", docs=docs,
    )
    assert_documented(s["spike_digest_sha256"][:16], what="whole-brain spike digest prefix", docs=docs)
    assert_documented(round(s["timing"]["measured_step_ms_mean"], 1),
                      what="whole-brain measured ms/step", docs=docs)
    assert_documented(round(s["timing"]["measured_steps_per_second"], 2),
                      what="whole-brain steps/s", docs=docs)
    assert_documented(round(s["timing"]["wall_seconds_total"], 1),
                      what="whole-brain wall seconds", docs=docs)
    assert_documented(s["requirements_estimate"]["peak_gb"],
                      what="whole-brain estimated peak GB", docs=docs)


def test_cli_accepts_a_whole_brain_artifact_run_directory():
    """The docs name the whole-brain artefact directory; it must be the real one."""
    assert WHOLE_BRAIN_RUN.is_dir(), (
        "docs/STATUS.md points at outputs/repro/wholebrain-1000ms/run for the "
        "whole-brain evidence, but that directory does not exist"
    )
    for name in ("summary.json", "config.json", "environment.json", "dataset.json",
                 "run.log", "spikes.parquet", "population_activity.parquet"):
        assert (WHOLE_BRAIN_RUN / name).is_file(), f"whole-brain run is missing {name}"


def test_the_two_whole_brain_runs_share_a_digest_if_both_are_present():
    """The reproducibility claim in STATUS.md, checked rather than remembered."""
    first = OUTPUTS / "experiments" / "sugar_stimulation" / "run" / "summary.json"
    if not first.exists():
        pytest.skip("the first whole-brain run's directory has been reused for another run")
    a = json.loads(first.read_text(encoding="utf-8"))
    b = load_json(WHOLE_BRAIN_RUN / "summary.json")
    if a.get("n_neurons") != 127_400 or a.get("dt_ms") != b.get("dt_ms"):
        pytest.skip("the first directory no longer holds a comparable whole-brain run")
    assert a["spike_digest_sha256"] == b["spike_digest_sha256"], (
        "two whole-brain runs of the same protocol must produce the same digest"
    )


# --------------------------------------------------------------------------------------
# the published comparison — the number that was once wrong
# --------------------------------------------------------------------------------------


def test_published_reference_and_the_like_for_like_comparison():
    """Guard the *basis* of the comparison, not just the arithmetic.

    The earlier claim compared FlyBrain's two-neuron MN9 population mean against the
    published value for the left neuron alone. Both numbers are legitimate; using them
    as a pair is not. This test recomputes all three quantities so the distinction
    cannot be lost again.
    """
    pd = pytest.importorskip("pandas")
    require(PUBLISHED)
    df = pd.read_parquet(PUBLISHED)
    trials = int(df["trial"].nunique())
    cnt = df.groupby("flywire_id").size()
    left = float(cnt.get(MN9_LEFT, 0) / trials)
    right = float(cnt.get(MN9_RIGHT, 0) / trials)
    mean = (left + right) / 2

    assert round(left, 2) == 67.03
    assert round(right, 2) == 48.63
    assert round(mean, 2) == 57.83
    assert left != mean, "the two are different quantities; the docs must not conflate them"

    docs = docs_text()
    assert_documented("67.03", what="published MN9-left", docs=docs)
    assert_documented("48.63", what="published MN9-right", docs=docs)
    assert_documented("57.83", what="published MN9 two-neuron mean", docs=docs)


def test_documented_mn9_comparison_is_per_neuron_and_matches_the_run():
    """The whole-brain MN9 figures in the docs must be the per-neuron rates.

    They are recomputed here from the recorded spike table, not read from the summary,
    because the summary only holds the population mean.
    """
    pd = pytest.importorskip("pandas")
    require(WHOLE_BRAIN_RUN / "spikes.parquet")
    s = load_json(WHOLE_BRAIN_RUN / "summary.json")
    df = pd.read_parquet(WHOLE_BRAIN_RUN / "spikes.parquet")
    seconds = s["duration_ms"] / 1000.0 * int(df["trial"].nunique())
    cnt = df.groupby("flywire_id").size()
    left = float(cnt.get(MN9_LEFT, 0)) / seconds
    right = float(cnt.get(MN9_RIGHT, 0)) / seconds

    # the summary's mn9 population rate must be the mean of exactly these two
    assert s["population_rates_hz"]["mn9"] == pytest.approx((left + right) / 2, abs=1e-6)

    docs = docs_text()
    assert_documented(round(left, 2), what="FlyBrain MN9-left rate", docs=docs)
    assert_documented(round(right, 2), what="FlyBrain MN9-right rate", docs=docs)

    # and the docs must state the per-neuron basis
    assert "perneuron" in docs.lower() or "per-neuron" in docs.lower()


def test_the_retracted_claim_is_marked_as_retracted():
    """The wrong figure must not silently disappear; the correction must be visible."""
    text = (DOCS / "STATUS.md").read_text(encoding="utf-8")
    assert "Correction" in text
    assert "like-for-unlike" in text or "like for unlike" in text
    assert "1.5 %" in text, "the retracted figure should be quoted so it can be recognised"


# --------------------------------------------------------------------------------------
# profiling and memory
# --------------------------------------------------------------------------------------


def test_documented_load_memory_numbers_match_the_measurement():
    path = OUTPUTS / "load_memory_flywire_630.json"
    m = load_json(path)
    docs = docs_text()
    assert_documented(round(m["deep_all_columns"]["rss_delta_mb"], 1),
                      what="all-columns decode RSS", docs=docs)
    assert_documented(round(m["minimal_columns"]["rss_delta_mb"], 1),
                      what="minimal-columns decode RSS", docs=docs)
    assert_documented(round(m["reduction_pct"], 1), what="decode RSS reduction %", docs=docs)
    assert m["reduction_pct"] > 50, (
        "narrowing the load path should be a large win, not a marginal one"
    )


def test_documented_profile_numbers_match_the_profile_artefacts():
    docs = docs_text()
    for name, keys in (
        ("profile_whole-brain_numpy.json", ("simulation.step",)),
        ("profile_subset_numpy.json", ("simulation.step",)),
    ):
        path = OUTPUTS / name
        if not path.exists():
            pytest.skip(f"profile artefact not present: {name}")
        prof = json.loads(path.read_text(encoding="utf-8"))
        step = prof["simulation.step"]["ms_per_step"]
        assert_documented(round(step, 1), what=f"{name} ms/step", docs=docs)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def test_every_run_directory_after_the_plotting_change_has_figures():
    """A run either wrote figures, or says why it did not."""
    found_any = False
    for summary_path in OUTPUTS.rglob("summary.json"):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if "plots" not in s:
            continue  # predates the runner-level plotting change; absence is visible
        found_any = True
        run_dir = summary_path.parent
        assert "plots_error" in s or s["plots"] or s.get("n_steps", 0) == 0, (
            f"{run_dir} recorded no figures and no error"
        )
        for name in s["plots"]:
            assert (run_dir / "plots" / name).is_file(), f"missing figure {name}"
    if not found_any:
        pytest.skip("no run has been made since figures moved into the runner")
