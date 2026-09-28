"""Runtime layer: backend equivalence, hardware reporting, recording, feasibility.

The backend-equivalence test is the important one. Two implementations that agree
on a hand-picked example prove little; two that agree bit-for-bit on a real network
over thousands of steps make it safe to choose between them on speed alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.config import RunConfig
from flybrain.data.synthetic import chain_connectome, make_synthetic_connectome
from flybrain.errors import ResourceLimitError
from flybrain.model import LIFNetwork, LIFParams, Stimulus
from flybrain.runtime import (
    NumpyBackend,
    RecordingConfig,
    Recorder,
    available_backends,
    check_feasibility,
    detect_hardware,
    estimate_requirements,
    format_runtime_banner,
    get_backend,
    plan_run,
)

from conftest import needs_dataset


# --------------------------------------------------------------------------------------
# hardware detection
# --------------------------------------------------------------------------------------


def test_detect_hardware_never_raises_and_reports_unknowns_as_none():
    hw = detect_hardware()
    assert hw.os_name
    assert hw.python_version
    # these may legitimately be None; the point is that they are present fields
    assert hasattr(hw, "ram_total_gb")
    assert hasattr(hw, "cpu_cores_logical")
    assert isinstance(hw.nvidia_gpus, list)
    assert isinstance(hw.jax_installed, bool)
    assert hw.to_dict()["os_name"] == hw.os_name


def test_hardware_separates_presence_from_usability():
    """Having an NVIDIA GPU and being able to use it are different statements."""
    hw = detect_hardware()
    if hw.nvidia_gpus and hw.jax_installed and not hw.jax_gpu_available:
        lines = "\n".join(hw.summary_lines())
        assert "gpu_available=False" in lines
        banner = format_runtime_banner(
            hw, backend_name="numpy", backend_is_gpu=False, dataset_id="x",
            n_neurons=1, n_edges=0, dtype="float32", dt_ms=0.1, mode="subset",
        )
        assert "no -" in banner
        assert "JAX reports" in banner
        assert "CUDA" not in banner.split("GPU usable")[1].split("\n")[0] or "no -" in banner


def test_banner_contains_every_field_the_objective_asks_for():
    hw = detect_hardware()
    banner = format_runtime_banner(
        hw, backend_name="numpy", backend_is_gpu=False, dataset_id="flywire_630",
        n_neurons=127_400, n_edges=14_687_178, dtype="float32", dt_ms=0.1, mode="subset",
    )
    for token in ("FlyBrain Runtime", "Backend", "GPU", "VRAM", "RAM total",
                  "Dataset", "Neurons", "Synaptic edges", "Precision", "dt"):
        assert token in banner, f"{token!r} missing from the runtime banner"


# --------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------


def test_available_backends_always_contains_numpy():
    assert "numpy" in available_backends()


def test_auto_backend_is_a_real_backend():
    bk = get_backend("auto")
    assert bk.name in ("numpy", "jax")
    assert bk.xp is not None


def test_unknown_backend_raises():
    with pytest.raises(Exception):
        get_backend("quantum")


def test_scatter_add_semantics_on_both_backends():
    """The single primitive that differs between backends must mean the same thing."""
    idx = np.array([1, 1, 3], dtype=np.int32)
    vals = np.array([2.0, 3.0, -1.0], dtype=np.float32)
    for name in available_backends():
        bk = get_backend(name)
        out = np.asarray(bk.scatter_add(5, bk.asarray(idx), bk.asarray(vals), dtype="float32"))
        assert out.dtype == np.float32
        assert out.tolist() == [0.0, 5.0, 0.0, -1.0, 0.0]


def test_index_arrays_are_not_coerced_to_float():
    """A default float dtype on asarray would break fancy indexing; guard it."""
    for name in available_backends():
        bk = get_backend(name)
        a = bk.asarray(np.array([0, 1, 2], dtype=np.int32))
        assert str(a.dtype) == "int32"
        f = bk.asarray(np.array([1.0, 2.0], dtype=np.float32))
        assert str(f.dtype) == "float32"


@pytest.mark.skipif("jax" not in available_backends(), reason="jax not installed")
def test_backends_produce_bit_identical_spike_counts():
    """The property that makes backend choice a performance decision only."""
    conn = chain_connectome(5, synapses=200)
    stim = Stimulus([int(conn.flywire_ids[0])], rate_hz=180.0)
    counts = {}
    for name in ("numpy", "jax"):
        net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=get_backend(name), seed=4)
        net.apply_stimulus(stim)
        for _ in range(1500):
            net.step()
        counts[name] = np.asarray(net.spike_count)
    assert np.array_equal(counts["numpy"], counts["jax"]), (
        "backends disagree; choose on correctness before speed"
    )


# --------------------------------------------------------------------------------------
# requirements estimation and refusal
# --------------------------------------------------------------------------------------


def test_estimate_reports_a_breakdown_and_the_dense_comparison():
    est = estimate_requirements(127_400, 14_687_178, 18)
    assert set(est["components_bytes"]) == {
        "edge_presynaptic_index", "edge_postsynaptic_index", "edge_weights",
        "delay_ring_buffer", "neuron_state_6_arrays", "flywire_id_table",
        "per_step_transient_peak",
    }
    assert est["peak_bytes"] > est["resident_bytes"]
    # the whole point of using a sparse representation
    assert est["dense_matrix_gb_if_materialised"] > 60
    assert est["resident_gb"] < 1.0, "14.7M edges must not need a GB of resident state"


def test_feasibility_refuses_when_ram_is_unknown():
    est = estimate_requirements(127_400, 14_687_178, 18)
    with pytest.raises(ResourceLimitError) as ei:
        check_feasibility(est, ram_available_gb=None)
    assert ei.value.estimate == est


def test_feasibility_refuses_when_the_estimate_exceeds_the_budget():
    est = estimate_requirements(127_400, 14_687_178, 18)
    with pytest.raises(ResourceLimitError) as ei:
        check_feasibility(est, ram_available_gb=0.2)
    assert "budget" in str(ei.value)


def test_feasibility_accepts_a_run_that_fits():
    est = estimate_requirements(127_400, 14_687_178, 18)
    check_feasibility(est, ram_available_gb=8.0)  # must not raise


@needs_dataset
def test_plan_refuses_whole_brain_rather_than_shrinking(real_connectome, monkeypatch):
    """The central safety property: refuse, never silently degrade."""
    import flybrain.runtime.runner as runner

    monkeypatch.setattr(runner, "available_ram_gb", lambda: 0.05)
    cfg = RunConfig(mode="whole-brain", duration_ms=10.0)
    stim = Stimulus([int(real_connectome.flywire_ids[0])], rate_hz=100.0, label="seed")
    with pytest.raises(ResourceLimitError) as ei:
        plan_run(real_connectome, cfg, [stim])
    assert ei.value.estimate["peak_gb"] > 0


@needs_dataset
def test_plan_fallback_labels_the_change_and_keeps_mode_separate(real_connectome, monkeypatch):
    import flybrain.runtime.runner as runner

    monkeypatch.setattr(runner, "available_ram_gb", lambda: 0.05)
    stim = Stimulus([int(real_connectome.flywire_ids[0])], rate_hz=100.0, label="seed")
    cfg = RunConfig(mode="whole-brain", duration_ms=10.0, allow_fallback=True,
                    subset_hops=1, subset_max_neurons=300)
    plan = plan_run(real_connectome, cfg, [stim])
    assert plan.requested_mode == "whole-brain"
    assert plan.effective_mode == "subset"
    assert plan.mode_changed
    assert plan.is_subset
    assert any("NOT a whole-brain" in n or "NOT DELIVERED" in n for n in plan.notes)


@needs_dataset
def test_subset_seeds_always_include_the_stimulus_targets(real_connectome):
    """A subset that excludes its own stimulus would silently measure nothing."""
    from flybrain.model.populations import PopulationRegistry

    reg = PopulationRegistry.builtin()
    present, _ = reg.resolve("sugar_grn", real_connectome.flywire_ids)
    stim = Stimulus(tuple(int(x) for x in present), rate_hz=100.0, label="sugar_grn")
    # explicit seeds deliberately chosen to NOT include the sugar GRNs
    cfg = RunConfig(mode="subset", duration_ms=5.0, subset_seed_ids=[720575940660219265],
                    subset_hops=1, subset_max_neurons=500)
    plan = plan_run(real_connectome, cfg, [stim])
    missing = [int(x) for x in present if int(x) not in plan.connectome.id_to_index]
    assert not missing, f"{len(missing)} stimulus neurons were dropped from the working set"


# --------------------------------------------------------------------------------------
# the refusal / fallback path, end to end through the artefacts
# --------------------------------------------------------------------------------------


@needs_dataset
def test_fallback_run_records_the_mode_change_in_every_artefact(real_connectome, monkeypatch, tmp_path):
    """A labelled subset must be labelled in the files, not only on stdout."""
    import json

    import flybrain.runtime.runner as runner

    monkeypatch.setattr(runner, "available_ram_gb", lambda: 0.05)
    stim = Stimulus([int(real_connectome.flywire_ids[0])], rate_hz=100.0, label="seed")
    cfg = RunConfig(
        mode="whole-brain", duration_ms=5.0, allow_fallback=True,
        subset_hops=1, subset_max_neurons=200, max_estimated_minutes=None,
        run_id="fallback-artefacts",
    )
    plan = plan_run(real_connectome, cfg, [stim])
    assert plan.mode_changed

    res = runner.run_experiment(
        plan, cfg, [stim],
        recording=RecordingConfig(record_spikes=False, record_population=False),
        run_dir=tmp_path / "fallback-artefacts",
    )
    assert res.is_subset and res.mode == "subset"

    summary = json.loads((tmp_path / "fallback-artefacts" / "summary.json").read_text(encoding="utf-8"))
    assert summary["requested_mode"] == "whole-brain"
    assert summary["effective_mode"] == "subset"
    assert summary["mode_changed"] is True
    assert summary["is_subset"] is True
    assert any("NOT DELIVERED" in n for n in summary["plan_notes"])
    assert "refusal_reason" in summary and summary["refusal_reason"]

    config = json.loads((tmp_path / "fallback-artefacts" / "config.json").read_text(encoding="utf-8"))
    assert config["requested_mode"] == "whole-brain"
    assert config["effective_mode"] == "subset"

    dataset = json.loads((tmp_path / "fallback-artefacts" / "dataset.json").read_text(encoding="utf-8"))
    assert dataset["is_subset"] is True
    assert dataset["subset"] is not None


# --------------------------------------------------------------------------------------
# recording
# --------------------------------------------------------------------------------------


def test_recorder_records_only_what_was_requested():
    conn = make_synthetic_connectome([(0, 1, 200)], 3)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=200.0))
    rec = Recorder(conn, RecordingConfig(
        record_spikes=False, record_population=False), dt_ms=0.1)
    rec.begin(net)
    for k in range(500):
        rec.on_step(k, net.step(), net)
    assert rec.spikes_frame() is None
    assert rec.population_frame() is None
    assert rec.summary()["recorded_spike_rows"] == 0


def test_recorder_spike_rows_match_the_network_spike_count():
    conn = make_synthetic_connectome([(0, 1, 200)], 3)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=1)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=200.0))
    rec = Recorder(conn, RecordingConfig(record_spikes=True, record_population=False), dt_ms=0.1)
    rec.begin(net)
    for k in range(2000):
        rec.on_step(k, net.step(), net)
    df = rec.spikes_frame()
    assert df is not None
    assert len(df) == int(np.asarray(net.spike_count).sum())
    assert set(df.columns) == {"t_ms", "step", "flywire_id", "index"}
    # t_ms must be consistent with the step index
    assert np.allclose(df["t_ms"].to_numpy(), df["step"].to_numpy() * 0.1)


def test_recorder_honours_the_spike_neuron_filter():
    conn = make_synthetic_connectome([(0, 1, 200), (1, 2, 200)], 3)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=200.0))
    rec = Recorder(conn, RecordingConfig(
        record_spikes=True, record_population=False,
        spike_neuron_ids=(int(conn.flywire_ids[2]),)), dt_ms=0.1)
    rec.begin(net)
    for k in range(3000):
        rec.on_step(k, net.step(), net)
    df = rec.spikes_frame()
    if df is not None:
        assert set(df["flywire_id"].unique().tolist()) <= {int(conn.flywire_ids[2])}


def test_recorder_truncation_is_flagged_not_silent():
    conn = make_synthetic_connectome([(0, 1, 200)], 2)
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=0)
    net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=300.0))
    rec = Recorder(conn, RecordingConfig(
        record_spikes=True, record_population=False, max_spike_rows=5), dt_ms=0.1)
    rec.begin(net)
    for k in range(3000):
        rec.on_step(k, net.step(), net)
    assert rec.summary()["spike_recording_truncated"] is True
    assert rec.summary()["recorded_spike_rows"] == 5


def test_recorder_population_rates_are_per_member_and_per_second():
    """A population rate must be a mean over its members and over its window.

    Uses an injected registry whose members are the synthetic connectome's own
    neurons, so the arithmetic is checkable by hand rather than depending on which
    real FlyWire IDs happen to be in a toy graph.
    """
    from flybrain.model.populations import Population, PopulationRegistry

    conn = make_synthetic_connectome([], 4)
    ids = [int(x) for x in conn.flywire_ids[:2]]
    reg = PopulationRegistry({
        "unit_pop": Population(name="unit_pop", ids=(ids[0], ids[1]),
                               description="test population", upstream_symbol="unit"),
    })
    net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=0)
    rec = Recorder(conn, RecordingConfig(
        record_spikes=False, record_population=True,
        population_names=("unit_pop",), population_interval_ms=10.0),
        registry=reg, dt_ms=0.1)
    assert rec._pop_names == ["unit_pop"], "the test population must resolve"

    rec.begin(net)
    # Sample windows are [0, 100) and [100, 200) steps, i.e. 0-10 ms and 10-20 ms.
    # Put two spikes in the first window and one in the second, on alternating
    # members, so both the member mean and the window boundary are exercised.
    #   window 1: 2 spikes / (2 members * 0.01 s) = 100 Hz
    #   window 2: 1 spike  / (2 members * 0.01 s) =  50 Hz
    injected = {10: 0, 20: 1, 150: 0}
    for k in range(200):
        spk = np.zeros(4, dtype=np.float32)
        if k in injected:
            spk[injected[k]] = 1.0
        net.spike_count = net.spike_count + net.backend.asarray(spk, dtype=net.dtype)
        net.step_index = k + 1
        rec.on_step(k, spk, net)

    df = rec.population_frame()
    assert df is not None
    first = df[(df["population"] == "unit_pop") & (df["t_ms"] == 10.0)]
    assert len(first) == 1
    assert int(first["n_neurons"].iloc[0]) == 2
    assert float(first["rate_hz"].iloc[0]) == pytest.approx(100.0)
    second = df[(df["population"] == "unit_pop") & (df["t_ms"] == 20.0)]
    assert len(second) == 1
    assert float(second["rate_hz"].iloc[0]) == pytest.approx(50.0)


# --------------------------------------------------------------------------------------
# determinism across the recorder boundary
# --------------------------------------------------------------------------------------


def test_two_identical_runs_produce_identical_spike_digests(tmp_path):
    from flybrain.runtime.runner import spike_digest

    def run_digest():
        conn = chain_connectome(4, synapses=200)
        net = LIFNetwork(conn, LIFParams(), dt_ms=0.1, backend=NumpyBackend(), seed=3)
        net.apply_stimulus(Stimulus([int(conn.flywire_ids[0])], rate_hz=200.0))
        rec = Recorder(conn, RecordingConfig(record_spikes=True, record_population=False), dt_ms=0.1)
        rec.begin(net)
        for k in range(1500):
            rec.on_step(k, net.step(), net)
        return spike_digest(rec.spikes_frame(), 1)

    assert run_digest() == run_digest()
