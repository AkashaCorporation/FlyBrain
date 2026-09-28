"""Dataset layer: loading, ID mapping, connectivity indexing, subsetting, validation.

The dataset-dependent tests skip cleanly when the dataset has not been staged, so
the suite is runnable on a fresh checkout. The synthetic tests always run: the
in-memory contract (indexing, subsetting, invariants) must hold regardless of
whether 180 MB of parquet happens to be on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from flybrain.data import (
    Connectome,
    load_connectome,
    validate_connectome,
)
from flybrain.data.synthetic import chain_connectome, make_synthetic_connectome
from flybrain.errors import DatasetError, DatasetValidationError, UnknownNeuronError

from conftest import needs_dataset


# --------------------------------------------------------------------------------------
# synthetic: the in-memory contract
# --------------------------------------------------------------------------------------


def test_connectome_basic_shape():
    conn = make_synthetic_connectome([(0, 1, 5), (0, 2, -3), (1, 2, 7)], 3)
    assert conn.n_neurons == 3
    assert conn.n_edges == 3
    assert conn.synapse_count == 5 + 3 + 7
    assert conn.synapse_count == 15


def test_edges_are_sorted_by_presynaptic_index():
    """The class invariant that lets CSR be rebuilt without a re-sort."""
    conn = make_synthetic_connectome([(2, 0, 1), (0, 1, 1), (1, 2, 1), (0, 2, 1)], 3)
    assert np.all(np.diff(conn.pre) >= 0)
    assert conn.pre.tolist() == [0, 0, 1, 2]


def test_id_mapping_round_trip():
    conn = make_synthetic_connectome([(0, 1, 1)], 4)
    for i, fid in enumerate(conn.flywire_ids):
        assert conn.index_of(int(fid)) == i
    assert conn.indices_of([int(conn.flywire_ids[2])]).tolist() == [2]


def test_unknown_id_raises_with_the_full_list():
    conn = make_synthetic_connectome([(0, 1, 1)], 3)
    with pytest.raises(UnknownNeuronError) as ei:
        conn.indices_of([int(conn.flywire_ids[0]), 1, 2, 3])
    msg = str(ei.value)
    assert "3 of 4" in msg and "1" in msg


def test_filter_present_does_not_raise():
    conn = make_synthetic_connectome([(0, 1, 1)], 3)
    present, missing = conn.filter_present([int(conn.flywire_ids[0]), 999])
    assert present.tolist() == [int(conn.flywire_ids[0])]
    assert missing.tolist() == [999]


def test_csr_indptr_consistent_with_out_degree():
    conn = make_synthetic_connectome([(0, 1, 1), (0, 2, 1), (2, 1, 1), (2, 0, 1)], 4)
    indptr, indices = conn.csr()
    assert indptr.shape == (conn.n_neurons + 1,)
    assert indptr[-1] == conn.n_edges
    counts = np.diff(indptr)
    assert counts.tolist() == conn.out_degree.tolist()
    # column indices for row i must be exactly the postsynaptic partners
    for i in range(conn.n_neurons):
        expected = sorted(conn.post[conn.pre == i].tolist())
        got = sorted(indices[indptr[i]:indptr[i + 1]].tolist())
        assert got == expected


def test_degree_matches_manual_count():
    conn = make_synthetic_connectome([(0, 1, 1), (0, 2, 1), (1, 2, 1)], 3)
    assert conn.out_degree.tolist() == [2, 1, 0]
    assert conn.in_degree.tolist() == [0, 1, 2]


def test_sign_is_per_presynaptic_neuron():
    conn = make_synthetic_connectome([(0, 1, 5), (1, 2, -5), (2, 0, -5)], 3)
    assert conn.sign.tolist() == [1, -1, -1]


def test_mixed_sign_neuron_is_reported_as_unknown():
    """A neuron with both signs has no well-defined per-neuron sign."""
    conn = make_synthetic_connectome([(0, 1, 5), (0, 2, -5)], 3)
    assert conn.sign[0] == 0
    rep = validate_connectome(conn)
    names = {c.name: c for c in rep.checks}
    assert names["sign_is_per_neuron"].status == "warn"
    assert rep.statistics["neurons_with_mixed_sign"] == 1


def test_self_loop_is_counted_but_not_fatal():
    conn = make_synthetic_connectome([(0, 0, 3), (0, 1, 3)], 2)
    rep = validate_connectome(conn)
    assert rep.statistics["self_connections"] == 1
    assert rep.ok


# --------------------------------------------------------------------------------------
# subsetting
# --------------------------------------------------------------------------------------


def test_subset_contains_all_seeds_and_closes_over_edges():
    conn = make_synthetic_connectome(
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)], 6
    )
    seeds = [int(conn.flywire_ids[0])]
    sub = conn.subset(seeds, hops=2, max_neurons=100)
    assert sub.n_neurons == 3, "2 hops from neuron 0 reaches neurons 0,1,2"
    assert int(conn.flywire_ids[0]) in sub.id_to_index
    # every retained edge must have both endpoints inside the subset
    assert int(sub.pre.max()) < sub.n_neurons
    assert int(sub.post.max()) < sub.n_neurons
    assert sub.subset_info is not None
    assert sub.subset_info.missing_seed_ids == []
    assert sub.subset_info.hop_histogram["0"] == 1


def test_subset_traverses_both_directions():
    """A neuron whose inputs are missing can never be driven, so in-edges count."""
    conn = make_synthetic_connectome([(0, 1, 1)], 2)
    sub = conn.subset([int(conn.flywire_ids[1])], hops=1, max_neurons=10)
    assert sub.n_neurons == 2, "neuron 1's presynaptic partner 0 must be pulled in"


def test_subset_respects_max_neurons_and_reports_truncation():
    edges = [(0, i, 1) for i in range(1, 51)]
    conn = make_synthetic_connectome(edges, 51)
    sub = conn.subset([int(conn.flywire_ids[0])], hops=1, max_neurons=10)
    assert sub.n_neurons == 10
    assert sub.subset_info.truncated is True


def test_subset_is_deterministic():
    conn = make_synthetic_connectome(
        [(0, i, 1) for i in range(1, 30)] + [(i, i + 1, 1) for i in range(1, 29)], 30
    )
    a = conn.subset([int(conn.flywire_ids[0])], hops=2, max_neurons=12)
    b = conn.subset([int(conn.flywire_ids[0])], hops=2, max_neurons=12)
    assert a.flywire_ids.tolist() == b.flywire_ids.tolist()
    assert a.pre.tolist() == b.pre.tolist()
    assert a.post.tolist() == b.post.tolist()


def test_subset_preserves_ids_and_keeps_pre_sorted():
    conn = chain_connectome(6, synapses=200)
    sub = conn.subset(list(conn.flywire_ids), hops=0, max_neurons=100)
    assert sub.flywire_ids.tolist() == conn.flywire_ids.tolist()
    assert np.all(np.diff(sub.pre) >= 0)


def test_subset_records_unknown_seeds_without_raising():
    conn = make_synthetic_connectome([(0, 1, 1)], 2)
    sub = conn.subset([int(conn.flywire_ids[0]), 123_456_789], hops=1, max_neurons=10)
    assert sub.subset_info.missing_seed_ids == [123_456_789]


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def test_validation_report_matches_independent_counts():
    # 0 and 1 are purely excitatory presynaptic neurons, 2 and 3 purely inhibitory,
    # and 4 has no outgoing edge at all so it carries no sign evidence.
    conn = make_synthetic_connectome(
        [(0, 1, 5), (1, 2, 5), (2, 0, -5), (3, 0, -5)], 5, dataset_id="unit"
    )
    rep = validate_connectome(conn)
    d = rep.to_dict()
    assert d["neurons"] == 5
    assert d["connections"] == 4
    assert d["excitatory_neurons"] == 2      # 0, 1
    assert d["inhibitory_neurons"] == 2      # 2, 3
    assert d["unknown_sign"] == 1            # 4 has no outgoing edge
    assert d["invalid_edges"] == 0
    assert d["status"] == "ok"
    assert rep.ok
    assert rep.statistics["neurons_with_mixed_sign"] == 0
    assert rep.statistics["neurons_no_outgoing"] == 1


def test_validation_refuses_out_of_range_endpoints():
    conn = Connectome(
        dataset_id="broken",
        flywire_ids=np.arange(1, 4, dtype=np.int64),
        pre=np.array([0, 5], dtype=np.int32),
        post=np.array([1, 2], dtype=np.int32),
        signed_count=np.array([1, 1], dtype=np.int16),
    )
    rep = validate_connectome(conn)
    assert not rep.ok
    with pytest.raises(DatasetValidationError):
        rep.raise_if_failed()


def test_validation_refuses_zero_magnitude_weights():
    conn = Connectome(
        dataset_id="broken",
        flywire_ids=np.arange(1, 3, dtype=np.int64),
        pre=np.array([0], dtype=np.int32),
        post=np.array([1], dtype=np.int32),
        signed_count=np.array([0], dtype=np.int16),
    )
    rep = validate_connectome(conn)
    assert not rep.ok
    assert rep.statistics["invalid_edges"] == 1


def test_validation_reports_the_missing_cell_type_column_honestly():
    conn = make_synthetic_connectome([(0, 1, 1)], 2)
    rep = validate_connectome(conn)
    names = {c.name: c for c in rep.checks}
    assert "neuron_types_available" in names
    assert names["neuron_types_available"].status == "warn"
    assert "no cell-type" in names["neuron_types_available"].detail


# --------------------------------------------------------------------------------------
# the real dataset
# --------------------------------------------------------------------------------------


@needs_dataset
def test_real_dataset_matches_the_audited_numbers(real_connectome):
    """Pin the numbers measured in the audit. A changed dataset must fail loudly."""
    conn = real_connectome
    assert conn.dataset_id == "flywire_630"
    assert conn.n_neurons == 127_400
    assert conn.n_edges == 14_687_178
    assert conn.synapse_count == 52_793_639
    assert np.all(np.diff(conn.pre) >= 0)
    assert np.all(np.diff(conn.flywire_ids) > 0)


@needs_dataset
def test_real_dataset_validation_is_clean(real_connectome):
    rep = validate_connectome(real_connectome)
    assert rep.ok, [c.detail for c in rep.failures]
    st = rep.statistics
    assert st["excitatory_neurons"] == 86_543
    assert st["inhibitory_neurons"] == 40_472
    assert st["neurons_unknown_sign"] == 385
    assert st["neurons_with_mixed_sign"] == 0
    assert st["invalid_edges"] == 0
    assert st["duplicate_edge_pairs"] == 0
    assert st["self_connections"] == 0
    assert st["neurons_no_outgoing"] == 385
    assert st["neurons_no_incoming"] == 607
    assert st["neurons_isolated"] == 0
    # a dense matrix would be tens of GB; the report must say so
    assert st["dense_matrix_bytes_if_materialised"] > 60e9


@needs_dataset
def test_real_dataset_orphans_are_kept_in_the_index_space(real_connectome):
    """Dropping orphan neurons would renumber everything and break FlyWire ID stability."""
    conn = real_connectome
    assert conn.out_degree.shape[0] == conn.n_neurons
    assert int((conn.out_degree == 0).sum()) == 385
    # an orphan still resolves by ID
    orphan = int(np.flatnonzero(conn.out_degree == 0)[0])
    assert conn.index_of(int(conn.flywire_ids[orphan])) == orphan


@needs_dataset
def test_real_dataset_subset_of_the_sugar_grns(real_connectome):
    from flybrain.model.populations import PopulationRegistry

    conn = real_connectome
    reg = PopulationRegistry.builtin()
    present, missing = reg.resolve("sugar_grn", conn.flywire_ids)
    assert present.size == 21, "all 21 declared sugar GRNs must be in the v630 dataset"
    assert missing.size == 0

    sub = conn.subset([int(x) for x in present], hops=1, max_neurons=2000)
    assert sub.is_subset
    assert sub.n_neurons > 21
    # every sugar GRN survives into the subset
    for fid in present.tolist():
        assert int(fid) in sub.id_to_index
    # and the subset validation is still clean
    rep = validate_connectome(sub)
    assert rep.ok, [c.detail for c in rep.failures]


@needs_dataset
def test_real_dataset_sign_is_never_mixed(real_connectome):
    """The audit's key structural finding: sign is a property of the presynaptic neuron."""
    conn = real_connectome
    sign = conn.sign[conn.pre]
    signed = conn.signed_count
    # for every edge, the presynaptic neuron's sign must equal the edge's sign
    consistent = np.all((sign > 0) == (signed > 0))
    assert consistent


@needs_dataset
def test_processed_cache_round_trip(tmp_path, real_connectome):
    real_connectome.save_cache(tmp_path)
    loaded = Connectome.load_cache(tmp_path, real_connectome.dataset_id)
    assert loaded is not None
    conn, meta = loaded
    assert meta["n_neurons"] == real_connectome.n_neurons
    assert np.array_equal(conn.flywire_ids, real_connectome.flywire_ids)
    assert np.array_equal(conn.pre, real_connectome.pre)
    assert np.array_equal(conn.post, real_connectome.post)
    assert np.array_equal(conn.signed_count, real_connectome.signed_count)


@needs_dataset
def test_loader_refuses_a_truncated_file(tmp_path, real_connectome):
    """A file whose size disagrees with the dataset card must not be simulated on."""
    import shutil

    from flybrain import config as fb_config
    from flybrain.data import load_dataset_card

    card = load_dataset_card("flywire_630", fb_config.METADATA_DIR)
    assert card is not None
    raw = tmp_path / "flywire_630"
    raw.mkdir(parents=True)
    for f in card.files:
        dst = raw / f.name
        shutil.copy2(fb_config.RAW_DIR / "flywire_630" / f.name, dst)
        dst.write_bytes(dst.read_bytes()[: 1024 * 1024])  # truncate

    with pytest.raises(DatasetError) as ei:
        load_connectome("flywire_630", tmp_path, None, fb_config.METADATA_DIR)
    assert "size" in str(ei.value) or "expected" in str(ei.value)


def test_loader_rejects_unknown_dataset_name():
    with pytest.raises(DatasetError):
        load_connectome("not_a_dataset", "nope")


@needs_dataset
def test_staged_cards_carry_measured_counts():
    """A card that says ``neuron_count: 0`` is a card that measured nothing.

    Staging only vouches for bytes; the counts are written by the validation step.
    This asserts the two did meet, because a provenance record with zero counts is
    worse than an obviously absent one.
    """
    from flybrain import config as fb_config
    from flybrain.data import list_dataset_cards

    cards = {c.dataset_id: c for c in list_dataset_cards(fb_config.METADATA_DIR)}
    assert "flywire_630" in cards
    card = cards["flywire_630"]
    assert card.neuron_count == 127_400
    assert card.edge_count == 14_687_178
    assert card.sha256
    assert len(card.files) == 2
    for f in card.files:
        assert len(f.sha256) == 64
        assert f.size_bytes > 0
        assert f.role in ("neuron_table", "connectivity_table")
    assert card.extra.get("synapses_total") == 52_793_639
    assert card.extra.get("excitatory_neurons") == 86_543
    assert card.extra.get("inhibitory_neurons") == 40_472
    assert card.extra.get("neurons_unknown_sign") == 385
