"""Tests for the degree-preserving rewire.

The point of these tests is not coverage. It is to make the control's claims
falsifiable: if the rewire does not preserve in-degree, out-degree, and the
per-neuron sign marginal exactly, then any difference measured against it is
uninterpretable, and these tests should fail loudly rather than let a broken
control quietly produce a conclusion.
"""
from __future__ import annotations

import numpy as np
import pytest

from flybrain.experiments.rewire import (
    compare_marginals,
    degree_preserving_rewire,
)

SIGN_EXC, SIGN_INH = 1, -1


def _toy_graph(n=40, seed=0):
    """Small signed weighted graph, one row per ordered pair.

    Unique pairs on purpose: the FlyWire connectome stores one row per pair with
    the synapse count as weight, and the v630 validation reports zero duplicates.
    A fixture with parallel edges would test a graph shape the real data never has.
    """
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, size=600).astype(np.int64)
    post = rng.integers(0, n, size=600).astype(np.int64)
    keep = pre != post
    pre, post = pre[keep], post[keep]
    keys = pre * n + post
    _u, first = np.unique(keys, return_index=True)
    pre, post = pre[first], post[first]
    sign = np.where(rng.random(len(pre)) < 0.7, SIGN_EXC, SIGN_INH).astype(np.int8)
    weights = rng.integers(1, 9, size=len(pre)).astype(np.int32)
    assert len(np.unique(pre * n + post)) == len(pre)
    return pre, post, sign, weights, n


def _canonical(pre, post):
    """Canonical form of a directed multigraph: sorted unique edge multiset."""
    key = np.sort(pre.astype(np.int64) * 10**9 + post.astype(np.int64))
    return key


def test_preserves_in_and_out_degree_exactly():
    pre, post, sign, _w, n = _toy_graph()
    p2, q2, s2, rep = degree_preserving_rewire(pre, post, sign, n, seed=1)
    assert rep.in_degree_preserved
    assert rep.out_degree_preserved
    assert np.array_equal(np.bincount(pre, minlength=n), np.bincount(p2, minlength=n))
    assert np.array_equal(np.bincount(post, minlength=n), np.bincount(q2, minlength=n))


def test_preserves_per_neuron_sign_marginals():
    """Every target keeps the same amount of excitation and inhibition."""
    pre, post, sign, _w, n = _toy_graph()
    _p, _q, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=2)
    assert rep.sign_marginals_preserved


def test_preserves_edge_multiset_size_and_sign_count():
    pre, post, sign, _w, n = _toy_graph()
    p2, q2, s2, _rep = degree_preserving_rewire(pre, post, sign, n, seed=3)
    assert len(p2) == len(pre)
    assert len(q2) == len(post)
    assert int((s2 == SIGN_EXC).sum()) == int((sign == SIGN_EXC).sum())


def test_produces_no_self_loops():
    pre, post, sign, _w, n = _toy_graph()
    p2, q2, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=4)
    assert rep.self_loops == 0
    assert not np.any(p2 == q2)


def test_does_not_mutate_inputs():
    pre, post, sign, _w, n = _toy_graph()
    pre0, post0, sign0 = pre.copy(), post.copy(), sign.copy()
    degree_preserving_rewire(pre, post, sign, n, seed=5)
    assert np.array_equal(pre, pre0)
    assert np.array_equal(post, post0)
    assert np.array_equal(sign, sign0)


def test_is_deterministic_for_a_fixed_seed():
    pre, post, sign, _w, n = _toy_graph()
    a = degree_preserving_rewire(pre, post, sign, n, seed=7)[:3]
    b = degree_preserving_rewire(pre, post, sign, n, seed=7)[:3]
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_different_seeds_give_different_graphs():
    pre, post, sign, _w, n = _toy_graph()
    _a, q_a, _s, _r = degree_preserving_rewire(pre, post, sign, n, seed=8)
    _b, q_b, _s2, _r2 = degree_preserving_rewire(pre, post, sign, n, seed=9)
    assert not np.array_equal(_canonical(pre, q_a), _canonical(pre, q_b))


def test_actually_changes_the_wiring():
    """A rewire that returns the input unchanged would pass every other test here."""
    pre, post, sign, _w, n = _toy_graph()
    _p, q2, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=10)
    assert not np.array_equal(_canonical(pre, post), _canonical(pre, q2))
    assert rep.swap_accepted > 0


def test_no_duplicate_edges_introduced():
    pre, post, sign, _w, n = _toy_graph()
    _p, q2, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=11)
    assert rep.duplicates == 0
    # independent of the report: the edge multiset itself must have no repeats
    assert len(np.unique(q2 * n + pre)) == len(pre)


def test_marginal_comparison_detects_a_deliberately_broken_control():
    """compare_marginals must actually discriminate, or it is decoration."""
    pre, post, sign, w, n = _toy_graph()
    good = degree_preserving_rewire(pre, post, sign, n, seed=12)
    same = compare_marginals(pre, post, w, good[0], good[1], w, n)
    assert same["out_degree_identical"] and same["in_degree_identical"]

    # a genuinely destroyed control: random targets, degrees no longer preserved
    rng = np.random.default_rng(13)
    broken_post = rng.integers(0, n, size=len(post))
    broken = compare_marginals(pre, post, w, pre, broken_post, w, n)
    assert not broken["in_degree_identical"]
    assert broken["in_degree_identical"] != same["in_degree_identical"]


def test_handles_a_node_with_a_single_edge():
    """Isolated structures are where rewiring kernels usually crash."""
    pre = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    post = np.array([1, 2, 3, 4, 0], dtype=np.int64)
    sign = np.array([SIGN_EXC, SIGN_EXC, SIGN_INH, SIGN_EXC, SIGN_INH], dtype=np.int8)
    p2, q2, _s, rep = degree_preserving_rewire(pre, post, sign, 5, seed=14, max_attempts=200)
    assert rep.in_degree_preserved and rep.out_degree_preserved
    assert len(p2) == 5


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_degree_vectors_survive_many_rewires(seed):
    """Repeated independent rewires must all preserve the degree vectors."""
    pre, post, sign, _w, n = _toy_graph(seed=seed)
    out_ref = np.bincount(pre, minlength=n)
    in_ref = np.bincount(post, minlength=n)
    for k in range(3):
        p2, q2, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=1000 + k)
        assert rep.in_degree_preserved and rep.out_degree_preserved
        assert np.array_equal(np.bincount(p2, minlength=n), out_ref)
        assert np.array_equal(np.bincount(q2, minlength=n), in_ref)


# --- the silent-degeneracy class, locked down --------------------------------


def test_unknown_sign_edges_also_move():
    """sign == 0 edges must be rewireable, not stranded.

    The v0 connectome has 385 neurons with undefined sign. An edge belonging to
    no sign bucket would never move, leaving the graph partly unmixed while
    still reporting success - the same shape as the annotation-filter bug.
    """
    n = 30
    rng = np.random.default_rng(31)
    pre = rng.integers(0, n, 300).astype(np.int64)
    post = rng.integers(0, n, 300).astype(np.int64)
    keep = pre != post
    pre, post = pre[keep][:120], post[keep][:120]
    sign = np.zeros(len(pre), np.int8)  # every edge unknown-sign
    _p, _q, _s, rep = degree_preserving_rewire(pre, post, sign, n, seed=32,
                                                max_attempts=20_000)
    assert rep.swap_accepted > 0, "unknown-sign edges were stranded"
    assert rep.in_degree_preserved and rep.out_degree_preserved


def test_rejects_an_empty_edge_list_instead_of_spinning():
    with pytest.raises(ValueError, match="empty"):
        degree_preserving_rewire(np.empty(0, np.int64), np.empty(0, np.int64),
                                 np.empty(0, np.int8), 3, seed=33)


def test_rejects_out_of_range_endpoints_with_a_readable_message():
    pre = np.array([0, 1, 2], np.int64)
    post = np.array([1, 2, 0], np.int64)
    sign = np.ones(3, np.int8)
    with pytest.raises(ValueError, match="out of range"):
        degree_preserving_rewire(pre, post, sign, 2, seed=34)


def test_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="same length"):
        degree_preserving_rewire(np.array([0], np.int64), np.array([1, 2], np.int64),
                                 np.array([1], np.int8), 3, seed=35)


def test_rejects_a_graph_too_small_to_rewire():
    """One edge per sign bucket means no swap exists; say so, do not return junk."""
    with pytest.raises(ValueError, match="no sign bucket"):
        degree_preserving_rewire(np.array([0, 1], np.int64), np.array([1, 2], np.int64),
                                 np.array([1, -1], np.int8), 3, seed=36)


def test_rejects_non_positive_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        degree_preserving_rewire(np.array([0, 1], np.int64), np.array([1, 2], np.int64),
                                 np.array([1, 1], np.int8), 3, seed=37, max_attempts=0)
