"""Degree-preserving connectome rewire, for the control condition.

A null model is only as good as what it preserves. A plain random rewire
destroys degree structure, so a difference in outcome cannot be attributed to
topology rather than to trivial network size. The standard fix is a
double-edge swap, which preserves the in- and out-degree of every node exactly.

But "preserves degree" is not sufficient here, and this is the part that
matters scientifically. A naive double-edge swap on a signed, weighted graph
also destroys:

- the sign marginals: how much excitation and inhibition a neuron receives;
- the weight distribution attached to a given degree;
- the excitatory/inhibitory balance of the whole network.

Any of those changes would show up in a firing-rate result and be
indistinguishable from a genuine topology effect. So the swap implemented here
is constrained to swap only among edges that agree in sign, and the caller is
expected to check the weight marginals separately (see
`compare_marginals`).

Reference: the double-edge swap is the standard rewiring kernel of
Maslov & Sneppen and of networkx's `double_edge_swap`. This is a
re-implementation, not a copy; the sign constraint is the addition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewireReport:
    """What the rewire actually preserved, measured rather than assumed."""

    n_neurons: int
    n_edges: int
    swap_attempts: int
    swap_accepted: int
    swaps_rejected_duplicate: int
    in_degree_preserved: bool
    out_degree_preserved: bool
    sign_marginals_preserved: bool
    self_loops: int
    duplicates: int

    @property
    def acceptance_rate(self) -> float:
        if self.swap_attempts == 0:
            return 0.0
        return self.swap_accepted / self.swap_attempts


def _build_csr(pre: np.ndarray, post: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR grouped by presynaptic node. Edge arrays must already be sorted by pre."""
    order = np.argsort(pre, kind="stable")
    pre_s, post_s = pre[order], post[order]
    counts = np.bincount(pre_s, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return indptr, np.ascontiguousarray(post_s)


def degree_preserving_rewire(
    pre: np.ndarray,
    post: np.ndarray,
    sign: np.ndarray,
    n_neurons: int,
    *,
    seed: int,
    max_attempts: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RewireReport]:
    """Rewire `post` while preserving every node's in- and out-degree exactly.

    `sign` is per-edge and is never changed; swaps are only accepted between two
    edges carrying the same sign, so each target keeps the same mix of
    excitation and inhibition it had.

    Returns (pre, post, sign, report). Input arrays are not mutated.
    """
    rng = np.random.default_rng(seed)
    pre = np.ascontiguousarray(pre, dtype=np.int64)
    post = np.ascontiguousarray(post, dtype=np.int64)
    sign = np.ascontiguousarray(sign, dtype=np.int8)
    m = len(pre)

    in_before = np.bincount(post, minlength=n_neurons)
    out_before = np.bincount(pre, minlength=n_neurons)
    sign_before = np.bincount(post, weights=(sign > 0).astype(np.float64),
                              minlength=n_neurons)

    order = np.argsort(pre, kind="stable")
    pre, post, sign = pre[order], post[order], sign[order]
    indptr, targets = _build_csr(pre, post, n_neurons)

    # The FlyWire connectome stores one row per ordered pair, with the synapse
    # count as the weight: the v630 validation reports zero duplicate pairs. A
    # control that introduced parallel edges would therefore no longer be the
    # same class of object, and a difference measured against it would be
    # confounded. Swaps that would create an existing pair are rejected.
    #
    # Cost: one Python int per edge. That is trivial for a circuit (the intended
    # use) and too heavy for whole-brain, which needs a sorted int64 key array
    # with searchsorted instead. Do not use this on 14.7M edges.
    present = set((pre * n_neurons + targets).tolist())

    # bucket edges by sign so a swap never crosses the excitatory/inhibitory split
    pos_edges = np.nonzero(sign > 0)[0]
    neg_edges = np.nonzero(sign < 0)[0]

    accepted = 0
    rejected_duplicate = 0
    for _ in range(max_attempts):
        pool = pos_edges if rng.random() < (len(pos_edges) / max(m, 1)) else neg_edges
        if len(pool) < 2:
            continue
        e1, e2 = rng.choice(pool, size=2, replace=False)
        p1, t1 = pre[e1], targets[e1]
        p2, t2 = pre[e2], targets[e2]
        # reject self-loops and the degenerate 2-cycle the swap would create
        if t1 == t2 or p1 == t2 or p2 == t1:
            continue
        k_new_1 = p1 * n_neurons + t2
        k_new_2 = p2 * n_neurons + t1
        if k_new_1 in present or k_new_2 in present:
            rejected_duplicate += 1
            continue
        # discard, not remove: if the input carries parallel edges the same key
        # can appear twice, and losing one copy is correct, raising is not
        present.discard(p1 * n_neurons + t1)
        present.discard(p2 * n_neurons + t2)
        present.add(k_new_1)
        present.add(k_new_2)
        targets[e1], targets[e2] = t2, t1
        accepted += 1
        if accepted >= m:  # every edge moved at least once: the kernel has mixed
            break

    pre_out = np.repeat(np.arange(n_neurons, dtype=np.int64), np.diff(indptr))
    post_out = targets
    sign_out = sign

    in_after = np.bincount(post_out, minlength=n_neurons)
    out_after = np.bincount(pre_out, minlength=n_neurons)
    sign_after = np.bincount(post_out, weights=(sign_out > 0).astype(np.float64),
                             minlength=n_neurons)

    key = post_out * n_neurons + pre_out
    return pre_out, post_out, sign_out, RewireReport(
        n_neurons=n_neurons,
        n_edges=m,
        swap_attempts=min(max_attempts, accepted + rejected_duplicate),
        swap_accepted=accepted,
        swaps_rejected_duplicate=rejected_duplicate,
        in_degree_preserved=bool(np.array_equal(in_before, in_after)),
        out_degree_preserved=bool(np.array_equal(out_before, out_after)),
        sign_marginals_preserved=bool(np.allclose(sign_before, sign_after, atol=0.5)),
        self_loops=int(np.count_nonzero(pre_out == post_out)),
        duplicates=int(len(key) - len(np.unique(key))),
    )


def compare_marginals(
    pre_a: np.ndarray, post_a: np.ndarray, w_a: np.ndarray,
    pre_b: np.ndarray, post_b: np.ndarray, w_b: np.ndarray,
    n_neurons: int,
) -> dict:
    """Distributional comparison of two graphs. Used to audit a control.

    A control is only adequate if it matches the real graph on everything that
    could produce a firing-rate difference other than topology.
    """
    def out_deg(pre):
        return np.bincount(pre, minlength=n_neurons)

    def in_deg(post):
        return np.bincount(post, minlength=n_neurons)

    def jensen(a, b, bins=64):
        hist_a, _ = np.histogram(a, bins=bins, density=False)
        hist_b, _ = np.histogram(b, bins=bins, density=False)
        pa_ = hist_a / max(hist_a.sum(), 1)
        pb_ = hist_b / max(hist_b.sum(), 1)
        m = 0.5 * (pa_ + pb_)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl = float(np.nansum(np.where(pa_ > 0, pa_ * np.log(pa_ / m), 0.0))
                       + np.nansum(np.where(pb_ > 0, pb_ * np.log(pb_ / m), 0.0)))
        return kl

    return {
        "out_degree_identical": bool(np.array_equal(out_deg(pre_a), out_deg(pre_b))),
        "in_degree_identical": bool(np.array_equal(in_deg(post_a), in_deg(post_b))),
        "out_degree_histogram_kl": jensen(out_deg(pre_a), out_deg(pre_b)),
        "weight_histogram_kl": jensen(w_a, w_b),
        "edges_a": int(len(pre_a)),
        "edges_b": int(len(pre_b)),
    }
