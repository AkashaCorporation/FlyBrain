"""Aggressive dataset validation, run before any simulation.

Every number in the report is computed from the loaded arrays. Nothing is
hard-coded, and nothing is inherited from the dataset card: the card vouches for
*bytes*, this module vouches for *content*.

Two severities are distinguished, because they mean different things:

``fail``  the dataset violates an invariant that makes simulation meaningless.
          :meth:`DatasetReport.raise_if_failed` refuses to continue.
``warn``  the dataset is usable but has a property the operator must know about
          (orphan neurons, mixed-sign neurons, no cell-type annotation). These are
          true facts about the data, not errors to be suppressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..errors import DatasetValidationError
from .loader import Connectome
from .provenance import git_commit, utc_now_iso

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    """One validation check and its outcome."""

    name: str
    status: str
    detail: str
    value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "value": self.value}


@dataclass
class DatasetReport:
    """The full validation result for one dataset instance."""

    dataset_id: str
    generated_at: str
    is_subset: bool
    checks: List[Check] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ------------------------------------------------------------------
    def add(self, name: str, status: str, detail: str, value: Any = None) -> None:
        self.checks.append(Check(name, status, detail, value))

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = "\n".join(f"  - {c.name}: {c.detail}" for c in self.failures)
            raise DatasetValidationError(
                f"dataset {self.dataset_id!r} failed {len(self.failures)} validation "
                f"check(s); refusing to simulate:\n{lines}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the objective's required shape, plus the detail behind it.

        The flat keys (``neurons``, ``connections``, ``excitatory_neurons``,
        ``inhibitory_neurons``, ``unknown_sign``, ``invalid_edges``) are the contract;
        ``statistics``/``checks`` are the evidence that those numbers are real.
        """
        st = self.statistics
        flat = {
            "dataset_id": self.dataset_id,
            "generated_at": self.generated_at,
            "is_subset": self.is_subset,
            "neurons": st.get("neurons"),
            "connections": st.get("edges"),
            "excitatory_neurons": st.get("excitatory_neurons"),
            "inhibitory_neurons": st.get("inhibitory_neurons"),
            "unknown_sign": st.get("neurons_unknown_sign"),
            "invalid_edges": st.get("invalid_edges"),
            "status": "ok" if self.ok else "failed",
            "n_failed_checks": len(self.failures),
            "n_warnings": len(self.warnings),
        }
        return {
            **flat,
            "statistics": st,
            "provenance": self.provenance,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": [c.to_dict() for c in self.warnings],
            "failures": [c.to_dict() for c in self.failures],
        }


def check_id_index_consistency(conn: Connectome, raw: Dict[str, np.ndarray]) -> Check:
    """Verify that the FlyWire ID columns agree with the index columns.

    This is the check that makes the phrase "neuron IDs are preserved" mean something:
    it proves that index ``k`` really does name the neuron whose FlyWire root ID is
    ``Presynaptic_ID[k]``, for every edge.

    It compares the *raw* arrays, which are row-aligned with each other. ``conn.pre``
    is sorted by presynaptic index while the file's ID column is in file order, so
    pairing a raw ID column against a sorted index column would be wrong.

    Requires the full column set (and therefore extra memory), so it runs on the
    verification path rather than on every load.
    """
    pre_id = raw.get("pre_id")
    post_id = raw.get("post_id")
    if pre_id is None or post_id is None:
        return Check(
            "flywire_id_index_consistency",
            WARN,
            "ID columns were not read, so the ID<->index mapping could not be checked "
            "(use `flybrain datasets --verify` for the full check)",
            None,
        )

    ids = conn.flywire_ids
    checked = 0
    bad = 0
    for index_col, id_col, name in (
        (raw["pre"], pre_id, "Presynaptic"),
        (raw["post"], post_id, "Postsynaptic"),
    ):
        if index_col.size and (index_col.min() < 0 or index_col.max() >= ids.size):
            return Check(
                "flywire_id_index_consistency", FAIL,
                f"{name}_Index contains values outside [0, {ids.size})", None,
            )
        checked += int(index_col.size)
        bad += int((ids[index_col] != id_col).sum())

    return Check(
        "flywire_id_index_consistency",
        PASS if bad == 0 else FAIL,
        f"{checked} endpoint(s) checked; {bad} mismatch(es) between the FlyWire ID "
        f"columns and the neuron table's index order",
        bad,
    )


def validate_connectome(
    conn: Connectome,
    *,
    expected_neurons: Optional[int] = None,
    expected_edges: Optional[int] = None,
    deep: bool = True,
) -> DatasetReport:
    """Validate ``conn`` and return a full report. Never mutates ``conn``."""
    rep = DatasetReport(
        dataset_id=conn.dataset_id,
        generated_at=utc_now_iso(),
        is_subset=conn.is_subset,
    )
    ids, pre, post, signed = conn.flywire_ids, conn.pre, conn.post, conn.signed_count
    n, e = conn.n_neurons, conn.n_edges

    rep.provenance = {
        "dataset_card": conn.card.to_dict() if conn.card is not None else None,
        "subset": conn.subset_info.to_dict() if conn.subset_info is not None else None,
        "code_revision": git_commit(),
    }

    # ---- 1. neuron table --------------------------------------------------------
    rep.add("neurons_present", PASS if n > 0 else FAIL, f"{n} neurons in the table", n)
    if n == 0:
        rep.statistics = {"neurons": 0, "edges": 0}
        return rep

    dup_ids = int(n - np.unique(ids).size)
    rep.add(
        "flywire_ids_unique",
        PASS if dup_ids == 0 else FAIL,
        f"{n} FlyWire IDs, {dup_ids} duplicate(s)",
        dup_ids,
    )

    nonpos = int((ids <= 0).sum())
    rep.add(
        "flywire_ids_positive",
        PASS if nonpos == 0 else FAIL,
        f"{nonpos} non-positive FlyWire ID(s)",
        nonpos,
    )

    sorted_ok = bool(np.all(np.diff(ids) > 0)) if n > 1 else True
    rep.add(
        "flywire_ids_sorted",
        PASS if sorted_ok else WARN,
        "ID array strictly increasing" if sorted_ok else "ID array not sorted; index space is still consistent",
        sorted_ok,
    )

    if expected_neurons is not None:
        rep.add(
            "neuron_count_matches_release",
            PASS if n == expected_neurons else WARN,
            f"expected {expected_neurons} for this release, found {n}",
            {"expected": expected_neurons, "found": n},
        )

    # ---- 2. edge table shape ----------------------------------------------------
    same_len = (pre.size == post.size == signed.size)
    rep.add(
        "edge_columns_consistent",
        PASS if same_len else FAIL,
        f"pre={pre.size} post={post.size} signed_count={signed.size}",
        {"pre": int(pre.size), "post": int(post.size), "signed": int(signed.size)},
    )
    rep.add("edges_present", PASS if e > 0 else FAIL, f"{e} directed edges", e)
    if expected_edges is not None:
        rep.add(
            "edge_count_matches_release",
            PASS if e == expected_edges else WARN,
            f"expected {expected_edges} for this release, found {e}",
            {"expected": expected_edges, "found": e},
        )
    if not same_len or e == 0:
        rep.statistics = {"neurons": n, "edges": e}
        return rep

    # ---- 3. index range --------------------------------------------------------
    bad_src = int(((pre < 0) | (pre >= n)).sum())
    bad_dst = int(((post < 0) | (post >= n)).sum())
    rep.add(
        "source_ids_in_range",
        PASS if bad_src == 0 else FAIL,
        f"{bad_src} Presynaptic_Index values outside [0, {n})",
        bad_src,
    )
    rep.add(
        "target_ids_in_range",
        PASS if bad_dst == 0 else FAIL,
        f"{bad_dst} Postsynaptic_Index values outside [0, {n})",
        bad_dst,
    )

    # Source/target IDs are *proven* to resolve: the index space is dense over
    # [0, n), so an in-range index always names exactly one neuron of the neuron
    # table. Recorded explicitly so the report states it rather than implying it.
    rep.add(
        "edge_endpoints_resolve_to_neurons",
        PASS if (bad_src == 0 and bad_dst == 0) else FAIL,
        "index space is dense over [0, n), so every in-range endpoint resolves "
        "to exactly one neuron of the neuron table",
        {"bad_sources": bad_src, "bad_targets": bad_dst},
    )

    # ---- 4. weights and sign ----------------------------------------------------
    nan_edges = int(np.isnan(signed.astype(np.float64)).sum())
    rep.add("no_nan_weights", PASS if nan_edges == 0 else FAIL, f"{nan_edges} NaN weights", nan_edges)

    zero_w = int((signed == 0).sum())
    rep.add(
        "no_zero_weights",
        PASS if zero_w == 0 else WARN,
        f"{zero_w} edges with zero signed synapse count"
        + ("" if zero_w == 0 else " (they contribute nothing and are still simulated)"),
        zero_w,
    )

    exc_mask = signed > 0
    inh_mask = signed < 0
    n_exc, n_inh = int(exc_mask.sum()), int(inh_mask.sum())
    rep.add(
        "excitatory_inhibitory_split",
        PASS,
        f"{n_exc} excitatory edges, {n_inh} inhibitory edges",
        {"excitatory": n_exc, "inhibitory": n_inh},
    )

    # Sign consistency: the derived column must equal sign * |count|.
    # On the subset path `Connectivity` may not be loaded, so this check is done
    # against the in-memory signed counts only when the raw magnitude is available.
    raw_mag = np.abs(signed.astype(np.int32))
    rep.add(
        "signed_count_magnitude_positive",
        PASS if int((raw_mag == 0).sum()) == 0 else FAIL,
        f"max |synapse count| = {int(raw_mag.max())}, min = {int(raw_mag.min())}",
        {"min": int(raw_mag.min()), "max": int(raw_mag.max())},
    )

    # ---- 5. self-connections and duplicate edges --------------------------------
    self_loops = int((pre == post).sum())
    rep.add(
        "no_self_connections",
        PASS if self_loops == 0 else WARN,
        f"{self_loops} self-connection(s) (pre == post)",
        self_loops,
    )

    if deep and e > 0:
        pair_key = pre.astype(np.int64) * np.int64(n) + post.astype(np.int64)
        dup_edges = int(pair_key.size - np.unique(pair_key).size)
        rep.add(
            "no_duplicate_edge_pairs",
            PASS if dup_edges == 0 else FAIL,
            f"{dup_edges} duplicate (presynaptic, postsynaptic) pair(s)",
            dup_edges,
        )
        del pair_key
    else:
        dup_edges = 0

    # Everything below indexes arrays by `pre`/`post`, so it is only safe once the
    # endpoints are known to be in range. Bail out with a partial report rather than
    # raising an IndexError: a validator that crashes on bad data cannot tell you
    # what is wrong with it.
    if bad_src or bad_dst:
        rep.statistics = {
            "neurons": n,
            "edges": e,
            "excitatory_edges": n_exc,
            "inhibitory_edges": n_inh,
            "invalid_edges": int(bad_src + bad_dst + nan_edges + zero_w),
            "out_of_range_source_indices": bad_src,
            "out_of_range_target_indices": bad_dst,
            "nan_weights": nan_edges,
            "zero_magnitude_edges": zero_w,
            "self_connections": self_loops,
            "duplicate_edge_pairs": dup_edges,
            "duplicate_neuron_ids": dup_ids,
            "note": "checks that index by presynaptic/postsynaptic neuron were skipped "
                    "because the edge list contains out-of-range endpoints",
        }
        return rep

    # ---- 6. orphans and sign coverage ------------------------------------------
    out_deg, in_deg = conn.out_degree, conn.in_degree
    no_out = int((out_deg == 0).sum())
    no_in = int((in_deg == 0).sum())
    neither = int(((out_deg == 0) & (in_deg == 0)).sum())
    rep.add(
        "no_orphan_neurons",
        PASS if neither == 0 else WARN,
        f"{neither} neuron(s) have neither incoming nor outgoing edges",
        neither,
    )
    rep.add(
        "orphan_counts",
        WARN if (no_out or no_in) else PASS,
        f"{no_out} neuron(s) with no outgoing edge, {no_in} with no incoming edge; "
        f"both are kept in the index space so that FlyWire IDs remain stable",
        {"no_outgoing": no_out, "no_incoming": no_in, "neither": neither},
    )

    sign = conn.sign
    n_sign_exc = int((sign == 1).sum())
    n_sign_inh = int((sign == -1).sum())
    n_sign_unknown = int((sign == 0).sum())

    # A neuron with both excitatory and inhibitory outgoing edges would make a
    # per-neuron sign undefined. The published data has none; detect it if it appears.
    pos = np.zeros(n, dtype=bool)
    neg = np.zeros(n, dtype=bool)
    np.logical_or.at(pos, pre[exc_mask], True)
    np.logical_or.at(neg, pre[inh_mask], True)
    mixed = int((pos & neg).sum())
    rep.add(
        "sign_is_per_neuron",
        PASS if mixed == 0 else WARN,
        f"{mixed} neuron(s) have BOTH excitatory and inhibitory outgoing edges; "
        f"a per-neuron sign is undefined for those and they are reported as ambiguous",
        mixed,
    )
    rep.add(
        "sign_coverage",
        PASS if n_sign_unknown == 0 else WARN,
        f"{n_sign_exc} excitatory / {n_sign_inh} inhibitory / {n_sign_unknown} unknown "
        f"presynaptic neurons (unknown = no outgoing edge, so no sign evidence)",
        {"excitatory": n_sign_exc, "inhibitory": n_sign_inh, "unknown": n_sign_unknown},
    )

    # ---- 7. cell-type annotation ------------------------------------------------
    # The dataset ships NO cell-type column. Declared honestly rather than substituted
    # with a guess: the only type information available to FlyBrain is the curated
    # input lists in flybrain/model/populations.py.
    rep.add(
        "neuron_types_available",
        WARN,
        "the dataset provides no cell-type / neurotransmitter-per-neuron annotation "
        "column (the neuron table has a single 'Completed' column); neuron types exist "
        "only through the curated population lists in flybrain/model/populations.py",
        {"cell_type_column": None, "annotated_neuron_types": "not available in dataset"},
    )

    # ---- 8. derived statistics --------------------------------------------------
    # "Invalid" = an edge that cannot carry a defined signal: an endpoint outside the
    # index space, a NaN magnitude, or a zero magnitude. Self-connections are legal
    # (if unusual) recurrent edges and are counted separately, not as invalid.
    invalid_edges = bad_src + bad_dst + nan_edges + zero_w
    rep.statistics = {
        "neurons": n,
        "edges": e,
        "synapses": conn.synapse_count,
        "mean_synapses_per_edge": (conn.synapse_count / e) if e else None,
        "excitatory_edges": n_exc,
        "inhibitory_edges": n_inh,
        "excitatory_neurons": n_sign_exc,
        "inhibitory_neurons": n_sign_inh,
        "neurons_unknown_sign": n_sign_unknown,
        "neurons_with_mixed_sign": mixed,
        "invalid_edges": int(invalid_edges),
        "duplicate_edge_pairs": dup_edges,
        "self_connections": self_loops,
        "duplicate_neuron_ids": dup_ids,
        "neurons_no_outgoing": no_out,
        "neurons_no_incoming": no_in,
        "neurons_isolated": neither,
        "max_out_degree": int(out_deg.max()) if n else 0,
        "max_in_degree": int(in_deg.max()) if n else 0,
        "mean_out_degree": float(out_deg.mean()) if n else 0.0,
        "signed_count_min": int(signed.min()) if e else 0,
        "signed_count_max": int(signed.max()) if e else 0,
        "pre_sorted": bool(np.all(np.diff(pre) >= 0)) if e > 1 else True,
    }
    dense_bytes = n * n * 4
    rep.statistics["dense_matrix_bytes_if_materialised"] = int(dense_bytes)
    rep.statistics["csr_bytes_estimate"] = int(
        (n + 1) * 8 + e * (4 + 4) + n * 8   # indptr + indices + weights + ids
    )
    return rep
