"""Connectivity analysis: degrees, neighbours, and the neuron inspector.

``neuron_profile`` backs the interactive console's ``neuron <flywire_id>`` command,
which is the fastest way to ask "what is this thing and who does it talk to".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def degree_stats(conn) -> Dict[str, Any]:
    out_deg, in_deg = conn.out_degree, conn.in_degree
    return {
        "n_neurons": conn.n_neurons,
        "n_edges": conn.n_edges,
        "out_degree": {
            "min": int(out_deg.min()), "max": int(out_deg.max()),
            "mean": float(out_deg.mean()), "median": float(np.median(out_deg)),
            "n_zero": int((out_deg == 0).sum()),
        },
        "in_degree": {
            "min": int(in_deg.min()), "max": int(in_deg.max()),
            "mean": float(in_deg.mean()), "median": float(np.median(in_deg)),
            "n_zero": int((in_deg == 0).sum()),
        },
    }


def neighbours(conn, flywire_id: int, direction: str = "out", limit: int = 20) -> Dict[str, Any]:
    """Neighbours of one neuron with the signed synapse counts between them.

    ``direction`` is ``out``, ``in``, or ``both``. Returned edges are sorted by
    |synapse count| descending, because the strongest connections are the ones
    that actually matter for whether anything propagates.
    """
    i = conn.index_of(int(flywire_id))
    rows: List[Dict[str, Any]] = []

    if direction in ("out", "both"):
        sel = np.flatnonzero(conn.pre == i)
    else:
        sel = np.empty(0, np.int64)
    if direction in ("in", "both"):
        sel = np.concatenate([sel, np.flatnonzero(conn.post == i)]) if sel.size else np.flatnonzero(conn.post == i)

    if sel.size == 0:
        return {"flywire_id": int(flywire_id), "index": int(i), "direction": direction,
                "n_connections": 0, "connections": []}

    pre = conn.pre[sel]
    post = conn.post[sel]
    signed = conn.signed_count[sel]
    order = np.argsort(-np.abs(signed.astype(np.int32)), kind="stable")[: max(limit, 0)]

    for k in order:
        is_outgoing = int(pre[k]) == i
        other_idx = int(post[k]) if is_outgoing else int(pre[k])
        rows.append(
            {
                "direction": "outgoing" if is_outgoing else "incoming",
                "flywire_id": int(conn.flywire_ids[other_idx]),
                "synapses": int(abs(int(signed[k]))),
                "sign": "excitatory" if int(signed[k]) > 0 else "inhibitory",
            }
        )
    return {
        "flywire_id": int(flywire_id),
        "index": int(i),
        "direction": direction,
        "n_connections": int(sel.size),
        "connections": rows,
    }


def neuron_profile(conn, flywire_id: int, *, registry=None, rate_hz: Optional[float] = None) -> Dict[str, Any]:
    """Everything FlyBrain knows about one neuron.

    Reports the *absence* of cell-type annotation explicitly (``neuron_type: None``
    plus a note) rather than leaving the field blank, since "we do not have this"
    and "this neuron has no type" are different statements.
    """
    i = conn.index_of(int(flywire_id))
    sign = int(conn.sign[i])
    sign_label = {1: "excitatory", -1: "inhibitory", 0: "unknown (no outgoing edge)"}[sign]

    memberships = []
    if registry is not None:
        fid = int(flywire_id)
        for pop in registry:
            if any(int(x) == fid for x in pop.ids):
                memberships.append(pop.name)

    out = {
        "flywire_id": int(flywire_id),
        "index": int(i),
        "neuron_type": None,
        "neuron_type_note": (
            "the FlyWire-derived dataset ships no cell-type column, so no type is available "
            "from data; `populations` lists curated input lists this neuron belongs to"
        ),
        "populations": memberships,
        "neurotransmitter_sign": sign_label,
        "in_degree": int(conn.in_degree[i]),
        "out_degree": int(conn.out_degree[i]),
        "out_synapse_count": int(np.abs(conn.signed_count[conn.pre == i].astype(np.int32)).sum()),
        "in_synapse_count": int(np.abs(conn.signed_count[conn.post == i].astype(np.int32)).sum()),
    }
    if rate_hz is not None:
        out["current_firing_rate_hz"] = float(rate_hz)
    return out


def connectivity_between(conn, source_ids: Sequence[int], target_ids: Sequence[int]) -> Dict[str, Any]:
    """Directed connection statistics between two neuron sets.

    The building block for asking "is there a monosynaptic route from the sugar
    GRNs to the neurons that responded", which is the reachability question a
    propagation claim has to answer.
    """
    src = conn.indices_of(list(source_ids))
    dst = conn.indices_of(list(target_ids))
    src_mask = np.zeros(conn.n_neurons, dtype=bool)
    dst_mask = np.zeros(conn.n_neurons, dtype=bool)
    src_mask[src] = True
    dst_mask[dst] = True

    sel = src_mask[conn.pre] & dst_mask[conn.post]
    k = int(sel.sum())
    if k == 0:
        return {
            "source_count": int(src.size),
            "target_count": int(dst.size),
            "connections": 0,
            "source_neurons_with_target_output": 0,
            "target_neurons_with_source_input": 0,
            "synapses": 0,
            "excitatory_connections": 0,
            "inhibitory_connections": 0,
        }
    signed = conn.signed_count[sel].astype(np.int32)
    return {
        "source_count": int(src.size),
        "target_count": int(dst.size),
        "connections": k,
        "source_neurons_with_target_output": int(np.unique(conn.pre[sel]).size),
        "target_neurons_with_source_input": int(np.unique(conn.post[sel]).size),
        "synapses": int(np.abs(signed).sum()),
        "excitatory_connections": int((signed > 0).sum()),
        "inhibitory_connections": int((signed < 0).sum()),
        "note": "counts are for the loaded connectome, which may be a subset of the whole brain",
    }
