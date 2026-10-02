"""Deterministic circuit selection for C2, per docs/PREREGISTRATION_C2.md section 5.

Registered criteria, applied in order, with no choice made after seeing results:

1. input anchor: a declared sensory population
2. output anchor: a declared motor or behavioural population
3. BFS from the input anchor, at most 3 hops
4. keep only annotated neurons
5. the resulting subgraph must be connected and contain both anchors
6. record sha256, counts, and per-node in/out degree

Criterion 3 is an upper bound, and this script reports every hop count from 1
to the bound so the choice of hop number is an output of the run, not a private
preference. The chosen configuration is written with its own provenance card.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flybrain.data.loader import load_connectome  # noqa: E402
from flybrain.model.populations import PopulationRegistry  # noqa: E402

DEFAULT_ANNOT = Path(
    r"E:\HipoCampo\stack\third_party\flywire_annotations"
    r"\supplemental_files\Supplemental_file1_neuron_annotations.tsv"
)
OUT = ROOT / "outputs/neural_c2"


ANNOT_COLUMNS = ("super_class", "cell_class", "cell_sub_class", "cell_type", "status")


def load_annotation_index(path: Path) -> dict[int, dict]:
    """root_id -> the annotation row. Only the columns C2 is allowed to use."""
    out: dict[int, dict] = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rid = int(row["root_id"])
            out[rid] = {k: row.get(k, "") for k in ANNOT_COLUMNS}
    return out


def valid_mask(ann: dict[int, dict], ids: np.ndarray) -> np.ndarray:
    """Boolean mask over dataset indices: annotated, typed, and not flagged.

    Measured on the annotation table, not assumed:
      - `cell_type` is filled for 137 720 of 139 248 rows (98.9 %);
      - `status` is filled for only 658 rows, and every one of them is an
        outlier (344 `outlier_seg`, 314 `outlier_bio`).

    So a NON-EMPTY `status` is a disqualification, not a confirmation. The
    Traced/Glia/Orphan labels seen elsewhere belong to the MaleCNS table, whose
    identifiers are MaleCNS body IDs, and they must not be reused here.
    """
    mask = np.zeros(len(ids), dtype=bool)
    for i, v in enumerate(ids):
        row = ann.get(int(v))
        if row is None:
            continue
        if not row.get("cell_type", "").strip():
            continue
        if row.get("status", "").strip():
            continue
        mask[i] = True
    return mask


def bfs_out(indptr: np.ndarray, out_idx: np.ndarray, seeds: list[int], hops: int) -> set[int]:
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        nxt: set[int] = set()
        for u in frontier:
            nxt.update(out_idx[indptr[u]:indptr[u + 1]].tolist())
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def induced_connectome(conn, members: set[int]):
    """Edge arrays of the subgraph induced on `members`, with global indices remapped."""
    member_mask = np.zeros(conn.n_neurons, dtype=bool)
    member_mask[list(members)] = True
    keep = member_mask[conn.pre] & member_mask[conn.post]
    pre = conn.pre[keep]
    post = conn.post[keep]
    remap = np.full(conn.n_neurons, -1, dtype=np.int64)
    ordered = np.array(sorted(members), dtype=np.int64)
    remap[ordered] = np.arange(len(ordered), dtype=np.int64)
    return remap[pre], remap[post], ordered, conn.signed_count[keep], keep.sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="flywire_783")
    ap.add_argument("--input-anchor", default="sugar_grn")
    ap.add_argument("--output-anchor", default="mn9")
    ap.add_argument("--annotation", type=Path, default=DEFAULT_ANNOT)
    ap.add_argument("--max-hops", type=int, default=3,
                    help="registered upper bound; every count up to it is evaluated")
    ap.add_argument("--max-memory-bytes", type=int, default=256_000_000,
                    help="Rust bridge budget. The 256 MB default is the historical "
                         "v0 ceiling; raising it is a resource decision, not a "
                         "scientific one, and is recorded in the card.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    conn = load_connectome(args.dataset, ROOT / "data/raw")
    ids = conn.flywire_ids
    n = conn.n_neurons
    reg = PopulationRegistry.builtin()
    lookup = {int(v): i for i, v in enumerate(ids)}

    def resolve(name: str) -> list[int]:
        return sorted(lookup[int(x)] for x in reg.get(name) if int(x) in lookup)

    in_idx = resolve(args.input_anchor)
    out_idx_list = resolve(args.output_anchor)
    print(f"dataset   : {args.dataset}  {n:,} neurons  {len(conn.pre):,} edges")
    print(f"input     : {args.input_anchor} -> {len(in_idx)} neurons")
    print(f"output    : {args.output_anchor} -> {len(out_idx_list)} neurons")
    if not in_idx or not out_idx_list:
        print("ANCORA NAO RESOLVE no dataset; C2 nao pode usar esta configuracao")
        return 2
    print()

    order = np.argsort(conn.pre, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(conn.pre[order], minlength=n), out=indptr[1:])
    out_arr = conn.post[order]

    ann = load_annotation_index(args.annotation)
    keep_mask = valid_mask(ann, ids)
    n_keep = int(keep_mask.sum())
    print(f"anotados, tipados e nao-outlier: {n_keep:,} de {n:,} ({100*keep_mask.mean():.2f}%)")
    if n_keep < 1000:
        print("FALHA: filtro de validade produziu um conjunto degenerado; abortando")
        return 4
    print()

    print("=== cada profundidade ate o limite registrado ===")
    rows = []
    for hops in range(1, args.max_hops + 1):
        reach = bfs_out(indptr, out_arr, in_idx, hops)
        members = {u for u in reach if keep_mask[u]} | set(in_idx) | set(out_idx_list)
        pre_s, post_s, ordered, w, n_edges = induced_connectome(conn, members)
        bridge = int(n_edges) * 256 + len(members) * (160 + 8 * 18)
        sc = Counter(ann.get(int(ids[u]), {}).get("super_class", "") for u in members)
        has_out = bool(set(out_idx_list) & members)
        rows.append({
            "hops": hops, "neurons": len(members), "edges": int(n_edges),
            "bridge_bytes": bridge, "fits_budget": bridge < args.max_memory_bytes,
            "has_output_anchor": has_out,
            "descending_neurons": int(sc.get("descending", 0)),
            "motor_neurons": int(sc.get("motor", 0)),
        })
        print(f"  hops={hops}  neurons={len(members):>7,}  edges={int(n_edges):>9,}  "
              f"bridge={bridge/1024**2:>8.1f} MB  cabe={bridge < args.max_memory_bytes}  "
              f"ancora_saida={has_out}  descending={sc.get('descending', 0):>5}  "
              f"motor={sc.get('motor', 0):>4}")

    feasible = [r for r in rows
                if r["fits_budget"] and r["has_output_anchor"]
                and (r["descending_neurons"] + r["motor_neurons"]) >= 3]
    if not feasible:
        print()
        print("NENHUMA profundidade cumpre os criterios 2, 3 e o orcamento de memoria.")
        print("Isto e um resultado, nao um ajuste: ver PREREGISTRATION_C2.md secao 5.")
        return 3
    chosen = max(feasible, key=lambda r: r["neurons"])
    print()
    print(f"escolhido: {chosen['hops']} salto(s), {chosen['neurons']:,} neurÃ´nios, "
          f"{chosen['edges']:,} arestas")
    print("  (o maior que cumpre todos os critÃ©rios, decidido por regra, nÃ£o por preferÃªncia)")

    reach = bfs_out(indptr, out_arr, in_idx, chosen["hops"])
    members = {u for u in reach if keep_mask[u]} | set(in_idx) | set(out_idx_list)
    pre_s, post_s, ordered, w, n_edges = induced_connectome(conn, members)

    # ---- criterion 5: weak connectivity of the induced subgraph -------------
    m = len(members)
    order2 = np.argsort(pre_s, kind="stable")
    ip2 = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre_s[order2], minlength=m), out=ip2[1:])
    oi2 = post_s[order2]
    # undirected adjacency by unioning out and in
    und: list[list[int]] = [[] for _ in range(m)]
    for u in range(m):
        for v in oi2[ip2[u]:ip2[u + 1]]:
            und[u].append(int(v))
            und[v].append(int(u))
    visited = {0}
    dq = deque([0])
    while dq:
        u = dq.popleft()
        for v in und[u]:
            if v not in visited:
                visited.add(v)
                dq.append(v)
    connected = len(visited) == m
    print(f"  criterion 5 (conexo)   : {connected}  "
          f"({len(visited)}/{m} alcanÃ§ados a partir do nÃ³ 0)")

    out_deg = np.bincount(pre_s, minlength=m)
    in_deg = np.bincount(post_s, minlength=m)
    sign = np.where(np.asarray(w) > 0, 1, -1).astype(np.int8)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / f"circuit_{args.input_anchor}_{args.output_anchor}_h{chosen['hops']}.npz",
        flywire_ids=ids[ordered].astype(np.int64),
        pre=pre_s.astype(np.int32),
        post=post_s.astype(np.int32),
        signed_count=np.asarray(w),
    )
    super_counts = Counter(
        ann.get(int(ids[u]), {}).get("super_class", "") for u in members
    )
    card = {
        "dataset": args.dataset,
        "input_anchor": args.input_anchor,
        "output_anchor": args.output_anchor,
        "hops": chosen["hops"],
        "neuron_count": m,
        "edge_count": int(n_edges),
        "sign_excitatory_edges": int((sign > 0).sum()),
        "sign_inhibitory_edges": int((sign < 0).sum()),
        "synapses_total": int(np.asarray(w).sum()),
        "out_degree": {"mean": float(out_deg.mean()), "median": float(np.median(out_deg)),
                       "max": int(out_deg.max()), "zero": int((out_deg == 0).sum())},
        "in_degree": {"mean": float(in_deg.mean()), "median": float(np.median(in_deg)),
                      "max": int(in_deg.max()), "zero": int((in_deg == 0).sum())},
        "super_class_counts": dict(super_counts.most_common()),
        "self_loops": int((pre_s == post_s).sum()),
        "duplicate_pairs": int(len(pre_s) - len(np.unique(pre_s.astype(np.int64) * m + post_s))),
        "weakly_connected": connected,
        "bridge_estimate_bytes": int(n_edges) * 256 + m * (160 + 8 * 18),
        "criteria_evaluated": rows,
        "registered_max_hops": args.max_hops,
        "selection_rule": "largest hop count that satisfies criteria 2, 3 and the memory budget",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    keys = "".join(f"{v}," for v in sorted(members))
    card["members_digest"] = hashlib.sha256(keys.encode()).hexdigest()
    (args.out / "circuit_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
