"""Loading, staging, indexing, subsetting and caching of the FlyWire-derived dataset.

Design constraints taken from ``docs/UPSTREAM_AUDIT.md``:

* The connectivity table already carries a precomputed excitatory/inhibitory sign
  per connection, so no neurotransmitter inference and no CAVE query is needed.
* ``Presynaptic_Index`` / ``Postsynaptic_Index`` are integer indices into the row
  order of the neuron table; the FlyWire root IDs live in the tables themselves.
  FlyBrain keeps **both** and treats the FlyWire ID as the durable identity: every
  subset, recording and report is keyed by FlyWire ID, and index space is an
  internal detail of whichever dataset instance is loaded.
* The whole graph is 14.7 M edges. It is never densified.

No network access is performed anywhere in this module. Once the two upstream
files are present, FlyBrain runs offline.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..errors import DatasetError, UnknownNeuronError
from . import schema
from .provenance import (
    DatasetCard,
    FileRecord,
    combined_digest,
    load_dataset_card,
    sha256_file,
    utc_now_iso,
    write_dataset_card,
)

# --------------------------------------------------------------------------------------
# Dataset registry: which physical files back which logical dataset id
# --------------------------------------------------------------------------------------

DATASET_SOURCES: Dict[str, Dict[str, Any]] = {
    "flywire_630": {
        "neuron_table": "2023_03_23_completeness_630_final.csv",
        "connectivity_table": "2023_03_23_connectivity_630_final.parquet",
        "release": "FlyWire 2023_03_23 materialization, completeness threshold 630",
        "source": "https://github.com/philshiu/Drosophila_brain_model",
        "notes": (
            "Version used by Shiu et al., Nature 634, 210-219 (2024). A byte-identical "
            "copy of the connectivity table is vendored by "
            "https://github.com/chaobrain/drosophila_whole_brain_snn_simulation."
        ),
    },
    "flywire_783": {
        "neuron_table": "Completeness_783.csv",
        "connectivity_table": "Connectivity_783.parquet",
        "release": "FlyWire public version 783",
        "source": "https://github.com/philshiu/Drosophila_brain_model",
        "notes": (
            "Newer public release, shipped alongside 630 by the upstream repository. "
            "Registered and validated by FlyBrain, but NOT the primary v0 dataset: no "
            "published reference output exists for it."
        ),
    },
}

KNOWN_DATASETS = tuple(DATASET_SOURCES)


# --------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------


def stage_dataset(
    dataset_id: str,
    from_dir: str | Path,
    raw_dir: str | Path,
    metadata_dir: str | Path,
    force: bool = False,
) -> DatasetCard:
    """Copy a dataset's two files into ``raw_dir``/``dataset_id`` and hash them.

    This is the ``import/preprocess`` boundary of the objective's data pipeline. It
    is explicit and idempotent: re-running without ``force`` verifies the existing
    copy instead of overwriting it.
    """
    if dataset_id not in DATASET_SOURCES:
        raise DatasetError(
            f"unknown dataset_id {dataset_id!r}; known: {list(DATASET_SOURCES)}"
        )
    spec = DATASET_SOURCES[dataset_id]
    src_dir = Path(from_dir)
    dest = Path(raw_dir) / dataset_id
    dest.mkdir(parents=True, exist_ok=True)

    records: List[FileRecord] = []
    for role, key in (("neuron_table", "neuron_table"), ("connectivity_table", "connectivity_table")):
        src = src_dir / spec[key]
        if not src.is_file():
            raise DatasetError(
                f"dataset source file not found: {src}\n"
                f"Expected the upstream repository to be cloned at {src_dir}."
            )
        dst = dest / spec[key]
        if force or not dst.is_file():
            shutil.copy2(src, dst)
        records.append(
            FileRecord(name=spec[key], role=role, sha256=sha256_file(dst), size_bytes=dst.stat().st_size)
        )

    card = DatasetCard(
        dataset_id=dataset_id,
        source=spec["source"],
        release=spec["release"],
        retrieved_at=utc_now_iso(),
        neuron_count=0,   # filled in by validate(); staging only vouches for bytes
        edge_count=0,
        sha256=combined_digest(records),
        files=records,
        notes=spec["notes"],
        upstream={"repository": spec["source"], "local_clone": str(src_dir)},
    )
    write_dataset_card(card, metadata_dir)
    return card


def raw_paths(dataset_id: str, raw_dir: str | Path) -> Tuple[Path, Path]:
    spec = DATASET_SOURCES.get(dataset_id)
    if spec is None:
        raise DatasetError(f"unknown dataset_id {dataset_id!r}; known: {list(DATASET_SOURCES)}")
    d = Path(raw_dir) / dataset_id
    return d / spec["neuron_table"], d / spec["connectivity_table"]


# --------------------------------------------------------------------------------------
# Subset description
# --------------------------------------------------------------------------------------


@dataclass
class SubsetInfo:
    """How a subset was derived from its parent connectome.

    Recorded so that a subset run is never mistaken for a whole-brain run, and so
    the exact selection can be reconstructed.
    """

    parent_dataset_id: str
    requested_seed_ids: List[int]
    present_seed_ids: List[int]
    missing_seed_ids: List[int]
    hops: int
    max_neurons: int
    truncated: bool
    n_neurons: int
    n_edges: int
    hop_histogram: Dict[str, int] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_dataset_id": self.parent_dataset_id,
            "requested_seed_ids": self.requested_seed_ids,
            "present_seed_ids": self.present_seed_ids,
            "missing_seed_ids": self.missing_seed_ids,
            "hops": self.hops,
            "max_neurons": self.max_neurons,
            "truncated": self.truncated,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "hop_histogram": self.hop_histogram,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# Connectome
# --------------------------------------------------------------------------------------


@dataclass(eq=False)
class Connectome:
    """A directed, signed, weighted whole-brain graph keyed by FlyWire ID.

    Invariants guaranteed by the constructors in this module:

    * ``pre`` is sorted in non-decreasing order, so CSR row pointers can be rebuilt
      with a single ``bincount``/``cumsum`` without a re-sort.
    * ``len(pre) == len(post) == len(signed_count)``.
    * all index values are in ``[0, n_neurons)``.
    * ``flywire_ids`` has length ``n_neurons`` and is strictly increasing.
    """

    dataset_id: str
    flywire_ids: np.ndarray          # (n,) int64
    pre: np.ndarray                  # (E,) int32, non-decreasing
    post: np.ndarray                 # (E,) int32
    signed_count: np.ndarray         # (E,) int16
    card: Optional[DatasetCard] = None
    subset_info: Optional[SubsetInfo] = None

    # cached derived state (not part of the logical value)
    _id2i: Optional[Dict[int, int]] = field(default=None, repr=False, compare=False)
    _sign: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _out_degree: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _in_degree: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _csr: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False, compare=False)

    # -- basics -------------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return int(self.flywire_ids.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.pre.shape[0])

    @property
    def is_subset(self) -> bool:
        return self.subset_info is not None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "subset" if self.is_subset else "full"
        return (
            f"<Connectome {self.dataset_id} ({kind}) neurons={self.n_neurons} "
            f"edges={self.n_edges}>"
        )

    # -- id <-> index -------------------------------------------------------------
    @property
    def id_to_index(self) -> Dict[int, int]:
        if self._id2i is None:
            self._id2i = {int(f): i for i, f in enumerate(self.flywire_ids)}
        return self._id2i

    @property
    def index_to_id(self) -> Dict[int, int]:
        return {i: int(f) for i, f in enumerate(self.flywire_ids)}

    def index_of(self, flywire_id: int) -> int:
        try:
            return self.id_to_index[int(flywire_id)]
        except KeyError:
            raise UnknownNeuronError(
                f"FlyWire ID {flywire_id} is not in dataset {self.dataset_id!r} "
                f"({self.n_neurons} neurons)"
            ) from None

    def indices_of(self, flywire_ids: Sequence[int]) -> np.ndarray:
        """Map IDs to indices, failing loudly and completely on unknown IDs.

        Failing on the *first* missing ID with the whole list of missing IDs is
        deliberate: an experiment that silently drops half its stimulus neurons is
        worse than one that refuses to run.
        """
        ids = [int(x) for x in flywire_ids]
        lookup = self.id_to_index
        missing = [x for x in ids if x not in lookup]
        if missing:
            preview = missing[:10]
            raise UnknownNeuronError(
                f"{len(missing)} of {len(ids)} requested FlyWire IDs are not in dataset "
                f"{self.dataset_id!r}; first missing: {preview}"
                + (" ..." if len(missing) > len(preview) else "")
            )
        return np.fromiter((lookup[x] for x in ids), dtype=schema.DTYPE_NEURON_INDEX, count=len(ids))

    def filter_present(self, flywire_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Split IDs into (present, missing) without raising. Used when reporting."""
        lookup = self.id_to_index
        ids = [int(x) for x in flywire_ids]
        present = np.fromiter((x for x in ids if x in lookup), dtype=np.int64)
        missing = np.fromiter((x for x in ids if x not in lookup), dtype=np.int64)
        return present, missing

    # -- degree / sign ------------------------------------------------------------
    @property
    def out_degree(self) -> np.ndarray:
        if self._out_degree is None:
            self._out_degree = np.bincount(
                self.pre, minlength=self.n_neurons
            ).astype(np.int32)
        return self._out_degree

    @property
    def in_degree(self) -> np.ndarray:
        if self._in_degree is None:
            self._in_degree = np.bincount(
                self.post, minlength=self.n_neurons
            ).astype(np.int32)
        return self._in_degree

    @property
    def sign(self) -> np.ndarray:
        """Per-neuron outgoing sign: +1 excitatory, -1 inhibitory, 0 unknown/ambiguous.

        The dataset encodes sign per *connection*, but the audit found that no neuron
        ever has both excitatory and inhibitory outgoing edges, so a per-neuron sign
        is well defined. Neurons with no outgoing edge at all have no sign evidence
        and get 0 ("unknown"), which the validator reports as ``unknown_sign``.
        """
        if self._sign is None:
            n = self.n_neurons
            pos = np.zeros(n, dtype=bool)
            neg = np.zeros(n, dtype=bool)
            exc = self.signed_count > 0
            np.logical_or.at(pos, self.pre[exc], True)
            np.logical_or.at(neg, self.pre[~exc], True)
            sign = np.zeros(n, dtype=np.int8)
            sign[pos & ~neg] = 1
            sign[neg & ~pos] = -1
            # pos & neg stays 0 -> ambiguous, reported by the validator
            self._sign = sign
        return self._sign

    @property
    def synapse_count(self) -> int:
        return int(np.abs(self.signed_count.astype(np.int64)).sum())

    # -- sparse representation ----------------------------------------------------
    def csr(self) -> Tuple[np.ndarray, np.ndarray]:
        """CSR row pointers and column indices over the raw signed counts.

        Returns ``(indptr, indices)`` where the value for edge ``k`` is
        ``signed_count[k]``. Callers that need mV weights multiply per-run, so a
        change of ``w_syn`` never invalidates this structure.
        """
        if self._csr is None:
            counts = np.bincount(self.pre, minlength=self.n_neurons)
            indptr = np.zeros(self.n_neurons + 1, dtype=np.int64)
            np.cumsum(counts, out=indptr[1:])
            self._csr = (indptr, self.post)
        return self._csr

    def weights_mv(self, weight_per_synapse_mv: float) -> np.ndarray:
        """Signed synaptic weight in mV for every edge, as float32."""
        return (
            self.signed_count.astype(schema.DTYPE_WEIGHT)
            * schema.DTYPE_WEIGHT(weight_per_synapse_mv)
        )

    # -- subsetting ---------------------------------------------------------------
    def subset(
        self,
        seed_ids: Sequence[int],
        hops: int = 2,
        max_neurons: int = 5000,
    ) -> "Connectome":
        """Extract a bounded, connected k-hop neighbourhood around ``seed_ids``.

        Both directions are traversed, because a neuron whose *inputs* are missing
        from the subset can never be driven. Seeds not present in the parent are
        recorded in ``SubsetInfo.missing_seed_ids`` rather than raising: a subset is
        an exploratory object, and the caller decides what a missing seed means
        (the smoke test deliberately runs on synthetic data, where seeds by
        definition do not exist).

        The expansion is deterministic: neighbours are deduplicated with
        ``np.unique`` and the retained neuron set is stored in ascending parent
        index order, which also keeps ``pre`` sorted in the child (see the class
        invariant).
        """
        n = self.n_neurons
        seed_present, seed_missing = self.filter_present(seed_ids)
        seeds = self.indices_of(seed_present.tolist()) if seed_present.size else np.empty(0, np.int32)

        selected = np.zeros(n, dtype=bool)
        hop_of: Dict[int, int] = {}
        for i in seeds:
            selected[i] = True
            hop_of[int(i)] = 0

        truncated = False
        frontier = np.asarray(seeds, dtype=np.int32)
        for h in range(1, hops + 1):
            if frontier.size == 0:
                break
            mask = np.zeros(n, dtype=bool)
            mask[frontier] = True
            # out-neighbours and in-neighbours of the frontier, in one pass each
            out_nbr = self.post[mask[self.pre]]
            in_nbr = self.pre[mask[self.post]]
            cand = np.unique(np.concatenate((out_nbr, in_nbr)))
            cand = cand[~selected[cand]]
            if cand.size == 0:
                continue
            room = max_neurons - int(selected.sum())
            if cand.size > room:
                cand = cand[: max(room, 0)]
                truncated = True
            selected[cand] = True
            for i in cand:
                hop_of[int(i)] = h
            frontier = cand.astype(np.int32)
            if truncated:
                break

        keep = np.flatnonzero(selected)
        remap = np.full(n, -1, dtype=np.int32)
        remap[keep] = np.arange(keep.size, dtype=np.int32)

        edge_keep = selected[self.pre] & selected[self.post]
        # `pre` is sorted and `remap` is monotonic in parent index, so the filtered
        # `pre` stays sorted -> the child satisfies its own class invariant.
        new_pre = remap[self.pre[edge_keep]]
        new_post = remap[self.post[edge_keep]]
        new_signed = self.signed_count[edge_keep]

        hist: Dict[str, int] = {}
        for i in keep:
            d = hop_of.get(int(i), -1)
            hist[str(d)] = hist.get(str(d), 0) + 1

        info = SubsetInfo(
            parent_dataset_id=self.dataset_id,
            requested_seed_ids=[int(x) for x in seed_ids],
            present_seed_ids=[int(x) for x in seed_present.tolist()],
            missing_seed_ids=[int(x) for x in seed_missing.tolist()],
            hops=hops,
            max_neurons=max_neurons,
            truncated=truncated,
            n_neurons=int(keep.size),
            n_edges=int(edge_keep.sum()),
            hop_histogram=hist,
            notes=(
                "k-hop in+out neighbourhood around the seed neurons, capped at "
                "max_neurons. NOT a whole brain."
            ),
        )
        return Connectome(
            dataset_id=self.dataset_id,
            flywire_ids=self.flywire_ids[keep].copy(),
            pre=new_pre.astype(schema.DTYPE_NEURON_INDEX),
            post=new_post.astype(schema.DTYPE_NEURON_INDEX),
            signed_count=new_signed.astype(schema.DTYPE_SIGNED_COUNT),
            card=self.card,
            subset_info=info,
        )

    # -- processed cache ----------------------------------------------------------
    def save_cache(self, processed_dir: str | Path) -> Path:
        """Persist the index-space arrays so the next run skips parquet decoding."""
        d = Path(processed_dir) / self.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        npz = d / "connectome.npz"
        np.savez(
            npz,
            flywire_ids=self.flywire_ids,
            pre=self.pre,
            post=self.post,
            signed_count=self.signed_count,
        )
        meta = {
            "dataset_id": self.dataset_id,
            "n_neurons": self.n_neurons,
            "n_edges": self.n_edges,
            "formatted_at": utc_now_iso(),
            "npz_sha256": sha256_file(npz),
            "source_digest": (self.card.sha256 if self.card is not None else None),
            "cache_format": 1,
        }
        (d / "cache.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return npz

    @staticmethod
    def load_cache(processed_dir: str | Path, dataset_id: str) -> Optional[Tuple["Connectome", Dict[str, Any]]]:
        d = Path(processed_dir) / dataset_id
        npz, meta_p = d / "connectome.npz", d / "cache.json"
        if not (npz.is_file() and meta_p.is_file()):
            return None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        with np.load(npz, allow_pickle=False) as z:
            conn = Connectome(
                dataset_id=dataset_id,
                flywire_ids=z["flywire_ids"].astype(np.int64),
                pre=z["pre"].astype(schema.DTYPE_NEURON_INDEX),
                post=z["post"].astype(schema.DTYPE_NEURON_INDEX),
                signed_count=z["signed_count"].astype(schema.DTYPE_SIGNED_COUNT),
            )
        return conn, meta


# --------------------------------------------------------------------------------------
# Construction from raw files
# --------------------------------------------------------------------------------------


def read_neuron_table(path: str | Path) -> np.ndarray:
    """Read the completeness CSV and return its index as an int64 FlyWire ID array."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"neuron table not found: {path}")
    df = pd.read_csv(path, index_col=0)
    ids = np.asarray(df.index, dtype=np.int64)
    if ids.size == 0:
        raise DatasetError(f"neuron table {path} is empty")
    if not np.all(np.diff(ids) > 0):
        raise DatasetError(
            f"neuron table {path} index is not strictly increasing; "
            "FlyBrain relies on the table's row order as the index space"
        )
    return ids


def read_connectivity_table(
    path: str | Path, columns: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    """Read the connectivity parquet into compact NumPy arrays.

    Only the columns that are needed are read, and each is narrowed to the smallest
    dtype that provably holds it (verified, not assumed).
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"connectivity table not found: {path}")
    cols = list(columns or schema.CONNECTIVITY_COLUMNS)
    df = pd.read_parquet(path, columns=cols)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DatasetError(
            f"connectivity table {path} is missing columns {missing}; "
            f"present columns: {list(df.columns)}"
        )

    signed = df[schema.COL_SIGNED_COUNT].to_numpy()
    info = np.iinfo(schema.DTYPE_SIGNED_COUNT)
    if signed.min() < info.min or signed.max() > info.max:
        raise DatasetError(
            f"signed synapse counts exceed int16 range [{signed.min()}, {signed.max()}]; "
            f"the in-memory dtype contract in flybrain/data/schema.py must be widened"
        )
    out = {
        "pre": df[schema.COL_PRE_INDEX].to_numpy(schema.DTYPE_NEURON_INDEX),
        "post": df[schema.COL_POST_INDEX].to_numpy(schema.DTYPE_NEURON_INDEX),
        "signed_count": signed.astype(schema.DTYPE_SIGNED_COUNT),
    }
    if schema.COL_PRE_ID in cols:
        out["pre_id"] = df[schema.COL_PRE_ID].to_numpy(np.int64)
        out["post_id"] = df[schema.COL_POST_ID].to_numpy(np.int64)
    if schema.COL_EXCITATORY in cols:
        out["excitatory"] = df[schema.COL_EXCITATORY].to_numpy(np.int8)
    if schema.COL_SYNAPSE_COUNT in cols:
        out["connectivity"] = df[schema.COL_SYNAPSE_COUNT].to_numpy(np.int32)
    return out


def build_connectome(
    dataset_id: str,
    neuron_table: str | Path,
    connectivity_table: str | Path,
    card: Optional[DatasetCard] = None,
) -> Connectome:
    """Build an in-memory :class:`Connectome` from the two raw files.

    The edge list is sorted by presynaptic index once, here, so that every consumer
    inherits the sorted-``pre`` invariant.
    """
    ids = read_neuron_table(neuron_table)
    n = ids.size
    raw = read_connectivity_table(connectivity_table)
    pre, post, signed = raw["pre"], raw["post"], raw["signed_count"]

    if pre.size != post.size or pre.size != signed.size:
        raise DatasetError("connectivity columns have inconsistent lengths")
    for name, arr in (("Presynaptic_Index", pre), ("Postsynaptic_Index", post)):
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise DatasetError(
                f"{name} contains values outside [0, {n}); the connectivity table and "
                f"the neuron table are not from the same dataset version"
            )

    order = np.argsort(pre, kind="stable")
    return Connectome(
        dataset_id=dataset_id,
        flywire_ids=ids,
        pre=pre[order],
        post=post[order],
        signed_count=signed[order],
        card=card,
    )


def load_connectome(
    dataset_id: str,
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> Connectome:
    """Load a dataset, preferring the processed cache, and attach its provenance card.

    Cache validity is decided by comparing the cache's recorded ``source_digest``
    against the dataset card's ``sha256``. If they disagree the raw files are
    re-read. Raw-file integrity itself is checked by size against the card; pass
    ``flybrain datasets --verify`` to force a full re-hash.
    """
    neuron_p, conn_p = raw_paths(dataset_id, raw_dir)
    card = load_dataset_card(dataset_id, metadata_dir) if metadata_dir is not None else None

    if card is not None:
        for rec in card.files:
            p = (Path(raw_dir) / dataset_id) / rec.name
            if not p.is_file():
                raise DatasetError(f"dataset {dataset_id!r} is registered but {p} is missing")
            if p.stat().st_size != rec.size_bytes:
                raise DatasetError(
                    f"{p} has size {p.stat().st_size}, expected {rec.size_bytes} from the "
                    f"dataset card; re-stage the dataset or run `flybrain datasets --verify`"
                )

    if use_cache and processed_dir is not None:
        cached = Connectome.load_cache(processed_dir, dataset_id)
        if cached is not None:
            conn, meta = cached
            if card is None or meta.get("source_digest") == card.sha256:
                if conn.n_neurons and conn.n_edges:
                    conn.card = card
                    return conn

    conn = build_connectome(dataset_id, neuron_p, conn_p, card=card)
    if write_cache and processed_dir is not None:
        conn.save_cache(processed_dir)
    return conn
