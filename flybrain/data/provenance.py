"""Dataset provenance: hashing, dataset cards, and git revision capture.

The objective requires that every dataset and every experiment record where its
numbers came from. This module is the single place that decides what a
"provenance record" contains, so there is exactly one format to keep honest.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_CHUNK = 1 << 22  # 4 MiB


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file, streamed. Used for every dataset and cache artefact."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(_CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def utc_now_iso() -> str:
    """Timestamp in ISO-8601 UTC with a 'Z' suffix, second resolution."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit(cwd: str | Path | None = None) -> Dict[str, Any]:
    """Return the current git revision, or an explicit 'not a repository' record.

    Never raises: a missing git repository must not stop a simulation. It is
    recorded as ``commit=None`` together with the reason, so that a run
    performed outside version control is visibly unreproducible rather than
    silently pretending otherwise.
    """
    cwd = Path(cwd) if cwd is not None else Path(__file__).resolve().parents[2]
    info: Dict[str, Any] = {"commit": None, "dirty": None, "branch": None, "error": None}

    def _git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # git absent / hung
            info["error"] = f"{type(exc).__name__}: {exc}"
            return None
        if out.returncode != 0:
            info["error"] = (out.stderr or out.stdout or "").strip() or f"exit {out.returncode}"
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return info
    info["commit"] = commit
    info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    info["dirty"] = None if status is None else bool(status)
    info["error"] = None
    return info


# --------------------------------------------------------------------------------------
# Dataset cards
# --------------------------------------------------------------------------------------


@dataclass
class FileRecord:
    """One physical file that a dataset card vouches for."""

    name: str
    role: str            # "neuron_table" | "connectivity_table"
    sha256: str
    size_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetCard:
    """Metadata describing one versioned local dataset.

    Field set follows the objective's required schema, extended with what was
    actually needed to make the numbers auditable (per-file hashes, and the
    upstream repository revision the files were taken from).
    """

    dataset_id: str
    source: str
    release: str
    retrieved_at: str
    neuron_count: int
    edge_count: int
    sha256: str                      # digest over the ordered file digests
    files: List[FileRecord] = field(default_factory=list)
    notes: str = ""
    upstream: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["files"] = [f if isinstance(f, dict) else f.to_dict() for f in self.files]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetCard":
        d = dict(d)
        d["files"] = [FileRecord(**f) for f in d.get("files", [])]
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def combined_digest(records: Iterable[FileRecord]) -> str:
    """Deterministic digest over an ordered set of file digests.

    Ordering is by ``(role, name)`` so the digest does not depend on directory
    iteration order.
    """
    ordered = sorted(records, key=lambda r: (r.role, r.name))
    h = hashlib.sha256()
    for r in ordered:
        h.update(f"{r.role}\0{r.name}\0{r.sha256}\0{r.size_bytes}\n".encode("utf-8"))
    return h.hexdigest()


def card_path(dataset_id: str, registry_dir: str | Path) -> Path:
    return Path(registry_dir) / f"{dataset_id}.json"


def write_dataset_card(card: DatasetCard, registry_dir: str | Path) -> Path:
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    p = card_path(card.dataset_id, registry_dir)
    p.write_text(json.dumps(card.to_dict(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return p


def load_dataset_card(dataset_id: str, registry_dir: str | Path) -> Optional[DatasetCard]:
    p = card_path(dataset_id, registry_dir)
    if not p.is_file():
        return None
    return DatasetCard.from_dict(json.loads(p.read_text(encoding="utf-8")))


def list_dataset_cards(registry_dir: str | Path) -> List[DatasetCard]:
    registry_dir = Path(registry_dir)
    if not registry_dir.is_dir():
        return []
    cards = []
    for p in sorted(registry_dir.glob("*.json")):
        try:
            cards.append(DatasetCard.from_dict(json.loads(p.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, TypeError):
            continue
    return cards


def update_dataset_card(
    dataset_id: str,
    registry_dir: str | Path,
    *,
    neuron_count: Optional[int] = None,
    edge_count: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[DatasetCard]:
    """Fill in content counts on an existing card and rewrite it.

    Staging vouches only for *bytes*; the neuron and edge counts can only come from
    reading the files. This is the step that writes those two numbers, so a card
    never claims a count nobody measured.
    """
    card = load_dataset_card(dataset_id, registry_dir)
    if card is None:
        return None
    if neuron_count is not None:
        card.neuron_count = int(neuron_count)
    if edge_count is not None:
        card.edge_count = int(edge_count)
    if extra:
        card.extra.update(extra)
    card.extra["counts_measured_at"] = utc_now_iso()
    write_dataset_card(card, registry_dir)
    return card


# --------------------------------------------------------------------------------------
# Environment snapshot
# --------------------------------------------------------------------------------------


def environment_snapshot() -> Dict[str, Any]:
    """Python / OS / dependency snapshot recorded with every run.

    Hardware and backend availability are added by
    :mod:`flybrain.runtime.backend`, which is the only module allowed to make
    claims about CUDA and GPUs.
    """
    deps = {}
    for name in ("numpy", "pandas", "pyarrow", "jax", "matplotlib", "pytest"):
        try:
            mod = __import__(name)
            deps[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[name] = None

    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": deps,
    }
