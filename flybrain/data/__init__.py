"""Dataset layer: staging, loading, indexing, subsetting, validation, provenance."""

from __future__ import annotations

from .loader import (
    DATASET_SOURCES,
    KNOWN_DATASETS,
    Connectome,
    SubsetInfo,
    build_connectome,
    load_connectome,
    raw_paths,
    read_connectivity_table,
    read_neuron_table,
    stage_dataset,
)
from .provenance import (
    DatasetCard,
    FileRecord,
    combined_digest,
    environment_snapshot,
    git_commit,
    load_dataset_card,
    list_dataset_cards,
    sha256_file,
    update_dataset_card,
    utc_now_iso,
    write_dataset_card,
)
from .validate import DatasetReport, check_id_index_consistency, validate_connectome

__all__ = [
    "DATASET_SOURCES",
    "KNOWN_DATASETS",
    "Connectome",
    "SubsetInfo",
    "build_connectome",
    "load_connectome",
    "raw_paths",
    "read_connectivity_table",
    "read_neuron_table",
    "stage_dataset",
    "DatasetCard",
    "FileRecord",
    "combined_digest",
    "environment_snapshot",
    "git_commit",
    "load_dataset_card",
    "list_dataset_cards",
    "sha256_file",
    "update_dataset_card",
    "utc_now_iso",
    "write_dataset_card",
    "DatasetReport",
    "validate_connectome",
    "check_id_index_consistency",
]
