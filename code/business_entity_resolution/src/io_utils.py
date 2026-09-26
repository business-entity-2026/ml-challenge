"""Backward-compatibility module re-exporting from load_data."""
from .load_data import (
    REQUIRED_GT_COLUMNS,
    REQUIRED_SOURCE_COLUMNS,
    load_ground_truth,
    load_sources,
    read_tsv,
    validate_ground_truth_frame,
    validate_source_frame,
    write_submission,
)

__all__ = [
    "REQUIRED_SOURCE_COLUMNS",
    "REQUIRED_GT_COLUMNS",
    "read_tsv",
    "validate_source_frame",
    "validate_ground_truth_frame",
    "load_sources",
    "load_ground_truth",
    "write_submission",
]
