from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a tab-separated file into a pandas DataFrame with string types."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def validate_source_frame(df: pd.DataFrame, name: str = "source") -> None:
    """Verify that a source DataFrame contains the required entity resolution columns."""
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_ground_truth_frame(df: pd.DataFrame) -> None:
    """Verify that the ground truth DataFrame contains the required evaluation columns."""
    missing = [c for c in REQUIRED_GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Ground truth is missing required columns: {missing}")


def load_sources(directory: str | Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate source1, source2, and source3 TSV files from the given directory."""
    directory = Path(directory)
    s1 = read_tsv(directory / f"{prefix}_source1.tsv")
    s2 = read_tsv(directory / f"{prefix}_source2.tsv")
    s3 = read_tsv(directory / f"{prefix}_source3.tsv")
    validate_source_frame(s1, f"{prefix}_source1")
    validate_source_frame(s2, f"{prefix}_source2")
    validate_source_frame(s3, f"{prefix}_source3")
    return s1, s2, s3


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """Load and validate the training ground truth TSV file."""
    df = read_tsv(path)
    validate_ground_truth_frame(df)
    return df


def write_submission(df: pd.DataFrame, path: str | Path, columns: Iterable[str]) -> None:
    """Write DataFrame columns to a UTF-8 tab-separated submission file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(columns)].to_csv(path, sep="\t", index=False, encoding="utf-8")
