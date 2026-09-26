from pathlib import Path
from typing import Optional

import pandas as pd

from .load_data import write_submission
from .model import MatchModel


def predict_candidate_matches(
    model: MatchModel,
    feature_frame: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """Predict match probabilities for candidate pairs and filter by threshold.

    Returns a DataFrame with columns: source1_entity_id, candidate_entity_id, match_probability, is_match.
    """
    if feature_frame.empty:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "match_probability", "is_match"]
        )

    probs = model.predict_proba(feature_frame)
    result = feature_frame[["source1_entity_id", "candidate_entity_id"]].copy()
    result["match_probability"] = probs
    result["is_match"] = result["match_probability"] >= threshold
    return result


def aggregate_matches(
    test_s1: pd.DataFrame,
    candidates: pd.DataFrame,
    predictions: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate candidate pairs and predictions into formatted submission DataFrames.

    Returns:
        (matching_results_df, candidate_pairs_df)
    """
    # Build candidate_pairs dataframe: source1_entity_id -> comma-separated candidate_entity_ids
    cand_df = candidates[["source1_entity_id", "candidate_entity_ids"]].copy()

    # Build matching_results dataframe: source1_entity_id -> comma-separated matched_entity_ids
    if predictions is not None and not predictions.empty:
        positive_matches = predictions[predictions["is_match"]]
        grouped = (
            positive_matches.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .reset_index(name="matched_entity_ids")
        )
    else:
        grouped = pd.DataFrame(columns=["source1_entity_id", "matched_entity_ids"])

    # Ensure every single test Source 1 entity is present exactly once
    s1_all = test_s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).drop_duplicates()
    matching_df = s1_all.merge(grouped, on="source1_entity_id", how="left")
    matching_df["matched_entity_ids"] = matching_df["matched_entity_ids"].fillna("")

    # Also ensure candidate_pairs covers all S1
    cand_all = s1_all.merge(cand_df, on="source1_entity_id", how="left")
    cand_all["candidate_entity_ids"] = cand_all["candidate_entity_ids"].fillna("")

    return matching_df, cand_all


def save_submission_outputs(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write matching_results.tsv and candidate_pairs.tsv to the output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    write_submission(
        matching_df,
        matching_path,
        ["source1_entity_id", "matched_entity_ids"],
    )
    write_submission(
        candidate_df,
        candidate_path,
        ["source1_entity_id", "candidate_entity_ids"],
    )

    return matching_path, candidate_path
