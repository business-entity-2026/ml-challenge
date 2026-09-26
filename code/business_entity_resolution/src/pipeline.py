from pathlib import Path

import pandas as pd

from .blocking import Blocker
from .features import build_pair_feature_frame
from .load_data import load_ground_truth, load_sources, write_submission
from .model import MatchModel
from .preprocessing import normalize_source


def ground_truth_map(df: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        raw = str(row["matched_entity_ids"] or "").strip()
        result[str(row["source1_entity_id"])] = {x for x in raw.split(",") if x}
    return result


def build_training_labels(features: pd.DataFrame, truth: dict[str, set[str]]) -> pd.Series:
    return features.apply(
        lambda row: int(row["candidate_entity_id"] in truth.get(row["source1_entity_id"], set())),
        axis=1,
    )


def run_train_baseline(train_dir: str | Path) -> tuple[MatchModel, pd.DataFrame, pd.DataFrame]:
    s1, s2, s3 = load_sources(train_dir, "train")
    gt = load_ground_truth(Path(train_dir) / "train_ground_truth.tsv")

    s1, s2, s3 = map(normalize_source, (s1, s2, s3))
    candidates = Blocker().generate(s1, s2, s3)
    lookup = pd.concat([s2, s3], ignore_index=True)
    features = build_pair_feature_frame(s1, candidates, lookup)
    truth = ground_truth_map(gt)
    labels = build_training_labels(features, truth)

    model = MatchModel().fit(features, labels)
    return model, candidates, features


def write_placeholder_test_outputs(test_s1: pd.DataFrame, candidates: pd.DataFrame, output_dir: str | Path) -> None:
    """Write valid candidate output and an empty-match baseline.

    This function is intentionally conservative. Replace the prediction stage with the
    trained model before submitting to the challenge.
    """
    output_dir = Path(output_dir)
    empty_matches = candidates[["source1_entity_id"]].copy()
    empty_matches["matched_entity_ids"] = ""

    # Ensure every test S1 exists once even if blocking returns an unexpected omission.
    empty_matches = test_s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}).merge(
        empty_matches, on="source1_entity_id", how="left"
    )
    empty_matches["matched_entity_ids"] = empty_matches["matched_entity_ids"].fillna("")

    write_submission(
        candidates,
        output_dir / "candidate_pairs.tsv",
        ["source1_entity_id", "candidate_entity_ids"],
    )
    write_submission(
        empty_matches,
        output_dir / "matching_results.tsv",
        ["source1_entity_id", "matched_entity_ids"],
    )
