from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = [
    "name_ratio",
    "address_ratio",
    "name_jaccard",
    "address_jaccard",
    "country_match",
    "name_empty_either",
    "address_empty_either",
]


@dataclass
class MatchModel:
    random_state: int = 42

    def __post_init__(self) -> None:
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self.random_state)

    def fit(self, feature_frame: pd.DataFrame, labels: pd.Series) -> "MatchModel":
        self.model.fit(feature_frame[FEATURE_COLUMNS], labels)
        return self

    def predict_proba(self, feature_frame: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(feature_frame[FEATURE_COLUMNS])[:, 1], index=feature_frame.index)
