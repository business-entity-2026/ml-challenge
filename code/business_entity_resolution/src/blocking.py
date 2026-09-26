from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Blocker:
    name_top_k: int = 25
    address_top_k: int = 25
    max_candidates_per_source1: int = 100

    def _fit_index(self, series: pd.Series) -> tuple[TfidfVectorizer, object]:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(series.fillna(""))
        return vectorizer, matrix

    def generate(self, source1: pd.DataFrame, source2: pd.DataFrame, source3: pd.DataFrame) -> pd.DataFrame:
        """Generate a conservative candidate set using country + TF-IDF name/address retrieval.

        This is a baseline blocker. During experimentation, add additional high-recall blockers
        and keep the last candidate set here identical to what the matcher receives.
        """
        target = pd.concat([source2.assign(_src="S2"), source3.assign(_src="S3")], ignore_index=True)
        name_vectorizer, name_matrix = self._fit_index(target["name_norm"])
        address_vectorizer, address_matrix = self._fit_index(target["address_norm"])

        target_country = target["country_norm"].fillna("").to_numpy()
        rows: list[dict[str, str]] = []

        for _, s1_row in source1.iterrows():
            allowed = np.ones(len(target), dtype=bool)
            country = s1_row["country_norm"]
            if country:
                allowed = target_country == country
                # If country has no exact candidates, fall back to all records.
                if not allowed.any():
                    allowed = np.ones(len(target), dtype=bool)

            name_query = name_vectorizer.transform([s1_row["name_norm"]])
            addr_query = address_vectorizer.transform([s1_row["address_norm"]])
            name_scores = cosine_similarity(name_query, name_matrix).ravel()
            address_scores = cosine_similarity(addr_query, address_matrix).ravel()

            scored_indices: set[int] = set()
            allowed_idx = np.flatnonzero(allowed)
            if len(allowed_idx):
                name_order = allowed_idx[np.argsort(name_scores[allowed_idx])[-self.name_top_k:]]
                addr_order = allowed_idx[np.argsort(address_scores[allowed_idx])[-self.address_top_k:]]
                scored_indices.update(name_order.tolist())
                scored_indices.update(addr_order.tolist())

            # Enforce an explicit cap while preserving highest combined similarity.
            ranked = sorted(
                scored_indices,
                key=lambda i: (name_scores[i] + address_scores[i]),
                reverse=True,
            )[: self.max_candidates_per_source1]

            candidate_ids = [str(target.iloc[i]["entity_id"]) for i in ranked]
            rows.append({
                "source1_entity_id": str(s1_row["entity_id"]),
                "candidate_entity_ids": ",".join(candidate_ids),
            })

        return pd.DataFrame(rows)
