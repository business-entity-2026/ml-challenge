import pandas as pd
from rapidfuzz.fuzz import ratio


def jaccard_tokens(a: str, b: str) -> float:
    left, right = set(a.split()) if a else set(), set(b.split()) if b else set()
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def pair_features(source1_row: pd.Series, candidate_row: pd.Series) -> dict[str, float]:
    name_a = source1_row["name_norm"]
    name_b = candidate_row["name_norm"]
    addr_a = source1_row["address_norm"]
    addr_b = candidate_row["address_norm"]
    country_a = source1_row["country_norm"]
    country_b = candidate_row["country_norm"]

    return {
        "name_ratio": ratio(name_a, name_b) / 100.0,
        "address_ratio": ratio(addr_a, addr_b) / 100.0,
        "name_jaccard": jaccard_tokens(name_a, name_b),
        "address_jaccard": jaccard_tokens(addr_a, addr_b),
        "country_match": float(bool(country_a) and country_a == country_b),
        "name_empty_either": float(not name_a or not name_b),
        "address_empty_either": float(not addr_a or not addr_b),
    }


def build_pair_feature_frame(source1: pd.DataFrame, candidates: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    by_s1 = source1.set_index("entity_id")
    by_target = lookup.set_index("entity_id")

    for _, row in candidates.iterrows():
        s1_id = str(row["source1_entity_id"])
        for candidate_id in filter(None, str(row["candidate_entity_ids"]).split(",")):
            if candidate_id not in by_target.index or s1_id not in by_s1.index:
                continue
            features = pair_features(by_s1.loc[s1_id], by_target.loc[candidate_id])
            features.update({"source1_entity_id": s1_id, "candidate_entity_id": candidate_id})
            records.append(features)

    return pd.DataFrame(records)
