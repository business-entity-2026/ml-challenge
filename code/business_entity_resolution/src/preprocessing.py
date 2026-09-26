import re
from typing import Iterable

import pandas as pd

# Conservative replacements. Keep this list small initially; add rules only when
# validation proves they improve matching without increasing false merges.
COMMON_REPLACEMENTS = {
    "&": " and ",
    "corporation": " corp ",
    "company": " co ",
    "incorporated": " inc ",
}


def normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).lower()
    for old, new in COMMON_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: object) -> list[str]:
    normalized = normalize_text(value)
    return normalized.split() if normalized else []


def normalize_source(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["name_norm"] = result["business_name"].map(normalize_text)
    result["address_norm"] = result["business_address"].map(normalize_text)
    result["country_norm"] = result["country"].map(normalize_text)
    result["name_tokens"] = result["name_norm"].map(tokenize)
    result["address_tokens"] = result["address_norm"].map(tokenize)
    return result


def canonical_token_set(tokens: Iterable[str]) -> set[str]:
    return {t for t in tokens if t}
