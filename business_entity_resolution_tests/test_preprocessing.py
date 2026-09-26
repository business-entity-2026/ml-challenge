import pandas as pd

from src.preprocessing import normalize_source, normalize_text


def test_normalize_text():
    assert normalize_text("Domino's Pizza & Co.") == "domino s pizza and co"


def test_normalize_source_adds_columns():
    df = pd.DataFrame(
        [{"entity_id": "S1-1", "business_name": "ABC Ltd", "business_address": "Main Road", "country": "India"}]
    )
    out = normalize_source(df)
    assert {"name_norm", "address_norm", "country_norm"}.issubset(out.columns)
