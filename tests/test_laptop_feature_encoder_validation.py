from __future__ import annotations

import pandas as pd
import pytest

from src.encoder import LaptopFeatureEncoder
from src.encoder.encoder_validation import validate_encoded_output


def test_encode_one_accepts_minimal_valid_input() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "Dell",
            "model": "XPS",
            "ram_gb": 16,
            "storage_gb": 512,
            "screen_size_inch": 13.3,
        }
    )

    assert X.shape[0] == 1
    assert X.columns.tolist() == encoder.feature_names
    assert X.isna().sum().sum() == 0


def test_encode_many_rejects_single_dict() -> None:
    encoder = LaptopFeatureEncoder()

    with pytest.raises(TypeError, match="encode_many expects a list of dicts"):
        encoder.encode_many({"brand": "Dell"})


def test_encode_many_rejects_non_dict_row() -> None:
    encoder = LaptopFeatureEncoder()

    with pytest.raises(TypeError, match="Each input row must be a dict"):
        encoder.encode_many(["bad row"])


def test_encode_many_rejects_empty_list() -> None:
    encoder = LaptopFeatureEncoder()

    with pytest.raises(ValueError, match="Input rows must not be empty"):
        encoder.encode_many([])


def test_encode_one_rejects_negative_numeric_value() -> None:
    encoder = LaptopFeatureEncoder()

    with pytest.raises(ValueError, match="ram_gb is out of valid range"):
        encoder.encode_one({"ram_gb": -8})


def test_encode_one_rejects_screen_size_above_valid_range() -> None:
    encoder = LaptopFeatureEncoder()

    with pytest.raises(ValueError, match="screen_size_inch is out of valid range"):
        encoder.encode_one({"screen_size_inch": 99})


def test_non_numeric_numeric_field_is_treated_as_missing() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"ram_gb": "abc"})

    assert X.loc[0, "ram_missing"] == 1
    assert X.loc[0, "ram_gb"] == 0


def test_missing_numeric_values_do_not_raise() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "ram_gb": None,
            "storage_gb": None,
            "screen_size_inch": None,
        }
    )

    assert X.loc[0, "ram_missing"] == 1
    assert X.loc[0, "storage_missing"] == 1
    assert X.loc[0, "screen_missing"] == 1


def test_validate_encoded_output_rejects_missing_column() -> None:
    encoder = LaptopFeatureEncoder()
    frame = pd.DataFrame([{name: 0 for name in encoder.feature_names[:-1]}])

    with pytest.raises(ValueError, match="Encoder failed to create final features"):
        validate_encoded_output(frame, encoder.feature_names, encoder.n_features)


def test_validate_encoded_output_rejects_nan() -> None:
    encoder = LaptopFeatureEncoder()
    row = {name: 0 for name in encoder.feature_names}
    row[encoder.feature_names[0]] = float("nan")
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Encoded output contains NaN values"):
        validate_encoded_output(frame, encoder.feature_names, encoder.n_features)


def test_validate_encoded_output_rejects_infinite() -> None:
    encoder = LaptopFeatureEncoder()
    row = {name: 0 for name in encoder.feature_names}
    row[encoder.feature_names[0]] = float("inf")
    frame = pd.DataFrame([row])

    with pytest.raises(ValueError, match="Encoded output contains infinite values"):
        validate_encoded_output(frame, encoder.feature_names, encoder.n_features)


def test_validate_encoded_output_rejects_non_numeric_string() -> None:
    encoder = LaptopFeatureEncoder()
    row = {name: 0 for name in encoder.feature_names}
    row[encoder.feature_names[0]] = "bad"
    frame = pd.DataFrame([row])

    with pytest.raises((ValueError, TypeError)):
        validate_encoded_output(frame, encoder.feature_names, encoder.n_features)
