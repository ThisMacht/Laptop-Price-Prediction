"""Validation helpers for production laptop feature encoding."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def validate_input_rows(rows: list[dict[str, Any]]) -> None:
    """Validate raw input rows before feature encoding."""
    if not rows:
        raise ValueError("Input rows must not be empty.")

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(
                f"Each input row must be a dict. Row {row_index} has type {type(row).__name__}."
            )
        _validate_numeric_range(row, row_index, "ram_gb", 0, 512)
        _validate_numeric_range(row, row_index, "storage_gb", 0, 100_000)
        _validate_numeric_range(row, row_index, "screen_size_inch", 0, 30)


def validate_encoded_output(
    X: pd.DataFrame,
    feature_names: list[str],
    n_features: int,
) -> pd.DataFrame:
    """Align encoded output to the schema and validate production invariants."""
    missing = [col for col in feature_names if col not in X.columns]
    if missing:
        raise ValueError(f"Encoder failed to create final features: {missing}")

    X = X.reindex(columns=feature_names)
    if X.shape[1] != n_features:
        raise ValueError(f"Expected {n_features} columns, got {X.shape[1]}.")

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    if X.isna().any().any():
        na_counts = X.isna().sum()
        raise ValueError(f"Encoded output contains NaN values: {na_counts[na_counts > 0].to_dict()}")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Encoded output contains infinite values.")

    object_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if object_cols:
        raise ValueError(f"Encoded output contains non-numeric columns: {object_cols}")

    if X.columns.tolist() != feature_names:
        raise ValueError("Encoded columns do not match the schema feature order.")

    return X


def _is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, dict, set)):
        return False

    try:
        is_missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(is_missing, (bool, np.bool_)):
        return bool(is_missing)
    return False


def _validate_numeric_range(
    row: dict[str, Any],
    row_index: int,
    field_name: str,
    min_value: float,
    max_value: float,
) -> None:
    if field_name not in row:
        return

    value = row[field_name]
    if _is_missing_like(value):
        return

    try:
        number = float(value)
    except (TypeError, ValueError):
        return

    if math.isnan(number):
        return

    if number < min_value or number > max_value:
        raise ValueError(f"{field_name} is out of valid range at row {row_index}: {number}")
