from __future__ import annotations

import pytest

from src.prediction.price_interval import (
    build_price_prediction,
    completeness_pct,
    missing_fields,
    price_segment,
    uncertainty_multiplier,
)


FULL_RAW = {
    "brand": "Dell",
    "model": "Inspiron",
    "ram_gb": 16,
    "storage_gb": 512,
    "storage_type": "SSD",
    "screen_size_inch": 15.6,
    "cpu_text": "Intel Core i5-1235U",
    "cpu_brand": "Intel",
    "cpu_family": "Intel Core i5",
    "cpu_generation": 12,
    "cpu_suffix": "U",
    "gpu_text": "integrated",
    "condition": "good",
    "warranty_status": "expired",
}


def test_price_segment_boundaries() -> None:
    assert price_segment(3.0) == "Q1_low"
    assert price_segment(6.0) == "Q2"
    assert price_segment(12.0) == "Q3"
    assert price_segment(20.0) == "Q4"
    assert price_segment(40.0) == "Q5_high"


def test_complete_input_has_narrower_interval_than_sparse_input() -> None:
    full = build_price_prediction(12.0, FULL_RAW)
    sparse = build_price_prediction(
        12.0,
        {field: None for field in FULL_RAW},
    )

    assert full["price_range"]["high"] - full["price_range"]["low"] < (
        sparse["price_range"]["high"] - sparse["price_range"]["low"]
    )
    assert full["uncertainty"]["level"] in {"low", "medium"}
    assert sparse["uncertainty"]["level"] == "high"


def test_missing_fields_and_completeness() -> None:
    partial = dict(FULL_RAW)
    partial["brand"] = None
    partial["gpu_text"] = None

    assert "brand" in missing_fields(partial)
    assert "gpu_text" in missing_fields(partial)
    assert completeness_pct(partial) < completeness_pct(FULL_RAW)


def test_interval_never_below_minimum_price() -> None:
    result = build_price_prediction(1.0, {field: None for field in FULL_RAW})
    assert result["price_range"]["low"] >= 0.5
    assert result["price_range"]["high"] >= result["predicted_price"]


def test_uncertainty_multiplier_caps() -> None:
    multiplier = uncertainty_multiplier({field: None for field in FULL_RAW})
    assert multiplier <= 2.6
