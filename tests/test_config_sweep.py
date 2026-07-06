from __future__ import annotations

from src.prediction.config_sweep import (
    DEFAULT_RAM_VALUES_GB,
    DEFAULT_STORAGE_VALUES_GB,
    format_storage_label,
    range_mean,
    sweep_ram_by_storage,
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


def test_format_storage_label() -> None:
    assert format_storage_label(256) == "256GB"
    assert format_storage_label(1024) == "1TB"
    assert format_storage_label(2048) == "2TB"


def test_range_mean() -> None:
    assert range_mean({"low": 10.0, "high": 14.0}) == 12.0


def test_sweep_ram_by_storage_structure() -> None:
    def fake_predict(features: dict) -> float:
        ram = float(features["ram_gb"])
        storage = float(features["storage_gb"])
        return round(8.0 + ram * 0.05 + storage / 1024.0, 3)

    result = sweep_ram_by_storage(
        FULL_RAW,
        predict_fn=fake_predict,
        ram_values=[8, 16],
        storage_values=[256, 512],
    )

    assert result["task"] == "config_sweep"
    assert result["ram_values"] == [8, 16]
    assert result["storage_values"] == [256, 512]
    assert len(result["series"]) == 2
    assert result["series"][0]["storage_label"] == "256GB"
    assert len(result["series"][0]["points"]) == 2

    first_point = result["series"][0]["points"][0]
    assert first_point["ram_gb"] == 8
    assert first_point["price_mean"] == range_mean(first_point["price_range"])
    assert first_point["price_range"]["low"] <= first_point["price_mean"] <= first_point["price_range"]["high"]


def test_sweep_defaults_cover_requested_axis() -> None:
    assert DEFAULT_RAM_VALUES_GB[0] == 8
    assert DEFAULT_RAM_VALUES_GB[-1] == 256
    assert DEFAULT_STORAGE_VALUES_GB[0] == 256
    assert DEFAULT_STORAGE_VALUES_GB[-1] == 2048
