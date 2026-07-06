"""Sweep RAM and storage values to compare predicted prices across configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .price_interval import build_price_prediction


DEFAULT_RAM_VALUES_GB = [8, 16, 32, 64, 128, 256]
DEFAULT_STORAGE_VALUES_GB = [256, 512, 1024, 2048]

HIGHLIGHT_COLORS = (
    "#72d8ff",
    "#e7ff65",
    "#ff8a70",
    "#80e18b",
    "#c9a0ff",
    "#ffd166",
)


def format_storage_label(storage_gb: int) -> str:
    if storage_gb >= 1024 and storage_gb % 1024 == 0:
        return f"{storage_gb // 1024}TB"
    return f"{storage_gb}GB"


def range_mean(price_range: dict[str, Any]) -> float:
    low = float(price_range["low"])
    high = float(price_range["high"])
    return round((low + high) / 2.0, 3)


def build_sweep_point(
    raw_features: dict[str, Any],
    predicted_price_million: float,
    encoded_features: dict[str, float | int] | None,
) -> dict[str, Any]:
    payload = build_price_prediction(predicted_price_million, raw_features, encoded_features)
    price_range = payload["price_range"]
    return {
        "predicted_price": payload["predicted_price"],
        "price_range": price_range,
        "price_mean": range_mean(price_range),
        "uncertainty": payload["uncertainty"],
    }


def sweep_ram_by_storage(
    base_features: dict[str, Any],
    predict_fn: Callable[[dict[str, Any]], float],
    encode_fn: Callable[[dict[str, Any]], tuple[dict[str, float | int], list[str]]] | None = None,
    ram_values: list[int] | None = None,
    storage_values: list[int] | None = None,
) -> dict[str, Any]:
    """Vary RAM on the X-axis and draw one series per storage capacity."""
    ram_axis = list(ram_values or DEFAULT_RAM_VALUES_GB)
    storage_axis = list(storage_values or DEFAULT_STORAGE_VALUES_GB)
    base = deepcopy(base_features)

    series: list[dict[str, Any]] = []
    for index, storage_gb in enumerate(storage_axis):
        points: list[dict[str, Any]] = []
        for ram_gb in ram_axis:
            variant = {**base, "ram_gb": ram_gb, "storage_gb": storage_gb}
            encoded_features = None
            if encode_fn is not None:
                encoded_features, _active = encode_fn(variant)
            predicted = float(predict_fn(variant))
            point = build_sweep_point(variant, predicted, encoded_features)
            points.append(
                {
                    "ram_gb": ram_gb,
                    **point,
                }
            )

        series.append(
            {
                "storage_gb": storage_gb,
                "storage_label": format_storage_label(storage_gb),
                "color": HIGHLIGHT_COLORS[index % len(HIGHLIGHT_COLORS)],
                "points": points,
            }
        )

    return {
        "task": "config_sweep",
        "vary_field": "ram_gb",
        "series_field": "storage_gb",
        "price_unit": "million_vnd",
        "ram_values": ram_axis,
        "storage_values": storage_axis,
        "base_features": base,
        "series": series,
    }
