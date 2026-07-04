"""Price interval estimation for production laptop price prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "models" / "price_interval_config.json"


def load_interval_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Price interval config not found at: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_CONFIG = load_interval_config()

RAW_LAPTOP_FIELDS = list(_CONFIG["raw_laptop_fields"])
CRITICAL_FIELDS = tuple(_CONFIG["critical_fields"])
SEGMENT_RMSE: dict[str, float] = dict(_CONFIG["segment_rmse"])
SEGMENT_UPPER_BOUNDS = tuple(_CONFIG["segment_upper_bounds"])
MISSING_FIELD_WEIGHTS: dict[str, float] = dict(_CONFIG["missing_field_weights"])
NO_INFO_FLAG_WEIGHT = float(_CONFIG["no_info_flag_weight"])
MAX_UNCERTAINTY_MULTIPLIER = float(_CONFIG["max_uncertainty_multiplier"])
MIN_PRICE_MILLION = float(_CONFIG["min_price_million"])
API_VERSION = int(_CONFIG.get("api_version", 2))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, float) and value != value:
        return True
    return False


def price_segment(predicted_price_million: float) -> str:
    """Map a point prediction to the holdout price segment used for error calibration."""
    price = max(float(predicted_price_million), 0.0)
    if price <= SEGMENT_UPPER_BOUNDS[0]:
        return "Q1_low"
    if price <= SEGMENT_UPPER_BOUNDS[1]:
        return "Q2"
    if price <= SEGMENT_UPPER_BOUNDS[2]:
        return "Q3"
    if price <= SEGMENT_UPPER_BOUNDS[3]:
        return "Q4"
    return "Q5_high"


def missing_fields(raw_features: dict[str, Any]) -> list[str]:
    return [field for field in RAW_LAPTOP_FIELDS if _is_missing(raw_features.get(field))]


def completeness_pct(raw_features: dict[str, Any]) -> float:
    present = len(RAW_LAPTOP_FIELDS) - len(missing_fields(raw_features))
    return round(100.0 * present / len(RAW_LAPTOP_FIELDS), 1)


def uncertainty_multiplier(
    raw_features: dict[str, Any],
    encoded_features: dict[str, float | int] | None = None,
) -> float:
    multiplier = 1.0
    for field, weight in MISSING_FIELD_WEIGHTS.items():
        if _is_missing(raw_features.get(field)):
            multiplier += weight

    if encoded_features:
        no_info_flags = sum(
            1
            for key, value in encoded_features.items()
            if key.startswith("no_info_") and float(value) == 1.0
        )
        multiplier += no_info_flags * NO_INFO_FLAG_WEIGHT

    return min(multiplier, MAX_UNCERTAINTY_MULTIPLIER)


def uncertainty_level(multiplier: float, completeness: float) -> str:
    if multiplier <= 1.2 and completeness >= 80:
        return "low"
    if multiplier <= 1.7 and completeness >= 55:
        return "medium"
    return "high"


def build_price_prediction(
    predicted_price_million: float,
    raw_features: dict[str, Any],
    encoded_features: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    """Return point estimate plus a calibrated price interval for Problem 1."""
    point = round(float(predicted_price_million), 3)
    segment = price_segment(point)
    base_margin = SEGMENT_RMSE[segment]
    multiplier = uncertainty_multiplier(raw_features, encoded_features)
    half_width = base_margin * multiplier
    low = max(MIN_PRICE_MILLION, point - half_width)
    high = point + half_width
    completeness = completeness_pct(raw_features)
    missing = missing_fields(raw_features)
    level = uncertainty_level(multiplier, completeness)

    return {
        "predicted_price": point,
        "price_unit": "million_vnd",
        "price_range": {
            "low": round(low, 2),
            "high": round(high, 2),
            "half_width_million_vnd": round(half_width, 2),
        },
        "price_segment": segment,
        "input_completeness_pct": completeness,
        "missing_fields": missing,
        "uncertainty": {
            "level": level,
            "multiplier": round(multiplier, 2),
            "base_rmse_million_vnd": base_margin,
            "reason": _uncertainty_reason(level, missing),
        },
    }


def _uncertainty_reason(level: str, missing: list[str]) -> str:
    if not missing:
        return "Đủ thông tin chính; khoảng giá dựa trên sai số holdout theo phân khúc giá."

    critical_missing = [field for field in missing if field in CRITICAL_FIELDS]
    if level == "high":
        if critical_missing:
            return (
                "Thiếu nhiều thông tin quan trọng "
                f"({', '.join(critical_missing[:4])}); khoảng giá được nới rộng."
            )
        return "Một số thông tin còn thiếu; khoảng giá được nới rộng để phản ánh độ không chắc chắn."

    if critical_missing:
        return (
            f"Thiếu {', '.join(critical_missing[:3])}; "
            "khoảng giá rộng hơn mức trung bình."
        )
    return "Thiếu vài thông tin phụ; khoảng giá điều chỉnh nhẹ theo mức độ hoàn thiện input."
