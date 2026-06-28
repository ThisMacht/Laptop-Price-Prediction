"""Production feature preprocessing for laptop price inference.

This module applies the feature schema fitted during training. It never fits
rare categories, mappings, one-hot columns, or final feature order on real
input data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "artifacts" / "preprocessors" / "feature_config.pkl"

TARGET_COLS = ["target_price", "log_target_price"]
NUMERIC_COLS = ["ram_gb", "storage_gb", "screen_size_inch"]
MISSING_FLAG_COLS = ["ram_missing", "storage_missing", "screen_missing", "cpu_missing", "gpu_missing"]
BOOLEAN_INPUT_COLS = [
    "brand_is_rare",
    "model_is_rare",
    "ram_missing",
    "storage_missing",
    "screen_missing",
    "cpu_missing",
    "gpu_missing",
    "is_soft_duplicate_spec",
    "no_info_brand",
    "no_info_model",
    "no_info_cpu_brand",
    "no_info_cpu_tier",
    "no_info_gpu",
    "ram_suspicious",
    "storage_suspicious",
    "screen_suspicious",
    "repair_mismatch",
    "potential_dedicated_gpu",
]
NO_INFO_VALUES = {"", "other", "unknown", "missing", "orther", "nan", "none", "<na>"}

DEFAULT_CPU_TIER_MAP = {
    "Other": 0,
    "Low-end": 1,
    "Low": 2,
    "Entry": 3,
    "Mid-range": 4,
    "Mid": 5,
    "Upper-mid": 6,
    "High": 7,
    "High-end": 8,
}

DEFAULT_GPU_TYPE_MAP = {
    "Integrated - Intel": "Integrated",
    "Intel Integrated": "Integrated",
    "Integrated - AMD Radeon": "Integrated",
    "Dedicated - Other/Entry": "Dedicated",
    "GTX": "Dedicated",
    "Other RTX": "Dedicated",
    "RTX 4000": "Dedicated",
    "RTX 5000": "Dedicated",
    "Apple GPU": "Apple SoC",
    "AMD Radeon": "Missing_Info",
    "Other": "Missing_Info",
}

DEFAULT_GPU_TIER_CLEAN_MAP = {
    "Integrated - Intel": "Integrated",
    "Intel Integrated": "Integrated",
    "Integrated - AMD Radeon": "Integrated",
    "Dedicated - Other/Entry": "Entry",
    "GTX": "Entry",
    "Other RTX": "Mid_High",
    "RTX 4000": "High_Workstation",
    "RTX 5000": "Very_High_Workstation",
    "Apple GPU": "Apple_SoC",
    "AMD Radeon": "Unknown",
    "Other": "Unknown",
}

DEFAULT_GPU_TIER_ORD_MAP = {
    "Unknown": -1,
    "Integrated": 0,
    "Apple_SoC": 1,
    "Entry": 2,
    "Mid_High": 3,
    "High_Workstation": 4,
    "Very_High_Workstation": 5,
}

DEFAULT_ONEHOT_SPECS = {
    "brand_grouped": "brand",
    "model_grouped": "model",
    "cpu_brand": "cpu_brand",
    "storage_type_clean": "storage_type",
    "condition_clean": "condition",
    "warranty_status": "warranty",
    "gpu_type": "gpu_type",
}

DEFAULT_INPUTS: dict[str, Any] = {
    "ram_gb": np.nan,
    "storage_gb": np.nan,
    "screen_size_inch": np.nan,
    "brand_grouped": "Other",
    "model_grouped": "Other",
    "cpu_brand": "Other",
    "cpu_tier": "Other",
    "gpu_tier": "Other",
    "storage_type_clean": "Other",
    "condition_clean": "Unknown",
    "warranty_status": "not_active",
    "no_info_gpu": False,
    "no_info_brand": False,
    "no_info_model": False,
    "no_info_cpu_brand": False,
    "no_info_cpu_tier": False,
}


def load_feature_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the fitted feature config artifact."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found at: {path}")
    feature_config = joblib.load(path)
    validate_feature_config(feature_config)
    return feature_config


def save_feature_config(feature_config: dict[str, Any], config_path: str | Path | None = None) -> None:
    """Save a fitted feature config artifact."""
    validate_feature_config(feature_config)
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_config, path)


def validate_feature_config(feature_config: dict[str, Any]) -> None:
    """Validate the minimal production contract for a fitted config."""
    if not isinstance(feature_config, dict):
        raise ValueError("feature_config must be a dict.")

    final_feature_cols = feature_config.get("final_feature_cols")
    if not isinstance(final_feature_cols, list) or not final_feature_cols:
        raise ValueError("feature_config must contain a non-empty list: final_feature_cols.")


def to_bool_series(series: pd.Series | None, index: pd.Index | None = None, default: bool = False) -> pd.Series:
    """Convert common bool/int/string encodings to bool."""
    if series is None:
        if index is None:
            return pd.Series(dtype=bool)
        return pd.Series(default, index=index, dtype=bool)

    result_index = series.index
    if series.dtype == bool:
        result = series.fillna(default).astype(bool)
    elif pd.api.types.is_numeric_dtype(series):
        result = series.fillna(int(default)).astype(int).ne(0)
    else:
        result = (
            series.astype("string")
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
            .fillna(default)
            .astype(bool)
        )

    if index is not None:
        result = result.reindex(index, fill_value=default)
    else:
        result = result.reindex(result_index, fill_value=default)
    return result.astype(bool)


def clean_text_series(series: pd.Series | None, index: pd.Index, default: str = "Other") -> pd.Series:
    """Strip text-like values and replace missing/no-info tokens."""
    if series is None:
        return pd.Series(default, index=index, dtype="string")
    cleaned = series.reindex(index).astype("string").str.strip()
    return cleaned.mask(cleaned.isna() | cleaned.str.lower().isin(NO_INFO_VALUES), default)


def ensure_columns(df: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    """Return a copy with any missing expected input columns created."""
    frame = df.copy(deep=True)
    for col, default in defaults.items():
        if col not in frame.columns:
            frame[col] = default
    return frame


def _onehot_specs(feature_config: dict[str, Any]) -> dict[str, str]:
    return dict(feature_config.get("onehot_specs") or DEFAULT_ONEHOT_SPECS)


def _onehot_prefixes(feature_config: dict[str, Any]) -> list[str]:
    return list(feature_config.get("onehot_prefixes") or _onehot_specs(feature_config).values())


def _is_managed_dummy_column(col: str, prefixes: list[str], raw_onehot_cols: set[str]) -> bool:
    protected_cols = set(BOOLEAN_INPUT_COLS + MISSING_FLAG_COLS + TARGET_COLS + NUMERIC_COLS) | raw_onehot_cols
    return col not in protected_cols and any(col.startswith(f"{prefix}_") for prefix in prefixes)


def drop_prefixed_dummy_columns(df: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    """Drop previously generated dummy columns while preserving semantic flags."""
    raw_onehot_cols = set(DEFAULT_ONEHOT_SPECS)
    dummy_cols = [col for col in df.columns if _is_managed_dummy_column(col, prefixes, raw_onehot_cols)]
    return df.drop(columns=dummy_cols, errors="ignore")


def one_hot_and_align(
    df: pd.DataFrame,
    columns: list[str],
    final_feature_cols: list[str],
    feature_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """One-hot encode configured columns and align to the fitted final schema."""
    onehot_specs = _onehot_specs(feature_config or {})
    prefixes = [onehot_specs[col] for col in columns]
    frame = drop_prefixed_dummy_columns(df.copy(), prefixes)
    encoded = pd.get_dummies(frame, columns=columns, prefix=prefixes, dtype=int)
    return encoded.reindex(columns=final_feature_cols, fill_value=0)


def _get_rare_config(feature_config: dict[str, Any]) -> dict[str, list[str]]:
    rare_config = feature_config.get("rare_config") or {}
    return {
        "rare_brands": list(feature_config.get("rare_brands", rare_config.get("rare_brands", [])) or []),
        "rare_models": list(feature_config.get("rare_models", rare_config.get("rare_models", [])) or []),
        "known_brands": list(feature_config.get("known_brands", rare_config.get("known_brands", [])) or []),
        "known_models": list(feature_config.get("known_models", rare_config.get("known_models", [])) or []),
    }


def _get_gpu_maps(feature_config: dict[str, Any]) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    gpu_config = feature_config.get("gpu_config") or {}
    type_map = (
        feature_config.get("gpu_type_map")
        or gpu_config.get("type_map")
        or gpu_config.get("gpu_type_map")
        or DEFAULT_GPU_TYPE_MAP
    )
    tier_map = (
        feature_config.get("gpu_tier_clean_map")
        or gpu_config.get("tier_map")
        or gpu_config.get("gpu_tier_clean_map")
        or DEFAULT_GPU_TIER_CLEAN_MAP
    )
    ord_map = (
        feature_config.get("gpu_tier_ord_map")
        or gpu_config.get("gpu_tier_ord_map")
        or DEFAULT_GPU_TIER_ORD_MAP
    )
    return dict(type_map), dict(tier_map), dict(ord_map)


def _impute_numeric(frame: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    impute_values = feature_config.get("impute_values") or feature_config.get("numeric_impute_values") or {}
    numeric_missing_map = {
        "ram_gb": "ram_missing",
        "storage_gb": "storage_missing",
        "screen_size_inch": "screen_missing",
    }

    for col in NUMERIC_COLS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    for value_col, missing_col in numeric_missing_map.items():
        frame[missing_col] = frame[value_col].isna().astype(int)
        frame[value_col] = frame[value_col].fillna(impute_values.get(value_col, 0))

    return frame


def _encode_levels(frame: pd.DataFrame) -> pd.DataFrame:
    frame["ram_level"] = frame["ram_gb"].apply(_encode_ram_level).astype(int)
    frame["storage_level"] = frame["storage_gb"].apply(_encode_storage_level).astype(int)
    frame["screen_size_level"] = frame["screen_size_inch"].apply(_encode_screen_size_level).astype(int)
    return frame


def _encode_ram_level(value: Any) -> int:
    if pd.isna(value) or value == 0:
        return 0
    return int(math.ceil(math.log2(float(value))))


def _encode_storage_level(value: Any) -> int:
    if pd.isna(value) or value == 0:
        return 0
    value = float(value)
    if value <= 128:
        return 1
    return int(math.ceil(math.log2(value / 128))) + 1


def _encode_screen_size_level(value: Any) -> int:
    if pd.isna(value) or value == 0:
        return 0
    value = float(value)
    if value < 14:
        return 1
    if value < 15.6:
        return 2
    if value <= 16:
        return 3
    return 4


def _apply_known_categories(frame: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    onehot_categories = feature_config.get("onehot_categories") or {}
    for col, categories in onehot_categories.items():
        if col not in frame.columns:
            continue
        known = set(categories or [])
        fallback = "Other" if "Other" in known else "Unknown" if "Unknown" in known else None
        if fallback is not None:
            frame.loc[~frame[col].isin(known), col] = fallback
    return frame


def _validate_output(X: pd.DataFrame, final_feature_cols: list[str]) -> None:
    if X.columns.tolist() != final_feature_cols:
        raise ValueError("Transformed columns do not match feature_config['final_feature_cols'].")
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Transformed output has duplicate columns: {dupes}")
    missing_count = int(X.isna().sum().sum())
    if missing_count:
        raise ValueError(f"Transformed output contains {missing_count} missing values.")
    object_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if object_cols:
        raise ValueError(f"Transformed output contains non-numeric columns: {object_cols}")


def transform_with_feature_config(df: pd.DataFrame, feature_config: dict[str, Any]) -> pd.DataFrame:
    """Apply a fitted feature config and return only final model features."""
    validate_feature_config(feature_config)
    final_feature_cols = feature_config["final_feature_cols"]
    onehot_specs = _onehot_specs(feature_config)
    onehot_cols = list(onehot_specs.keys())
    onehot_prefixes = _onehot_prefixes(feature_config)

    frame = ensure_columns(df, DEFAULT_INPUTS)
    frame = drop_prefixed_dummy_columns(frame, onehot_prefixes)

    for col in BOOLEAN_INPUT_COLS:
        if col in frame.columns:
            frame[col] = to_bool_series(frame[col], index=frame.index)

    frame = _impute_numeric(frame, feature_config)

    text_defaults = {
        "brand_grouped": "Other",
        "model_grouped": "Other",
        "cpu_brand": "Other",
        "cpu_tier": "Other",
        "gpu_tier": "Other",
        "storage_type_clean": "Other",
        "condition_clean": "Unknown",
        "warranty_status": "not_active",
    }
    for col, default in text_defaults.items():
        frame[col] = clean_text_series(frame[col], index=frame.index, default=default)

    rare_config = _get_rare_config(feature_config)
    rare_brands = set(rare_config["rare_brands"])
    rare_models = set(rare_config["rare_models"])
    known_brands = set(rare_config["known_brands"])
    known_models = set(rare_config["known_models"])

    if known_brands or rare_brands:
        frame["brand_is_rare"] = (
            frame["brand_grouped"].isin(rare_brands) | ~frame["brand_grouped"].isin(known_brands)
        ).astype(int)
    else:
        frame["brand_is_rare"] = 0

    if known_models or rare_models:
        frame["model_is_rare"] = (
            frame["model_grouped"].isin(rare_models) | ~frame["model_grouped"].isin(known_models)
        ).astype(int)
    else:
        frame["model_is_rare"] = 0

    frame.loc[frame["brand_is_rare"].eq(1), "brand_grouped"] = "Other"
    frame.loc[frame["model_is_rare"].eq(1), "model_grouped"] = "Other"

    cpu_missing = (
        frame["cpu_brand"].str.lower().isin(NO_INFO_VALUES)
        | frame["cpu_brand"].eq("Other")
        | frame["cpu_tier"].str.lower().isin(NO_INFO_VALUES)
        | to_bool_series(frame.get("no_info_cpu_brand"), index=frame.index)
        | to_bool_series(frame.get("no_info_cpu_tier"), index=frame.index)
    )
    frame["cpu_missing"] = cpu_missing.astype(int)
    cpu_tier_map = dict(feature_config.get("cpu_tier_map") or DEFAULT_CPU_TIER_MAP)
    frame["cpu_tier_encoded"] = frame["cpu_tier"].map(cpu_tier_map).fillna(0).astype(int)

    gpu_type_map, gpu_tier_clean_map, gpu_tier_ord_map = _get_gpu_maps(feature_config)
    gpu_raw = frame["gpu_tier"].astype("string").str.strip()
    gpu_missing = gpu_raw.isna() | gpu_raw.eq("Other") | to_bool_series(frame.get("no_info_gpu"), index=frame.index)
    frame["gpu_missing"] = gpu_missing.astype(int)
    frame["gpu_raw"] = gpu_raw.fillna("Other")
    frame["gpu_type"] = frame["gpu_raw"].map(gpu_type_map).fillna("Missing_Info")
    frame.loc[frame["gpu_missing"].eq(1), "gpu_type"] = "Missing_Info"
    frame["gpu_tier_clean"] = frame["gpu_raw"].map(gpu_tier_clean_map).fillna("Unknown")
    frame.loc[frame["gpu_missing"].eq(1), "gpu_tier_clean"] = "Unknown"
    frame["gpu_tier_ord_filled"] = (
        frame["gpu_tier_clean"].map(gpu_tier_ord_map).fillna(-1).astype(int)
    )

    frame = _encode_levels(frame)
    frame = _apply_known_categories(frame, feature_config)

    for col in onehot_cols:
        if col not in frame.columns:
            default = DEFAULT_INPUTS.get(col, "Other")
            frame[col] = default

    X = one_hot_and_align(frame, onehot_cols, final_feature_cols, feature_config)

    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    object_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if object_cols:
        X[object_cols] = X[object_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    X = X.fillna(0)

    _validate_output(X, final_feature_cols)
    return X


def transform_real_input(
    input_df: pd.DataFrame,
    feature_config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """Transform real/inference input into model-ready features."""
    config = load_feature_config(config_path) if feature_config is None else feature_config
    validate_feature_config(config)
    X_real = transform_with_feature_config(input_df.copy(deep=True), config)
    _validate_output(X_real, config["final_feature_cols"])
    return X_real


if __name__ == "__main__":
    feature_config = load_feature_config()
    sample_df = pd.DataFrame(
        [
            {
                "ram_gb": 16,
                "storage_gb": 512,
                "screen_size_inch": 13.6,
                "brand_grouped": "Apple",
                "model_grouped": "MacBook Air",
                "cpu_brand": "Apple",
                "cpu_tier": "High",
                "gpu_tier": "Apple GPU",
                "storage_type_clean": "SSD",
                "condition_clean": "Mới",
                "warranty_status": "Active",
                "no_info_gpu": False,
            },
            {
                "ram_gb": 8,
                "storage_gb": 256,
                "screen_size_inch": 15.6,
                "brand_grouped": "Dell",
                "model_grouped": "Latitude",
                "cpu_brand": "Intel",
                "cpu_tier": "Mid",
                "gpu_tier": "Other",
                "storage_type_clean": "SSD",
                "condition_clean": "Đã sử dụng (chưa sửa chữa)",
                "warranty_status": "not_active",
                "no_info_gpu": True,
            },
            {
                "ram_gb": 32,
                "storage_gb": 1024,
                "screen_size_inch": 16,
                "brand_grouped": "Lenovo",
                "model_grouped": "ThinkPad",
                "cpu_brand": "Intel",
                "cpu_tier": "High-end",
                "gpu_tier": "RTX 4000",
                "storage_type_clean": "SSD",
                "condition_clean": "Mới",
                "warranty_status": "Manufacturer",
                "no_info_gpu": False,
            },
            {
                "ram_gb": None,
                "storage_gb": None,
                "screen_size_inch": None,
                "brand_grouped": "UnknownBrandXYZ",
                "model_grouped": "UnknownModelXYZ",
                "cpu_brand": "Other",
                "cpu_tier": "Other",
                "gpu_tier": None,
                "storage_type_clean": "Other",
                "condition_clean": "Unknown",
                "warranty_status": "not_active",
                "no_info_gpu": False,
            },
        ]
    )
    X = transform_real_input(sample_df, feature_config=feature_config)
    object_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    print("Shape:", X.shape)
    print("Number of columns:", X.shape[1])
    print("Missing count:", int(X.isna().sum().sum()))
    print("Object columns:", object_columns)
    assert X.columns.tolist() == feature_config["final_feature_cols"]
    assert X.isna().sum().sum() == 0
