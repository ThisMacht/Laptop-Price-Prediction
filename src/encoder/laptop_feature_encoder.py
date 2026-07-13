"""Production encoder from raw structured laptop fields to final features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
import unicodedata

import pandas as pd

from .cpu_parser import parse_cpu
from .encoder_validation import validate_encoded_output, validate_input_rows
from .feature_maps import BRAND_ALIASES, CONDITION_SCORE_MAP, FINAL_BRANDS
from .gpu_parser import parse_gpu_tier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "models" / "final_laptop_price_feature_schema.json"


class LaptopFeatureEncoder:
    """Encode raw structured laptop rows into the final production schema."""

    def __init__(self, schema_path: str | Path | None = None):
        """Load the final feature schema and derive managed one-hot columns."""
        self.schema_path = Path(schema_path) if schema_path is not None else DEFAULT_SCHEMA_PATH
        self.schema = self._load_schema(self.schema_path)
        self.feature_names = list(self.schema["feature_names"])
        self.n_features = int(self.schema["n_features"])
        self.feature_set = set(self.feature_names)
        self.brand_columns = [
            col
            for col in self.feature_names
            if col.startswith("brand_") and col != "brand_is_rare" and not col.startswith("brand_segment_")
        ]
        self.model_columns = [
            col for col in self.feature_names if col.startswith("model_") and col != "model_is_rare"
        ]
        self.cpu_brand_columns = [col for col in self.feature_names if col.startswith("cpu_brand_")]
        self.cpu_family_group_columns = [
            col for col in self.feature_names if col.startswith("cpu_family_group_")
        ]
        self.final_models = {col.removeprefix("model_"): col for col in self.model_columns}
        self.model_aliases = {normalize_key(model): model for model in self.final_models}

    def encode_one(self, raw: dict) -> pd.DataFrame:
        """Encode one raw structured laptop row into a one-row DataFrame."""
        return self.encode_many([raw])

    def encode_many(self, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
        """Encode many raw structured laptop rows into a numeric DataFrame."""
        if isinstance(rows, dict):
            raise TypeError(
                "encode_many expects a list of dicts or a DataFrame. "
                "Use encode_one for a single dict."
            )

        if isinstance(rows, pd.DataFrame):
            raw_rows = rows.to_dict(orient="records")
        else:
            raw_rows = list(rows)

        validate_input_rows(raw_rows)

        encoded_rows = [self._encode_row(row) for row in raw_rows]
        frame = pd.DataFrame(encoded_rows)
        return self._align_and_validate(frame)

    @staticmethod
    def _load_schema(schema_path: Path) -> dict[str, Any]:
        if not schema_path.exists():
            raise FileNotFoundError(f"Feature schema not found at: {schema_path}")
        with schema_path.open("r", encoding="utf-8") as f:
            schema = json.load(f)
        feature_names = schema.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Feature schema must contain a non-empty feature_names list.")
        if int(schema.get("n_features", len(feature_names))) != len(feature_names):
            raise ValueError("Schema n_features does not match feature_names length.")
        return schema

    def _encode_row(self, raw: dict[str, Any]) -> dict[str, float | int]:
        features: dict[str, float | int] = {name: 0 for name in self.feature_names}

        ram, ram_missing = _numeric_or_zero(raw.get("ram_gb"))
        storage, storage_missing = _numeric_or_zero(raw.get("storage_gb"))
        screen, screen_missing = _numeric_or_zero(raw.get("screen_size_inch"))
        features["ram_gb"] = ram
        features["storage_gb"] = storage
        features["screen_size_inch"] = screen
        features["ram_missing"] = ram_missing
        features["storage_missing"] = storage_missing
        features["screen_missing"] = screen_missing

        self._encode_brand(raw.get("brand"), raw.get("model"), features)
        self._encode_model(raw.get("model"), features)
        self._encode_storage(raw.get("storage_type"), features)
        self._encode_condition(raw.get("condition"), features)
        self._encode_warranty(raw.get("warranty_status"), features)
        self._encode_cpu(raw, features)
        self._encode_gpu(raw.get("gpu_text"), features)
        self._encode_ram_storage_interactions(ram, storage, ram_missing, storage_missing, features)

        return features

    def _encode_brand(self, raw_brand: Any, raw_model: Any, features: dict[str, float | int]) -> None:
        brand = canonical_brand(raw_brand)
        if brand is None:
            features["no_info_brand"] = 1
            return

        features["no_info_brand"] = 0
        brand_col = f"brand_{brand}"
        if brand_col in self.feature_set:
            features[brand_col] = 1
        else:
            features["brand_is_rare"] = 1

        self._encode_brand_segment(brand, raw_model, features)

    def _encode_brand_segment(
        self,
        brand: str,
        raw_model: Any,
        features: dict[str, float | int],
    ) -> None:
        model_key = normalize_key(raw_model)
        rare_brand = int(features.get("brand_is_rare", 0))

        premium = brand in {"Apple", "Microsoft", "LG"} or any(
            token in model_key for token in ["macbook", "surface", "gram"]
        )
        business = brand in {"Dell", "HP", "Lenovo"} or any(
            token in model_key
            for token in ["thinkpad", "latitude", "elitebook", "probook", "precision", "xps"]
        )
        gaming = brand in {"ASUS", "MSI", "Acer", "Gigabyte"} or any(
            token in model_key for token in ["legion", "rog", "tuf", "nitro", "gaming thin"]
        )

        if premium:
            features["brand_segment_premium"] = 1
        elif business:
            features["brand_segment_business"] = 1
        elif gaming:
            features["brand_segment_gaming_value"] = 1
        elif rare_brand:
            features["brand_segment_rare"] = 1

        if rare_brand:
            features["brand_segment_rare"] = 1

    def _encode_model(self, raw_model: Any, features: dict[str, float | int]) -> None:
        text = normalize_text(raw_model)
        if text is None:
            features["no_info_model"] = 1
            return

        canonical = self.model_aliases.get(normalize_key(text))
        features["no_info_model"] = 0
        if canonical is None:
            features["model_is_rare"] = 1
            return

        features[self.final_models[canonical]] = 1

    @staticmethod
    def _encode_storage(raw_storage_type: Any, features: dict[str, float | int]) -> None:
        text = normalize_text(raw_storage_type)
        if text is None:
            features["no_info_storage"] = 1
            return
        low = text.casefold()
        features["storage_ssd"] = int("ssd" in low or "nvme" in low or "solid state" in low)
        features["storage_hdd"] = int("hdd" in low)
        features["no_info_storage"] = int(not features["storage_ssd"] and not features["storage_hdd"])

    @staticmethod
    def _encode_condition(raw_condition: Any, features: dict[str, float | int]) -> None:
        key = normalize_key(raw_condition)
        # Match notebook training semantics: missing/unrecognized condition is
        # imputed to the neutral/used level rather than treated as a new class.
        features["condition_score"] = CONDITION_SCORE_MAP.get(key, 2)

    @staticmethod
    def _encode_warranty(raw_warranty: Any, features: dict[str, float | int]) -> None:
        key = normalize_key(raw_warranty)
        if key in {"active", "manufacturer", "con bao hanh"}:
            features["warranty_active"] = 1
        elif key in {"expired", "het bao hanh"}:
            features["warranty_expired"] = 1
        elif key in {"not activated", "not active", "chua kich hoat"}:
            features["warranty_not_activated"] = 1

    def _encode_cpu(self, raw: dict[str, Any], features: dict[str, float | int]) -> None:
        parsed = parse_cpu(raw)
        features["no_info_cpu_brand"] = parsed.no_info_brand
        features["no_info_cpu_tier"] = parsed.no_info_tier
        features["cpu_tier_encoded"] = parsed.tier_encoded
        features["cpu_family_ord_filled"] = parsed.family_ord
        features["cpu_intel_generation_ord"] = parsed.intel_generation_ord
        features["cpu_amd_generation_ord"] = parsed.amd_generation_ord
        features["cpu_apple_core_spec"] = parsed.apple_core_spec
        features["cpu_suffix_power_ord_filled"] = parsed.suffix_power_ord

        if parsed.brand:
            col = f"cpu_brand_{parsed.brand}"
            if col in self.feature_set:
                features[col] = 1

        if parsed.family_group:
            col = f"cpu_family_group_{parsed.family_group}"
            if col in self.feature_set:
                features[col] = 1

    @staticmethod
    def _encode_gpu(raw_gpu_text: Any, features: dict[str, float | int]) -> None:
        tier, no_info = parse_gpu_tier(raw_gpu_text)
        features["gpu_tier_ord_filled"] = tier
        features["no_info_gpu"] = no_info

    @staticmethod
    def _encode_ram_storage_interactions(
        ram: float,
        storage: float,
        ram_missing: int,
        storage_missing: int,
        features: dict[str, float | int],
    ) -> None:
        features["ram_storage_product_scaled"] = (ram * storage) / (16 * 512)
        features["ram_storage_balance"] = min(ram / 16, storage / 512)

        if ram_missing and storage_missing:
            features["memory_storage_score"] = 0
            return

        features["memory_storage_score"] = 2.0 * math.log1p(ram) + math.log1p(storage)
        features["is_entry_memory_storage"] = int(ram <= 8 and storage <= 256)
        features["is_mid_memory_storage"] = int(ram >= 16 and storage >= 512)
        features["is_premium_memory_storage"] = int(ram >= 32 and storage >= 1024)

    def _align_and_validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        return validate_encoded_output(
            frame,
            feature_names=self.feature_names,
            n_features=self.n_features,
        )


def normalize_text(value: Any) -> str | None:
    """Normalize missing-like values without inventing category tokens."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.casefold() in {"nan", "none", "<na>", "null"}:
        return None
    return text


def normalize_key(value: Any) -> str:
    text = normalize_text(value)
    if text is None:
        return ""
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return " ".join(ascii_text.casefold().replace("_", " ").replace("-", " ").split())


def canonical_brand(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    key = normalize_key(text)
    if key in BRAND_ALIASES:
        return BRAND_ALIASES[key]
    for brand in FINAL_BRANDS:
        if key == normalize_key(brand):
            return brand
    return text.strip()


def _numeric_or_zero(value: Any) -> tuple[float, int]:
    if value is None or pd.isna(value):
        return 0.0, 1
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, 1
    if math.isnan(number):
        return 0.0, 1
    return number, 0
