from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests


from dotenv import load_dotenv


from src.encoder import LaptopFeatureEncoder
from src.encoder.encoder_validation import validate_input_rows
from src.prediction import (
    build_comparison_row,
    build_price_prediction,
    rank_comparison_rows,
    summarize_comparison,
    sweep_ram_by_storage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "final_laptop_price_model_full_data.joblib"

RAW_LAPTOP_FIELDS = [
    "brand",
    "model",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "screen_size_inch",
    "cpu_text",
    "cpu_brand",
    "cpu_family",
    "cpu_generation",
    "cpu_suffix",
    "gpu_text",
    "condition",
    "warranty_status",
]

DETAILED_EXTRACTION_PROMPT = """
You are a strict laptop-specification extraction assistant for a production laptop price prediction system.

Your only job is to convert free-form user text into one JSON object that can be passed to
src.encoder.LaptopFeatureEncoder. The encoder will create all final 86 numeric model features later.
Do not output final engineered features such as brand_Dell, model_Inspiron, storage_ssd,
condition_score, cpu_tier_encoded, gpu_tier_ord_filled, warranty_active, no_info_*,
ram_storage_product_scaled, or any other one-hot/ordinal/interaction feature.

Return exactly these raw encoder input fields and no extra fields:
brand, model, ram_gb, storage_gb, storage_type, screen_size_inch,
cpu_text, cpu_brand, cpu_family, cpu_generation, cpu_suffix,
gpu_text, condition, warranty_status.

General JSON rules:
- Return only valid JSON. No markdown, no comments, no explanation.
- Use JSON null for missing/unstated/unparseable values. Do not use "Other", "orther",
  "N/A", empty strings, NaN, or guessed placeholders.
- Do not invent specs. If the user does not state a value and it cannot be directly inferred from
  a stated model/CPU/storage phrase, use null.
- Numeric fields must be JSON numbers, not strings:
  ram_gb, storage_gb, screen_size_inch, cpu_generation.

Capacity and screen parsing:
- ram_gb: parse RAM capacity into GB. Examples: "8GB RAM" -> 8, "16 gb" -> 16,
  "32GB DDR5" -> 32. Ignore RAM type.
- storage_gb: parse total storage capacity into GB. Use 1TB = 1024GB, 2TB = 2048GB.
  Examples: "512GB SSD" -> 512, "1TB NVMe" -> 1024, "256GB SSD + 1TB HDD" -> 1280.
- storage_type: extract technology only. Use values such as "SSD", "HDD", "NVMe SSD",
  "SSD + HDD". If storage capacity is present but technology is absent, storage_gb is a number
  and storage_type is null.
- screen_size_inch: parse screen size in inches, e.g. "13.3 inch" -> 13.3,
  "14-inch" -> 14, "15.6\"" -> 15.6.

Brand parsing:
- Canonical common brands: ASUS, Acer, Apple, Dell, HP, LG, Lenovo, MSI, Microsoft.
- Preserve rare/unseen brands if explicitly stated, such as Gigabyte, Sony, Toshiba.
- Normalize aliases: "hewlett packard" -> "HP"; "surface" as a brand hint -> "Microsoft";
  "micro star" -> "MSI".
- If no brand is stated, brand must be null.

Model/series parsing:
- Extract the laptop series/model line, not the whole product title when a cleaner series exists.
- Prefer canonical model strings when clearly present:
  Aspire, Elitebook, Elitebook 800, Gaming Thin GF, IdeaPad, Inspiron, Latitude,
  Latitude 14 7000, Latitude E Series, Legion, Legion 5, MacBook Air, MacBook Air M1,
  MacBook Air M2, MacBook Pro, MacBook Pro M1, MacBook Pro M2, Macbook air m4,
  Nitro 5, Pavilion 15, Precision, ProBook, ROG Strix, TUF Gaming, TUF Gaming F15,
  ThinkPad, ThinkPad X1 Carbon, Vivobook 15, Vostro, X Series, XPS 13.
- If user states an unsupported but real model, preserve it as raw model. Do not use "Other".
- If model is not stated, model must be null.

CPU parsing:
- cpu_text: keep the most complete CPU phrase available, e.g. "Intel Core i5-1235U",
  "AMD Ryzen 7 7840HS", "Apple M2", "Snapdragon X Elite".
- cpu_brand: Intel, AMD, Apple, Qualcomm, or null.
- cpu_family examples: Intel Core i3/i5/i7/i9; Intel Core Ultra 5/7/9; Intel Celeron;
  Intel Pentium; Intel N-Series; AMD Ryzen 3/5/7/9; AMD Ryzen AI;
  Apple M1/M2/M3/M4/M5; Snapdragon X Plus; Snapdragon X Elite.
- cpu_generation: infer only when clear. Examples: i5-1235U -> 12, i9-13900HX -> 13,
  Ryzen 5 5500U -> 5, Ryzen 7 7840HS -> 7, Apple M2 -> 2. Otherwise null.
- cpu_suffix: extract U, P, H, HS, HX, HK, HQ, G7, G4, G1, G, K, Y, M when present.

GPU parsing:
- gpu_text: preserve useful GPU text. Examples: "integrated", "Intel Iris Xe", "Intel UHD",
  "AMD Radeon Graphics", "Apple GPU", "GTX 1650", "RTX 3050", "RTX 4060".
- If no GPU information is stated, gpu_text must be null.

Condition parsing:
- Output one of these raw condition values when possible:
  "new", "moi", "unknown", "unknow", "like new", "da mua", "good", "used",
  "da su dung chua sua chua", "repaired", "da sua chua",
  "da su dung qua sua chua", "fair", "poor".
- The encoder condition_score scale is:
  new/moi/unknown/unknow -> 3;
  like new/da mua/98%/99%/used/good/da su dung chua sua chua -> 2;
  repaired/da sua chua/da su dung qua sua chua/fair/poor -> 1.
- Vietnamese mapping:
  "moi", "may moi", "new", "nguyen seal", "chua dung" -> "new";
  "unknown", "unknow", "khong ro tinh trang", "khong biet tinh trang" -> "unknown";
  "nhu moi", "like new", "da mua", "moi mua", "98%", "99%" -> "like new" or "da mua";
  "da su dung chua sua chua", "used, not repaired", "used good" -> "da su dung chua sua chua";
  "da su dung qua sua chua", "repaired", "da sua", "qua sua chua" -> "da su dung qua sua chua";
  poor/fair words map to "poor"/"fair" and are treated as score 1.
- If condition is not stated, use null.

Warranty parsing:
- Output one of: "active", "expired", "not activated",
  "con bao hanh", "het bao hanh", "chua kich hoat", or null.
- Map "con bao hanh", "manufacturer warranty", "warranty active" -> "active".
- Map "het bao hanh", "no warranty", "expired warranty" -> "expired".
- Map "chua kich hoat", "not activated", "not active" -> "not activated".
- If warranty information is absent, use null.
""".strip()

MIN_COMPARE_LAPTOPS = 2
MAX_COMPARE_LAPTOPS = 10

COMPARE_EXTRACTION_PROMPT = f"""
You are a strict laptop comparison extraction assistant for a production laptop price ranking system.

Your job is to read free-form Vietnamese or English text that describes MULTIPLE laptops for sale.
Each laptop has configuration details and an actual asking/selling price.

Return exactly one JSON object with this shape:
{{ "laptops": [ laptop_object, ... ] }}

Each laptop_object MUST contain:
- label: short name from the user text, such as "Laptop A", "Máy 1", "Dell Inspiron", "Option 2"
- actual_price_million_vnd: the listed selling price converted to million VND
- brand, model, ram_gb, storage_gb, storage_type, screen_size_inch,
  cpu_text, cpu_brand, cpu_family, cpu_generation, cpu_suffix,
  gpu_text, condition, warranty_status

Price conversion rules for actual_price_million_vnd:
- 12.000.000 VND -> 12.0
- 14 triệu -> 14.0
- 9.5tr -> 9.5
- Always output million VND as a JSON number, never as a formatted string.

Comparison extraction rules:
- Extract every distinct laptop listing in the text.
- Require at least {MIN_COMPARE_LAPTOPS} laptops and at most {MAX_COMPARE_LAPTOPS}.
- Every laptop MUST include actual_price_million_vnd. If a laptop has no price, omit it.
- Keep laptops in the same order as the user wrote them before ranking happens later.
- Do not merge two laptops into one object.
- Do not output engineered model features or one-hot columns.

Apply the same per-field parsing rules as the single-laptop extractor:
{DETAILED_EXTRACTION_PROMPT}
""".strip()


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / "app/backend/.env")


def raw_laptop_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "brand": {"type": ["string", "null"]},
            "model": {"type": ["string", "null"]},
            "ram_gb": {"type": ["number", "null"]},
            "storage_gb": {"type": ["number", "null"]},
            "storage_type": {"type": ["string", "null"]},
            "screen_size_inch": {"type": ["number", "null"]},
            "cpu_text": {"type": ["string", "null"]},
            "cpu_brand": {"type": ["string", "null"]},
            "cpu_family": {"type": ["string", "null"]},
            "cpu_generation": {"type": ["integer", "null"]},
            "cpu_suffix": {"type": ["string", "null"]},
            "gpu_text": {"type": ["string", "null"]},
            "condition": {"type": ["string", "null"]},
            "warranty_status": {"type": ["string", "null"]},
        },
        "required": RAW_LAPTOP_FIELDS,
    }


def gemini_response_schema() -> dict[str, Any]:
    """Schema format accepted by Gemini generationConfig.responseSchema."""
    string_or_null = {"type": "STRING", "nullable": True}
    number_or_null = {"type": "NUMBER", "nullable": True}
    integer_or_null = {"type": "INTEGER", "nullable": True}

    return {
        "type": "OBJECT",
        "properties": {
            "brand": string_or_null,
            "model": string_or_null,
            "ram_gb": number_or_null,
            "storage_gb": number_or_null,
            "storage_type": string_or_null,
            "screen_size_inch": number_or_null,
            "cpu_text": string_or_null,
            "cpu_brand": string_or_null,
            "cpu_family": string_or_null,
            "cpu_generation": integer_or_null,
            "cpu_suffix": string_or_null,
            "gpu_text": string_or_null,
            "condition": string_or_null,
            "warranty_status": string_or_null,
        },
        "required": RAW_LAPTOP_FIELDS,
        "propertyOrdering": RAW_LAPTOP_FIELDS,
    }


def gemini_compare_response_schema() -> dict[str, Any]:
    string_field = {"type": "STRING", "nullable": True}
    number_field = {"type": "NUMBER", "nullable": True}
    integer_field = {"type": "INTEGER", "nullable": True}

    laptop_schema = {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING"},
            "actual_price_million_vnd": {"type": "NUMBER"},
            "brand": string_field,
            "model": string_field,
            "ram_gb": number_field,
            "storage_gb": number_field,
            "storage_type": string_field,
            "screen_size_inch": number_field,
            "cpu_text": string_field,
            "cpu_brand": string_field,
            "cpu_family": string_field,
            "cpu_generation": integer_field,
            "cpu_suffix": string_field,
            "gpu_text": string_field,
            "condition": string_field,
            "warranty_status": string_field,
        },
        "required": ["label", "actual_price_million_vnd", *RAW_LAPTOP_FIELDS],
        "propertyOrdering": ["label", "actual_price_million_vnd", *RAW_LAPTOP_FIELDS],
    }

    return {
        "type": "OBJECT",
        "properties": {
            "laptops": {
                "type": "ARRAY",
                "items": laptop_schema,
            }
        },
        "required": ["laptops"],
        "propertyOrdering": ["laptops"],
    }


def call_gemini_json_api(user_input: str, api_key: str) -> dict[str, Any]:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "systemInstruction": {
            "parts": [{"text": DETAILED_EXTRACTION_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_input}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": gemini_response_schema(),
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def call_gemini_compare_api(user_input: str, api_key: str) -> dict[str, Any]:
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "systemInstruction": {
            "parts": [{"text": COMPARE_EXTRACTION_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_input}],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": gemini_compare_response_schema(),
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


def gemini_content_text(response_data: dict[str, Any]) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response did not include any candidates.")

    parts = candidates[0].get("content", {}).get("parts") or []
    text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
    raw_content = "".join(text_parts).strip()
    if not raw_content:
        raise ValueError("Gemini response candidate did not include JSON text.")
    return raw_content


def parse_llm_json_content(raw_content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(1))

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def parse_llm_compare_content(raw_content: str) -> list[dict[str, Any]]:
    parsed = parse_llm_json_content(raw_content)
    laptops = parsed.get("laptops")
    if not isinstance(laptops, list):
        raise ValueError("LLM compare output must include a laptops array.")
    if len(laptops) < MIN_COMPARE_LAPTOPS:
        raise ValueError(f"Cần ít nhất {MIN_COMPARE_LAPTOPS} laptop để so sánh.")
    if len(laptops) > MAX_COMPARE_LAPTOPS:
        raise ValueError(f"Chỉ hỗ trợ tối đa {MAX_COMPARE_LAPTOPS} laptop mỗi lần so sánh.")

    normalized: list[dict[str, Any]] = []
    for index, laptop in enumerate(laptops, start=1):
        if not isinstance(laptop, dict):
            raise TypeError(f"Laptop #{index} must be a JSON object.")

        label = str(laptop.get("label") or f"Laptop {index}").strip() or f"Laptop {index}"
        actual_price = laptop.get("actual_price_million_vnd")
        if actual_price is None:
            raise ValueError(f"{label} thiếu actual_price_million_vnd.")
        try:
            actual_price_value = float(actual_price)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} có actual_price_million_vnd không hợp lệ.") from exc
        if actual_price_value <= 0:
            raise ValueError(f"{label} phải có giá bán lớn hơn 0.")

        raw_features = {field: laptop.get(field) for field in RAW_LAPTOP_FIELDS}
        validate_encoder_ready_json(raw_features)
        normalized.append(
            {
                "label": label,
                "actual_price_million_vnd": actual_price_value,
                "raw_features": raw_features,
            }
        )
    return normalized


def validate_encoder_ready_json(raw_features: dict[str, Any]) -> None:
    if not isinstance(raw_features, dict):
        raise TypeError(f"LLM output must be a dict, got {type(raw_features).__name__}.")

    missing_fields = [field for field in RAW_LAPTOP_FIELDS if field not in raw_features]
    extra_fields = [field for field in raw_features if field not in RAW_LAPTOP_FIELDS]
    if missing_fields:
        raise ValueError(f"LLM output is missing required encoder fields: {missing_fields}")
    if extra_fields:
        raise ValueError(f"LLM output contains unsupported encoder fields: {extra_fields}")

    validate_input_rows([raw_features])


def extract_laptop_features(user_input: str, api_key: str) -> dict[str, Any]:
    response_data = call_gemini_json_api(user_input, api_key)
    raw_content = gemini_content_text(response_data)
    raw_features = parse_llm_json_content(raw_content)
    validate_encoder_ready_json(raw_features)
    return raw_features


def extract_compare_laptops(user_input: str, api_key: str) -> list[dict[str, Any]]:
    response_data = call_gemini_compare_api(user_input, api_key)
    raw_content = gemini_content_text(response_data)
    return parse_llm_compare_content(raw_content)


def normalize_manual_features(raw_features: dict[str, Any]) -> dict[str, Any]:
    return {field: raw_features.get(field) for field in RAW_LAPTOP_FIELDS}


def encode_features(raw_features: dict[str, Any]) -> tuple[dict[str, float | int], list[str]]:
    encoder = LaptopFeatureEncoder()
    encoded_frame = encoder.encode_one(raw_features)
    row = encoded_frame.iloc[0].to_dict()
    active = [
        name
        for name, value in row.items()
        if value not in (0, 0.0) and not name.endswith("_missing")
    ]
    return row, active[:32]


def predict_price(raw_features: dict[str, Any]) -> float:
    import joblib

    encoder = LaptopFeatureEncoder()
    model = joblib.load(MODEL_PATH)
    encoded = encoder.encode_one(raw_features)
    return float(model.predict(encoded)[0])


def build_prediction_response(
    raw_features: dict[str, Any],
    encoded_features: dict[str, float | int],
    active_features: list[str],
    validation: list[str],
) -> dict[str, Any]:
    prediction = predict_price(raw_features)
    price_payload = build_price_prediction(prediction, raw_features, encoded_features)
    return {
        "raw_features": raw_features,
        "encoded_features": encoded_features,
        "active_features": active_features,
        **price_payload,
        "validation": validation,
    }


def predict_from_text(description: str) -> dict[str, Any]:
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    raw_features = extract_laptop_features(description, api_key)
    encoded_features, active_features = encode_features(raw_features)
    return build_prediction_response(
        raw_features,
        encoded_features,
        active_features,
        validation=[
            "Đã trích xuất thông tin cấu hình từ mô tả.",
            "Dữ liệu đã sẵn sàng để ước tính giá.",
        ],
    )


def predict_from_raw(raw_features: dict[str, Any]) -> dict[str, Any]:
    raw_features = normalize_manual_features(raw_features)
    validate_encoder_ready_json(raw_features)
    encoded_features, active_features = encode_features(raw_features)
    return build_prediction_response(
        raw_features,
        encoded_features,
        active_features,
        validation=[
            "Đã kiểm tra thông tin nhập từ form.",
            "Dữ liệu đã sẵn sàng để ước tính giá.",
        ],
    )


def sweep_from_raw(raw_features: dict[str, Any]) -> dict[str, Any]:
    raw_features = normalize_manual_features(raw_features)
    validate_encoder_ready_json(raw_features)
    result = sweep_ram_by_storage(
        raw_features,
        predict_fn=predict_price,
        encode_fn=encode_features,
    )
    result["validation"] = [
        "Giữ nguyên cấu hình gốc, chỉ thay đổi RAM và dung lượng ổ cứng.",
        f"Đã dự đoán {len(result['ram_values']) * len(result['storage_values'])} cấu hình.",
    ]
    return result


def compare_from_text(description: str) -> dict[str, Any]:
    load_environment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    laptops = extract_compare_laptops(description, api_key)
    comparison_rows: list[dict[str, Any]] = []

    for laptop in laptops:
        raw_features = laptop["raw_features"]
        encoded_features, _active_features = encode_features(raw_features)
        prediction = predict_price(raw_features)
        prediction_payload = build_price_prediction(prediction, raw_features, encoded_features)
        prediction_payload["raw_features"] = raw_features
        comparison_rows.append(
            build_comparison_row(
                laptop["label"],
                laptop["actual_price_million_vnd"],
                prediction_payload,
                rank=0,
            )
        )

    rankings = rank_comparison_rows(comparison_rows)
    summary = summarize_comparison(rankings)

    return {
        "task": "compare",
        "laptop_count": len(rankings),
        "best_pick": summary["best_pick"],
        "summary": summary["summary"],
        "rankings": rankings,
        "validation": [
            f"Đã phân tích {len(rankings)} laptop kèm giá bán.",
            "Mỗi máy đã được ước tính giá tham khảo.",
            "Xếp hạng theo mức chênh lệch giữa giá dự đoán và giá rao bán.",
        ],
    }
