from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: dict[str, Any] = {
    "raw_path": PROJECT_ROOT / "data" / "raw" / "chotot_laptop_data.csv",
    "issues_path": PROJECT_ROOT / "docs" / "chotot_issues_list.csv",
    "output_dir": PROJECT_ROOT / "data" / "intern",
    "cleaned_path": PROJECT_ROOT / "data" / "intern" / "chotot_cleaned.csv",
    "report_path": PROJECT_ROOT / "docs" / "chotot_cleaning_report.csv",
    "issue_action_plan_path": PROJECT_ROOT / "docs" / "chotot_issue_action_plan.csv",
    "log_path": PROJECT_ROOT / "docs" / "chotot_cleaning_log.json",
    "price_min": 1_000_000,
    "price_max": 60_000_000,
    "rare_count_threshold": 30,
    "valid_ram_levels": {2, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64},
    "title_valid_ram_levels": {2, 4, 6, 8, 12, 16, 24, 32, 36, 48, 64, 96, 128},
    "screen_min": 8,
    "screen_max": 22,
    "price_bins": [0, 5_000_000, 10_000_000, 20_000_000, np.inf],
    "price_labels": ["low", "mid", "high", "premium"],
    "drop_columns": ["Thông tin sử dụng"],
    "brand_aliases": {
        "ASUS": ["asus", "vivobook", "zenbook", "rog", "tuf"],
        "Dell": ["dell", "latitude", "inspiron", "xps", "vostro", "precision"],
        "HP": ["hp", "hewlett", "elitebook", "probook", "envy", "pavilion"],
        "Lenovo": ["lenovo", "thinkpad", "ideapad", "legion", "yoga", "loq"],
        "Acer": ["acer", "aspire", "swift", "nitro", "predator"],
        "MSI": ["msi", "stealth", "modern", "prestige", "katana"],
        "Apple": ["apple", "macbook", "mac book"],
        "Microsoft": ["microsoft", "surface"],
        "Razer": ["razer", "blade"],
        "Gigabyte": ["gigabyte", "aorus"],
        "LG": ["lg", "gram"],
        "Huawei": ["huawei", "matebook"],
        "Xiaomi": ["xiaomi", "redmibook", "mi notebook"],
        "Samsung": ["samsung", "galaxy book"],
        "Sony": ["sony", "vaio"],
        "Toshiba": ["toshiba", "dynabook"],
        "Panasonic": ["panasonic", "toughbook"],
        "Honor": ["honor", "magicbook"],
    },
    "model_recovery_keywords": {
        "Latitude": ["latitude"],
        "ThinkPad": ["thinkpad"],
        "MacBook Pro": ["macbook pro", "mac book pro"],
        "MacBook Air": ["macbook air", "mac book air"],
        "Inspiron": ["inspiron"],
        "Precision": ["precision"],
        "EliteBook": ["elitebook"],
        "ProBook": ["probook"],
        "Pavilion": ["pavilion"],
        "XPS": ["xps"],
        "Legion": ["legion"],
        "IdeaPad": ["ideapad"],
        "Yoga": ["yoga"],
        "ROG": ["rog"],
        "TUF Gaming": ["tuf"],
        "Nitro": ["nitro"],
        "Aspire": ["aspire"],
        "Swift": ["swift"],
        "Surface": ["surface"],
        "Victus": ["victus"],
        "Omen": ["omen"],
        "VivoBook": ["vivobook"],
        "ZenBook": ["zenbook"],
        "LOQ": ["loq"],
    },
}


@dataclass
class CleaningArtifacts:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    issues: pd.DataFrame
    issue_action_plan: pd.DataFrame
    report: pd.DataFrame
    log: dict[str, Any]
    validation: dict[str, bool]


def load_inputs(config: dict[str, Any] = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(config["raw_path"])
    issues = pd.read_csv(config["issues_path"])
    print("Raw schema:")
    print(raw.dtypes)
    print(f"\nRaw shape: {raw.shape[0]:,} rows x {raw.shape[1]:,} columns")
    print("\nSample records:")
    print(raw.head(5).to_string())
    print(f"\nFull duplicate rows: {raw.duplicated().sum():,}")
    if "url" in raw.columns:
        print(f"Duplicate listing URLs: {raw['url'].duplicated().sum():,}")
    print(f"\nIssues loaded: {len(issues):,}")
    return raw, issues


def _normalize_ascii_key(text: Any) -> str:
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")


def infer_issue_group(issue_key: str, description: str, action: str) -> str:
    text = " ".join([issue_key, description, action]).lower()
    if "drop" in text or "leakage" in text or "constant" in text:
        return "columns_to_drop_keep"
    if "missing" in text or "thiếu" in text:
        return "missing_null_empty_values"
    if "duplicate" in text or "trùng" in text:
        return "duplicate_records_listings"
    if "outlier" in text or "bất thường" in text or "ngưỡng" in text:
        return "outliers"
    if "price" in text or "giá" in text:
        return "price_related_issues"
    if "title" in text or "text" in text or "regex" in text:
        return "text_normalization_specs_parsing"
    if "brand" in text or "hãng" in text:
        return "brand_model_extraction_issues"
    if "model" in text or "dòng" in text:
        return "brand_model_extraction_issues"
    if "rare" in text or "hiếm" in text or "low-sample" in text or "tần suất thấp" in text:
        return "low_count_categories_interactions"
    if "category" in text or "nhóm" in text or "gộp" in text:
        return "inconsistent_categories"
    return "invalid_values"


def infer_affected_column(issue_key: str) -> str:
    mapping = {
        "price": "price",
        "title": "title",
        "brand": "Hãng",
        "model": "Dòng máy",
        "dong_khac": "Dòng máy",
        "condition": "Tình trạng",
        "warranty": "Chính sách bảo hành",
        "screen": "Kích cỡ màn hình",
        "cpu": "Bộ vi xử lý",
        "ram": "RAM",
        "gpu": "Card màn hình",
        "storage": "Ổ cứng/Loại ổ cứng",
        "usage_info": "Thông tin sử dụng",
    }
    for prefix, column in mapping.items():
        if issue_key.startswith(prefix) or prefix in issue_key:
            return column
    return ""


def infer_severity(issue_key: str, description: str, action: str) -> str:
    text = " ".join([issue_key, description, action]).lower()
    high_terms = ["drop", "safe_to_drop", "leakage", "outlier", "missing", "suspicious", "không hợp lệ"]
    medium_terms = ["imbalance", "bias", "long-tail", "rare", "inconsistency", "mismatch", "không ổn định"]
    if any(term in text for term in high_terms):
        return "high"
    if any(term in text for term in medium_terms):
        return "medium"
    return "low"


def needs_note_lookup(issue_key: str, description: str, action: str) -> str:
    text = " ".join([issue_key, description, action]).lower()
    lookup_terms = ["threshold", "ngưỡng", "regex", "mapping", "impute", "recover", "outlier", "rare", "hiếm"]
    return "yes" if any(term in text for term in lookup_terms) else "no"


def decide_final_action(issue_key: str, proposed_action: str) -> tuple[str, str]:
    text = f"{issue_key} {proposed_action}".lower()
    if "price_market_bias" in issue_key:
        return "Document local-market scope; no row-level cleaning.", "needs_manual_review"
    if "ordinal_encoding" in issue_key or "encoding" in text or "modeling" in text or "mô hình" in text:
        return "Leave for modeling; document as modeling decision.", "needs_manual_review"
    if issue_key.startswith("usage_info"):
        return "Drop `Thông tin sử dụng` because it is constant and marked safe to drop.", "resolved"
    if "outlier" in text or "bất thường" in text:
        return "Create explicit flags; do not drop uncertain rows.", "partially_resolved"
    if "missing" in text or "thiếu" in text:
        return "Create missing indicators and conservative fallback extraction where supported.", "partially_resolved"
    if "recover" in text or "extract" in text or "parse" in text or "regex" in text:
        return "Apply conservative parser; never overwrite structured non-null values.", "partially_resolved"
    if "rare" in text or "hiếm" in text or "gộp" in text:
        return "Create grouped category columns with rare values mapped to Other; keep raw columns.", "resolved"
    if "giữ" in text or "use" in text or "sử dụng" in text:
        return "Preserve as feature or helper signal in cleaned dataset.", "resolved"
    return "Documented; no deterministic cleaning required.", "needs_manual_review"


def build_issue_action_plan(issues: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in issues.iterrows():
        issue_key = str(row["Column"])
        desc = str(row["Issue (với số liệu)"])
        proposed = str(row["Proposed Action"])
        final_action, status = decide_final_action(issue_key, proposed)
        rows.append(
            {
                "issue_key": issue_key,
                "severity": infer_severity(issue_key, desc, proposed),
                "affected_column": infer_affected_column(issue_key),
                "issue_group": infer_issue_group(issue_key, desc, proposed),
                "issue_description": desc,
                "proposed_action": proposed,
                "need_note_lookup": needs_note_lookup(issue_key, desc, proposed),
                "final_action": final_action,
                "resolution_status": status,
            }
        )
    return pd.DataFrame(rows)


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include="object").columns:
        s = out[col].astype("string").str.replace("\n", " ", regex=False)
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        out[col] = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return out


def parse_price_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(value))
    return float(digits) if digits else np.nan


def clean_price(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["_price"] = out["price"].apply(parse_price_value)
    out["is_price_missing"] = out["_price"].isna()
    out["is_price_outlier"] = out["_price"].notna() & (
        (out["_price"] < config["price_min"]) | (out["_price"] > config["price_max"])
    )
    out["price_segment"] = pd.cut(
        out["_price"],
        bins=config["price_bins"],
        labels=config["price_labels"],
        include_lowest=True,
    ).astype("string")
    return out


def clean_location_or_origin(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["origin_clean"] = out["Xuất xứ"].fillna("Unknown").astype("string")
    out["origin_clean"] = out["origin_clean"].replace({"Đang cập nhật": "Unknown", "Nước khác": "Other"})
    out["origin_missing_or_unknown"] = out["origin_clean"].eq("Unknown")
    return out


def normalize_brand_value(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    aliases = {
        "hp": "HP",
        "hewlett packard": "HP",
        "apple": "Apple",
        "macbook": "Apple",
        "dell": "Dell",
        "lenovo": "Lenovo",
        "asus": "ASUS",
        "acer": "Acer",
        "msi": "MSI",
        "microsoft": "Microsoft",
    }
    return aliases.get(text.lower(), text if text else "Unknown")


def recover_brand_from_title(title: Any, config: dict[str, Any] = CONFIG) -> str:
    if pd.isna(title):
        return "Unknown"
    lowered = str(title).lower()
    for brand, aliases in config["brand_aliases"].items():
        if any(alias in lowered for alias in aliases):
            return brand
    return "Unknown"


def clean_model_text(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else "Unknown"


def recover_model_from_title(title: Any, config: dict[str, Any] = CONFIG) -> str:
    if pd.isna(title):
        return "Unknown"
    lowered = str(title).lower()
    for model, keywords in config["model_recovery_keywords"].items():
        if any(keyword in lowered for keyword in keywords):
            return model
    return "Unknown"


def extract_brand_model(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["brand_clean"] = out["Hãng"].apply(normalize_brand_value)
    out["brand_from_title"] = out["title"].apply(lambda x: recover_brand_from_title(x, config))
    out["brand_mismatch_title"] = out["brand_from_title"].ne("Unknown") & out["brand_clean"].ne(out["brand_from_title"])
    out["model_clean"] = out["Dòng máy"].apply(clean_model_text)
    out["model_from_title"] = out["title"].apply(lambda x: recover_model_from_title(x, config))
    out["model_recovered"] = np.where(
        out["model_clean"].eq("Dòng Khác") & out["model_from_title"].ne("Unknown"),
        out["model_from_title"],
        out["model_clean"],
    )
    return out


def parse_ram_value(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    nums = re.findall(r"\d+", text)
    if not nums:
        return np.nan
    if "<" in text:
        return 1.0
    if ">" in text:
        return float(nums[0]) + 1
    return float(nums[0])


def parse_storage_gb(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).upper().replace(",", ".")
    nums = re.findall(r"[\d.]+", text)
    if not nums:
        return np.nan
    val = float(nums[0]) * 1024 if "TB" in text else float(nums[0])
    if "<" in text:
        return max(val - 1, 0)
    if ">" in text:
        return val + 1
    return val


def extract_ram_from_title(title: Any, config: dict[str, Any] = CONFIG) -> float:
    if pd.isna(title):
        return np.nan
    text = str(title).lower()
    match = re.search(r"\bram\s*(\d{1,3})\s?gb\b|\b(\d{1,3})\s?gb\s*ram\b", text)
    if match:
        return float(match.group(1) or match.group(2))
    pair = re.search(r"\b(2|4|6|8|12|16|24|32|36|48|64)\s*/\s*(128|256|512|1024)\b", text)
    if pair:
        return float(pair.group(1))
    vals = [int(x) for x in re.findall(r"\b(\d{1,3})\s?g(?:b)?\b", text) if int(x) in config["title_valid_ram_levels"]]
    return float(vals[0]) if vals else np.nan


def extract_storage_from_title(title: Any) -> float:
    if pd.isna(title):
        return np.nan
    text = str(title).lower().replace(",", ".")
    pair = re.search(r"\b(?:2|4|6|8|12|16|24|32|36|48|64)\s*/\s*(128|256|512|1024)\b", text)
    if pair:
        return float(pair.group(1))
    match = re.search(r"\b(\d+(?:\.\d+)?)\s?tb\b", text)
    if match:
        return float(match.group(1)) * 1024
    vals = re.findall(r"\b(128|250|256|320|480|500|512|640|750|1000|1024)\s?gb\b", text)
    return float(vals[-1]) if vals else np.nan


def extract_cpu_from_title(title: Any) -> str:
    if pd.isna(title):
        return "Unknown"
    text = str(title).lower()
    for pat in [
        r"\b(i[3579])[-\s]?(\d{4,5}[a-z]{0,2})\b",
        r"\bintel core\s?(i[3579])\b",
        r"\bryzen\s?([3579])\b",
        r"\b(m[12345])\s?(pro|max|ultra)?\b",
        r"\bcore ultra\s?([579])\b",
        r"\bn\d{3,5}\b",
        r"\bceleron\b",
        r"\bpentium\b",
    ]:
        match = re.search(pat, text)
        if match:
            return match.group(0).strip()
    return "Unknown"


def extract_screen_from_title(title: Any, config: dict[str, Any] = CONFIG) -> float:
    if pd.isna(title):
        return np.nan
    text = str(title).lower().replace(",", ".")
    for pat in [r'(?<!\d)(\d{1,2}(?:\.\d)?)\s*(?:inch|inches)\b', r'(?<!\d)(\d{1,2}(?:\.\d)?)\s*["”]']:
        match = re.search(pat, text)
        if match:
            val = float(match.group(1))
            if config["screen_min"] <= val <= config["screen_max"]:
                return val
    return np.nan


def parse_screen_bucket(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(value))]
    if not nums:
        return np.nan
    if "<" in str(value):
        return nums[0] - 0.1
    if ">" in str(value):
        return nums[0]
    return float(np.mean(nums[:2])) if len(nums) >= 2 else nums[0]


def map_cpu_brand(value: Any, title_cpu: Any = "Unknown") -> str:
    text = " ".join(str(x) for x in [value, title_cpu] if not pd.isna(x)).lower()
    if "ryzen" in text or re.search(r"\bamd\b", text):
        return "AMD"
    if re.search(r"\bm[1-5]\b", text) or "apple" in text:
        return "Apple"
    if "intel" in text or re.search(r"\bi[3579]\b", text) or "celeron" in text or "pentium" in text or "core ultra" in text:
        return "Intel"
    return "Unknown"


def map_cpu_tier(value: Any, title_cpu: Any = "Unknown") -> str:
    text = " ".join(str(x) for x in [value, title_cpu] if not pd.isna(x)).lower()
    rules = [
        (r"core ultra\s*9|i9|ryzen\s*9|m[1-5]\s*(max|ultra)", "High"),
        (r"core ultra\s*7|i7|ryzen\s*7|m[1-5]\s*pro", "Upper-mid"),
        (r"core ultra\s*5|i5|ryzen\s*5|\bm[1-5]\b", "Mid"),
        (r"i3|ryzen\s*3", "Entry"),
        (r"celeron|pentium|athlon|\bn\d{3,5}\b", "Low"),
    ]
    for pat, tier in rules:
        if re.search(pat, text):
            return tier
    if pd.isna(value) or str(value).strip().lower() in {"", "nan"}:
        return "Missing"
    return "Other"


def gpu_tier(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value)
    if re.search(r"RTX\s?[34]\d{3}", text, re.I):
        return "Dedicated - High (RTX 30/40)"
    if re.search(r"RTX\s?2\d{3}|GTX\s?16[56]0", text, re.I):
        return "Dedicated - Mid (RTX 20 / GTX 16)"
    if re.search(r"GTX|Quadro|GeForce|NVIDIA|Radeon\s?RX|RX\s?\d{3,4}|Radeon\s?Pro", text, re.I):
        return "Dedicated - Other/Entry"
    if re.search(r"Intel|Iris|UHD|HD\s?Graphics|integrated|tích hợp|Onboard", text, re.I):
        return "Integrated - Intel"
    if re.search(r"Radeon|AMD", text, re.I) and not re.search(r"RX\s?\d", text, re.I):
        return "Integrated - AMD Radeon"
    return "Unclear"


def parse_laptop_specs(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["_ram_gb"] = out["RAM"].apply(parse_ram_value)
    out["_storage_gb"] = out["Ổ cứng"].apply(parse_storage_gb)
    out["_screen_size_inch"] = out["Kích cỡ màn hình"].apply(parse_screen_bucket)
    out["_title_ram_gb"] = out["title"].apply(lambda x: extract_ram_from_title(x, config))
    out["_title_storage_gb"] = out["title"].apply(extract_storage_from_title)
    out["_title_cpu"] = out["title"].apply(extract_cpu_from_title)
    out["_title_screen_size_inch"] = out["title"].apply(lambda x: extract_screen_from_title(x, config))
    out["_ram_gb_filled"] = out["_ram_gb"].fillna(out["_title_ram_gb"])
    out["_storage_gb_filled"] = out["_storage_gb"].fillna(out["_title_storage_gb"])
    out["_screen_size_inch_filled"] = out["_screen_size_inch"].fillna(out["_title_screen_size_inch"])
    out["ram_missing"] = out["_ram_gb"].isna()
    out["storage_capacity_missing"] = out["_storage_gb"].isna()
    out["screen_missing"] = out["_screen_size_inch"].isna()
    out["ram_suspicious"] = out["_ram_gb_filled"].notna() & (
        ~out["_ram_gb_filled"].isin(config["valid_ram_levels"])
        | (out["_ram_gb_filled"] < 2)
        | (out["_ram_gb_filled"] > 64)
    )
    out["storage_missing_all"] = out["Ổ cứng"].isna() & out["Loại ổ cứng"].isna()
    out["cpu_missing"] = out["Bộ vi xử lý"].isna()
    out["cpu_brand"] = [map_cpu_brand(a, b) for a, b in zip(out["Bộ vi xử lý"], out["_title_cpu"])]
    out["cpu_tier"] = [map_cpu_tier(a, b) for a, b in zip(out["Bộ vi xử lý"], out["_title_cpu"])]
    out["gpu_missing"] = out["Card màn hình"].isna()
    out["gpu_tier"] = out["Card màn hình"].apply(gpu_tier)
    title_lower = out["title"].fillna("").str.lower()
    model_lower = out["Dòng máy"].fillna("").str.lower()
    gpu_kw = r"\b(?:rtx|gtx|nvidia|geforce|quadro|mx\d+|vga|radeon|rx\s?\d+|iris xe)\b"
    gaming_kw = r"\b(?:gaming|rog|tuf|nitro|legion|victus|omen|loq|predator|msi|g15|g16)\b"
    out["potential_dedicated_gpu"] = out["gpu_missing"] & (
        title_lower.str.contains(gpu_kw, regex=True, na=False)
        | title_lower.str.contains(gaming_kw, regex=True, na=False)
        | model_lower.str.contains(gaming_kw, regex=True, na=False)
    )
    return out


def parse_warranty_months(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    if "hết" in text:
        return 0.0
    if ">12" in text:
        return 13.0
    nums = [int(x) for x in re.findall(r"\d+", text)]
    return float(max(nums)) if nums else np.nan


def map_warranty_status(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).lower()
    if "hết" in text:
        return "Expired"
    if "hãng" in text:
        return "Manufacturer"
    if "còn" in text or re.search(r"\d|>", text):
        return "Active"
    return "Unknown"


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["warranty_months"] = out["Chính sách bảo hành"].apply(parse_warranty_months)
    out["warranty_status"] = out["Chính sách bảo hành"].apply(map_warranty_status)
    out["has_warranty"] = out["warranty_status"].isin(["Manufacturer", "Active"])
    out["storage_type_clean"] = out["Loại ổ cứng"].fillna("Unknown")
    title_lower = out["title"].fillna("").str.lower()
    out["was_storage_type_imputed"] = False
    missing_type = out["storage_type_clean"].eq("Unknown")
    title_ssd = title_lower.str.contains(r"\bssd\b", regex=True, na=False)
    title_hdd = title_lower.str.contains(r"\bhdd\b", regex=True, na=False)
    out.loc[missing_type & title_ssd, "storage_type_clean"] = "SSD"
    out.loc[missing_type & title_hdd, "storage_type_clean"] = "HDD"
    out.loc[missing_type & (title_ssd | title_hdd), "was_storage_type_imputed"] = True
    out["condition_clean"] = out["Tình trạng"].fillna("Unknown")
    out["new_low_price"] = out["condition_clean"].eq("Mới") & out["_price"].notna() & (out["_price"] < 5_000_000)
    repair_kw = r"\b(?:lỗi|loi|sửa|sua|main|vỡ|vo|bể|be|xác|xac|hỏng|hong|mất nguồn|mat nguon)\b"
    out["repair_mismatch"] = title_lower.str.contains(repair_kw, regex=True, na=False) & ~out[
        "condition_clean"
    ].str.contains("qua sửa chữa", case=False, na=False)
    out["condition_warranty_inconsistent"] = out["condition_clean"].eq("Mới") & out["warranty_status"].eq("Expired")
    for col in ["brand_clean", "model_clean", "model_recovered", "cpu_brand", "cpu_tier", "gpu_tier", "origin_clean"]:
        out[col] = out[col].fillna("Unknown")
    return out


def group_low_count(series: pd.Series, threshold: int, other_label: str = "Other") -> pd.Series:
    counts = series.value_counts(dropna=False)
    return series.where(~series.isin(set(counts[counts < threshold].index)), other_label)


def handle_low_count_categories(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    threshold = config["rare_count_threshold"]
    out["brand_grouped"] = group_low_count(out["brand_clean"], threshold)
    model_base = out["model_recovered"].fillna(out["model_clean"])
    out["model_grouped"] = group_low_count(model_base, threshold)
    out["model_grouped"] = np.where(out["model_clean"].eq("Dòng Khác"), "Dòng Khác", out["model_grouped"])
    out["brand_is_rare"] = out["brand_grouped"].eq("Other")
    out["model_is_rare"] = out["model_grouped"].eq("Other")
    return out


def drop_leakage_or_bad_columns(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    return df.drop(columns=[c for c in config["drop_columns"] if c in df.columns])


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    out = df.drop_duplicates().copy()
    full_removed = before - len(out)
    url_removed = 0
    if "url" in out.columns:
        before_url = len(out)
        out = out.drop_duplicates(subset=["url"], keep="first").copy()
        url_removed = before_url - len(out)
    return out, {"full_duplicate_removed": int(full_removed), "url_duplicate_removed": int(url_removed)}


def distribution_summary(df: pd.DataFrame, column: str, top_n: int = 10) -> list[dict[str, Any]]:
    if column not in df.columns:
        return []
    counts = df[column].value_counts(dropna=False).head(top_n)
    return [{"value": str(k), "count": int(v)} for k, v in counts.items()]


def numeric_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    s = pd.to_numeric(df[column], errors="coerce")
    desc = s.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {str(k): (float(v) if pd.notna(v) else None) for k, v in desc.items()}


def build_cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame, issue_action_plan: pd.DataFrame, duplicate_log: dict[str, int]) -> pd.DataFrame:
    rows = [
        {"section": "shape", "metric": "row_count_before", "value": len(raw)},
        {"section": "shape", "metric": "row_count_after", "value": len(cleaned)},
        {"section": "shape", "metric": "column_count_before", "value": raw.shape[1]},
        {"section": "shape", "metric": "column_count_after", "value": cleaned.shape[1]},
        {"section": "duplicates", "metric": "full_duplicate_removed", "value": duplicate_log["full_duplicate_removed"]},
        {"section": "duplicates", "metric": "url_duplicate_removed", "value": duplicate_log["url_duplicate_removed"]},
        {"section": "price", "metric": "price_missing_flagged", "value": int(cleaned["is_price_missing"].sum())},
        {"section": "price", "metric": "price_outlier_flagged", "value": int(cleaned["is_price_outlier"].sum())},
        {"section": "specs", "metric": "ram_suspicious_flagged", "value": int(cleaned["ram_suspicious"].sum())},
        {"section": "specs", "metric": "storage_missing_all_flagged", "value": int(cleaned["storage_missing_all"].sum())},
        {"section": "specs", "metric": "screen_missing_flagged", "value": int(cleaned["screen_missing"].sum())},
        {"section": "specs", "metric": "potential_dedicated_gpu_flagged", "value": int(cleaned["potential_dedicated_gpu"].sum())},
        {"section": "condition", "metric": "new_low_price_flagged", "value": int(cleaned["new_low_price"].sum())},
        {"section": "condition", "metric": "repair_mismatch_flagged", "value": int(cleaned["repair_mismatch"].sum())},
    ]
    for col in sorted(set(raw.columns).union(cleaned.columns)):
        rows.append(
            {
                "section": "missing_before_after",
                "metric": col,
                "value": json.dumps(
                    {
                        "before": int(raw[col].isna().sum()) if col in raw.columns else None,
                        "after": int(cleaned[col].isna().sum()) if col in cleaned.columns else None,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    for status, count in issue_action_plan["resolution_status"].value_counts().items():
        rows.append({"section": "issue_status", "metric": status, "value": int(count)})
    for col in ["brand_grouped", "_ram_gb_filled", "_storage_gb_filled", "cpu_tier", "condition_clean", "origin_clean"]:
        rows.append({"section": "distribution", "metric": col, "value": json.dumps(distribution_summary(cleaned, col), ensure_ascii=False)})
    rows.append({"section": "distribution", "metric": "_price", "value": json.dumps(numeric_summary(cleaned, "_price"))})
    return pd.DataFrame(rows)


def validate_cleaned_data(cleaned: pd.DataFrame, issue_action_plan: pd.DataFrame) -> dict[str, bool]:
    validations = {
        "no_duplicate_url": cleaned["url"].duplicated().sum() == 0,
        "price_numeric": pd.api.types.is_numeric_dtype(cleaned["_price"]),
        "invalid_price_flagged": bool((cleaned["_price"].isna() | (cleaned["_price"] > 0) | cleaned["is_price_missing"] | cleaned["is_price_outlier"]).all()),
        "ram_numeric": pd.api.types.is_numeric_dtype(cleaned["_ram_gb"]),
        "storage_numeric": pd.api.types.is_numeric_dtype(cleaned["_storage_gb"]),
        "screen_numeric": pd.api.types.is_numeric_dtype(cleaned["_screen_size_inch"]),
        "brand_normalized": cleaned["brand_clean"].notna().all(),
        "model_normalized": cleaned["model_clean"].notna().all(),
        "usage_info_dropped": "Thông tin sử dụng" not in cleaned.columns,
        "all_issues_have_actions": issue_action_plan["final_action"].notna().all() and issue_action_plan["resolution_status"].notna().all(),
    }
    failed = [name for name, ok in validations.items() if not ok]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")
    return validations


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(make_json_safe(v) for v in value)
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_outputs(cleaned: pd.DataFrame, report: pd.DataFrame, issue_action_plan: pd.DataFrame, log: dict[str, Any], config: dict[str, Any] = CONFIG) -> None:
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config["cleaned_path"], index=False, encoding="utf-8-sig")
    report.to_csv(config["report_path"], index=False, encoding="utf-8-sig")
    issue_action_plan.to_csv(config["issue_action_plan_path"], index=False, encoding="utf-8-sig")
    config["log_path"].write_text(json.dumps(make_json_safe(log), ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: dict[str, Any] = CONFIG) -> CleaningArtifacts:
    raw, issues = load_inputs(config)
    issue_action_plan = build_issue_action_plan(issues)
    duplicate_free, duplicate_log = remove_duplicates(raw)
    cleaned = (
        duplicate_free.pipe(normalize_text_columns)
        .pipe(clean_price, config=config)
        .pipe(clean_location_or_origin)
        .pipe(extract_brand_model, config=config)
        .pipe(parse_laptop_specs, config=config)
        .pipe(handle_missing_values)
        .pipe(handle_low_count_categories, config=config)
        .pipe(drop_leakage_or_bad_columns, config=config)
    )
    report = build_cleaning_report(raw, cleaned, issue_action_plan, duplicate_log)
    validation = validate_cleaned_data(cleaned, issue_action_plan)
    log = {
        "config": make_json_safe(config),
        "raw_shape": list(raw.shape),
        "cleaned_shape": list(cleaned.shape),
        "duplicate_log": duplicate_log,
        "validation": validation,
        "issue_status_counts": issue_action_plan["resolution_status"].value_counts().to_dict(),
        "important_rule_counts": {
            "price_missing": int(cleaned["is_price_missing"].sum()),
            "price_outlier": int(cleaned["is_price_outlier"].sum()),
            "ram_suspicious": int(cleaned["ram_suspicious"].sum()),
            "potential_dedicated_gpu": int(cleaned["potential_dedicated_gpu"].sum()),
            "storage_type_imputed": int(cleaned["was_storage_type_imputed"].sum()),
            "new_low_price": int(cleaned["new_low_price"].sum()),
            "repair_mismatch": int(cleaned["repair_mismatch"].sum()),
        },
    }
    save_outputs(cleaned, report, issue_action_plan, log, config)
    print("\nValidation:")
    for name, ok in validation.items():
        print(f"- {name}: {ok}")
    print("\nSaved outputs:")
    for key in ["cleaned_path", "report_path", "issue_action_plan_path", "log_path"]:
        print(f"- {config[key]}")
    return CleaningArtifacts(raw, cleaned, issues, issue_action_plan, report, log, validation)
