from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG: dict[str, Any] = {
    "raw_path": PROJECT_ROOT / "data" / "raw" / "clean_laptop_features.csv",
    "issues_path": PROJECT_ROOT / "docs" / "websach_issues_list.csv",
    "output_dir": PROJECT_ROOT / "data" / "intern",
    "cleaned_path": PROJECT_ROOT / "data" / "intern" / "websach_cleaned.csv",
    "report_path": PROJECT_ROOT / "docs" / "websach_cleaning_report.csv",
    "issue_action_plan_path": PROJECT_ROOT / "docs" / "websach_issue_action_plan.csv",
    "log_path": PROJECT_ROOT / "docs" / "websach_cleaning_log.json",
    "price_cols": ["shop_1_price", "shop_2_price", "shop_3_price"],
    "shop_name_cols": ["shop_1_name", "shop_2_name", "shop_3_name"],
    "price_min": 3_000_000,
    "price_max": 200_000_000,
    "relative_low_factor": 0.5,
    "relative_high_factor": 2.0,
    "price_spread_warn": 0.30,
    "price_spread_critical": 0.50,
    "screen_min": 0,
    "screen_max": 25,
    "rare_count_threshold": 30,
    "min_brand_screen_cell_n": 10,
    "min_interaction_cell_n": 20,
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
    print(f"Issues loaded: {len(issues):,}")
    return raw, issues


def infer_issue_group(issue_key: str, description: str, recommendation: str) -> str:
    text = f"{issue_key} {description} {recommendation}".lower()
    if "price" in text or "giá" in text or "target" in text:
        return "price_related_issues"
    if "missing" in text or "unknown" in text or "thiếu" in text:
        return "missing_unknown_values"
    if "outlier" in text or "bất thường" in text:
        return "invalid_values_outliers"
    if "duplicate" in text or "trùng" in text:
        return "duplicate_records"
    if "screen" in text or "màn hình" in text:
        return "screen_size_cleaning"
    if "cpu" in text:
        return "cpu_parsing_grouping"
    if "ram" in text:
        return "ram_parsing_grouping"
    if "storage" in text or "ổ cứng" in text:
        return "storage_parsing_grouping"
    if "gpu" in text:
        return "gpu_parsing_grouping"
    if "interaction" in text or "tương tác" in text:
        return "low_count_interactions"
    if "sparse" in text or "imbalance" in text or "hiếm" in text:
        return "low_count_categories"
    return "modeling_or_documentation"


def infer_affected_column(issue_key: str) -> str:
    if issue_key.startswith("price") or issue_key in {"target_right_skewed", "high_price_tail_influence"}:
        return "shop_*_price/price_median"
    if issue_key.startswith("screen") or "screen" in issue_key:
        return "Kích thước (inch)"
    if issue_key.startswith("cpu") or "cpu" in issue_key:
        return "Công nghệ CPU/Loại CPU"
    if issue_key.startswith("ram") or "ram" in issue_key:
        return "Dung lượng RAM/Loại RAM"
    if issue_key.startswith("storage") or "storage" in issue_key:
        return "Loại ổ cứng/Dung lượng ổ cứng (GB)"
    if issue_key.startswith("gpu") or "gpu" in issue_key:
        return "Đồ họa đã làm sạch"
    if "brand" in issue_key:
        return "Hãng sản xuất"
    if "duplicate" in issue_key:
        return "all/spec_key"
    return ""


def infer_severity(issue_key: str, description: str, recommendation: str) -> str:
    text = f"{issue_key} {description} {recommendation}".lower()
    if any(term in text for term in ["outlier", "missing", "unknown", "lỗi", "parse", "duplicate"]):
        return "high"
    if any(term in text for term in ["imbalance", "sparse", "low_count", "confounding", "influence"]):
        return "medium"
    return "low"


def needs_note_lookup(issue_key: str, description: str, recommendation: str) -> str:
    text = f"{issue_key} {description} {recommendation}".lower()
    terms = ["<", ">", "0.5", "2×", "mapping", "ngưỡng", "threshold", "tier", "group", "gom"]
    return "yes" if any(term in text for term in terms) else "no"


def decide_final_action(issue_key: str, recommendation: str) -> tuple[str, str]:
    text = f"{issue_key} {recommendation}".lower()
    if any(term in text for term in ["modeling", "mô hình", "log-transform", "feature importance", "baseline", "evaluation"]):
        return "Document as modeling guidance; keep engineered helper columns where useful.", "needs_manual_review"
    if "outlier" in text or "null-out" in text or "filter" in text:
        return "Apply cell/field-level null-out or flag; do not drop whole products.", "resolved"
    if "unknown" in text or "missing" in text or "thiếu" in text:
        return "Keep Unknown/tier flags instead of dropping rows.", "partially_resolved"
    if "gom" in text or "group" in text or "tier" in text or "mapping" in text:
        return "Create normalized grouped/tier feature and preserve raw column for audit.", "resolved"
    if "duplicate" in text:
        return "Report exact and soft duplicates; do not drop soft spec duplicates.", "resolved"
    return "Documented; no deterministic cleaning required.", "needs_manual_review"


def build_issue_action_plan(issues: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in issues.iterrows():
        issue_key = str(row["issue"])
        description = str(row["description"])
        recommendation = str(row["recommendation"])
        final_action, status = decide_final_action(issue_key, recommendation)
        rows.append(
            {
                "issue_key": issue_key,
                "severity": infer_severity(issue_key, description, recommendation),
                "affected_column": infer_affected_column(issue_key),
                "issue_group": infer_issue_group(issue_key, description, recommendation),
                "issue_description": description,
                "proposed_action": recommendation,
                "need_note_lookup": needs_note_lookup(issue_key, description, recommendation),
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


def clean_prices(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    price_cols = config["price_cols"]
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        domain_col = f"{col}_domain"
        clean_col = f"{col}_clean"
        bad = out[col].notna() & ((out[col] < config["price_min"]) | (out[col] > config["price_max"]))
        out[f"{col}_domain_outlier"] = bad
        out[domain_col] = out[col].mask(bad)
        out[clean_col] = out[domain_col]

    domain_cols = [f"{col}_domain" for col in price_cols]
    clean_cols = [f"{col}_clean" for col in price_cols]
    row_median = out[domain_cols].median(axis=1, skipna=True)
    out["price_row_median_domain"] = row_median

    for col, clean_col in zip(price_cols, clean_cols):
        low = out[clean_col] < config["relative_low_factor"] * row_median
        high = out[clean_col] > config["relative_high_factor"] * row_median
        rel_bad = out[clean_col].notna() & row_median.notna() & (low | high)
        out[f"{col}_relative_outlier"] = rel_bad
        out[clean_col] = out[clean_col].mask(rel_bad)

    clean_wide = out[clean_cols]
    out["n_prices_raw"] = out[price_cols].notna().sum(axis=1)
    out["n_prices_clean"] = clean_wide.notna().sum(axis=1)
    out["price_median"] = clean_wide.median(axis=1, skipna=True)
    out["price_min_clean"] = clean_wide.min(axis=1, skipna=True)
    out["price_max_clean"] = clean_wide.max(axis=1, skipna=True)
    out["price_spread_clean_pct"] = (out["price_max_clean"] - out["price_min_clean"]) / out["price_median"]
    out.loc[out["n_prices_clean"] < 2, "price_spread_clean_pct"] = np.nan
    out["flag_price_spread_warn"] = out["price_spread_clean_pct"].gt(config["price_spread_warn"])
    out["flag_price_spread_critical"] = out["price_spread_clean_pct"].gt(config["price_spread_critical"])
    out["log_price_median"] = np.log1p(out["price_median"])
    out["price_segment"] = pd.cut(
        out["price_median"],
        bins=[0, 15_000_000, 30_000_000, 60_000_000, np.inf],
        labels=["low", "mid", "high", "high_end"],
        include_lowest=True,
    ).astype("string")
    return out


def clean_brand(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    brand = out["Hãng sản xuất"].astype("string").str.strip()
    brand_map = {"Lg": "LG", "Asus": "ASUS", "Hp": "HP", "Msi": "MSI"}
    out["brand_clean"] = brand.replace(brand_map).fillna("Unknown")
    counts = out["brand_clean"].value_counts(dropna=False)
    rare = set(counts[counts < config["rare_count_threshold"]].index)
    out["brand_grouped"] = out["brand_clean"].where(~out["brand_clean"].isin(rare), "Other")
    out["brand_is_rare"] = out["brand_grouped"].eq("Other")
    return out


def clean_screen(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["screen_size_clean"] = pd.to_numeric(out["Kích thước (inch)"], errors="coerce")
    invalid = out["screen_size_clean"].notna() & (
        (out["screen_size_clean"] <= config["screen_min"]) | (out["screen_size_clean"] > config["screen_max"])
    )
    out["screen_size_outlier"] = invalid
    out["screen_size_clean"] = out["screen_size_clean"].mask(invalid)
    bins = [-np.inf, 13, 14, 15, 16, 17, np.inf]
    labels = ["<13in", "13-13.9in", "14-14.9in", "15-15.9in", "16-16.9in", ">=17in"]
    out["screen_size_group"] = pd.cut(out["screen_size_clean"], bins=bins, labels=labels, right=False).astype("string")
    out["screen_size_group"] = out["screen_size_group"].fillna("Unknown")
    return out


def extract_cpu_brand(cpu_tech: Any, cpu_model: Any) -> str:
    text = f"{cpu_tech} {cpu_model}".lower()
    if "ryzen" in text or re.search(r"\bamd\b", text):
        return "AMD"
    if re.search(r"\bm[1-5]\b", text) or "apple" in text:
        return "Apple"
    if "snapdragon" in text or "qualcomm" in text:
        return "Qualcomm"
    if "microsoft sq" in text:
        return "Microsoft SQ"
    if "intel" in text or re.search(r"\b\d{4,5}[a-z]{0,2}\b", text) or "core ultra" in text:
        return "Intel"
    return "Unknown"


def extract_cpu_tier(cpu_tech: Any, cpu_model: Any) -> str:
    text = f"{cpu_tech} {cpu_model}".lower()
    if re.search(r"m[1-5]\s*max|core ultra\s*9|i9|ryzen\s*9", text):
        return "High-end"
    if re.search(r"m[1-5]\s*pro|core ultra\s*7|i7|ryzen\s*7|snapdragon x elite", text):
        return "Upper-mid"
    if re.search(r"\bm[1-5]\b|core ultra\s*5|i5|ryzen\s*5|snapdragon x plus", text):
        return "Mid-range"
    if re.search(r"i3|ryzen\s*3", text):
        return "Entry"
    if re.search(r"celeron|pentium|athlon|n\d{3}", text):
        return "Low-end"
    return "Unknown"


def clean_cpu(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cpu_brand"] = [extract_cpu_brand(a, b) for a, b in zip(out["Công nghệ CPU"], out["Loại CPU"])]
    out["cpu_tier"] = [extract_cpu_tier(a, b) for a, b in zip(out["Công nghệ CPU"], out["Loại CPU"])]
    out["cpu_model_clean"] = out["Loại CPU"].fillna("Unknown").astype("string").str.upper()
    return out


def clean_ram_type(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", "", str(value).upper())
    mapping = {
        "DR4": "DDR4",
        "DR5": "DDR5",
        "DDR5X": "DDR5",
        "LPDDR5X": "LPDDR5X",
        "LPDDR5": "LPDDR5",
        "LPDDR4X": "LPDDR4X",
        "LPDDR4": "LPDDR4",
        "DDR4": "DDR4",
        "DDR5": "DDR5",
        "DDR3": "DDR3",
        "DDR3L": "DDR3L",
    }
    if text.isdigit():
        return "Unknown"
    return mapping.get(text, "Other")


def ram_tier(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= 8:
        return "<=8GB"
    if value == 16:
        return "16GB"
    if value == 32:
        return "32GB"
    if value > 32:
        return ">32GB"
    return "Other"


def clean_ram(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ram_gb_clean"] = pd.to_numeric(out["Dung lượng RAM"], errors="coerce")
    invalid = out["ram_gb_clean"].notna() & ((out["ram_gb_clean"] <= 0) | (out["ram_gb_clean"] > 256))
    out["ram_capacity_outlier"] = invalid
    out["ram_gb_clean"] = out["ram_gb_clean"].mask(invalid)
    out["ram_tier_clean"] = out["ram_gb_clean"].apply(ram_tier)
    out["ram_type_clean"] = out["Loại RAM"].apply(clean_ram_type)
    return out


def clean_storage_type(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = re.sub(r"\s+", "", str(value).upper())
    if re.search(r"\d+\s*GB|\d+\s*TB", text):
        return "Unknown"
    if text in {"SSD", "SS", "SSD."}:
        return "SSD"
    if text == "HDD":
        return "HDD"
    if "SSD" in text and "HDD" in text:
        return "SSD + HDD"
    if text == "EMMC":
        return "Other"
    return "Other"


def storage_tier(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    if value <= 256:
        return "<=256GB"
    if value <= 512:
        return "512GB"
    if value <= 1024:
        return "1TB"
    if value <= 2048:
        return "2TB"
    return ">2TB"


def clean_storage(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["storage_type_clean"] = out["Loại ổ cứng"].apply(clean_storage_type)
    out["storage_gb_clean"] = pd.to_numeric(out["Dung lượng ổ cứng (GB)"], errors="coerce")
    out["storage_gb_outlier"] = out["storage_gb_clean"].notna() & ((out["storage_gb_clean"] <= 0) | (out["storage_gb_clean"] > 8192))
    out["storage_gb_clean"] = out["storage_gb_clean"].mask(out["storage_gb_outlier"])
    out["storage_tier_clean"] = out["storage_gb_clean"].apply(storage_tier)
    return out


def gpu_brand(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).lower()
    if "rtx" in text or "gtx" in text or "geforce" in text or "nvidia" in text or "quadro" in text:
        return "NVIDIA"
    if "radeon" in text or re.search(r"\brx\b", text):
        return "AMD"
    if "iris" in text or "uhd" in text or "intel" in text:
        return "Intel"
    if "apple" in text or re.search(r"\bm[1-5]\b", text):
        return "Apple"
    return "Other"


def gpu_tier_v2(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value).lower()
    if "unknown" in text:
        return "Unknown"
    if "apple" in text:
        return "Apple GPU"
    if re.search(r"rtx\s?50\d{2}", text):
        return "RTX 5000"
    if re.search(r"rtx\s?40\d{2}", text):
        return "RTX 4000"
    if re.search(r"rtx", text):
        return "Other RTX"
    if re.search(r"gtx", text):
        return "GTX"
    if "mx" in text or "nvidia" in text or "geforce" in text or "quadro" in text:
        return "Other NVIDIA"
    if "iris" in text or "uhd" in text or "intel" in text:
        return "Intel Integrated"
    if "radeon" in text or "amd" in text:
        return "AMD Radeon"
    return "Other GPU"


def clean_gpu(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gpu_brand"] = out["Đồ họa đã làm sạch"].apply(gpu_brand)
    out["gpu_tier_v2"] = out["Đồ họa đã làm sạch"].apply(gpu_tier_v2)
    return out


def add_quality_and_interaction_flags(df: pd.DataFrame, config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    out = df.copy()
    out["product_id"] = np.arange(len(out))
    spec_cols = ["brand_clean", "cpu_model_clean", "ram_gb_clean", "storage_gb_clean", "screen_size_clean"]
    out["spec_key"] = out[spec_cols].astype("string").fillna("Unknown").agg("|".join, axis=1)
    out["is_soft_duplicate_spec"] = out.duplicated(subset=["spec_key"], keep=False)
    out["high_end_config_flag"] = (
        out["gpu_tier_v2"].isin(["RTX 4000", "RTX 5000", "Other RTX", "Apple GPU"])
        | out["ram_tier_clean"].isin(["32GB", ">32GB"])
        | out["storage_tier_clean"].isin(["2TB", ">2TB"])
        | out["cpu_tier"].eq("High-end")
    )
    out["brand_screen_cell_n"] = out.groupby(["brand_grouped", "screen_size_group"])["product_id"].transform("size")
    out["low_count_brand_screen_cell"] = out["brand_screen_cell_n"] < config["min_brand_screen_cell_n"]
    out["cpu_ram_cell_n"] = out.groupby(["cpu_brand", "ram_tier_clean"])["product_id"].transform("size")
    out["gpu_ram_cell_n"] = out.groupby(["gpu_tier_v2", "ram_tier_clean"])["product_id"].transform("size")
    out["low_count_cpu_ram_cell"] = out["cpu_ram_cell_n"] < config["min_interaction_cell_n"]
    out["low_count_gpu_ram_cell"] = out["gpu_ram_cell_n"] < config["min_interaction_cell_n"]
    return out


def remove_full_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(df)
    out = df.drop_duplicates().copy()
    return out, {"full_duplicate_removed": int(before - len(out))}


def distribution_summary(df: pd.DataFrame, column: str, top_n: int = 10) -> list[dict[str, Any]]:
    counts = df[column].value_counts(dropna=False).head(top_n)
    return [{"value": str(k), "count": int(v)} for k, v in counts.items()]


def numeric_summary(df: pd.DataFrame, column: str) -> dict[str, Any]:
    desc = pd.to_numeric(df[column], errors="coerce").describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {str(k): (float(v) if pd.notna(v) else None) for k, v in desc.items()}


def build_cleaning_report(raw: pd.DataFrame, cleaned: pd.DataFrame, issue_plan: pd.DataFrame, duplicate_log: dict[str, int], config: dict[str, Any] = CONFIG) -> pd.DataFrame:
    domain_count = int(sum(cleaned[f"{c}_domain_outlier"].sum() for c in config["price_cols"]))
    relative_count = int(sum(cleaned[f"{c}_relative_outlier"].sum() for c in config["price_cols"]))
    rows: list[dict[str, Any]] = [
        {"section": "shape", "metric": "row_count_before", "value": len(raw)},
        {"section": "shape", "metric": "row_count_after", "value": len(cleaned)},
        {"section": "shape", "metric": "column_count_before", "value": raw.shape[1]},
        {"section": "shape", "metric": "column_count_after", "value": cleaned.shape[1]},
        {"section": "duplicates", "metric": "full_duplicate_removed", "value": duplicate_log["full_duplicate_removed"]},
        {"section": "duplicates", "metric": "soft_duplicate_spec_rows", "value": int(cleaned["is_soft_duplicate_spec"].sum())},
        {"section": "price", "metric": "domain_price_cells_nulled", "value": domain_count},
        {"section": "price", "metric": "relative_price_cells_nulled", "value": relative_count},
        {"section": "price", "metric": "price_spread_warn_rows", "value": int(cleaned["flag_price_spread_warn"].sum())},
        {"section": "price", "metric": "price_spread_critical_rows", "value": int(cleaned["flag_price_spread_critical"].sum())},
        {"section": "screen", "metric": "screen_size_outlier_nulled", "value": int(cleaned["screen_size_outlier"].sum())},
        {"section": "ram", "metric": "ram_capacity_outlier_nulled", "value": int(cleaned["ram_capacity_outlier"].sum())},
        {"section": "storage", "metric": "storage_gb_outlier_nulled", "value": int(cleaned["storage_gb_outlier"].sum())},
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
    for status, count in issue_plan["resolution_status"].value_counts().items():
        rows.append({"section": "issue_status", "metric": status, "value": int(count)})
    for col in ["brand_grouped", "screen_size_group", "cpu_brand", "cpu_tier", "ram_tier_clean", "ram_type_clean", "storage_tier_clean", "storage_type_clean", "gpu_tier_v2"]:
        rows.append({"section": "distribution", "metric": col, "value": json.dumps(distribution_summary(cleaned, col), ensure_ascii=False)})
    rows.append({"section": "distribution", "metric": "price_median", "value": json.dumps(numeric_summary(cleaned, "price_median"))})
    return pd.DataFrame(rows)


def validate_cleaned_data(cleaned: pd.DataFrame, issue_plan: pd.DataFrame) -> dict[str, bool]:
    validations = {
        "price_median_numeric": pd.api.types.is_numeric_dtype(cleaned["price_median"]),
        "price_median_not_missing": cleaned["price_median"].notna().all(),
        "domain_prices_in_range": all(
            cleaned[f"{c}_clean"].dropna().between(CONFIG["price_min"], CONFIG["price_max"]).all() for c in CONFIG["price_cols"]
        ),
        "screen_range_valid_or_missing": cleaned["screen_size_clean"].dropna().between(0, CONFIG["screen_max"]).all(),
        "ram_range_valid_or_missing": cleaned["ram_gb_clean"].dropna().between(0, 256).all(),
        "storage_positive_or_missing": cleaned["storage_gb_clean"].dropna().gt(0).all(),
        "unknown_categories_available": all(col in cleaned.columns for col in ["screen_size_group", "ram_tier_clean", "storage_tier_clean", "gpu_tier_v2"]),
        "all_issues_have_actions": issue_plan["final_action"].notna().all() and issue_plan["resolution_status"].notna().all(),
    }
    failed = [name for name, ok in validations.items() if not bool(ok)]
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


def save_outputs(cleaned: pd.DataFrame, report: pd.DataFrame, issue_plan: pd.DataFrame, log: dict[str, Any], config: dict[str, Any] = CONFIG) -> None:
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(config["cleaned_path"], index=False, encoding="utf-8-sig")
    report.to_csv(config["report_path"], index=False, encoding="utf-8-sig")
    issue_plan.to_csv(config["issue_action_plan_path"], index=False, encoding="utf-8-sig")
    config["log_path"].write_text(json.dumps(make_json_safe(log), ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(config: dict[str, Any] = CONFIG) -> CleaningArtifacts:
    raw, issues = load_inputs(config)
    issue_plan = build_issue_action_plan(issues)
    deduped, duplicate_log = remove_full_duplicates(raw)
    cleaned = (
        deduped.pipe(normalize_text_columns)
        .pipe(clean_prices, config=config)
        .pipe(clean_brand, config=config)
        .pipe(clean_screen, config=config)
        .pipe(clean_cpu)
        .pipe(clean_ram)
        .pipe(clean_storage)
        .pipe(clean_gpu)
        .pipe(add_quality_and_interaction_flags, config=config)
    )
    report = build_cleaning_report(raw, cleaned, issue_plan, duplicate_log, config)
    validation = validate_cleaned_data(cleaned, issue_plan)
    log = {
        "config": make_json_safe(config),
        "raw_shape": list(raw.shape),
        "cleaned_shape": list(cleaned.shape),
        "duplicate_log": duplicate_log,
        "validation": validation,
        "issue_status_counts": issue_plan["resolution_status"].value_counts().to_dict(),
        "important_rule_counts": {
            "domain_price_cells_nulled": int(sum(cleaned[f"{c}_domain_outlier"].sum() for c in config["price_cols"])),
            "relative_price_cells_nulled": int(sum(cleaned[f"{c}_relative_outlier"].sum() for c in config["price_cols"])),
            "price_spread_warn_rows": int(cleaned["flag_price_spread_warn"].sum()),
            "price_spread_critical_rows": int(cleaned["flag_price_spread_critical"].sum()),
            "screen_size_outlier_nulled": int(cleaned["screen_size_outlier"].sum()),
            "ram_capacity_outlier_nulled": int(cleaned["ram_capacity_outlier"].sum()),
            "soft_duplicate_spec_rows": int(cleaned["is_soft_duplicate_spec"].sum()),
        },
    }
    save_outputs(cleaned, report, issue_plan, log, config)
    print("\nValidation:")
    for name, ok in validation.items():
        print(f"- {name}: {ok}")
    print("\nSaved outputs:")
    for key in ["cleaned_path", "report_path", "issue_action_plan_path", "log_path"]:
        print(f"- {config[key]}")
    return CleaningArtifacts(raw, cleaned, issues, issue_plan, report, log, validation)
