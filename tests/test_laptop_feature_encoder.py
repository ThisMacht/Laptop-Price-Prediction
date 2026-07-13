from __future__ import annotations

import pandas as pd

from src.encoder import LaptopFeatureEncoder


def test_full_intel_input() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "Dell",
            "model": "Latitude",
            "ram_gb": 16,
            "storage_gb": 512,
            "storage_type": "SSD",
            "screen_size_inch": 14,
            "cpu_text": "Intel Core i5-1235U",
            "gpu_text": "integrated",
            "condition": "good",
            "warranty_status": "expired",
        }
    )

    assert X.shape == (1, 86)
    row = X.iloc[0]
    assert row["brand_Dell"] == 1
    assert row["model_Latitude"] == 1
    assert row["ram_gb"] == 16
    assert row["storage_ssd"] == 1
    assert row["cpu_brand_Intel"] == 1
    assert row["cpu_family_group_Intel Core i"] == 1
    assert row["cpu_intel_generation_ord"] == 12
    assert row["cpu_tier_encoded"] == 6
    assert row["no_info_cpu_tier"] == 0
    assert row["warranty_expired"] == 1


def test_rare_brand_and_model() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "Gigabyte",
            "model": "Aero 15",
            "ram_gb": 16,
            "storage_gb": 512,
            "storage_type": "SSD",
        }
    )

    row = X.iloc[0]
    brand_onehot_cols = [
        col
        for col in X.columns
        if col.startswith("brand_") and col != "brand_is_rare" and not col.startswith("brand_segment_")
    ]
    model_onehot_cols = [col for col in X.columns if col.startswith("model_") and col != "model_is_rare"]

    assert row[brand_onehot_cols].sum() == 0
    assert row["brand_is_rare"] == 1
    assert row["no_info_brand"] == 0
    assert row[model_onehot_cols].sum() == 0
    assert row["model_is_rare"] == 1
    assert row["no_info_model"] == 0


def test_missing_brand_model_cpu_gpu() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "ram_gb": None,
            "storage_gb": None,
            "screen_size_inch": None,
        }
    )

    row = X.iloc[0]
    assert row["ram_missing"] == 1
    assert row["storage_missing"] == 1
    assert row["screen_missing"] == 1
    assert row["no_info_brand"] == 1
    assert row["no_info_model"] == 1
    assert row["no_info_cpu_brand"] == 1
    assert row["no_info_cpu_tier"] == 1
    assert row["no_info_gpu"] == 1
    assert X.shape == (1, 86)
    assert X.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_apple_input() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "Apple",
            "model": "MacBook Air M2",
            "ram_gb": 16,
            "storage_gb": 512,
            "storage_type": "SSD",
            "cpu_text": "Apple M2",
            "gpu_text": "integrated",
            "condition": "like_new",
            "warranty_status": "active",
        }
    )

    row = X.iloc[0]
    assert row["brand_Apple"] == 1
    assert row["model_MacBook Air M2"] == 1
    assert row["cpu_brand_Apple"] == 1
    assert row["cpu_family_group_Apple Silicon"] == 1
    assert row["cpu_apple_core_spec"] == 2
    assert row["warranty_active"] == 1


def test_cpu_partial_information() -> None:
    encoder = LaptopFeatureEncoder()

    X_family = encoder.encode_one({"cpu_text": "Intel Core i5"})
    row_family = X_family.iloc[0]
    assert row_family["cpu_brand_Intel"] == 1
    assert row_family["cpu_family_group_Intel Core i"] == 1
    assert row_family["cpu_tier_encoded"] == 6
    assert row_family["no_info_cpu_tier"] == 0

    X_brand = encoder.encode_one({"cpu_text": "Intel"})
    row_brand = X_brand.iloc[0]
    assert row_brand["cpu_brand_Intel"] == 1
    assert row_brand["cpu_tier_encoded"] == 2
    assert row_brand["no_info_cpu_tier"] == 1


def test_output_columns_follow_schema_order() -> None:
    encoder = LaptopFeatureEncoder()
    X = encoder.encode_one({"brand": "Dell"})

    assert X.columns.tolist() == encoder.feature_names
    assert X.shape[1] == len(encoder.feature_names)


def test_encode_many_with_list_of_dicts() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_many(
        [
            {
                "brand": "Dell",
                "model": "Latitude",
                "ram_gb": 16,
                "storage_gb": 512,
                "storage_type": "SSD",
                "cpu_text": "Intel Core i5-1235U",
            },
            {
                "brand": "Apple",
                "model": "MacBook Air M2",
                "ram_gb": 16,
                "storage_gb": 512,
                "storage_type": "SSD",
                "cpu_text": "Apple M2",
            },
        ]
    )

    assert X.shape == (2, 86)
    assert X.isna().sum().sum() == 0
    assert X.iloc[0]["brand_Dell"] == 1
    assert X.iloc[1]["brand_Apple"] == 1
    assert X.iloc[1]["model_MacBook Air M2"] == 1


def test_encode_many_with_dataframe() -> None:
    encoder = LaptopFeatureEncoder()

    raw_df = pd.DataFrame(
        [
            {
                "brand": "Lenovo",
                "model": "ThinkPad",
                "ram_gb": 16,
                "storage_gb": 512,
                "storage_type": "SSD",
                "cpu_text": "Intel Core i7-1260P",
            },
            {
                "brand": "ASUS",
                "model": "TUF Gaming F15",
                "ram_gb": 16,
                "storage_gb": 512,
                "storage_type": "SSD",
                "cpu_text": "Intel Core i5-12500H",
            },
        ]
    )

    X = encoder.encode_many(raw_df)

    assert X.shape == (2, 86)
    assert X.iloc[0]["brand_Lenovo"] == 1
    assert X.iloc[0]["model_ThinkPad"] == 1
    assert X.iloc[1]["brand_ASUS"] == 1
    assert X.iloc[1]["model_TUF Gaming F15"] == 1


def test_brand_model_normalization() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "  dell ",
            "model": " latitude ",
        }
    )

    row = X.iloc[0]
    assert row["brand_Dell"] == 1
    assert row["model_Latitude"] == 1
    assert row["no_info_brand"] == 0
    assert row["no_info_model"] == 0


def test_storage_ssd_hdd_combo() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "storage_gb": 1256,
            "storage_type": "SSD + HDD",
        }
    )

    row = X.iloc[0]
    assert row["storage_gb"] == 1256
    assert row["storage_ssd"] == 1
    assert row["storage_hdd"] == 1
    assert row["no_info_storage"] == 0
    assert row["storage_missing"] == 0


def test_missing_storage_type() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "storage_gb": 512,
            "storage_type": None,
        }
    )

    row = X.iloc[0]
    assert row["storage_gb"] == 512
    assert row["storage_missing"] == 0
    assert row["no_info_storage"] == 1
    assert row["storage_ssd"] == 0
    assert row["storage_hdd"] == 0


def test_cpu_high_end_i9_hx() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"cpu_text": "Intel Core i9-13900HX"})
    row = X.iloc[0]

    assert row["cpu_brand_Intel"] == 1
    assert row["cpu_family_group_Intel Core i"] == 1
    assert row["cpu_intel_generation_ord"] == 13
    assert row["cpu_suffix_power_ord_filled"] > 0
    assert row["cpu_tier_encoded"] == 7
    assert row["no_info_cpu_tier"] == 0


def test_cpu_upper_mid_i7_h() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"cpu_text": "Intel Core i7-12700H"})
    row = X.iloc[0]

    assert row["cpu_brand_Intel"] == 1
    assert row["cpu_family_group_Intel Core i"] == 1
    assert row["cpu_intel_generation_ord"] == 12
    assert row["cpu_tier_encoded"] == 0
    assert row["no_info_cpu_tier"] == 0


def test_cpu_amd_ryzen5_mid_range() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"cpu_text": "AMD Ryzen 5 5500U"})
    row = X.iloc[0]

    assert row["cpu_brand_AMD"] == 1
    assert row["cpu_family_group_AMD Ryzen"] == 1
    assert row["cpu_amd_generation_ord"] >= 5000
    assert row["cpu_tier_encoded"] == 6
    assert row["no_info_cpu_tier"] == 0


def test_cpu_amd_ryzen9_high_end() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"cpu_text": "AMD Ryzen 9 7945HX"})
    row = X.iloc[0]

    assert row["cpu_brand_AMD"] == 1
    assert row["cpu_family_group_AMD Ryzen"] == 1
    assert row["cpu_amd_generation_ord"] >= 7000
    assert row["cpu_tier_encoded"] == 7
    assert row["no_info_cpu_tier"] == 0


def test_integrated_gpu_is_not_missing() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"gpu_text": "integrated graphics"})
    row = X.iloc[0]

    assert row["no_info_gpu"] == 0
    assert row["gpu_tier_ord_filled"] >= 0


def test_missing_gpu() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one({"gpu_text": None})
    row = X.iloc[0]

    assert row["no_info_gpu"] == 1
    assert row["gpu_tier_ord_filled"] == 0


def test_rare_brand_model_is_different_from_missing() -> None:
    encoder = LaptopFeatureEncoder()

    X_rare = encoder.encode_one({"brand": "Gigabyte", "model": "Aero 15"})
    rare = X_rare.iloc[0]

    assert rare["brand_is_rare"] == 1
    assert rare["model_is_rare"] == 1
    assert rare["no_info_brand"] == 0
    assert rare["no_info_model"] == 0

    X_missing = encoder.encode_one({"brand": None, "model": None})
    missing = X_missing.iloc[0]

    assert missing["brand_is_rare"] == 0
    assert missing["model_is_rare"] == 0
    assert missing["no_info_brand"] == 1
    assert missing["no_info_model"] == 1


def test_no_other_unknown_columns_in_output() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "UnknownBrand",
            "model": "UnknownModel",
            "cpu_text": "Unknown CPU",
        }
    )

    assert "brand_Other" not in X.columns
    assert "model_Other" not in X.columns
    assert "cpu_brand_Other" not in X.columns


def test_output_is_all_numeric() -> None:
    encoder = LaptopFeatureEncoder()

    X = encoder.encode_one(
        {
            "brand": "HP",
            "model": "ProBook",
            "ram_gb": 8,
            "storage_gb": 256,
            "storage_type": "SSD",
            "cpu_text": "Intel Core i3-1115G4",
        }
    )

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    assert X.isna().sum().sum() == 0


def test_condition_score_mapping_for_new_unknown_and_like_new() -> None:
    encoder = LaptopFeatureEncoder()

    assert encoder.encode_one({"condition": "new"}).loc[0, "condition_score"] == 3
    assert encoder.encode_one({"condition": "moi"}).loc[0, "condition_score"] == 3
    assert encoder.encode_one({"condition": "unknown"}).loc[0, "condition_score"] == 2
    assert encoder.encode_one({"condition": "unknow"}).loc[0, "condition_score"] == 2
    assert encoder.encode_one({"condition": None}).loc[0, "condition_score"] == 2
    assert encoder.encode_one({"condition": "like new"}).loc[0, "condition_score"] == 2
    assert encoder.encode_one({"condition": "da mua"}).loc[0, "condition_score"] == 2
