from __future__ import annotations

from pathlib import Path

import pytest

from src.encoder import LaptopFeatureEncoder


def test_encoder_output_can_be_used_by_production_model() -> None:
    model_path = Path("models/final_laptop_price_model.joblib")

    if not model_path.exists():
        pytest.skip("Production model file is not available.")

    joblib = pytest.importorskip("joblib")
    pytest.importorskip("catboost")

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

    model = joblib.load(model_path)
    pred = model.predict(X)

    assert len(pred) == 1
    assert float(pred[0]) == pred[0]
