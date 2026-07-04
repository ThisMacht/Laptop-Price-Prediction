from __future__ import annotations

import pytest

from app.backend.predictor_service import parse_llm_compare_content


def test_parse_llm_compare_content_accepts_multiple_laptops() -> None:
    raw = """
    {
      "laptops": [
        {
          "label": "Laptop A",
          "actual_price_million_vnd": 12.0,
          "brand": "Dell",
          "model": null,
          "ram_gb": 8,
          "storage_gb": 512,
          "storage_type": "SSD",
          "screen_size_inch": 15.6,
          "cpu_text": "Intel Core i5",
          "cpu_brand": "Intel",
          "cpu_family": "Intel Core i5",
          "cpu_generation": null,
          "cpu_suffix": null,
          "gpu_text": "integrated",
          "condition": "good",
          "warranty_status": "expired"
        },
        {
          "label": "Laptop B",
          "actual_price_million_vnd": 14.0,
          "brand": "HP",
          "model": null,
          "ram_gb": 16,
          "storage_gb": 512,
          "storage_type": "SSD",
          "screen_size_inch": 15.6,
          "cpu_text": "Intel Core i5",
          "cpu_brand": "Intel",
          "cpu_family": "Intel Core i5",
          "cpu_generation": null,
          "cpu_suffix": null,
          "gpu_text": "integrated",
          "condition": "good",
          "warranty_status": "expired"
        }
      ]
    }
    """
    laptops = parse_llm_compare_content(raw)
    assert len(laptops) == 2
    assert laptops[0]["label"] == "Laptop A"
    assert laptops[1]["actual_price_million_vnd"] == 14.0


def test_parse_llm_compare_content_requires_at_least_two_laptops() -> None:
    with pytest.raises(ValueError, match="ít nhất 2"):
        parse_llm_compare_content('{"laptops": []}')
