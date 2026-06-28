"""GPU parsing helpers for production laptop feature encoding."""

from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_text(value: Any) -> str | None:
    """Return a stripped string or None for missing/no-info values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.casefold() in {"nan", "none", "<na>", "null"}:
        return None
    return text


def parse_gpu_tier(gpu_text: Any) -> tuple[int, int]:
    """Return (gpu_tier_ord_filled, no_info_gpu)."""
    text = normalize_text(gpu_text)
    if text is None:
        return 0, 1

    low = text.casefold()
    compact = low.replace("-", " ")

    high_terms = [
        "rtx 3080",
        "rtx3080",
        "rtx 4080",
        "rtx4080",
        "rtx 4090",
        "rtx4090",
        "rtx 5070",
        "rtx5070",
        "rtx 5080",
        "rtx5080",
        "rtx 5090",
        "rtx5090",
    ]
    upper_terms = [
        "rtx 3060",
        "rtx3060",
        "rtx 4060",
        "rtx4060",
        "rtx 3070",
        "rtx3070",
        "rtx 4070",
        "rtx4070",
    ]
    mid_terms = ["rtx 3050 ti", "rtx3050ti", "rtx 3050", "rtx3050", "rtx 4050", "rtx4050"]
    entry_terms = ["mx", "gtx 1050", "gtx1050", "gtx 1650", "gtx1650", "rtx 2050", "rtx2050"]
    integrated_terms = ["integrated", "onboard", "iris xe", "uhd", "vega", "radeon graphics"]
    apple_terms = ["apple gpu", "apple integrated", "m1 gpu", "m2 gpu", "m3 gpu", "m4 gpu"]

    if any(term in compact for term in high_terms):
        return 5, 0
    if any(term in compact for term in upper_terms):
        return 4, 0
    if any(term in compact for term in mid_terms):
        return 3, 0
    if any(term in compact for term in entry_terms):
        return 2, 0
    if any(term in compact for term in apple_terms):
        return 1, 0
    if any(term in compact for term in integrated_terms):
        return 0, 0
    if "rtx" in compact:
        return 3, 0
    if "gtx" in compact or "dedicated" in compact:
        return 2, 0

    return 0, 0

