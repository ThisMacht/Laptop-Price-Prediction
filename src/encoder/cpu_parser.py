"""CPU parsing helpers for structured laptop feature encoding."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd

from .feature_maps import (
    CPU_FAMILY_GROUP_MAP,
    CPU_FAMILY_ORD_MAP,
    CPU_FAMILY_TIER_FALLBACK,
    CPU_SUFFIX_POWER_MAP,
    CPU_TIER_LOOKUP,
    CPU_TIER_NUMERIC_MAP,
)


@dataclass(frozen=True)
class ParsedCPU:
    brand: str | None
    family: str | None
    generation: int
    suffix: str | None
    model_code_group: str
    family_group: str | None
    family_ord: int
    intel_generation_ord: int
    amd_generation_ord: int
    apple_core_spec: int
    suffix_power_ord: int
    tier_encoded: int
    no_info_brand: int
    no_info_tier: int


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


def parse_cpu(raw: dict[str, Any]) -> ParsedCPU:
    """Parse CPU fields into production numeric CPU features."""
    cpu_text = normalize_text(raw.get("cpu_text"))
    brand = _canonical_cpu_brand(raw.get("cpu_brand")) or _parse_cpu_brand(cpu_text, raw.get("cpu_family"))
    family = _canonical_cpu_family(raw.get("cpu_family"), brand) or _parse_cpu_family(cpu_text, brand)
    generation = _coerce_int(raw.get("cpu_generation"))
    suffix = _canonical_suffix(raw.get("cpu_suffix"))

    if cpu_text:
        generation = generation or _parse_generation_from_text(cpu_text, brand, family)
        suffix = suffix or _parse_suffix_from_text(cpu_text)

    if family and family.startswith("Apple M"):
        generation = _apple_generation_from_family(family)

    model_code_group = _build_model_code_group(brand, family, generation, suffix, cpu_text)
    family_group = CPU_FAMILY_GROUP_MAP.get(family or "")
    family_ord = CPU_FAMILY_ORD_MAP.get(family or "", 0)
    intel_generation_ord = generation if brand == "Intel" else 0
    amd_generation_ord = _amd_generation_ord(generation) if brand == "AMD" else 0
    apple_core_spec = generation if brand == "Apple" and generation else 0
    suffix_power_ord = CPU_SUFFIX_POWER_MAP.get((suffix or "NOSUFFIX").upper(), 0)

    tier_encoded, no_info_tier = _encode_cpu_tier(brand, family, model_code_group, bool(cpu_text))
    no_info_brand = 0 if brand else 1

    return ParsedCPU(
        brand=brand,
        family=family,
        generation=generation,
        suffix=suffix,
        model_code_group=model_code_group,
        family_group=family_group,
        family_ord=family_ord,
        intel_generation_ord=intel_generation_ord,
        amd_generation_ord=amd_generation_ord,
        apple_core_spec=apple_core_spec,
        suffix_power_ord=suffix_power_ord,
        tier_encoded=tier_encoded,
        no_info_brand=no_info_brand,
        no_info_tier=no_info_tier,
    )


def _canonical_cpu_brand(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    key = text.casefold()
    if "intel" in key:
        return "Intel"
    if "amd" in key or "ryzen" in key:
        return "AMD"
    if "apple" in key or re.search(r"\bm[1-5]\b", key):
        return "Apple"
    if "qualcomm" in key or "snapdragon" in key:
        return "Qualcomm"
    return text.strip()


def _parse_cpu_brand(cpu_text: str | None, cpu_family: Any) -> str | None:
    text = " ".join(v for v in [normalize_text(cpu_family), cpu_text] if v).casefold()
    if not text:
        return None
    if "intel" in text or re.search(r"\bcore\s+(?:ultra|i[3579])\b", text):
        return "Intel"
    if "amd" in text or "ryzen" in text:
        return "AMD"
    if "apple" in text or re.search(r"\bm[1-5]\b", text):
        return "Apple"
    if "qualcomm" in text or "snapdragon" in text:
        return "Qualcomm"
    return None


def _canonical_cpu_family(value: Any, brand: str | None) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    return _parse_cpu_family(text, brand)


def _parse_cpu_family(cpu_text: str | None, brand: str | None) -> str | None:
    if not cpu_text:
        return None
    text = cpu_text.casefold()

    ultra = re.search(r"core\s+ultra\s*([579])", text)
    if ultra:
        return f"Intel Core Ultra {ultra.group(1)}"

    core_i = re.search(r"core\s+i\s*([3579])|i([3579])", text)
    if core_i and (brand in {None, "Intel"} or "intel" in text):
        tier = core_i.group(1) or core_i.group(2)
        return f"Intel Core i{tier}"

    if "celeron" in text:
        return "Intel Celeron"
    if "pentium" in text:
        return "Intel Pentium"
    if brand == "Intel" and re.search(r"\bn\d{2,4}\b|n-series|n series", text):
        return "Intel N-Series"

    if "ryzen ai" in text:
        return "AMD Ryzen AI"
    ryzen = re.search(r"ryzen\s*([3579])", text)
    if ryzen:
        return f"AMD Ryzen {ryzen.group(1)}"

    apple = re.search(r"\bm([1-5])\b", text)
    if apple and (brand in {None, "Apple"} or "apple" in text):
        return f"Apple M{apple.group(1)}"

    if "snapdragon x elite" in text:
        return "Snapdragon X Elite"
    if "snapdragon x plus" in text:
        return "Snapdragon X Plus"

    return None


def _parse_generation_from_text(cpu_text: str, brand: str | None, family: str | None) -> int:
    text = cpu_text.casefold()

    if brand == "Apple" or (family or "").startswith("Apple M"):
        match = re.search(r"\bm([1-5])\b", text)
        return int(match.group(1)) if match else 0

    if brand == "Intel":
        ultra_code = re.search(r"core\s+ultra\s+[579]\s+(\d{3})", text)
        if ultra_code:
            first_digit = int(ultra_code.group(1)[0])
            return {1: 14, 2: 15, 3: 16}.get(first_digit, 0)

        code = re.search(r"\b(\d{4,5})(?:[a-z]{1,3}\d?)?\b", text)
        if code:
            number = code.group(1)
            if len(number) >= 5:
                return int(number[:2])
            if number.startswith(("10", "11", "12", "13", "14")):
                return int(number[:2])
            return int(number[0])

        low_end = re.search(r"\bn\d{2,4}\b", text)
        return 1 if low_end else 0

    if brand == "AMD":
        code = re.search(r"\b(\d{4})(?:[a-z]{1,3})?\b", text)
        if code:
            return int(code.group(1)[0])
        if "ryzen ai" in text:
            return 9

    return 0


def _parse_suffix_from_text(cpu_text: str) -> str | None:
    text = cpu_text.upper()
    code_suffix = re.search(r"\b\d{3,5}([A-Z]{1,3}\d?)\b", text)
    if code_suffix:
        return _canonical_suffix(code_suffix.group(1))
    explicit = re.search(r"\b(HX|HS|HK|HQ|UC|MQ|G7|G4|G1|U|P|H|K|Y|M|G)\b", text)
    return _canonical_suffix(explicit.group(1)) if explicit else None


def _canonical_suffix(value: Any) -> str | None:
    text = normalize_text(value)
    if not text:
        return None
    suffix = text.strip().upper()
    return suffix if suffix in CPU_SUFFIX_POWER_MAP else suffix


def _build_model_code_group(
    brand: str | None,
    family: str | None,
    generation: int,
    suffix: str | None,
    cpu_text: str | None,
) -> str:
    clean_suffix = suffix or "NoSuffix"
    if brand == "Intel":
        if family and family.startswith("Intel Core Ultra"):
            series = {14: 100, 15: 200, 16: 300}.get(generation)
            return f"Intel Core Ultra {series}-Series / {clean_suffix}" if series else "Other"
        if family == "Intel N-Series":
            return "Intel N-Series / NoSuffix"
        if generation:
            if clean_suffix.startswith("G"):
                return f"Intel Gen {generation} G-Series / {clean_suffix}"
            return f"Intel Gen {generation} / {clean_suffix}"
    if brand == "AMD":
        if family == "AMD Ryzen AI":
            return f"AMD Ryzen AI HX 300-Series / {clean_suffix}"
        if generation:
            return f"AMD Ryzen {generation}000 / {clean_suffix}"
    if brand == "Apple" and (family or cpu_text):
        return "Apple Core Count Spec"
    return "Other"


def _encode_cpu_tier(
    brand: str | None,
    family: str | None,
    model_code_group: str,
    has_cpu_text: bool,
) -> tuple[int, int]:
    if not brand and not family and not has_cpu_text:
        return CPU_TIER_NUMERIC_MAP["Other"], 1
    if not family:
        return CPU_TIER_NUMERIC_MAP["Other"], 1

    lookup_keys = [
        (brand or "", family, model_code_group),
        (brand or "", family, "Other"),
    ]
    for key in lookup_keys:
        tier = CPU_TIER_LOOKUP.get(key)
        if tier is not None:
            return CPU_TIER_NUMERIC_MAP[tier], 0

    fallback_tier = CPU_FAMILY_TIER_FALLBACK.get(family)
    if fallback_tier is not None:
        return CPU_TIER_NUMERIC_MAP[fallback_tier], 0

    return CPU_TIER_NUMERIC_MAP["Other"], 1


def _coerce_int(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _apple_generation_from_family(family: str) -> int:
    match = re.search(r"M([1-5])", family)
    return int(match.group(1)) if match else 0


def _amd_generation_ord(generation: int) -> int:
    if not generation:
        return 0
    if generation >= 1000:
        return generation
    return generation * 1000
