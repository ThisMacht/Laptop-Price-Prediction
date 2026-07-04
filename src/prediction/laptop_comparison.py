"""Rank laptops by comparing actual selling price to model prediction."""

from __future__ import annotations

from typing import Any


VERDICT_THRESHOLDS_PCT = (
    ("Đáng mua", -8.0),
    ("Hợp lý", 5.0),
    ("Hơi đắt", 15.0),
)


def price_gap_million(actual_million: float, predicted_million: float) -> float:
    """Positive gap means the listing is more expensive than the model expects."""
    return round(actual_million - predicted_million, 3)


def price_gap_pct(actual_million: float, predicted_million: float) -> float | None:
    if predicted_million <= 0:
        return None
    return round(100.0 * (actual_million - predicted_million) / predicted_million, 2)


def verdict_from_gap_pct(gap_pct: float | None, rank: int) -> str:
    if gap_pct is None:
        return "Không đánh giá được"

    if rank == 1 and gap_pct <= 0:
        return "Đáng mua nhất"

    for label, upper_bound in VERDICT_THRESHOLDS_PCT:
        if gap_pct <= upper_bound:
            return label

    return "Đắt so với cấu hình"


def build_comparison_row(
    label: str,
    actual_price_million_vnd: float,
    prediction_payload: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    predicted = float(prediction_payload["predicted_price"])
    gap = price_gap_million(actual_price_million_vnd, predicted)
    gap_pct = price_gap_pct(actual_price_million_vnd, predicted)
    value_score = round(predicted - actual_price_million_vnd, 3)

    return {
        "rank": rank,
        "label": label,
        "actual_price_million_vnd": round(actual_price_million_vnd, 3),
        "predicted_price": prediction_payload["predicted_price"],
        "price_range": prediction_payload["price_range"],
        "price_gap_million_vnd": gap,
        "price_gap_pct": gap_pct,
        "value_score_million_vnd": value_score,
        "verdict": verdict_from_gap_pct(gap_pct, rank),
        "input_completeness_pct": prediction_payload.get("input_completeness_pct"),
        "missing_fields": prediction_payload.get("missing_fields", []),
        "uncertainty": prediction_payload.get("uncertainty"),
        "raw_features": prediction_payload.get("raw_features"),
    }


def rank_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by buyer value: higher predicted-minus-actual is a better deal."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["value_score_million_vnd"]),
            float(row.get("price_gap_pct") or 0.0),
            row["label"],
        ),
    )

    ranked: list[dict[str, Any]] = []
    for index, row in enumerate(ordered, start=1):
        updated = dict(row)
        updated["rank"] = index
        updated["verdict"] = verdict_from_gap_pct(row.get("price_gap_pct"), index)
        ranked.append(updated)
    return ranked


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"best_pick": None, "summary": "Không có laptop để so sánh."}

    best = rows[0]
    return {
        "best_pick": best["label"],
        "summary": (
            f"{best['label']} đang có lợi nhất: giá rao "
            f"{best['actual_price_million_vnd']:.2f} triệu, model dự đoán "
            f"{best['predicted_price']:.2f} triệu ({best['verdict'].lower()})."
        ),
    }
