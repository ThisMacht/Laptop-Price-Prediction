from __future__ import annotations

from src.prediction.laptop_comparison import (
    build_comparison_row,
    price_gap_million,
    price_gap_pct,
    rank_comparison_rows,
    summarize_comparison,
    verdict_from_gap_pct,
)


def _prediction(predicted: float) -> dict:
    return {
        "predicted_price": predicted,
        "price_range": {"low": predicted - 1, "high": predicted + 1},
        "input_completeness_pct": 90.0,
        "missing_fields": [],
        "uncertainty": {"level": "low"},
        "raw_features": {"brand": "Dell"},
    }


def test_ranking_prefers_lower_actual_price_for_same_prediction() -> None:
    rows = [
        build_comparison_row("A", 12.0, _prediction(15.0), rank=0),
        build_comparison_row("B", 14.0, _prediction(15.5), rank=0),
        build_comparison_row("C", 18.0, _prediction(18.2), rank=0),
    ]
    ranked = rank_comparison_rows(rows)

    assert ranked[0]["label"] == "A"
    assert ranked[0]["rank"] == 1
    assert ranked[0]["verdict"] == "Đáng mua nhất"
    assert ranked[0]["value_score_million_vnd"] == 3.0


def test_price_gap_helpers() -> None:
    assert price_gap_million(14.0, 15.5) == -1.5
    assert price_gap_pct(14.0, 15.5) == round(-9.68, 2)
    assert verdict_from_gap_pct(-9.68, 1) == "Đáng mua nhất"


def test_summarize_comparison() -> None:
    rows = rank_comparison_rows(
        [
            build_comparison_row("A", 12.0, _prediction(11.0), rank=0),
            build_comparison_row("B", 14.0, _prediction(15.5), rank=0),
        ]
    )
    summary = summarize_comparison(rows)
    assert summary["best_pick"] == "B"
    assert "B" in summary["summary"]
