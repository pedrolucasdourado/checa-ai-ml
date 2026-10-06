from argparse import Namespace
from pathlib import Path

from scripts.evaluate_retrieval import (
    build_report,
    choose_best_threshold,
    evaluate_at_threshold,
    write_report,
)


def test_evaluate_threshold_counts_wrong_match_as_error():
    rows = [
        {"expected_status": "matched", "best_score": 0.9, "rank": 1},
        {"expected_status": "matched", "best_score": 0.8, "rank": None},
        {"expected_status": "matched", "best_score": 0.4, "rank": 1},
        {"expected_status": "abstained", "best_score": 0.7, "rank": None},
        {"expected_status": "abstained", "best_score": 0.2, "rank": None},
    ]

    metrics = evaluate_at_threshold(rows, 0.6)

    assert metrics["tp"] == 1
    assert metrics["wrong_match"] == 1
    assert metrics["fn"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 1 / 3
    assert metrics["recall"] == 1 / 3


def test_choose_best_threshold_prefers_f1_then_accuracy_then_higher_threshold():
    metrics = [
        {"threshold": 0.5, "f1": 0.8, "accuracy": 0.7},
        {"threshold": 0.6, "f1": 0.8, "accuracy": 0.8},
        {"threshold": 0.7, "f1": 0.8, "accuracy": 0.8},
    ]

    assert choose_best_threshold(metrics)["threshold"] == 0.7


def test_build_and_write_report(tmp_path):
    rows = [{"expected_status": "abstained", "best_score": 0.2, "rank": None}]
    args = Namespace(
        collection="fact_checks_silver",
        eval=Path("data/eval/retrieval_eval.jsonl"),
        top_k=5,
        threshold=0.6,
    )

    report = build_report(rows, [0.5, 0.6], args)
    path = tmp_path / "report.json"
    write_report(report, path)

    assert report["recommended_threshold"] == 0.6
    assert path.read_text(encoding="utf-8").startswith("{")
