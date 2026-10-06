"""
evaluate_retrieval.py
─────────────────────────────────────────────────────────────────────
Avalia a recuperacao semantica do indice Qdrant usando um JSONL pequeno
de consultas rotuladas.

Uso:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --eval data/eval/retrieval_eval.jsonl --thresholds 0.50 0.55 0.60
    python scripts/evaluate_retrieval.py --report reports/retrieval/retrieval_eval_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    COLLECTION_NAME,
    CONTEXT_SCORE_MARGIN,
    MAX_DOCUMENTS,
    QDRANT_PATH,
    QDRANT_URL,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.rag.chunking import document_id_for
from src.rag.embeddings import get_embedding_provider
from src.rag.qdrant import get_qdrant_client
from src.rag.retriever import evidence_from_hit, select_evidences, unique_sources

DEFAULT_EVAL = Path("data/eval/retrieval_eval.jsonl")
DEFAULT_REPORT = Path("reports/retrieval/retrieval_eval_report.json")


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    expected_url: str | None
    expected_status: str

    @property
    def expected_document_id(self) -> str | None:
        return document_id_for(self.expected_url) if self.expected_url else None


def load_cases(path: Path) -> list[EvalCase]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append(
                EvalCase(
                    id=row.get("id") or f"case_{line_no}",
                    query=row["query"],
                    expected_url=row.get("expected_url"),
                    expected_status=row.get("expected_status", "matched"),
                )
            )
    return cases


def search(client, collection: str, query_vec: list[float], top_k: int):
    hits = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k * RETRIEVAL_OVERFETCH,
        with_payload=True,
    ).points
    evidences = [evidence_from_hit(hit) for hit in hits]
    return select_evidences(
        evidences,
        max_chunks=top_k,
        max_documents=MAX_DOCUMENTS,
        min_score=0.0,
    )


def rank_of_expected(case: EvalCase, sources) -> int | None:
    expected_id = case.expected_document_id
    if expected_id is None:
        return None
    for rank, ev in enumerate(sources, 1):
        if ev.document_id == expected_id:
            return rank
    return None


def evaluate_at_threshold(rows: list[dict], threshold: float) -> dict:
    true_positive = false_positive = true_negative = false_negative = wrong_match = 0
    hit_at_k = 0
    reciprocal_ranks = []

    for row in rows:
        expected_matched = row["expected_status"] == "matched"
        predicted_matched = row["best_score"] >= threshold
        found_expected = row["rank"] is not None

        if expected_matched and predicted_matched and found_expected:
            true_positive += 1
        elif expected_matched and predicted_matched:
            wrong_match += 1
            false_negative += 1
        elif expected_matched:
            false_negative += 1
        elif predicted_matched:
            false_positive += 1
        else:
            true_negative += 1

        if expected_matched and row["rank"]:
            hit_at_k += 1
            reciprocal_ranks.append(1 / row["rank"])

    total = len(rows)
    predicted_positive = true_positive + wrong_match + false_positive
    expected_positive = sum(1 for row in rows if row["expected_status"] == "matched")
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": threshold,
        "accuracy": (true_positive + true_negative) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_at_k": hit_at_k / expected_positive if expected_positive else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "tp": true_positive,
        "fp": false_positive,
        "tn": true_negative,
        "fn": false_negative,
        "wrong_match": wrong_match,
    }


def choose_best_threshold(metrics: list[dict]) -> dict | None:
    if not metrics:
        return None
    return max(metrics, key=lambda row: (row["f1"], row["accuracy"], row["threshold"]))


def build_report(rows: list[dict], thresholds: list[float], args: argparse.Namespace) -> dict[str, Any]:
    metrics = [evaluate_at_threshold(rows, threshold) for threshold in thresholds]
    best = choose_best_threshold(metrics)
    return {
        "collection": args.collection,
        "eval_file": str(args.eval),
        "top_k": args.top_k,
        "current_threshold": args.threshold,
        "recommended_threshold": best["threshold"] if best else None,
        "cases": rows,
        "thresholds": metrics,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.eval)
    provider = get_embedding_provider()
    client = get_qdrant_client(
        url=None if args.local else args.qdrant_url,
        path=args.qdrant_path,
        local_fallback=args.local,
    )

    rows = []
    for case in cases:
        query_vec = provider.embed_query(case.query)
        evidences = search(client, args.collection, query_vec, args.top_k)
        sources = unique_sources(evidences)
        best_score = evidences[0].score if evidences else 0.0
        rank = rank_of_expected(case, sources)
        predicted_status = "matched" if best_score >= args.threshold else "abstained"
        rows.append(
            {
                "id": case.id,
                "expected_status": case.expected_status,
                "predicted_status": predicted_status,
                "best_score": best_score,
                "rank": rank,
                "top_url": sources[0].url if sources else None,
                "expected_url": case.expected_url,
            }
        )

    thresholds = args.thresholds or [
        round(SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN, 2),
        SIMILARITY_THRESHOLD,
        round(SIMILARITY_THRESHOLD + CONTEXT_SCORE_MARGIN, 2),
    ]
    thresholds = sorted(set(thresholds))
    report = build_report(rows, thresholds, args)
    if args.report:
        write_report(report, args.report)

    print(f"Cases: {len(rows)} | collection: {args.collection} | top_k: {args.top_k}")
    print("\nPer-case:")
    for row in rows:
        marker = "ok" if (
            (row["expected_status"] == "abstained" and row["predicted_status"] == "abstained")
            or (row["expected_status"] == "matched" and row["predicted_status"] == "matched" and row["rank"])
        ) else "review"
        print(
            f"- {marker} {row['id']}: expected={row['expected_status']} predicted={row['predicted_status']} "
            f"score={row['best_score']:.4f} rank={row['rank']} top_url={row['top_url']}"
        )

    print("\nThreshold sweep:")
    for metrics in report["thresholds"]:
        print(
            f"- threshold={metrics['threshold']:.2f} accuracy={metrics['accuracy']:.3f} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"hit@k={metrics['hit_at_k']:.3f} mrr={metrics['mrr']:.3f} "
            f"tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} "
            f"fn={metrics['fn']} wrong_match={metrics['wrong_match']}"
        )
    if report["recommended_threshold"] is not None:
        print(f"\nRecommended threshold: {report['recommended_threshold']:.2f}")
    if args.report:
        print(f"Report: {args.report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia retrieval do Checa-AI no Qdrant local")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL, help="JSONL de casos de avaliacao")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Colecao Qdrant")
    parser.add_argument("--qdrant-url", default=QDRANT_URL, help="URL do Qdrant server")
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH, help="Diretorio Qdrant local")
    parser.add_argument("--local", action="store_true", help="Usa Qdrant local em --qdrant-path em vez do server")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K, help="Top-K apos selecao")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD, help="Limiar principal")
    parser.add_argument("--thresholds", type=float, nargs="*", help="Limiares para varredura")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Caminho para salvar relatorio JSON")
    run(parser.parse_args())
