"""
build_silver_clusters.py
─────────────────────────────────────────────────────────────────────
Cria clusters temáticos documentais para a camada Silver usando os
embeddings já indexados no Qdrant.

Esta etapa evita PyTorch/scikit-learn: agrega chunks por documento,
clusteriza os vetores médios com KMeans em NumPy e salva relatórios
para revisão humana antes de escrever qualquer coisa de volta no Qdrant.

Uso:
    python scripts/build_silver_clusters.py --collection fact_checks_silver
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLLECTION_NAME, QDRANT_PATH, QDRANT_URL
from src.rag.qdrant import get_qdrant_client


STOPWORDS = {
    "a",
    "as",
    "ao",
    "aos",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
}


@dataclass
class DocumentAggregate:
    document_id: str
    vector_sum: np.ndarray
    chunk_count: int
    title: str
    url: str
    domain: str
    topic: str
    published_at: str
    topic_keywords: Counter[str]

    @property
    def vector(self) -> np.ndarray:
        vector = self.vector_sum / max(1, self.chunk_count)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clusteriza documentos Silver a partir dos vetores no Qdrant")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--url", default=QDRANT_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--local", action="store_true", help="Usa Qdrant local em vez do server")
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=0, help="0 carrega todos os pontos")
    parser.add_argument("--output-dir", type=Path, default=Path("reports") / "silver_clusters")
    return parser.parse_args()


def as_vector(vector: Any) -> list[float] | None:
    if vector is None:
        return None
    if isinstance(vector, dict):
        if not vector:
            return None
        first_value = next(iter(vector.values()))
        return list(first_value) if first_value is not None else None
    return list(vector)


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", (text or "").casefold())
    return [word for word in words if word not in STOPWORDS]


def update_aggregate(aggregate: DocumentAggregate, payload: dict[str, Any], vector: np.ndarray) -> None:
    aggregate.vector_sum += vector
    aggregate.chunk_count += 1
    for keyword in payload.get("topic_keywords") or []:
        aggregate.topic_keywords[str(keyword)] += 1


def aggregate_points(client, collection: str, batch_size: int, max_points: int) -> list[DocumentAggregate]:
    aggregates: dict[str, DocumentAggregate] = {}
    offset = None
    seen_points = 0

    while True:
        limit = batch_size
        if max_points:
            remaining = max_points - seen_points
            if remaining <= 0:
                break
            limit = min(limit, remaining)

        records, offset = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        for record in records:
            payload = record.payload or {}
            raw_vector = as_vector(record.vector)
            if raw_vector is None:
                continue
            vector = np.asarray(raw_vector, dtype=np.float32)
            document_id = str(payload.get("document_id") or payload.get("url") or record.id)
            if document_id not in aggregates:
                aggregates[document_id] = DocumentAggregate(
                    document_id=document_id,
                    vector_sum=np.zeros_like(vector, dtype=np.float32),
                    chunk_count=0,
                    title=str(payload.get("titulo") or ""),
                    url=str(payload.get("url") or ""),
                    domain=str(payload.get("dominio") or ""),
                    topic=str(payload.get("topic") or "outros"),
                    published_at=str(payload.get("data_publicacao") or ""),
                    topic_keywords=Counter(),
                )
            update_aggregate(aggregates[document_id], payload, vector)
            seen_points += 1

        if offset is None:
            break

    if not aggregates:
        raise RuntimeError(f"Nenhum vetor encontrado na collection '{collection}'.")
    return list(aggregates.values())


def matrix_from_documents(documents: list[DocumentAggregate]) -> np.ndarray:
    return np.vstack([doc.vector for doc in documents]).astype(np.float32)


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, float]:
    if vectors.shape[0] < 2:
        raise RuntimeError("PCA precisa de pelo menos dois documentos.")
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :2] * singular_values[:2]
    eigenvalues = (singular_values**2) / max(1, vectors.shape[0] - 1)
    total = math.fsum(float(value) for value in eigenvalues)
    explained = math.fsum(float(value) for value in eigenvalues[:2]) / total if total else 0.0
    return projected.astype(np.float32), explained


def kmeans(vectors: np.ndarray, requested_clusters: int, iterations: int = 80, seed: int = 42) -> np.ndarray:
    n_points = vectors.shape[0]
    n_clusters = max(1, min(requested_clusters, n_points))
    if n_clusters == 1:
        return np.zeros(n_points, dtype=int)

    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(n_points, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n_points, dtype=int)

    for _ in range(iterations):
        distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for label in range(n_clusters):
            mask = labels == label
            if mask.any():
                centers[label] = vectors[mask].mean(axis=0)
            else:
                centers[label] = vectors[distances.min(axis=1).argmax()]
    return labels


def cluster_summary(documents: list[DocumentAggregate], labels: np.ndarray) -> list[dict[str, Any]]:
    summaries = []
    for label in sorted(set(int(value) for value in labels)):
        cluster_docs = [doc for doc, doc_label in zip(documents, labels) if int(doc_label) == label]
        topics = Counter(doc.topic for doc in cluster_docs)
        domains = Counter(doc.domain for doc in cluster_docs if doc.domain)
        keywords = Counter()
        title_terms = Counter()
        for doc in cluster_docs:
            keywords.update(doc.topic_keywords)
            title_terms.update(tokenize(doc.title))

        examples = [
            {
                "title": doc.title,
                "url": doc.url,
                "topic": doc.topic,
                "domain": doc.domain,
            }
            for doc in cluster_docs[:8]
        ]
        summaries.append(
            {
                "cluster_id": label,
                "document_count": len(cluster_docs),
                "top_topics": topics.most_common(6),
                "top_domains": domains.most_common(6),
                "top_keywords": keywords.most_common(10),
                "top_title_terms": title_terms.most_common(12),
                "examples": examples,
            }
        )
    return summaries


def save_documents_csv(path: Path, documents: list[DocumentAggregate], projected: np.ndarray, labels: np.ndarray) -> None:
    fieldnames = [
        "document_id",
        "cluster_id",
        "x",
        "y",
        "title",
        "url",
        "domain",
        "topic",
        "published_at",
        "chunk_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc, point, label in zip(documents, projected, labels):
            writer.writerow(
                {
                    "document_id": doc.document_id,
                    "cluster_id": int(label),
                    "x": f"{float(point[0]):.6f}",
                    "y": f"{float(point[1]):.6f}",
                    "title": doc.title,
                    "url": doc.url,
                    "domain": doc.domain,
                    "topic": doc.topic,
                    "published_at": doc.published_at,
                    "chunk_count": doc.chunk_count,
                }
            )


def save_summary(path: Path, summary: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.write_text(
        json.dumps({"metadata": metadata, "clusters": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_html(path: Path, documents: list[DocumentAggregate], projected: np.ndarray, labels: np.ndarray, metadata: dict[str, Any]) -> None:
    width, height, pad = 1280, 860, 56
    colors = (
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ca8a04",
        "#0891b2",
        "#db2777",
        "#4b5563",
        "#65a30d",
        "#ea580c",
        "#7c3aed",
        "#0f766e",
    )
    x_min, x_max = float(projected[:, 0].min()), float(projected[:, 0].max())
    y_min, y_max = float(projected[:, 1].min()), float(projected[:, 1].max())
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def sx(value: float) -> float:
        return pad + ((value - x_min) / x_span) * (width - 2 * pad)

    def sy(value: float) -> float:
        return height - pad - ((value - y_min) / y_span) * (height - 2 * pad)

    points = []
    for doc, point, label in zip(documents, projected, labels):
        tooltip = f"C{int(label)} | {doc.topic} | {doc.title} | {doc.url}"
        points.append(
            f'<circle cx="{sx(float(point[0])):.2f}" cy="{sy(float(point[1])):.2f}" '
            f'r="4" fill="{colors[int(label) % len(colors)]}" opacity="0.68">'
            f"<title>{html.escape(tooltip)}</title></circle>"
        )

    cluster_labels = []
    for label in sorted(set(int(value) for value in labels)):
        mask = labels == label
        center = projected[mask].mean(axis=0)
        cluster_labels.append(
            f'<text x="{sx(float(center[0])):.2f}" y="{sy(float(center[1])):.2f}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-size="14" font-weight="700" fill="#111827">C{label} ({int(mask.sum())})</text>'
        )

    content = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Checa-AI Silver clusters</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Arial, sans-serif; background: #f8fafc; color: #111827; }}
    main {{ max-width: 1320px; margin: 0 auto; }}
    h1 {{ font-size: 24px; margin: 0 0 6px; }}
    p {{ margin: 0 0 18px; color: #4b5563; }}
    svg {{ width: 100%; height: auto; background: #fff; border: 1px solid #d1d5db; }}
  </style>
</head>
<body>
  <main>
    <h1>Checa-AI Silver clusters</h1>
    <p>{metadata["documents"]} documentos, {metadata["clusters"]} clusters, PCA 2D explica {metadata["pca_explained_variance_pct"]:.2f}% da variância.</p>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Clusters documentais da camada Silver">
      <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#fff" stroke="#d1d5db" />
      {"".join(points)}
      {"".join(cluster_labels)}
    </svg>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = get_qdrant_client(
        url=None if args.local else args.url,
        path=args.qdrant_path,
        api_key=args.api_key,
        local_fallback=args.local,
    )
    documents = aggregate_points(client, args.collection, args.batch_size, args.max_points)
    vectors = matrix_from_documents(documents)
    projected, explained = pca_2d(vectors)
    labels = kmeans(vectors, args.clusters)
    summary = cluster_summary(documents, labels)

    metadata = {
        "collection": args.collection,
        "documents": len(documents),
        "clusters": len(set(int(value) for value in labels)),
        "requested_clusters": args.clusters,
        "pca_explained_variance_pct": round(explained * 100, 4),
    }

    csv_path = args.output_dir / "silver_document_clusters.csv"
    summary_path = args.output_dir / "silver_cluster_summary.json"
    html_path = args.output_dir / "silver_clusters_pca.html"
    save_documents_csv(csv_path, documents, projected, labels)
    save_summary(summary_path, summary, metadata)
    save_html(html_path, documents, projected, labels, metadata)

    print(f"Documents: {len(documents)} | clusters: {metadata['clusters']}")
    for item in summary:
        topics = ", ".join(f"{topic}:{count}" for topic, count in item["top_topics"][:3])
        terms = ", ".join(term for term, _ in item["top_title_terms"][:6])
        print(f"C{item['cluster_id']}: {item['document_count']} docs | topics={topics} | terms={terms}")
    print("\nSaved:")
    print(f"  csv:     {csv_path}")
    print(f"  summary: {summary_path}")
    print(f"  html:    {html_path}")


if __name__ == "__main__":
    main()
