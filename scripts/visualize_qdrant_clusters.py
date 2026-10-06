"""
Visualize Qdrant vectors as 2D clusters.

Examples:
    python scripts/visualize_qdrant_clusters.py
    python scripts/visualize_qdrant_clusters.py --clusters 12
    python scripts/visualize_qdrant_clusters.py --url http://localhost:6333
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLLECTION_NAME, QDRANT_PATH, QDRANT_URL
from src.rag.qdrant import get_qdrant_client


PAYLOAD_COLUMNS = (
    "titulo",
    "dominio",
    "document_id",
    "chunk_index",
    "data_publicacao",
    "url",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project Qdrant vectors to 2D and save cluster plots."
    )
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument(
        "--url",
        default=QDRANT_URL,
        help="Qdrant server URL. If omitted, uses local Qdrant path from config.",
    )
    parser.add_argument("--api-key", help="Qdrant API key, when using --url.")
    parser.add_argument("--local", action="store_true", help="Use local Qdrant path instead of server URL.")
    parser.add_argument(
        "--method",
        choices=("pca",),
        default="pca",
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters used only for plot colors.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Maximum points to load from Qdrant.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "qdrant_clusters",
    )
    return parser.parse_args()


def get_client(args: argparse.Namespace):
    return get_qdrant_client(
        url=None if args.local else args.url,
        path=args.qdrant_path,
        api_key=args.api_key,
        local_fallback=args.local,
    )


def as_vector(vector: Any) -> list[float] | None:
    if vector is None:
        return None
    if isinstance(vector, dict):
        if not vector:
            return None
        first_value = next(iter(vector.values()))
        return list(first_value) if first_value is not None else None
    return list(vector)


def load_points(
    client: QdrantClient,
    collection: str,
    limit: int,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    vectors: list[list[float]] = []
    offset = None

    while len(vectors) < limit:
        records, offset = client.scroll(
            collection_name=collection,
            limit=min(batch_size, limit - len(vectors)),
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            break

        for record in records:
            vector = as_vector(record.vector)
            if vector is None:
                continue
            payload = record.payload or {}
            vectors.append(vector)
            rows.append({"id": str(record.id), **payload})

        if offset is None:
            break

    if not vectors:
        raise RuntimeError(f"No vectors found in collection '{collection}'.")

    return np.asarray(vectors, dtype=np.float32), rows


def project(vectors: np.ndarray, method: str) -> np.ndarray:
    if vectors.shape[0] < 2:
        raise RuntimeError("At least two vectors are required for a 2D plot.")

    if method == "pca":
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
        return u[:, :2] * singular_values[:2]

    raise ValueError(f"Unsupported method: {method}")


def cluster(projected: np.ndarray, requested_clusters: int) -> np.ndarray:
    n_points = projected.shape[0]
    n_clusters = max(1, min(requested_clusters, n_points))
    if n_clusters == 1:
        return np.zeros(n_points, dtype=int)

    rng = np.random.default_rng(42)
    centers = projected[rng.choice(n_points, size=n_clusters, replace=False)].copy()
    labels = np.zeros(n_points, dtype=int)

    for _ in range(80):
        distances = ((projected[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for label in range(n_clusters):
            mask = labels == label
            if mask.any():
                centers[label] = projected[mask].mean(axis=0)
            else:
                farthest = distances.min(axis=1).argmax()
                centers[label] = projected[farthest]

    return labels


def pca_explained_variance(vectors: np.ndarray) -> float:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = (singular_values**2) / max(1, vectors.shape[0] - 1)
    total = math.fsum(float(value) for value in eigenvalues)
    if total == 0:
        return 0.0
    return math.fsum(float(value) for value in eigenvalues[:2]) / total


def save_csv(
    output_path: Path,
    projected: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = ["id", "x", "y", "cluster", *PAYLOAD_COLUMNS]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for point, label, row in zip(projected, labels, rows):
            writer.writerow(
                {
                    **row,
                    "x": f"{point[0]:.6f}",
                    "y": f"{point[1]:.6f}",
                    "cluster": int(label),
                }
            )


def save_plot(
    output_path: Path,
    projected: np.ndarray,
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    collection: str,
    method: str,
) -> None:
    width = 1200
    height = 850
    pad = 56
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
    )

    x_min, x_max = float(projected[:, 0].min()), float(projected[:, 0].max())
    y_min, y_max = float(projected[:, 1].min()), float(projected[:, 1].max())
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def sx(value: float) -> float:
        return pad + ((value - x_min) / x_span) * (width - 2 * pad)

    def sy(value: float) -> float:
        return height - pad - ((value - y_min) / y_span) * (height - 2 * pad)

    circles: list[str] = []
    for index, (point, label, row) in enumerate(zip(projected, labels, rows)):
        row_title = str(row.get("titulo") or "").strip()
        title = f"Point {index} | cluster C{int(label)}"
        if row_title:
            title = f"{title} | {row_title}"
        circles.append(
            (
                f'<circle cx="{sx(float(point[0])):.2f}" '
                f'cy="{sy(float(point[1])):.2f}" r="4" '
                f'fill="{colors[int(label) % len(colors)]}" opacity="0.72">'
                f"<title>{html.escape(title)}</title></circle>"
            )
        )

    labels_svg: list[str] = []
    for label in sorted(set(labels)):
        mask = labels == label
        center = projected[mask].mean(axis=0)
        labels_svg.append(
            (
                f'<text x="{sx(float(center[0])):.2f}" y="{sy(float(center[1])):.2f}" '
                f'text-anchor="middle" dominant-baseline="central" '
                f'font-size="14" font-weight="700" fill="#111">'
                f'C{int(label)} ({int(mask.sum())})</text>'
            )
        )

    legend_items = []
    for label in sorted(set(labels)):
        color = colors[int(label) % len(colors)]
        count = int((labels == label).sum())
        legend_items.append(
            f'<span><i style="background:{color}"></i>C{int(label)} ({count})</span>'
        )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qdrant clusters - {html.escape(collection)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f7f8fa;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 4px;
    }}
    p {{
      margin: 0 0 18px;
      color: #52606d;
    }}
    svg {{
      width: 100%;
      height: auto;
      background: #fff;
      border: 1px solid #d9e2ec;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-top: 14px;
      color: #323f4b;
      font-size: 14px;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend i {{
      width: 12px;
      height: 12px;
      display: inline-block;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Qdrant clusters: {html.escape(collection)}</h1>
    <p>{html.escape(method.upper())} projection colored by KMeans cluster.</p>
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="2D projection of Qdrant vectors">
      <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#fff" stroke="#d9e2ec" />
      <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#9fb3c8" />
      <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#9fb3c8" />
      {"".join(circles)}
      {"".join(labels_svg)}
    </svg>
    <div class="legend">{"".join(legend_items)}</div>
  </main>
</body>
</html>
"""
    output_path.write_text(content, encoding="utf-8")


def print_summary(labels: np.ndarray, rows: list[dict[str, Any]]) -> None:
    print("\nCluster summary")
    print("---------------")
    for label in sorted(set(labels)):
        cluster_rows = [row for row, row_label in zip(rows, labels) if row_label == label]
        size = len(cluster_rows)
        titles = [str(row.get("titulo") or "").strip() for row in cluster_rows]
        titles = [title for title in titles if title]
        sample = titles[: min(3, len(titles))]
        print(f"C{label}: {size} points")
        for title in sample:
            print(f"  - {title[:120]}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(args)
    vectors, rows = load_points(
        client=client,
        collection=args.collection,
        limit=args.max_points,
        batch_size=args.batch_size,
    )
    projected = project(vectors, args.method)
    labels = cluster(projected, args.clusters)

    base = f"{args.collection}_{args.method}_{vectors.shape[0]}pts"
    plot_path = args.output_dir / f"{base}.html"
    csv_path = args.output_dir / f"{base}.csv"

    save_plot(plot_path, projected, labels, rows, args.collection, args.method)
    save_csv(csv_path, projected, labels, rows)
    print_summary(labels, rows)

    print("\nSaved:")
    print(f"  plot: {plot_path}")
    print(f"  csv:  {csv_path}")
    print(
        f"\nLoaded {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions "
        f"across {len(set(labels))} clusters."
    )
    if args.method == "pca":
        explained_pct = pca_explained_variance(vectors) * 100
        print(f"PCA explained variance in 2D: {explained_pct:.2f}%")


if __name__ == "__main__":
    main()
