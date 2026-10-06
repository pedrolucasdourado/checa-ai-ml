"""
apply_silver_clusters_to_qdrant.py
─────────────────────────────────────────────────────────────────────
Grava os metadados de cluster da camada Silver nos payloads do Qdrant.

Por seguranca, roda em dry-run por padrao. Use --apply para escrever.

Uso:
    python scripts/apply_silver_clusters_to_qdrant.py --collection fact_checks_silver
    python scripts/apply_silver_clusters_to_qdrant.py --collection fact_checks_silver --apply
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLLECTION_NAME, QDRANT_PATH, QDRANT_URL
from src.rag.qdrant import get_qdrant_client


DEFAULT_DOCUMENT_CLUSTERS = Path("reports") / "silver_clusters" / "silver_document_clusters.csv"
DEFAULT_CURATION = Path("reports") / "silver_clusters" / "silver_cluster_curation.json"
CLUSTER_VERSION = "silver-clusters-v1"


def cluster_review_status(cluster: dict[str, Any]) -> str:
    """Aprova somente clusters tematicos com confianca alta/media."""
    if cluster.get("cluster_type") == "tema" and cluster.get("confidence") in {"alta", "media"}:
        return "approved"
    return "needs_subcluster"


def build_cluster_payload(cluster: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_id": int(cluster["cluster_id"]),
        "cluster_label": cluster["suggested_label"],
        "cluster_type": cluster["cluster_type"],
        "cluster_confidence": cluster["confidence"],
        "cluster_version": CLUSTER_VERSION,
        "cluster_review_status": cluster_review_status(cluster),
    }


def load_cluster_payloads(path: Path) -> dict[int, dict[str, Any]]:
    curation = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(cluster["cluster_id"]): build_cluster_payload(cluster)
        for cluster in curation.get("clusters", [])
    }


def load_document_payloads(csv_path: Path, curation_path: Path) -> dict[str, dict[str, Any]]:
    cluster_payloads = load_cluster_payloads(curation_path)
    document_payloads: dict[str, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            document_id = row.get("document_id", "").strip()
            if not document_id:
                continue
            cluster_id = int(row["cluster_id"])
            payload = cluster_payloads.get(cluster_id)
            if payload is None:
                continue
            document_payloads[document_id] = payload
    return document_payloads


def payload_key(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def iter_batches(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collect_point_groups(
    client,
    collection: str,
    document_payloads: dict[str, dict[str, Any]],
    batch_size: int,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]], int, int]:
    groups: dict[str, list[str]] = defaultdict(list)
    payload_by_key: dict[str, dict[str, Any]] = {}
    offset = None
    scanned = matched = 0

    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            break

        for record in records:
            scanned += 1
            payload = record.payload or {}
            document_id = str(payload.get("document_id") or "")
            cluster_payload = document_payloads.get(document_id)
            if cluster_payload is None:
                continue
            matched += 1
            key = payload_key(cluster_payload)
            payload_by_key[key] = cluster_payload
            groups[key].append(str(record.id))

        if offset is None:
            break

    return dict(groups), payload_by_key, scanned, matched


def apply_payload_groups(client, collection: str, groups: dict[str, list[str]], payload_by_key: dict[str, dict[str, Any]], update_batch_size: int) -> int:
    updated = 0
    for key, point_ids in groups.items():
        payload = payload_by_key[key]
        for batch in iter_batches(point_ids, update_batch_size):
            client.set_payload(
                collection_name=collection,
                payload=payload,
                points=batch,
                wait=True,
            )
            updated += len(batch)
    return updated


def run(args: argparse.Namespace) -> dict[str, Any]:
    document_payloads = load_document_payloads(args.document_clusters, args.curation)
    client = get_qdrant_client(
        url=None if args.local else args.url,
        path=args.qdrant_path,
        api_key=args.api_key,
        local_fallback=args.local,
    )
    groups, payload_by_key, scanned, matched = collect_point_groups(
        client=client,
        collection=args.collection,
        document_payloads=document_payloads,
        batch_size=args.scroll_batch_size,
    )

    status_counts: dict[str, int] = defaultdict(int)
    label_counts: dict[str, int] = defaultdict(int)
    for key, point_ids in groups.items():
        payload = payload_by_key[key]
        status_counts[payload["cluster_review_status"]] += len(point_ids)
        label_counts[payload["cluster_label"]] += len(point_ids)

    updated = 0
    if args.apply:
        updated = apply_payload_groups(
            client=client,
            collection=args.collection,
            groups=groups,
            payload_by_key=payload_by_key,
            update_batch_size=args.update_batch_size,
        )

    result = {
        "mode": "apply" if args.apply else "dry-run",
        "collection": args.collection,
        "documents_with_cluster": len(document_payloads),
        "points_scanned": scanned,
        "points_matched": matched,
        "points_updated": updated,
        "status_counts": dict(sorted(status_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplica cluster_id/cluster_label da Silver no Qdrant")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--document-clusters", type=Path, default=DEFAULT_DOCUMENT_CLUSTERS)
    parser.add_argument("--curation", type=Path, default=DEFAULT_CURATION)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--url", default=QDRANT_URL)
    parser.add_argument("--api-key")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--scroll-batch-size", type=int, default=512)
    parser.add_argument("--update-batch-size", type=int, default=512)
    parser.add_argument("--apply", action="store_true", help="Escreve no Qdrant. Sem isso, apenas simula.")
    run(parser.parse_args())
