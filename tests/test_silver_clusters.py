from types import SimpleNamespace

import numpy as np

from scripts.build_silver_clusters import (
    aggregate_points,
    cluster_summary,
    kmeans,
    matrix_from_documents,
    pca_2d,
)


class FakeClient:
    def __init__(self, records):
        self.records = records
        self.called = False

    def scroll(self, **kwargs):
        if self.called:
            return [], None
        self.called = True
        return self.records, None


def record(point_id, document_id, vector, title="Titulo", topic="saude_ciencia"):
    return SimpleNamespace(
        id=point_id,
        vector=vector,
        payload={
            "document_id": document_id,
            "titulo": title,
            "url": f"https://example.com/{document_id}",
            "dominio": "example.com",
            "topic": topic,
            "data_publicacao": "2020-01-01",
            "topic_keywords": ["covid", "vacina"],
        },
    )


def test_aggregate_points_groups_chunks_by_document():
    client = FakeClient(
        [
            record("1", "doc-a", [1.0, 0.0], title="Vacina contra covid"),
            record("2", "doc-a", [0.0, 1.0], title="Vacina contra covid"),
            record("3", "doc-b", [0.0, 2.0], title="Urna eletronica", topic="politica_eleicoes"),
        ]
    )

    docs = aggregate_points(client, "collection", batch_size=10, max_points=0)

    assert len(docs) == 2
    doc_a = next(doc for doc in docs if doc.document_id == "doc-a")
    assert doc_a.chunk_count == 2
    assert np.isclose(np.linalg.norm(doc_a.vector), 1.0)
    assert doc_a.topic_keywords["covid"] == 2


def test_pca_and_kmeans_shapes():
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )

    projected, explained = pca_2d(vectors)
    labels = kmeans(vectors, requested_clusters=2)

    assert projected.shape == (4, 2)
    assert 0.0 <= explained <= 1.0
    assert set(labels) <= {0, 1}


def test_cluster_summary_includes_topics_and_examples():
    client = FakeClient(
        [
            record("1", "doc-a", [1.0, 0.0], title="Vacina contra covid"),
            record("2", "doc-b", [0.0, 1.0], title="Urna eletronica", topic="politica_eleicoes"),
        ]
    )
    docs = aggregate_points(client, "collection", batch_size=10, max_points=0)
    vectors = matrix_from_documents(docs)
    labels = np.asarray([0, 1])

    summary = cluster_summary(docs, labels)

    assert vectors.shape == (2, 2)
    assert [item["cluster_id"] for item in summary] == [0, 1]
    assert summary[0]["examples"][0]["url"].startswith("https://example.com/")
