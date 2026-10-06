import sys
import types
from types import SimpleNamespace

import pytest

# O pipeline é testado com Qdrant/LLM mockados; o stub evita importar o cliente real.
_qc = types.ModuleType("qdrant_client")
_qc.QdrantClient = object
sys.modules["qdrant_client"] = _qc

from src import fact_check_service as fcs  # noqa: E402


def hit(
    doc,
    score,
    idx=0,
    texto="texto do laudo",
    cluster_review_status="",
    cluster_confidence="",
    cluster_label="",
):
    return SimpleNamespace(
        score=score,
        payload={
            "document_id": doc, "chunk_index": idx, "texto_chunk": texto,
            "titulo": f"Título {doc}", "dominio": "g1.globo.com",
            "url": f"https://g1.globo.com/{doc}", "data_publicacao": "2024-05-01",
            "cluster_label": cluster_label,
            "cluster_confidence": cluster_confidence,
            "cluster_review_status": cluster_review_status,
        },
    )


@pytest.fixture
def service(monkeypatch):
    svc = object.__new__(fcs.FactCheckService)  # evita carregar modelo/Qdrant
    svc.hits = []
    svc.prompts = []
    monkeypatch.setattr(svc, "_search", lambda q, limit: svc.hits, raising=False)

    def fake_llm(system, user):
        svc.prompts.append((system, user))
        return "Contranarrativa gerada."

    monkeypatch.setattr(svc, "_call_llm", fake_llm, raising=False)
    return svc


def test_abstains_when_no_hits(service):
    r = service.verify_claim("alegação qualquer")
    assert r["status"] == "abstained" and r["sources"] == [] and r["evidence"] is None
    assert service.prompts == []


def test_abstains_below_threshold_without_calling_llm(service):
    service.hits = [hit("a", fcs.SIMILARITY_THRESHOLD - 0.05)]
    r = service.verify_claim("alegação qualquer")
    assert r["status"] == "abstained"
    assert r["score"] == round(fcs.SIMILARITY_THRESHOLD - 0.05, 4)
    assert service.prompts == []


def test_matched_uses_multiple_documents_and_returns_sources(service):
    service.hits = [hit("a", 0.9), hit("a", 0.85, 1), hit("b", 0.8), hit("c", 0.7)]
    r = service.verify_claim("alegação qualquer")

    assert r["status"] == "matched" and r["score"] == 0.9
    assert [s["url"] for s in r["sources"]] == [
        "https://g1.globo.com/a", "https://g1.globo.com/b", "https://g1.globo.com/c",
    ]
    assert r["evidence"]["url"] == "https://g1.globo.com/a"
    user_prompt = service.prompts[0][1]
    assert "[1]" in user_prompt and "[2]" in user_prompt and "[3]" in user_prompt


def test_context_never_exceeds_limit(service, monkeypatch):
    monkeypatch.setattr(fcs, "MAX_CONTEXT_CHARS", 400)
    service.hits = [hit(d, 0.9 - i / 100, texto="palavra " * 40) for i, d in enumerate("abc")]
    r = service.verify_claim("alegação qualquer")
    assert r["status"] == "matched"
    assert len(r["sources"]) < 3


def test_cluster_rerank_changes_source_order_without_changing_best_similarity(service):
    service.hits = [
        hit(
            "pending",
            0.81,
            cluster_review_status="needs_subcluster",
            cluster_confidence="baixa",
            cluster_label="misto",
        ),
        hit(
            "approved",
            0.80,
            cluster_review_status="approved",
            cluster_confidence="alta",
            cluster_label="urnas_eleicoes_fraude",
        ),
    ]
    r = service.verify_claim("alegação qualquer")

    assert r["status"] == "matched"
    assert r["score"] == 0.81
    assert [s["url"] for s in r["sources"]] == [
        "https://g1.globo.com/approved",
        "https://g1.globo.com/pending",
    ]
    assert r["sources"][0]["cluster_review_status"] == "approved"
