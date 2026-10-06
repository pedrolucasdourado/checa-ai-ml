from types import SimpleNamespace

from src.rag.retriever import (
    Evidence,
    build_context,
    cluster_rerank_score,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)


def ev(
    doc,
    score,
    idx=0,
    texto="texto",
    cluster_review_status="",
    cluster_confidence="",
    rerank_score=None,
):
    return Evidence(
        document_id=doc, titulo=f"T-{doc}", dominio="d.com", url=f"https://d.com/{doc}",
        data_publicacao="2024-01-01", texto=texto, score=score, chunk_index=idx,
        cluster_review_status=cluster_review_status, cluster_confidence=cluster_confidence,
        rerank_score=rerank_score,
    )


def test_select_orders_by_score_and_limits_chunks():
    items = [ev("a", 0.6), ev("a", 0.9, 1), ev("b", 0.7), ev("c", 0.65)]
    out = select_evidences(items, max_chunks=3, max_documents=3)
    assert [e.score for e in out] == [0.9, 0.7, 0.65]


def test_select_limits_documents():
    items = [ev("a", 0.9), ev("b", 0.8), ev("c", 0.7), ev("d", 0.6), ev("a", 0.5, 1)]
    out = select_evidences(items, max_chunks=5, max_documents=2)
    assert {e.document_id for e in out} == {"a", "b"}
    assert [e.score for e in out] == [0.9, 0.8, 0.5]


def test_select_applies_min_score():
    items = [ev("a", 0.9), ev("b", 0.3)]
    assert [e.document_id for e in select_evidences(items, 5, 3, min_score=0.45)] == ["a"]


def test_select_orders_by_rerank_score_but_filters_by_original_score():
    items = [
        ev("a", 0.81, rerank_score=0.79),
        ev("b", 0.80, rerank_score=0.83),
        ev("c", 0.40, rerank_score=0.99),
    ]
    out = select_evidences(items, max_chunks=3, max_documents=3, min_score=0.75)
    assert [e.document_id for e in out] == ["b", "a"]


def test_cluster_rerank_boosts_approved_and_penalizes_uncurated_clusters():
    approved = ev("a", 0.80, cluster_review_status="approved", cluster_confidence="alta")
    pending = ev("b", 0.81, cluster_review_status="needs_subcluster", cluster_confidence="baixa")

    ranked = select_evidences(with_cluster_rerank([pending, approved]), 2, 2)

    assert cluster_rerank_score(approved) > approved.score
    assert cluster_rerank_score(pending) < pending.score
    assert [e.document_id for e in ranked] == ["a", "b"]


def test_select_empty():
    assert select_evidences([], 5, 3) == []


def test_unique_sources_keeps_best_per_document():
    out = unique_sources([ev("a", 0.9), ev("a", 0.8, 1), ev("b", 0.7)])
    assert [e.document_id for e in out] == ["a", "b"]
    assert out[0].score == 0.9


def test_build_context_numbers_and_respects_limit():
    items = [ev("a", 0.9, texto="x" * 100), ev("b", 0.8, texto="y" * 100)]
    ctx, used = build_context(items, max_chars=10_000)
    assert "[1]" in ctx and "[2]" in ctx and len(used) == 2

    ctx, used = build_context(items, max_chars=250)
    assert len(used) == 1 and "[2]" not in ctx


def test_build_context_groups_chunks_by_document():
    items = [
        ev("a", 0.9, idx=0, texto="primeiro trecho"),
        ev("a", 0.8, idx=1, texto="segundo trecho"),
        ev("b", 0.7, texto="outro documento"),
    ]
    ctx, used = build_context(items, max_chars=10_000)

    assert [e.document_id for e in used] == ["a", "b"]
    assert ctx.count("[1]") == 1
    assert ctx.count("[2]") == 1
    assert "Trecho 1: primeiro trecho" in ctx
    assert "Trecho 2: segundo trecho" in ctx


def test_build_context_truncates_single_oversized_evidence():
    ctx, used = build_context([ev("a", 0.9, texto="palavra " * 500)], max_chars=300)
    assert len(used) == 1 and len(ctx) <= 310 and ctx.endswith("[...]")


def test_evidence_from_hit_supports_new_and_legacy_payload():
    new = SimpleNamespace(score=0.8, payload={
        "document_id": "doc1", "texto_chunk": "chunk", "url": "u", "chunk_index": 2,
        "cluster_id": 7, "cluster_label": "vacinas_covid_imunizacao",
        "cluster_type": "tema", "cluster_confidence": "alta",
        "cluster_review_status": "approved",
    })
    legacy = SimpleNamespace(score=0.7, payload={"texto_completo": "tudo", "url": "u2"})

    e_new, e_old = evidence_from_hit(new), evidence_from_hit(legacy)
    assert (e_new.document_id, e_new.texto, e_new.chunk_index) == ("doc1", "chunk", 2)
    assert (
        e_new.cluster_id,
        e_new.cluster_label,
        e_new.cluster_confidence,
        e_new.cluster_review_status,
    ) == (7, "vacinas_covid_imunizacao", "alta", "approved")
    assert (e_old.document_id, e_old.texto) == ("u2", "tudo")
