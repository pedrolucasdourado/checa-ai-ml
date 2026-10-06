"""
src/rag/retriever.py
─────────────────────────────────────────────────────────────────────
Lógica pura de pós-processamento da recuperação: deduplicação por
documento, seleção de evidências e montagem do contexto para o LLM.
(Sem dependência de modelo ou Qdrant, para ser testável isoladamente.)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.config import (
    CLUSTER_APPROVED_BONUS,
    CLUSTER_HIGH_CONFIDENCE_BONUS,
    CLUSTER_LOW_CONFIDENCE_PENALTY,
    CLUSTER_NEEDS_SUBCLUSTER_PENALTY,
    CLUSTER_RERANK_ENABLED,
)


@dataclass(frozen=True)
class Evidence:
    document_id: str
    titulo: str
    dominio: str
    url: str
    data_publicacao: str
    texto: str
    score: float
    chunk_index: int = 0
    cluster_id: int | None = None
    cluster_label: str = ""
    cluster_type: str = ""
    cluster_confidence: str = ""
    cluster_review_status: str = ""
    rerank_score: float | None = None


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evidence_from_hit(hit) -> Evidence:
    """Converte um ScoredPoint do Qdrant em `Evidence` (aceita o payload legado)."""
    p = hit.payload or {}
    return Evidence(
        document_id=p.get("document_id") or p.get("url", ""),
        titulo=p.get("titulo", ""),
        dominio=p.get("dominio", ""),
        url=p.get("url", ""),
        data_publicacao=p.get("data_publicacao", "") or "",
        texto=p.get("texto_chunk") or p.get("texto_completo", ""),
        score=float(hit.score),
        chunk_index=int(p.get("chunk_index", 0)),
        cluster_id=_optional_int(p.get("cluster_id")),
        cluster_label=p.get("cluster_label", "") or "",
        cluster_type=p.get("cluster_type", "") or "",
        cluster_confidence=p.get("cluster_confidence", "") or "",
        cluster_review_status=p.get("cluster_review_status", "") or "",
    )


def cluster_rerank_score(ev: Evidence) -> float:
    """
    Calcula um score de ordenação que usa cluster como sinal fraco.

    O `score` original de similaridade continua intacto para limiar e métricas.
    """
    adjusted = ev.score
    if not CLUSTER_RERANK_ENABLED:
        return adjusted

    if ev.cluster_review_status == "approved":
        adjusted += CLUSTER_APPROVED_BONUS
    elif ev.cluster_review_status == "needs_subcluster":
        adjusted -= CLUSTER_NEEDS_SUBCLUSTER_PENALTY

    if ev.cluster_confidence == "alta":
        adjusted += CLUSTER_HIGH_CONFIDENCE_BONUS
    elif ev.cluster_confidence == "baixa":
        adjusted -= CLUSTER_LOW_CONFIDENCE_PENALTY

    return max(0.0, min(1.0, adjusted))


def with_cluster_rerank(evidences: list[Evidence]) -> list[Evidence]:
    """Retorna evidências com `rerank_score` preenchido, preservando `score`."""
    return [replace(ev, rerank_score=cluster_rerank_score(ev)) for ev in evidences]


def select_evidences(
    evidences: list[Evidence],
    max_chunks: int,
    max_documents: int,
    min_score: float = 0.0,
) -> list[Evidence]:
    """
    Ordena por score de ranking, descarta os abaixo de `min_score` original e
    limita a `max_chunks` chunks de no máximo `max_documents` documentos distintos.
    O documento do melhor chunk sempre entra primeiro.
    """
    selected: list[Evidence] = []
    documents: list[str] = []
    eligible = [ev for ev in evidences if ev.score >= min_score]
    ranked = sorted(
        eligible,
        key=lambda e: (e.rerank_score if e.rerank_score is not None else e.score, e.score),
        reverse=True,
    )
    for ev in ranked:
        if ev.document_id not in documents:
            if len(documents) >= max_documents:
                continue
            documents.append(ev.document_id)
        selected.append(ev)
        if len(selected) >= max_chunks:
            break
    return selected


def unique_sources(evidences: list[Evidence]) -> list[Evidence]:
    """Um item por documento (o de maior score), na ordem de relevância."""
    seen: set[str] = set()
    out: list[Evidence] = []
    for ev in evidences:
        if ev.document_id not in seen:
            seen.add(ev.document_id)
            out.append(ev)
    return out


def build_context(evidences: list[Evidence], max_chars: int) -> tuple[str, list[Evidence]]:
    """
    Monta o bloco de evidências numeradas [1], [2]… por documento/fonte.

    Vários chunks do mesmo documento são agrupados sob o mesmo número, para que
    as citações geradas pelo LLM correspondam às fontes expostas na API/UI.
    Retorna (contexto, fontes efetivamente incluídas).
    """
    grouped: list[tuple[Evidence, list[Evidence]]] = []
    positions: dict[str, int] = {}
    for ev in evidences:
        if ev.document_id in positions:
            grouped[positions[ev.document_id]][1].append(ev)
        else:
            positions[ev.document_id] = len(grouped)
            grouped.append((ev, [ev]))

    parts: list[str] = []
    used: list[Evidence] = []
    total = 0
    for source, chunks in grouped:
        chunk_lines = [
            f"Trecho {i}: {chunk.texto}"
            for i, chunk in enumerate(chunks, 1)
        ]
        trechos = "\n".join(chunk_lines)
        block = (
            f"[{len(used) + 1}] Título: {source.titulo}\n"
            f"Agência: {source.dominio}\n"
            f"Data de publicação: {source.data_publicacao}\n"
            f"URL: {source.url}\n"
            f"{trechos}"
        )
        if used and total + len(block) > max_chars:
            break
        if not used and len(block) > max_chars:
            block = block[:max_chars].rsplit(" ", 1)[0] + " [...]"
        parts.append(block)
        used.append(source)
        total += len(block)
    return "\n\n".join(parts), used
