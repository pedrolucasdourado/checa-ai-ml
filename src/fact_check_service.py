"""
src/fact_check_service.py
─────────────────────────────────────────────────────────────────────
Serviço de verificação de fatos via pipeline RAG:
  1. Converte a mensagem do usuário em embedding (sentence-transformers)
  2. Busca no Qdrant os chunks mais similares (top-k) e deduplica por documento
  3. Se o melhor score >= limiar, gera contranarrativa via GPT-4o-mini
     usando múltiplas evidências numeradas
  4. Caso contrário, retorna abstinência

Singleton por processo: o modelo e o cliente Qdrant são carregados
apenas uma vez no startup do servidor.
"""

from __future__ import annotations

import logging
import os

from src import observability
from src.config import (
    COLLECTION_NAME,
    CONTEXT_SCORE_MARGIN,
    EMBED_MODEL,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_DOCUMENTS,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.rag.corpus_version import corpus_version
from src.rag.embeddings import get_embedding_provider
from src.rag.guardrails import RESPONSE_FORMAT, enforce_grounding, parse_answer
from src.rag.prompts import (
    ABSTENTION_MESSAGE as _ABSTENTION_MESSAGE,
    PROMPT_VERSION,
    current_prompt,
    load_prompt,
)
from src.rag.qdrant import get_qdrant_client
from src.rag.retriever import (
    Evidence,
    build_context,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)

log = logging.getLogger("checa-ai.service")


def _evidence_to_dict(ev: Evidence) -> dict:
    return {
        "titulo": ev.titulo,
        "texto_completo": ev.texto,
        "dominio": ev.dominio,
        "url": ev.url,
        "data_publicacao": ev.data_publicacao,
    }


def _source_to_dict(ev: Evidence) -> dict:
    return {
        "title": ev.titulo,
        "domain": ev.dominio,
        "url": ev.url,
        "published_at": ev.data_publicacao,
        "score": round(ev.score, 4),
        "rerank_score": round(ev.rerank_score, 4) if ev.rerank_score is not None else None,
        "cluster_id": ev.cluster_id,
        "cluster_label": ev.cluster_label,
        "cluster_type": ev.cluster_type,
        "cluster_confidence": ev.cluster_confidence,
        "cluster_review_status": ev.cluster_review_status,
    }


class FactCheckService:
    """
    Singleton que carrega modelo e cliente Qdrant uma única vez.
    Use `FactCheckService.get_instance()` em vez de instanciar diretamente.
    """

    _instance: "FactCheckService | None" = None

    def __init__(self) -> None:
        self._embedder = get_embedding_provider()
        self._qdrant = get_qdrant_client()

    @classmethod
    def get_instance(cls) -> "FactCheckService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── Embedding ────────────────────────────────────────────────────
    def _embed(self, text: str) -> list[float]:
        with observability.observation(
            "embed-query", as_type="embedding", model=EMBED_MODEL, input=text
        ) as obs:
            vec = self._embedder.embed_query(text)
            tokens = getattr(self._embedder, "last_usage_tokens", None)
            obs.update(
                usage_details={"input": tokens} if tokens else None,
                metadata={"provider": getattr(self._embedder, "name", "?"), "dimension": len(vec)},
            )
        return vec

    # ─── Retrieval ────────────────────────────────────────────────────
    def _search(self, query: str, limit: int):
        vec = self._embed(query)
        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=limit,
            with_payload=True,
        ).points

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Evidence]:
        """Busca candidatos, deduplica por documento e limita chunks/documentos."""
        with observability.observation(
            "retrieval",
            as_type="retriever",
            input=query,
            metadata={
                "collection": COLLECTION_NAME,
                "top_k": top_k,
                "overfetch": RETRIEVAL_OVERFETCH,
                "max_documents": MAX_DOCUMENTS,
                "threshold": SIMILARITY_THRESHOLD,
            },
        ) as obs:
            hits = self._search(query, limit=top_k * RETRIEVAL_OVERFETCH)
            candidates = [evidence_from_hit(h) for h in hits]
            # Guardrail: laudos raspados da web são entrada não confiável.
            blocked = [ev for ev in candidates if ev.injection_risk == "high"]
            if blocked:
                log.warning(
                    "Guardrail: %d chunk(s) com prompt injection excluídos: %s",
                    len(blocked), sorted({ev.url for ev in blocked}),
                )
            evidences = with_cluster_rerank([ev for ev in candidates if ev.injection_risk != "high"])
            selected = select_evidences(
                evidences,
                max_chunks=top_k,
                max_documents=MAX_DOCUMENTS,
                min_score=SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN,
            )
            obs.update(
                output=[
                    {"url": ev.url, "score": round(ev.score, 4), "cluster": ev.cluster_label}
                    for ev in selected
                ],
                metadata={
                    "candidates": len(hits),
                    "selected": len(selected),
                    "injection_blocked": len(blocked),
                    "injection_medium": sum(ev.injection_risk == "medium" for ev in selected),
                },
            )
        return selected

    # ─── LLM ─────────────────────────────────────────────────────────
    _llm_client = None

    def _get_llm_client(self):
        if self._llm_client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY não configurada. "
                    "Defina a variável de ambiente ou adicione ao arquivo .env."
                )
            # Um cliente por processo (pool de conexões), com timeout e retry.
            self._llm_client = openai.OpenAI(
                api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
            )
        return self._llm_client

    def _call_llm(self, system: str, user: str) -> str:
        client = self._get_llm_client()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        bundle = current_prompt.get()
        prompt_version = bundle.version if bundle else PROMPT_VERSION
        with observability.observation(
            "llm-generation",
            as_type="generation",
            model=LLM_MODEL,
            model_parameters={"temperature": LLM_TEMPERATURE},
            input=messages,
            version=prompt_version,
            **({"prompt": bundle.langfuse_prompt} if bundle and bundle.langfuse_prompt else {}),
        ) as obs:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=messages,
                response_format=RESPONSE_FORMAT,  # {texto, fontes_usadas}
            )
            text = (resp.choices[0].message.content or "").strip()
            usage = getattr(resp, "usage", None)
            obs.update(
                output=text,
                # Langfuse calcula o custo a partir de model + usage_details.
                usage_details=(
                    {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    if usage
                    else None
                ),
                metadata={"finish_reason": resp.choices[0].finish_reason, "prompt_version": prompt_version},
            )
        return text

    # ─── Public API ───────────────────────────────────────────────────
    @staticmethod
    def _abstention(score: float, evidences: list[Evidence]) -> dict:
        sources = unique_sources(evidences)
        return {
            "status": "abstained",
            "score": round(score, 4),
            "evidence": _evidence_to_dict(evidences[0]) if evidences else None,
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": _ABSTENTION_MESSAGE,
        }

    def verify_claim(self, user_message: str, *, session_id: str | None = None) -> dict:
        """
        Executa o pipeline RAG completo, dentro de um trace do Langfuse.

        Retorno:
            {
                "status":            "matched" | "abstained",
                "score":             float,            # melhor score entre os chunks
                "evidence":          dict | None,      # melhor fonte (compatibilidade)
                "sources":           list[dict],       # uma entrada por documento
                "counter_narrative": str,
                "trace_id":          str | None,       # id p/ feedback; None sem Langfuse
            }
        """
        prompt = load_prompt()
        token = current_prompt.set(prompt)
        try:
            return self._verify_traced(user_message, session_id, prompt)
        finally:
            current_prompt.reset(token)

    def _verify_traced(self, user_message: str, session_id: str | None, prompt) -> dict:
        with observability.request_trace(
            "verify-claim",
            input=user_message,
            session_id=session_id,
            version=prompt.version,
            tags=[f"prompt:{prompt.version}", f"llm:{LLM_MODEL}", f"corpus:{corpus_version()}"],
            metadata={
                "prompt_version": prompt.version,
                "prompt_source": prompt.source,
                "llm_model": LLM_MODEL,
                "embedding_model": EMBED_MODEL,
                "collection": COLLECTION_NAME,
                "corpus_version": corpus_version(),
                "threshold": SIMILARITY_THRESHOLD,
            },
        ) as trace:
            result = self._verify(user_message, prompt)
            result["trace_id"] = trace.trace_id
            self._score_result(trace.trace_id, result)
            trace.update(
                output=result["counter_narrative"],
                metadata={
                    "status": result["status"],
                    "top_score": result["score"],
                    "n_sources": len(result["sources"]),
                },
            )
        return result

    def _verify(self, user_message: str, prompt) -> dict:
        evidences = self._retrieve(user_message)

        if not evidences:
            return self._abstention(0.0, [])

        best_score = max(ev.score for ev in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._abstention(best_score, evidences)

        contexto, used = build_context(evidences, MAX_CONTEXT_CHARS)
        user_prompt = prompt.render_user(query=user_message, contexto=contexto)

        raw = self._call_llm(prompt.render_system(), user_prompt)
        sources = unique_sources(used)
        answer, structured = parse_answer(raw)
        counter_narrative, report = enforce_grounding(
            answer, [(s.url, s.dominio) for s in sources], structured
        )
        if report.sanitized:
            log.warning("Guardrail: URLs fora do contexto removidas: %s", report.ungrounded_urls)

        return {
            "status": "matched",
            "score": round(best_score, 4),
            "evidence": _evidence_to_dict(used[0]),
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": counter_narrative,
            "guardrails": report.to_dict(),
        }

    @staticmethod
    def _score_result(trace_id: str | None, result: dict) -> None:
        """Scores online automáticos: viram séries/filtros no dashboard do Langfuse."""
        if not trace_id:
            return
        observability.score_trace(trace_id, "top_similarity", result["score"])
        observability.score_trace(
            trace_id, "abstained", float(result["status"] == "abstained"), data_type="BOOLEAN"
        )
        if result["status"] == "matched":
            # Mede a saída crua do LLM (antes da sanitização): taxa de link inventado.
            bad = result["guardrails"]["ungrounded_urls"]
            observability.score_trace(
                trace_id,
                "sources_grounded",
                float(not bad),
                data_type="BOOLEAN",
                comment=None if not bad else f"URLs fora do contexto: {bad}"[:500],
            )
            observability.score_trace(
                trace_id,
                "structured_output",
                float(result["guardrails"]["structured"]),
                data_type="BOOLEAN",
            )
