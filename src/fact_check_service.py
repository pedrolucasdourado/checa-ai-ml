"""
src/fact_check_service.py
─────────────────────────────────────────────────────────────────────
Serviço de verificação de fatos via pipeline RAG:
  1. Moderação e Guardrails de Entrada (Toxicidade e Injection)
  2. Converte a mensagem do usuário em embedding (sentence-transformers)
  3. Busca no Qdrant os chunks mais similares (top-k) e deduplica por documento
  4. Se o melhor score >= limiar, gera contranarrativa via GPT-4o-mini
     usando múltiplas evidências numeradas
  5. Validação de Guardrails de Saída
  6. Caso contrário, retorna abstinência
"""

from __future__ import annotations

import os
import logging

from src.config import (
    COLLECTION_NAME,
    CONTEXT_SCORE_MARGIN,
    EMBED_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MAX_CONTEXT_CHARS,
    MAX_DOCUMENTS,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
    SAFE_REFUSAL_MESSAGE,
    SAFE_INJECTION_REFUSAL_MESSAGE,
)
from src.rag.embeddings import get_embedding_provider
from src.rag.qdrant import get_qdrant_client
from src.rag.retriever import (
    Evidence,
    build_context,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)
from src.rag.web_search import perform_web_search
from src.safety.moderation import check_moderation
from src.safety.injection import check_injection
from src.safety.validator import validate_output

# Configure logging for safety events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("checa-ai.safety")

# ─── Prompt Templates ────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
Você é um verificador de fatos especializado em combate à desinformação no Brasil.
Sua missão é redigir contranarrativas curtas, didáticas e embasadas em evidências \
jornalísticas reais para serem compartilhadas via WhatsApp ou redes sociais.

Regras estritas:
1. COMECE sempre afirmando o FATO verídico logo na primeira frase (Truth Sandwich).
2. Mencione brevemente a alegação falsa circulando — sem amplificá-la.
3. Use APENAS as evidências contidas nos "Laudos de Checagem" fornecidos. \
PROIBIDO inventar dados, estatísticas ou declarações não presentes nos laudos. \
Se os laudos não tratarem da alegação, diga isso em vez de supor.
4. Tom: cortês, claro, direto e acessível — adequado para leigos.
5. Comprimento: 3 a 5 parágrafos curtos. Sem markdown, sem bullets, só texto corrido.
6. Ao usar uma informação, indique a evidência de origem entre colchetes, por exemplo [1].
7. Finalize com uma linha para cada evidência efetivamente usada, no formato exato:
   "Fonte: <dominio> — Leia mais em: <url>"
   (substitua pelos valores reais da evidência correspondente).
"""

_USER_PROMPT = """\
--- ALEGAÇÃO RECEBIDA ---
{query}

--- LAUDOS DE CHECAGEM RECUPERADOS ---
{contexto}

--- TAREFA ---
Com base EXCLUSIVAMENTE nos laudos acima, redija uma contranarrativa clara e \
fundamentada para desmentir a alegação recebida. Siga as regras do sistema.
"""

_ABSTENTION_MESSAGE = "Abstenção: Nenhuma checagem oficial encontrada para essa alegação."


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
        return self._embedder.embed_query(text)

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
        hits = self._search(query, limit=top_k * RETRIEVAL_OVERFETCH)
        evidences = with_cluster_rerank([evidence_from_hit(h) for h in hits])
        return select_evidences(
            evidences,
            max_chunks=top_k,
            max_documents=MAX_DOCUMENTS,
            min_score=SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN,
        )

    # ─── LLM ─────────────────────────────────────────────────────────
    def _call_llm(self, system: str, user: str) -> str:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. "
                "Defina a variável de ambiente ou adicione ao arquivo .env."
            )

        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    def _deep_search_fallback(self, query: str, score: float, evidences: list[Evidence]) -> dict:
        """
        Fallback quando não há match no RAG: pesquisa na web e sintetiza a resposta.
        """
        from src.config import DEEP_SEARCH_SYSTEM_PROMPT, DEEP_SEARCH_USER_PROMPT

        logger.info("Iniciando DeepSearch para query: %s", query)
        
        # 1. Busca Web
        web_results = perform_web_search(query)
        logger.info("Resultados da web encontrados: %d", len(web_results))
        
        if not web_results:
            # Se nem a web retornou nada, voltamos para a abstenção clássica
            return self._abstention(score, evidences)

        # Formata resultados para o prompt
        formatted_results = "\n\n".join([
            f"Fonte: {r['title']} ({r['url']})\nConteúdo: {r['snippet']}" 
            for r in web_results
        ])
        
        # DEBUG: Print dos resultados reais da web para transparência
        logger.info("--- RESULTADOS REAIS DA WEB ---\n%s\n--------------------------------", formatted_results)
        
        user_prompt = DEEP_SEARCH_USER_PROMPT.format(
            query=query, 
            web_results=formatted_results
        )
        
        # 2. Geração de síntese cautelosa
        try:
            narrative = self._call_llm(DEEP_SEARCH_SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            logger.error("Erro ao gerar síntese DeepSearch: %s", e)
            return self._abstention(score, evidences)

        # 3. Validação de Segurança (Saída)
        val_result = validate_output(narrative)
        if not val_result.is_safe:
            logger.error("Blocked harmful deep search output: category=%s", val_result.category)
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": "Desculpe, mas a pesquisa web gerou um conteúdo que não atende aos nossos critérios de segurança.",
            }

        return {
            "status": "deep_searched",
            "score": round(score, 4),
            "evidence": None,
            "sources": [], # Em um sistema real, poderíamos adicionar as URLs da web aqui
            "counter_narrative": narrative,
        }

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

    def verify_claim(self, user_message: str) -> dict:
        """
        Executa o pipeline RAG completo com camadas de segurança.

        Retorno:
            {
                "status":            "matched" | "abstained" | "blocked",
                "score":             float | None,
                "evidence":          dict | None,
                "sources":           list[dict],
                "counter_narrative": str,
            }
        """
        # 1. Input Guardrail: Moderation (Toxicity)
        mod_result = check_moderation(user_message)
        if not mod_result.is_safe:
            logger.warning(f"Blocked toxic input: category={mod_result.category}")
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": SAFE_REFUSAL_MESSAGE,
            }

        # 2. Input Guardrail: Prompt Injection
        inj_result = check_injection(user_message)
        if not inj_result.is_safe:
            logger.warning(f"Blocked injection attempt: pattern={inj_result.pattern}")
            return {
                "status": "blocked",
                "score": None,
                "evidence": None,
                "sources": [],
                "counter_narrative": SAFE_INJECTION_REFUSAL_MESSAGE,
            }

        # 3. RAG Retrieval
        evidences = self._retrieve(user_message)

        if not evidences:
            return self._deep_search_fallback(user_message, 0.0, [])

        best_score = max(ev.score for ev in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._deep_search_fallback(user_message, best_score, evidences)

        # 4. Generation
        contexto, used = build_context(evidences, MAX_CONTEXT_CHARS)
        user_prompt = _USER_PROMPT.format(query=user_message, contexto=contexto)

        counter_narrative = self._call_llm(_SYSTEM_PROMPT, user_prompt)

        # 5. Output Guardrail: Safety Validation
        val_result = validate_output(counter_narrative)
        if not val_result.is_safe:
            logger.error(f"Blocked harmful output: category={val_result.category}")
            # In case of harmful output, we return a fallback refusal
            return {
                "status": "blocked",
                "score": round(best_score, 4),
                "evidence": _evidence_to_dict(used[0]),
                "sources": [_source_to_dict(s) for s in unique_sources(used)],
                "counter_narrative": "Desculpe, mas a resposta gerada não atende aos nossos critérios de segurança e não pode ser exibida.",
            }

        sources = unique_sources(used)
        return {
            "status": "matched",
            "score": round(best_score, 4),
            "evidence": _evidence_to_dict(used[0]),
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": counter_narrative,
        }
