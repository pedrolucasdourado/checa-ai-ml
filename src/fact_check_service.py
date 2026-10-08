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

import os
import re
from urllib.parse import urlsplit

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

_BACKEND_SYSTEM_PROMPT = """\
Você verifica alegações usando somente os laudos fornecidos. Escreva uma
contranarrativa clara, em português, de até 500 caracteres para WhatsApp.
Cite cada laudo usado pelo número entre colchetes, como [1]. Não inclua
links nem uma lista de fontes: a API acrescentará as fontes citadas.
Não invente fatos ou citações.
"""

_BACKEND_USER_PROMPT = """\
Alegação: {query}

Laudos de checagem:
{contexto}

Responda apenas com a contranarrativa curta e suas citações numeradas.
"""

_NARRATIVE_LIMIT = 500
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class VerificationGenerationError(RuntimeError):
    """A geração não produziu um resultado fundamentado para o contrato v1."""


def _safe_source_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() for char in url)
        )
    except (TypeError, ValueError):
        return False


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
        self._llm_client = None

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

        client = getattr(self, "_llm_client", None)
        if client is None:
            client = openai.OpenAI(api_key=api_key)
            self._llm_client = client
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

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
        Executa o pipeline RAG completo.

        Retorno:
            {
                "status":            "matched" | "abstained",
                "score":             float,            # melhor score entre os chunks
                "evidence":          dict | None,      # melhor fonte (compatibilidade)
                "sources":           list[dict],       # uma entrada por documento
                "counter_narrative": str,
            }
        """
        evidences = self._retrieve(user_message)

        if not evidences:
            return self._abstention(0.0, [])

        best_score = max(ev.score for ev in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._abstention(best_score, evidences)

        contexto, used = build_context(evidences, MAX_CONTEXT_CHARS)
        user_prompt = _USER_PROMPT.format(query=user_message, contexto=contexto)

        counter_narrative = self._call_llm(_SYSTEM_PROMPT, user_prompt)
        sources = unique_sources(used)

        return {
            "status": "matched",
            "score": round(best_score, 4),
            "evidence": _evidence_to_dict(used[0]),
            "sources": [_source_to_dict(s) for s in sources],
            "counter_narrative": counter_narrative,
        }

    def verify_claim_for_backend(self, text: str) -> dict:
        """Return the grounded, compact v1 result consumed by the backend."""
        evidences = self._retrieve(text)
        best_score = max((ev.score for ev in evidences), default=0.0)
        result = {
            "schema_version": 1,
            "verdict": "insufficient_evidence",
            "similarity_score": round(best_score, 4),
            "counter_narrative": None,
            "sources": [],
        }
        if best_score < SIMILARITY_THRESHOLD:
            return result

        selected_ids = {ev.document_id for ev in unique_sources(evidences)[:2]}
        selected = [ev for ev in evidences if ev.document_id in selected_ids]
        context, used = build_context(selected, MAX_CONTEXT_CHARS)
        prompt = _BACKEND_USER_PROMPT.format(query=text, contexto=context)
        narrative = self._call_llm(_BACKEND_SYSTEM_PROMPT, prompt)
        if len(narrative) > _NARRATIVE_LIMIT:
            condensation_prompt = (
                f"Reduza o texto a no máximo {_NARRATIVE_LIMIT} caracteres, "
                "mantendo somente fatos dos laudos e as citações numeradas. "
                f"Responda apenas com o texto reduzido.\n\n{narrative}"
            )
            narrative = self._call_llm(_BACKEND_SYSTEM_PROMPT, condensation_prompt)
        if len(narrative) > _NARRATIVE_LIMIT:
            raise VerificationGenerationError("contranarrativa excede o tamanho permitido")

        cited = {int(number) for number in _CITATION_PATTERN.findall(narrative)}
        if not cited or any(number < 1 or number > len(used) for number in cited):
            raise VerificationGenerationError("citação sem fonte recuperada")

        sources = []
        seen_urls: set[str] = set()
        for number, ev in enumerate(used, 1):
            if number not in cited:
                continue
            if not _safe_source_url(ev.url):
                raise VerificationGenerationError("fonte citada possui URL inválida")
            if ev.url not in seen_urls:
                sources.append({"title": ev.titulo, "domain": ev.dominio, "url": ev.url})
                seen_urls.add(ev.url)

        result.update(
            verdict="matched",
            counter_narrative=narrative,
            sources=sources,
        )
        return result
