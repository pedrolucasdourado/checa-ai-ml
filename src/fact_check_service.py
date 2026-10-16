"""Evidence-first claim verification service."""

from __future__ import annotations

import logging
import os
import re
import sys
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from src import observability
from src.config import (
    COLLECTION_NAME,
    CONTEXT_SCORE_MARGIN,
    EMBED_MODEL,
    LLM_MAX_RETRIES,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONTEXT_CHARS,
    MAX_DOCUMENTS,
    RETRIEVAL_OVERFETCH,
    RETRIEVAL_TOP_K,
    SAFE_INJECTION_REFUSAL_MESSAGE,
    SAFE_REFUSAL_MESSAGE,
    SIMILARITY_THRESHOLD,
    HYBRID_ENABLED,
    HYBRID_LEXICAL_WEIGHT,
    NLI_ABSTAIN_ON_DISAGREEMENT,
)
from src.models import (
    ClaimAnalysis,
    EvidenceConclusion,
    EvidenceLevel,
    EvidenceSource,
    FactCheckResponse,
    OperationalStatus,
    SourceOrigin,
)
from src.rag.corpus_version import corpus_version
from src.rag.embeddings import get_embedding_provider
from src.rag.guardrails import (
    CLAIM_RESPONSE_FORMAT,
    RESPONSE_FORMAT,
    enforce_grounding,
    enforce_evidence_consistency,
    parse_answer,
    parse_claim_analysis,
)
from src.rag.prompts import (
    ABSTENTION_MESSAGE as _ABSTENTION_MESSAGE,
    PROMPT_VERSION,
    PromptBundle,
    current_prompt,
    load_claim_prompt,
    load_prompt,
)
from src.rag.retriever import (
    Evidence,
    build_context,
    evidence_from_hit,
    select_evidences,
    unique_sources,
    with_cluster_rerank,
)
from src.rag.entities import all_entities_present, extract_entities, normalize
from src.rag.hybrid import hybrid_rerank
from src.rag.reranker import cross_encoder_rerank
from src.rag import nli
from src.rag.web_search import perform_web_search, validate_web_results
from src.classifier_runtime import narrative as classifier_narrative
from src.safety.injection import check_injection
from src.safety.moderation import check_moderation
from src.safety.validator import validate_output

logger = logging.getLogger("checa-ai.fact-check")

_BACKEND_SYSTEM_PROMPT = """\
Você verifica alegações usando somente os laudos fornecidos. Escreva uma
contranarrativa clara, em português, de até {max_chars} caracteres para WhatsApp.
Cite cada laudo usado pelo número entre colchetes, como [1]. Não inclua
links nem uma lista de fontes: a API acrescentará as fontes citadas.
Não invente fatos ou citações. Preencha a saída estruturada exigida; use
`inconclusiva` e `evidence_sufficient=false` quando os laudos não tratarem da
mesma alegação. Em `source_urls`, informe as URLs completas dos laudos citados.
"""

_BACKEND_USER_PROMPT = """\
Alegação: {query}

Laudos de checagem:
{contexto}

Produza a contranarrativa curta e suas citações numeradas no campo `explanation`.
"""

_NARRATIVE_LIMIT = 500
_DEEP_SEARCH_NARRATIVE_LIMIT = 900
_REPLY_FIXED_RESERVE = 350
_MIN_NARRATIVE_CHARS = 80
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_CHUNK_CITATION_PATTERN = re.compile(r"\[(\d+),\s*trecho\s+\d+\]", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://[^\s\)\]\"'<>]+")
_DIRECT_STATEMENT_PATTERN = re.compile(
    r"(?P<subject>[A-ZÀ-Ý][\wÀ-ÿ]*(?:\s+(?:[A-ZÀ-Ý][\wÀ-ÿ]*|da|das|de|do|dos|e)){0,5})"
    r"\s+(?P<verb>defend\w*|apoi\w*|afirm\w*|declar\w*|disse|diz|"
    r"manifest\w*|posicion\w*|sustent\w*|admit\w*|confirm\w*)",
)
_DIRECT_STATEMENT_ROOTS = (
    "defend", "apoi", "afirm", "declar", "disse", "diz", "manifest",
    "posicion", "sustent", "admit", "confirm",
)
_DIRECT_STATEMENT_WORDS = re.compile(
    r"\b(?:" + "|".join(_DIRECT_STATEMENT_ROOTS) + r")\w*\b"
)
_DIRECT_STATEMENT_NEGATION = re.compile(
    r"\b(?:nao|nega\w*|desment\w*|rejeit\w*)\b(?:\s+\w+){0,2}\s+"
    r"(?:" + "|".join(_DIRECT_STATEMENT_ROOTS) + r")\w*\b"
)
_DIRECT_STATEMENT_STOP_WORDS = frozenset({
    "alegacao", "crime", "crimes", "como", "com", "contra", "defendeu",
    "defende", "disse", "diz", "dos", "das", "ele", "ela", "essa", "este",
    "isso", "lula", "maior", "para", "pela", "pelo", "que", "reduzir",
    "sobre", "uma", "verdade",
})
_generation_version: ContextVar[str | None] = ContextVar("generation_version", default=None)


class VerificationGenerationError(RuntimeError):
    """A geração não produziu um resultado fundamentado para o contrato v1."""


def _safe_source_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        port = parsed.port
        return (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and (port is None or port > 0)
            and parsed.username is None
            and parsed.password is None
            and not any(char.isspace() for char in url)
        )
    except (TypeError, ValueError):
        return False


def _specific_named_entities(claim: str) -> list[str]:
    """Extract named entities with the configured backend and safe fallback."""
    return extract_entities(claim)


def _contradiction_has_entity_anchor(claim: str, sources: list[EvidenceSource]) -> bool:
    """Exige que a refutação mencione a entidade própria principal da alegação."""
    source_text = " ".join(f"{source.title} {source.relevant_excerpt}" for source in sources)
    return all_entities_present(claim, source_text)


def _claim_topic_terms(claim: str, subject: str) -> set[str]:
    """Return conservative content terms for matching a direct statement."""
    subject_terms = set(re.findall(r"[a-z0-9]+", _normalize_text(subject)))
    terms = set()
    for word in re.findall(r"[a-z0-9]+", _normalize_text(claim)):
        stem = word.removesuffix("s")
        if (
            len(stem) >= 5
            and word not in _DIRECT_STATEMENT_STOP_WORDS
            and stem not in subject_terms
            and not any(stem.startswith(root) for root in _DIRECT_STATEMENT_ROOTS)
        ):
            terms.add(stem)
    return terms


def _normalize_text(value: str) -> str:
    return normalize(value)


def _direct_statement_source(query: str, sources: list[EvidenceSource]) -> tuple[str, EvidenceSource] | None:
    """Find explicit web evidence for a person's attributed public position.

    This narrow override only handles an otherwise inconclusive web synthesis.  It
    requires the subject, a statement verb, and at least two topical terms in the
    same validated source; a denial never qualifies as support.
    """
    match = _DIRECT_STATEMENT_PATTERN.search(query)
    if not match:
        return None
    subject = match.group("subject").strip()
    normalized_subject = _normalize_text(subject)
    topic_terms = _claim_topic_terms(query, subject)
    if len(topic_terms) < 2:
        return None

    subject_pattern = re.compile(rf"(?<!\w){re.escape(normalized_subject)}(?!\w)")
    for source in sources:
        source_text = _normalize_text(f"{source.title} {source.relevant_excerpt}")
        if (
            not subject_pattern.search(source_text)
            or not _DIRECT_STATEMENT_WORDS.search(source_text)
            or _DIRECT_STATEMENT_NEGATION.search(source_text)
        ):
            continue
        source_terms = {
            word.removesuffix("s")
            for word in re.findall(r"[a-z0-9]+", source_text)
        }
        if len(topic_terms & source_terms) >= 2:
            return subject, source
    return None


def _nli_check(claim: str, conclusion: EvidenceConclusion, sources: list[EvidenceSource]) -> dict[str, Any] | None:
    """Segunda opinião NLI (desligada por padrão). Retorna None sem opinião."""
    if not sources:
        return None
    result = nli.classify(claim, " ".join(s.relevant_excerpt for s in sources))
    if result is None:
        return None
    return {
        "label": result.label,
        "conclusion": result.conclusion,
        "agrees": nli.agrees(conclusion.value, result),
    }


def _evidence_to_dict(ev: Evidence) -> dict[str, Any]:
    return {
        "titulo": ev.titulo,
        "texto_completo": ev.texto,
        "dominio": ev.dominio,
        "url": ev.url,
        "data_publicacao": ev.data_publicacao,
    }


def _local_source(ev: Evidence) -> EvidenceSource:
    domain = ev.dominio or (urlsplit(ev.url).hostname or "").removeprefix("www.")
    return EvidenceSource(
        title=ev.titulo or domain or "Evidência local",
        domain=domain,
        url=ev.url,
        relevant_excerpt=ev.texto[:2_000] or "Trecho não disponível.",
        origin=SourceOrigin.LOCAL,
        published_at=ev.data_publicacao or None,
    )


def _source_to_dict(ev: Evidence) -> dict[str, Any]:
    """Public fields plus legacy retrieval metadata for internal callers."""
    source = _local_source(ev).model_dump(mode="json")
    source.update(
        score=round(ev.score, 4),
        rerank_score=round(ev.rerank_score, 4) if ev.rerank_score is not None else None,
        cluster_id=ev.cluster_id,
        cluster_label=ev.cluster_label,
        cluster_type=ev.cluster_type,
        cluster_confidence=ev.cluster_confidence,
        cluster_review_status=ev.cluster_review_status,
    )
    return source


def _context_from_sources(sources: list[EvidenceSource], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for index, source in enumerate(sources, 1):
        block = (
            f"[{index}] Título: {source.title}\n"
            f"Domínio: {source.domain}\nURL: {source.url}\n"
            f"Origem: {source.origin.value}\nTrecho: {source.relevant_excerpt}"
        )
        if blocks and used + len(block) + 2 > max_chars:
            break
        if len(block) > max_chars:
            block = block[: max(0, max_chars - 5)].rstrip() + " [...]"
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def _is_standalone_url(value: str) -> bool:
    stripped = value.strip()
    if any(char.isspace() for char in stripped):
        return False
    parsed = urlsplit(stripped)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


class FactCheckService:
    """Orchestrates safety, local retrieval, web fallback and grounded synthesis."""

    _instance: "FactCheckService | None" = None
    _llm_client: Any | None = None

    def __init__(self) -> None:
        from src.rag.qdrant import get_qdrant_client

        self._embedder = get_embedding_provider()
        self._qdrant = get_qdrant_client()
        self._llm_client = None

    @classmethod
    def get_instance(cls) -> "FactCheckService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _embed(self, text: str) -> list[float]:
        with observability.observation(
            "embed-query", as_type="embedding", model=EMBED_MODEL, input=text
        ) as obs:
            vector = self._embedder.embed_query(text)
            tokens = getattr(self._embedder, "last_usage_tokens", None)
            obs.update(
                usage_details={"input": tokens} if tokens else None,
                metadata={
                    "provider": getattr(self._embedder, "name", "?"),
                    "dimension": len(vector),
                },
            )
        return vector

    def _search(self, query: str, limit: int) -> list[Any]:
        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=self._embed(query),
            limit=limit,
            with_payload=True,
        ).points

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Evidence]:
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
            candidates = [evidence_from_hit(hit) for hit in hits]
            blocked = [item for item in candidates if item.injection_risk == "high"]
            if blocked:
                logger.warning("Evidências locais com prompt injection descartadas: %d", len(blocked))
            valid: list[Evidence] = []
            for item in candidates:
                if item.injection_risk == "high":
                    continue
                try:
                    _local_source(item)
                except ValidationError:
                    logger.warning("Evidência local inválida descartada: %s", item.url or "(sem URL)")
                    continue
                valid.append(item)
            ranked = with_cluster_rerank(valid)
            if HYBRID_ENABLED:
                ranked = hybrid_rerank(query, ranked, lexical_weight=HYBRID_LEXICAL_WEIGHT)
            ranked = cross_encoder_rerank(query, ranked)
            selected = select_evidences(
                ranked,
                max_chunks=top_k,
                max_documents=MAX_DOCUMENTS,
                min_score=SIMILARITY_THRESHOLD - CONTEXT_SCORE_MARGIN,
            )
            obs.update(
                output=[{"url": item.url, "score": round(item.score, 4)} for item in selected],
                metadata={
                    "candidates": len(hits),
                    "valid": len(valid),
                    "selected": len(selected),
                    "injection_blocked": len(blocked),
                    "best_score": max((item.score for item in selected), default=None),
                },
            )
        return selected

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            import openai

            if LLM_PROVIDER != "openai" and not LLM_BASE_URL:
                raise RuntimeError(
                    "LLM_BASE_URL é obrigatório quando LLM_PROVIDER não é 'openai'."
                )
            api_key = LLM_API_KEY or os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY ou OPENAI_API_KEY não configurada. "
                    "Defina a credencial do provedor de geração no ambiente."
                )
            client_options = dict(
                api_key=api_key,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_MAX_RETRIES,
            )
            if LLM_BASE_URL:
                client_options["base_url"] = LLM_BASE_URL
            self._llm_client = openai.OpenAI(**client_options)
        return self._llm_client

    def _call_llm(self, system: str, user: str) -> str:
        return self._chat(system, user, RESPONSE_FORMAT)

    def _call_llm_text(self, system: str, user: str) -> str:
        """Compatibility seam for web synthesis; output remains strict JSON."""
        return self._chat(system, user, RESPONSE_FORMAT)

    def _call_claim_analyzer(self, system: str, user: str) -> str:
        return self._chat(system, user, CLAIM_RESPONSE_FORMAT)

    def _analyze_claim(
        self,
        user_message: str,
        url_sources: list[EvidenceSource] | None = None,
        prompt: PromptBundle | None = None,
    ) -> ClaimAnalysis:
        url_context = _context_from_sources(url_sources or [], MAX_CONTEXT_CHARS)
        claim_prompt = prompt or load_claim_prompt()
        token = current_prompt.set(claim_prompt)
        try:
            raw = self._call_claim_analyzer(
                claim_prompt.render_system(),
                claim_prompt.render_user(message=user_message, url_context=url_context or "(não disponível)"),
            )
        finally:
            current_prompt.reset(token)
        return parse_claim_analysis(raw)

    def _chat(self, system: str, user: str, response_format: dict | None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        bundle = current_prompt.get()
        prompt_version = _generation_version.get() or (bundle.version if bundle else PROMPT_VERSION)
        with observability.observation(
            "llm-generation",
            as_type="generation",
            model=LLM_MODEL,
            model_parameters={"temperature": LLM_TEMPERATURE},
            input=messages,
            version=prompt_version,
            **({"prompt": bundle.langfuse_prompt} if bundle and bundle.langfuse_prompt else {}),
        ) as obs:
            response = self._get_llm_client().chat.completions.create(
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                messages=messages,
                **({"response_format": response_format} if response_format else {}),
            )
            text = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            obs.update(
                output=text,
                usage_details=(
                    {
                        "input": usage.prompt_tokens,
                        "output": usage.completion_tokens,
                        "total": usage.total_tokens,
                    }
                    if usage
                    else None
                ),
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_version": prompt_version,
                },
            )
        return text

    @staticmethod
    def _result(
        *,
        status: OperationalStatus,
        claim: str | None,
        conclusion: EvidenceConclusion,
        sufficient: bool,
        level: EvidenceLevel,
        explanation: str,
        sources: list[EvidenceSource],
        score: float | None,
        evidence: Evidence | None = None,
        guardrails: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        public = FactCheckResponse(
            status=status,
            claim=claim,
            conclusion=conclusion,
            evidence_sufficient=sufficient,
            evidence_level=level,
            explanation=explanation,
            sources=sources,
        ).model_dump(mode="json")
        # These aliases are intentionally internal/backward-compatible and are filtered by FastAPI.
        public.update(
            score=round(score, 4) if score is not None else None,
            evidence=_evidence_to_dict(evidence) if evidence else None,
            counter_narrative=public["explanation"],
        )
        if guardrails is not None:
            public["guardrails"] = guardrails
        return public

    @classmethod
    def _abstention(
        cls,
        score: float,
        evidences: list[Evidence],
        *,
        claim: str | None,
        explanation: str = _ABSTENTION_MESSAGE,
    ) -> dict[str, Any]:
        unique = unique_sources(evidences)
        sources = [_local_source(item) for item in unique]
        result = cls._result(
            status=OperationalStatus.ABSTAINED,
            claim=claim,
            conclusion=EvidenceConclusion.INCONCLUSIVE,
            sufficient=False,
            level=EvidenceLevel.LIMITED if sources else EvidenceLevel.NONE,
            explanation=explanation,
            sources=sources,
            score=score,
            evidence=evidences[0] if evidences else None,
        )
        result["sources"] = [_source_to_dict(item) for item in unique]
        return result

    @classmethod
    def _blocked(
        cls,
        message: str,
        score: float | None = None,
        *,
        claim: str | None = None,
    ) -> dict[str, Any]:
        return cls._result(
            status=OperationalStatus.BLOCKED,
            claim=claim,
            conclusion=EvidenceConclusion.INCONCLUSIVE,
            sufficient=False,
            level=EvidenceLevel.NONE,
            explanation=message,
            sources=[],
            score=score,
        )

    @staticmethod
    def _log_output_block(stage: str, category: str | None, *, trace_id: str | None = None) -> None:
        """Audit a blocked output without recording user or model content."""
        logger.error(
            "Saída bloqueada: stage=%s category=%s trace_id=%s",
            stage,
            category or "unspecified",
            trace_id or "unavailable",
        )

    def _synthesize(
        self,
        query: str,
        sources: list[EvidenceSource],
        prompt: PromptBundle,
        *,
        use_web_seam: bool = False,
        max_explanation_chars: int | None = None,
        include_source_credit: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        context = _context_from_sources(sources, MAX_CONTEXT_CHARS)
        user_prompt = prompt.render_user(query=query, contexto=context)
        system_prompt = prompt.render_system()
        if max_explanation_chars is not None:
            system_prompt += (
                "\nLimite o campo explanation a no máximo "
                f"{max_explanation_chars} caracteres."
            )
        raw = (
            self._call_llm_text(system_prompt, user_prompt)
            if use_web_seam
            else self._call_llm(system_prompt, user_prompt)
        )
        answer, structured = parse_answer(raw)
        explanation, report = enforce_grounding(
            answer,
            [(source.url, source.domain) for source in sources],
            structured,
            append_credit=include_source_credit,
        )
        used_urls = set(report.used_urls)
        used_sources = [source for source in sources if source.url in used_urls]
        if report.invalid_citations:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE
            answer.evidence_level = EvidenceLevel.LIMITED if used_sources else EvidenceLevel.NONE
            explanation = (
                "A resposta gerada citou uma referência que não pertence às evidências "
                "recuperadas; por segurança, a conclusão foi marcada como inconclusiva."
            )
        enforce_evidence_consistency(
            answer,
            used_urls=used_urls,
            invalid_citations=report.invalid_citations,
        )
        if not used_sources and answer.evidence_sufficient:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE
            answer.evidence_level = EvidenceLevel.NONE
        result = {
            "conclusion": answer.conclusion,
            "sufficient": answer.evidence_sufficient,
            "level": answer.evidence_level,
            "explanation": explanation,
            "sources": used_sources,
            "model_source_urls": tuple(answer.source_urls),
            "guardrails": report.to_dict(),
        }
        return result, structured

    def _deep_search_fallback(
        self,
        query: str,
        score: float,
        abstention_evidences: list[Evidence],
        prompt: PromptBundle | None = None,
        *,
        web_sources: list[EvidenceSource] | None = None,
        abstention_explanation: str = _ABSTENTION_MESSAGE,
        include_model_sources: bool = False,
    ) -> dict[str, Any]:
        if web_sources is None:
            raw_results = perform_web_search(query)
            web_sources = validate_web_results(raw_results)
        if not web_sources:
            return self._abstention(
                score,
                abstention_evidences,
                claim=query,
                explanation=abstention_explanation,
            )

        prompt = prompt or load_prompt()
        token = current_prompt.set(None)
        try:
            decision, _ = self._synthesize(
                query,
                web_sources,
                prompt,
                use_web_seam=True,
                max_explanation_chars=_DEEP_SEARCH_NARRATIVE_LIMIT,
                include_source_credit=not include_model_sources,
            )
            if len(decision["explanation"]) > _DEEP_SEARCH_NARRATIVE_LIMIT:
                decision, _ = self._synthesize(
                    query,
                    web_sources,
                    prompt,
                    use_web_seam=True,
                    max_explanation_chars=_DEEP_SEARCH_NARRATIVE_LIMIT,
                    include_source_credit=not include_model_sources,
                )
        except Exception:
            # Não registra a mensagem do provedor (pode repetir a alegação), mas registra
            # tipo e status para diagnosticar (quota, 401, timeout, schema...).
            exc = sys.exc_info()[1]
            logger.error(
                "Síntese do fallback web falhou: error=%s status=%s sources=%d",
                type(exc).__name__, getattr(exc, "status_code", None), len(web_sources),
            )
            return self._abstention(
                score,
                abstention_evidences,
                claim=query,
                explanation=abstention_explanation,
            )
        finally:
            current_prompt.reset(token)

        if (
            decision["conclusion"] is EvidenceConclusion.INCONCLUSIVE
            or not decision["sufficient"]
        ):
            direct_source = _direct_statement_source(query, web_sources)
            if direct_source is not None:
                subject, source = direct_source
                decision.update(
                    conclusion=EvidenceConclusion.SUPPORTED,
                    sufficient=True,
                    level=EvidenceLevel.MODERATE,
                    explanation=(
                        f"Segundo a fonte, {source.title} [1]. Isso confirma a posição "
                        f"atribuída a {subject}; não equivale, por si só, a uma decisão formal."
                    ),
                    sources=[source],
                    model_source_urls=(source.url,),
                )
                logger.info("Síntese web inconclusiva corrigida por fonte direta validada.")

        if not decision["explanation"]:
            logger.warning("Síntese DeepSearch vazia")
            return self._abstention(
                score,
                abstention_evidences,
                claim=query,
                explanation=abstention_explanation,
            )
        if len(decision["explanation"]) > _DEEP_SEARCH_NARRATIVE_LIMIT:
            logger.warning("Síntese DeepSearch acima do limite de caracteres")
            if not include_model_sources or not decision["sources"]:
                return self._abstention(
                    score,
                    abstention_evidences,
                    claim=query,
                    explanation=abstention_explanation,
                )
            decision["explanation"] = (
                "Pesquisei a alegação, mas não encontrei uma checagem conclusiva "
                "suficiente para confirmar ou refutar o que foi dito."
            )

        output_validation = validate_output(decision["explanation"])
        if not output_validation.is_safe:
            self._log_output_block("web_synthesis", output_validation.category)
            return self._blocked(
                "A resposta gerada pela busca externa não atende aos critérios de segurança.",
                claim=query,
            )
        result = self._result(
            status=OperationalStatus.DEEP_SEARCHED,
            claim=query,
            conclusion=decision["conclusion"],
            sufficient=decision["sufficient"],
            level=decision["level"],
            explanation=decision["explanation"],
            sources=decision["sources"],
            score=score,
            guardrails=decision["guardrails"],
        )
        if include_model_sources:
            result["_model_source_urls"] = decision["model_source_urls"]
        return result

    def verify_claim(self, user_message: str, *, session_id: str | None = None) -> dict[str, Any]:
        prompt = load_prompt()
        token = current_prompt.set(prompt)
        try:
            return self._verify_traced(user_message, session_id, prompt)
        finally:
            current_prompt.reset(token)

    def _verify_traced(
        self,
        user_message: str,
        session_id: str | None,
        prompt: PromptBundle,
    ) -> dict[str, Any]:
        claim_prompt = load_claim_prompt()
        with observability.request_trace(
            "verify-claim",
            input=user_message,
            session_id=session_id,
            version=prompt.version,
            tags=[
                f"prompt:{prompt.version}",
                f"claim-prompt:{claim_prompt.version}",
                f"llm:{LLM_MODEL}",
                f"corpus:{corpus_version()}",
            ],
            metadata={
                "prompt_version": prompt.version,
                "prompt_hash": prompt.sha256,
                "claim_prompt_version": claim_prompt.version,
                "claim_prompt_hash": claim_prompt.sha256,
                "prompt_source": prompt.source,
                "llm_model": LLM_MODEL,
                "embedding_model": EMBED_MODEL,
                "collection": COLLECTION_NAME,
                "corpus_version": corpus_version(),
                "retrieval_threshold": SIMILARITY_THRESHOLD,
            },
        ) as trace:
            result = self._verify(user_message, prompt, claim_prompt)
            result["trace_id"] = trace.trace_id
            self._score_result(trace.trace_id, result)
            trace.update(
                output=result["explanation"],
                metadata={
                    "status": result["status"],
                    "conclusion": result["conclusion"],
                    "retrieval_top_score": result["score"],
                    "n_sources": len(result["sources"]),
                },
            )
        return result

    def _verify(
        self,
        user_message: str,
        prompt: PromptBundle,
        claim_prompt: PromptBundle | None = None,
    ) -> dict[str, Any]:
        moderation = check_moderation(user_message)
        if not moderation.is_safe:
            logger.warning("Entrada bloqueada por moderação: %s", moderation.category)
            return self._blocked(SAFE_REFUSAL_MESSAGE)

        injection = check_injection(user_message)
        if not injection.is_safe:
            logger.warning("Entrada bloqueada por prompt injection: %s", injection.pattern)
            return self._blocked(SAFE_INJECTION_REFUSAL_MESSAGE)

        prefetched_web_sources: list[EvidenceSource] | None = None
        if _is_standalone_url(user_message):
            prefetched_web_sources = validate_web_results(perform_web_search(user_message))

        claim_analysis = self._analyze_claim(
            user_message,
            prefetched_web_sources,
            prompt=claim_prompt,
        )
        if not claim_analysis.has_verifiable_claim or not claim_analysis.claim:
            explanation = claim_analysis.explanation
            if claim_analysis.missing_context:
                missing = "; ".join(claim_analysis.missing_context)
                explanation = f"{explanation} Para verificar, informe: {missing}."
            if not validate_output(explanation).is_safe:
                explanation = (
                    "Não foi possível identificar uma alegação verificável com o contexto "
                    "fornecido. Inclua o fato, a data ou o local relevante."
                )
            return self._abstention(
                0.0,
                [],
                claim=None,
                explanation=explanation,
            )

        claim = claim_analysis.claim
        evidences = self._retrieve(claim)
        if not evidences:
            return self._deep_search_fallback(
                claim,
                0.0,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        best_score = max(item.score for item in evidences)
        if best_score < SIMILARITY_THRESHOLD:
            return self._deep_search_fallback(
                claim,
                best_score,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        context, used_evidences = build_context(evidences, MAX_CONTEXT_CHARS)
        # Keep the mature grouping/truncation behavior from the local RAG path.
        local_sources = [_local_source(item) for item in unique_sources(used_evidences)]
        raw = self._call_llm(
            prompt.render_system(),
            prompt.render_user(query=claim, contexto=context),
        )
        answer, structured = parse_answer(raw)
        explanation, report = enforce_grounding(
            answer,
            [(source.url, source.domain) for source in local_sources],
            structured,
        )
        used_urls = set(report.used_urls)
        result_sources = [source for source in local_sources if source.url in used_urls]

        if report.invalid_citations:
            answer.evidence_sufficient = False
            answer.conclusion = EvidenceConclusion.INCONCLUSIVE

        nli_opinion = _nli_check(claim, answer.conclusion, result_sources)
        if nli_opinion is not None:
            logger.info("NLI auxiliar: %s", nli_opinion)
            if nli_opinion["agrees"] is False and NLI_ABSTAIN_ON_DISAGREEMENT:
                answer.evidence_sufficient = False
                answer.conclusion = EvidenceConclusion.INCONCLUSIVE

        if (
            answer.conclusion is EvidenceConclusion.CONTRADICTED
            and not _contradiction_has_entity_anchor(claim, result_sources)
        ):
            evidence_by_url = {item.url: item for item in unique_sources(used_evidences)}
            cited_evidences = [
                evidence_by_url[url]
                for url in report.used_urls
                if url in evidence_by_url
            ]
            return self._deep_search_fallback(
                claim,
                best_score,
                cited_evidences,
                prompt,
                web_sources=prefetched_web_sources,
                abstention_explanation=(
                    "As fontes recuperadas tratam de entidade diferente da alegação; "
                    "não há evidência suficiente para contradizê-la."
                ),
            )

        if answer.conclusion is EvidenceConclusion.INCONCLUSIVE or not answer.evidence_sufficient:
            evidence_by_url = {item.url: item for item in unique_sources(used_evidences)}
            cited_evidences = [
                evidence_by_url[url]
                for url in report.used_urls
                if url in evidence_by_url
            ]
            result = self._deep_search_fallback(
                claim,
                best_score,
                cited_evidences,
                prompt,
                web_sources=prefetched_web_sources,
                abstention_explanation=(
                    explanation if cited_evidences else _ABSTENTION_MESSAGE
                ),
            )
            if result["status"] == "abstained":
                result["abstained_by"] = "model"
            return result

        if not result_sources:
            return self._deep_search_fallback(
                claim,
                best_score,
                [],
                prompt,
                web_sources=prefetched_web_sources,
            )

        output_validation = validate_output(explanation)
        if not output_validation.is_safe:
            self._log_output_block("local_synthesis", output_validation.category)
            return self._blocked(
                "A resposta gerada não atende aos critérios de segurança e não pode ser exibida.",
                best_score,
                claim=claim,
            )

        result = self._result(
            status=OperationalStatus.MATCHED,
            claim=claim,
            conclusion=answer.conclusion,
            sufficient=answer.evidence_sufficient,
            level=answer.evidence_level,
            explanation=explanation,
            sources=result_sources,
            score=best_score,
            evidence=used_evidences[0],
            guardrails=report.to_dict(),
        )
        by_url = {item.url: item for item in unique_sources(used_evidences)}
        result["sources"] = [_source_to_dict(by_url[source.url]) for source in result_sources]
        return result

    def verify_claim_for_backend(
        self, text: str, *, user_id: str | None = None, session_id: str | None = None
    ) -> dict:
        """Return the grounded, compact v1 result consumed by the backend.

        `user_id`/`session_id` são pseudônimos do backend (nunca telefone): agrupam os traces
        por pessoa/conversa no Langfuse para medir uso e custo.
        """
        with observability.request_trace(
            "verify-v1",
            input=text,
            user_id=user_id,
            session_id=session_id,
            version="verify-v1",
            tags=["prompt:verify-v1", f"llm:{LLM_MODEL}", f"corpus:{corpus_version()}"]
            + (["channel:whatsapp"] if user_id else []),
            metadata={"collection": COLLECTION_NAME, "threshold": SIMILARITY_THRESHOLD},
        ) as trace:
            result = self._verify_claim_for_backend(text)
            trace.update(
                output=result["counter_narrative"],
                metadata={"verdict": result["verdict"], "top_score": result["similarity_score"]},
            )
            return result

    def _verify_claim_for_backend(self, text: str) -> dict:
        mod_result = check_moderation(text)
        if not mod_result.is_safe:
            logger.warning("Blocked toxic v1 input: category=%s", mod_result.category)
            return self._backend_result("blocked", None, SAFE_REFUSAL_MESSAGE)
        inj_result = check_injection(text)
        if not inj_result.is_safe:
            logger.warning("Blocked v1 injection: pattern=%s", inj_result.pattern)
            return self._backend_result("blocked", None, SAFE_INJECTION_REFUSAL_MESSAGE)

        claim_analysis = self._analyze_claim(text, prompt=load_claim_prompt())
        if not claim_analysis.has_verifiable_claim or not claim_analysis.claim:
            return self._backend_result("insufficient_evidence", 0.0)

        claim = claim_analysis.claim
        evidences = self._retrieve(claim)
        best_score = max((ev.score for ev in evidences), default=0.0)
        result = self._backend_result("insufficient_evidence", best_score)
        if best_score < SIMILARITY_THRESHOLD:
            return self._backend_fallback(claim, best_score, evidences)

        selected_sources: list[Evidence] = []
        selected_urls: set[str] = set()
        source_chars = 0
        max_source_chars = 1000 - _REPLY_FIXED_RESERVE - _MIN_NARRATIVE_CHARS
        for ev in unique_sources(evidences):
            if not _safe_source_url(ev.url):
                continue
            additional = 0 if ev.url in selected_urls else len(ev.url) + bool(selected_urls)
            if source_chars + additional > max_source_chars:
                continue
            selected_sources.append(ev)
            selected_urls.add(ev.url)
            source_chars += additional
            if len(selected_sources) == 2:
                break
        if not selected_sources or not any(
            ev.score >= SIMILARITY_THRESHOLD for ev in selected_sources
        ):
            raise VerificationGenerationError("nenhuma fonte válida cabe na resposta")

        narrative_limit = min(
            _NARRATIVE_LIMIT, 1000 - _REPLY_FIXED_RESERVE - source_chars
        )
        selected_ids = {ev.document_id for ev in selected_sources}
        selected = [ev for ev in evidences if ev.document_id in selected_ids]
        context, used = build_context(selected, MAX_CONTEXT_CHARS)
        prompt = _BACKEND_USER_PROMPT.format(query=claim, contexto=context)
        system_prompt = _BACKEND_SYSTEM_PROMPT.format(max_chars=narrative_limit)
        token = _generation_version.set("verify-v1")
        try:
            raw = self._call_llm(system_prompt, prompt)
        finally:
            _generation_version.reset(token)
        answer, _ = parse_answer(raw)
        if not answer.laudos_tratam_alegacao:
            return self._backend_fallback(claim, best_score, used)
        narrative = answer.texto.strip()
        if len(narrative) > narrative_limit:
            condensation_prompt = (
                f"Reduza o texto a no máximo {narrative_limit} caracteres, "
                "mantendo somente fatos dos laudos e as citações numeradas. "
                f"Responda apenas com o texto reduzido.\n\n{narrative}"
            )
            token = _generation_version.set("verify-v1")
            try:
                raw = self._call_llm(system_prompt, condensation_prompt)
            finally:
                _generation_version.reset(token)
            answer, _ = parse_answer(raw)
            if not answer.laudos_tratam_alegacao:
                return self._backend_fallback(claim, best_score, used)
            narrative = answer.texto.strip()
        narrative = _CHUNK_CITATION_PATTERN.sub(r"[\1]", narrative)
        if len(narrative) > narrative_limit:
            raise VerificationGenerationError("contranarrativa excede o tamanho permitido")

        cited = {int(number) for number in _CITATION_PATTERN.findall(narrative)}
        if not cited or any(number < 1 or number > len(used) for number in cited):
            raise VerificationGenerationError("citação sem fonte recuperada")

        sources = []
        seen_urls: set[str] = set()
        source_numbers: dict[str, int] = {}
        citation_numbers: dict[int, int] = {}
        for number, ev in enumerate(used, 1):
            if number not in cited:
                continue
            if not _safe_source_url(ev.url):
                raise VerificationGenerationError("fonte citada possui URL inválida")
            if ev.url not in seen_urls:
                sources.append({"title": ev.titulo, "domain": ev.dominio, "url": ev.url})
                seen_urls.add(ev.url)
                source_numbers[ev.url] = len(sources)
            citation_numbers[number] = source_numbers[ev.url]
        allowed_urls = seen_urls
        narrative_urls = {url.rstrip(".,;:!?") for url in _URL_PATTERN.findall(narrative)}
        reported_urls: set[str] = set()
        for reference in answer.fontes_usadas:
            numbered = _CITATION_PATTERN.fullmatch(reference)
            if numbered or reference.isdecimal():
                number = int(numbered.group(1) if numbered else reference)
                if number not in citation_numbers:
                    raise VerificationGenerationError("fonte sem citação recuperada")
                reported_urls.add(used[number - 1].url)
            else:
                reported_urls.add(reference)
        if narrative_urls - allowed_urls or reported_urls - allowed_urls:
            raise VerificationGenerationError("URL sem fonte recuperada")
        narrative = _CITATION_PATTERN.sub(
            lambda match: f"[{citation_numbers[int(match.group(1))]}]", narrative
        )

        val_result = validate_output(narrative)
        if not val_result.is_safe:
            self._log_output_block("v1_generation", val_result.category)
            return self._backend_result(
                "blocked", best_score,
                "Desculpe, mas a resposta gerada não atende aos nossos critérios de segurança e não pode ser exibida.",
            )

        result.update(verdict="matched", counter_narrative=narrative, sources=sources)
        return result

    @staticmethod
    def _backend_result(verdict: str, score: float | None, narrative: str | None = None) -> dict:
        return {
            "schema_version": 1,
            "verdict": verdict,
            "similarity_score": round(score, 4) if score is not None else None,
            "counter_narrative": narrative,
            "sources": [],
        }

    def _backend_fallback(self, text: str, score: float, evidences: list[Evidence]) -> dict:
        result = self._deep_search_fallback(
            text, score, evidences, include_model_sources=True,
        )
        if result["status"] == "deep_searched":
            # A web search may find pages that are real but do not establish the
            # claim. In that case, expose the textual-pattern classifier as an
            # auxiliary signal. If it is unavailable, retain the cautious web
            # response below rather than degrading a useful explanation.
            if (
                result.get("conclusion") == EvidenceConclusion.INCONCLUSIVE
                or result.get("evidence_sufficient") is False
            ):
                auxiliary = classifier_narrative(text)
                if auxiliary:
                    return self._backend_result("deep_searched", score, auxiliary)
            declared = set(result.get("_model_source_urls", ()))
            sources = []
            source_chars = 0
            for source in result["sources"]:
                url = source["url"]
                if url not in declared or not _safe_source_url(url):
                    continue
                added_chars = len(url) + bool(sources)
                if source_chars + added_chars > 570:
                    continue
                sources.append({key: source[key] for key in ("title", "domain", "url")})
                source_chars += added_chars
                if len(sources) == 2:
                    break
            if sources:
                response = self._backend_result(
                    "deep_searched", score, result["counter_narrative"],
                )
                response["sources"] = sources
                return response
        if result["status"] == "blocked":
            return self._backend_result("blocked", result["score"], result["counter_narrative"])
        auxiliary = classifier_narrative(text)
        if auxiliary:
            return self._backend_result("deep_searched", score, auxiliary)
        return self._backend_result("insufficient_evidence", score)

    @staticmethod
    def _score_result(trace_id: str | None, result: dict[str, Any]) -> None:
        if not trace_id:
            return
        if result["score"] is not None:
            observability.score_trace(trace_id, "top_similarity", result["score"])
        for flag in ("abstained", "deep_searched", "blocked"):
            observability.score_trace(
                trace_id,
                flag,
                float(result["status"] == flag),
                data_type="BOOLEAN",
            )
        if result["status"] in {"matched", "deep_searched"} and result.get("guardrails"):
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
