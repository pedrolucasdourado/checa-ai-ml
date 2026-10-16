"""Structured, validated web-search fallback."""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from pydantic import ValidationError

from src.models import EvidenceSource, SourceOrigin
from src.rag.guardrails import injection_risk

log = logging.getLogger("checa-ai.web-search")

_FACT_CHECK_DOMAINS = (
    "aosfatos.org",
    "lupa.uol.com.br",
    "projetocomprova.com.br",
    "checamos.afp.com",
    "factcheck.org",
    "snopes.com",
)
_RECOGNIZED_NEWS_DOMAINS = (
    "agenciabrasil.ebc.com.br",
    "bbc.com",
    "cnn.com.br",
    "estadao.com.br",
    "folha.uol.com.br",
    "g1.globo.com",
    "reuters.com",
    "uol.com.br",
)


def _domain(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().removeprefix("www.")


def _trust_rank(domain: str) -> int:
    if any(domain == item or domain.endswith(f".{item}") for item in _FACT_CHECK_DOMAINS):
        return 0
    if domain.endswith(".gov.br") or domain == "gov.br":
        return 1
    if any(domain == item or domain.endswith(f".{item}") for item in _RECOGNIZED_NEWS_DOMAINS):
        return 2
    return 3


def validate_web_results(raw_results: list[dict], max_results: int = 5) -> list[EvidenceSource]:
    """Normalize provider output and reject invalid or injected evidence."""
    unique: dict[str, EvidenceSource] = {}
    for raw in raw_results:
        url = str(raw.get("url") or "").strip()
        title = str(raw.get("title") or "").strip()
        excerpt = str(raw.get("snippet") or raw.get("content") or "").strip()
        domain = _domain(url)
        if not title or not excerpt or injection_risk(f"{title}\n{excerpt}") == "high":
            continue
        try:
            source = EvidenceSource(
                title=title,
                domain=domain,
                url=url,
                relevant_excerpt=excerpt[:2_000],
                origin=SourceOrigin.WEB,
                published_at=raw.get("published_at"),
            )
        except ValidationError:
            log.warning("Resultado web inválido descartado: %s", url or "(sem URL)")
            continue
        unique.setdefault(source.url, source)
    return sorted(unique.values(), key=lambda item: (_trust_rank(item.domain), item.title))[:max_results]


def perform_web_search(query: str, max_results: int = 5) -> list[dict[str, str | None]]:
    """Query Tavily. Validation is performed by ``validate_web_results``."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        log.info("TAVILY_API_KEY ausente; fallback web indisponível.")
        return []

    try:
        from tavily import TavilyClient

        response = TavilyClient(api_key=api_key).search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
        )
    except Exception:
        # Alguns provedores repetem a consulta dentro da exceção.
        log.warning("Busca Tavily falhou.")
        return []

    return [
        {
            "title": str(item.get("title") or ""),
            "snippet": str(item.get("content") or ""),
            "url": str(item.get("url") or ""),
            "published_at": str(item.get("published_date") or "") or None,
        }
        for item in response.get("results", [])
    ]
