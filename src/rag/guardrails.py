"""
src/rag/guardrails.py
─────────────────────────────────────────────────────────────────────
Guardrails determinísticos (sem custo extra de LLM) em volta da geração:

  1. Saída estruturada: o LLM responde {texto, fontes_usadas}, validado por
     Pydantic. Resposta fora do schema não derruba o pipeline: o texto cru
     é aproveitado e as URLs são extraídas por regex.
  2. Grounding de links: toda URL citada (campo ou texto) precisa estar entre
     as evidências recuperadas; link inventado é removido da resposta.
  3. Prompt injection nos laudos raspados: chunks de risco alto são excluídos
     do contexto; os de risco médio entram, mas ficam registrados.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from src.rag.preprocessing import assess_prompt_injection

_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")
_TRAILING_PUNCT = ".,;:!?"
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class GeneratedAnswer(BaseModel):
    """Contrato de saída do LLM."""

    texto: str = Field(min_length=1, description="Contranarrativa em texto corrido, com citações [n].")
    fontes_usadas: list[str] = Field(
        default_factory=list, description="URLs das evidências efetivamente usadas."
    )


# JSON Schema estrito (OpenAI structured outputs exige additionalProperties=false
# e todos os campos em `required`).
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "contranarrativa",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "texto": {"type": "string"},
                "fontes_usadas": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["texto", "fontes_usadas"],
            "additionalProperties": False,
        },
    },
}


def normalize_url(url: str) -> str:
    return url.strip().rstrip(_TRAILING_PUNCT)


def parse_answer(raw: str) -> tuple[GeneratedAnswer, bool]:
    """Valida a saída do LLM. Retorna (resposta, estruturada?)."""
    raw = (raw or "").strip()
    try:
        return GeneratedAnswer.model_validate(json.loads(raw)), True
    except (ValueError, ValidationError):
        urls = [normalize_url(u) for u in _URL_RE.findall(raw)]
        return GeneratedAnswer(texto=raw or "-", fontes_usadas=list(dict.fromkeys(urls))), False


@dataclass(frozen=True)
class GroundingReport:
    structured: bool
    ungrounded_urls: tuple[str, ...] = ()
    sanitized: bool = False
    used_urls: tuple[str, ...] = field(default=())

    @property
    def grounded(self) -> bool:
        return not self.ungrounded_urls

    def to_dict(self) -> dict:
        return {
            "structured": self.structured,
            "ungrounded_urls": list(self.ungrounded_urls),
            "sanitized": self.sanitized,
        }


def enforce_grounding(
    answer: GeneratedAnswer,
    allowed: list[tuple[str, str]],
    structured: bool = True,
) -> tuple[str, GroundingReport]:
    """
    `allowed`: [(url, dominio)] das evidências recuperadas.
    Remove da resposta qualquer URL fora do contexto e garante que ao menos
    uma fonte real seja creditada. Retorna (texto_final, relatório).
    """
    allowed_urls = {normalize_url(u) for u, _ in allowed}
    cited = {normalize_url(u) for u in answer.fontes_usadas} | {
        normalize_url(u) for u in _URL_RE.findall(answer.texto)
    }
    bad = sorted(cited - allowed_urls)
    texto = answer.texto.strip()

    if bad:
        kept: list[str] = []
        for line in texto.splitlines():
            urls = {normalize_url(u) for u in _URL_RE.findall(line)}
            if urls & set(bad):
                if line.lstrip().lower().startswith("fonte"):
                    continue  # linha de crédito com link inventado: descarta
                for u in urls & set(bad):
                    line = line.replace(u, "")
            kept.append(line.rstrip())
        texto = "\n".join(kept).strip()

    present = {normalize_url(u) for u in _URL_RE.findall(texto)} & allowed_urls
    if allowed and not present:
        credit = "\n".join(f"Fonte: {dom} — Leia mais em: {url}" for url, dom in allowed)
        texto = f"{texto}\n\n{credit}".strip()
        present = allowed_urls

    return texto, GroundingReport(
        structured=structured,
        ungrounded_urls=tuple(bad),
        sanitized=bool(bad),
        used_urls=tuple(sorted(present)),
    )


def injection_risk(text: str, payload_risk: str = "") -> str:
    """Maior risco entre o gravado na ingestão e o recalculado agora (cobre payloads antigos)."""
    now = assess_prompt_injection(text).risk
    return max((now, payload_risk or "low"), key=lambda r: _RISK_ORDER.get(r, 0))
