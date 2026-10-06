"""
Pre-processamento para camadas Bronze/Silver do RAG.

A camada Silver enriquece os laudos com metadados de governanca:
deduplicacao, risco de prompt injection, tema heuristico e entidades
candidatas. A implementacao e propositalmente leve para rodar no Windows
sem depender de PyTorch/scikit-learn.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
import unicodedata
from typing import Iterable

from src.rag.chunking import document_id_for

PREPROCESSING_VERSION = "silver-v1"

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ignore_previous", re.compile(r"\b(ignore|ignorem|desconsidere|desconsidere[m]?)\b.{0,80}\b(instru[cç][oõ]es|prompt|regras|anteriores)\b", re.I)),
    ("system_prompt", re.compile(r"\b(system prompt|prompt do sistema|mensagem de sistema|developer message)\b", re.I)),
    ("role_override", re.compile(r"\b(aja como|act as|voce agora e|you are now|finja ser|pretend to be)\b", re.I)),
    ("instruction_leak", re.compile(r"\b(revele|mostre|imprima|exiba|leak|dump)\b.{0,80}\b(prompt|instru[cç][oõ]es|segredo|chave|api key)\b", re.I)),
    ("tool_override", re.compile(r"\b(execute|rode|chame|use a ferramenta|call tool)\b.{0,80}\b(shell|powershell|bash|python|terminal)\b", re.I)),
)

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "saude_ciencia": (
        "vacina",
        "covid",
        "coronavirus",
        "virus",
        "doenca",
        "medico",
        "hospital",
        "anvisa",
        "ciencia",
        "tratamento",
        "remedio",
        "imunizacao",
    ),
    "politica_eleicoes": (
        "eleicao",
        "eleicoes",
        "voto",
        "urna",
        "tse",
        "presidente",
        "governo",
        "deputado",
        "senador",
        "ministro",
        "bolsonaro",
        "lula",
    ),
    "seguranca_justica": (
        "policia",
        "prisao",
        "crime",
        "justica",
        "supremo",
        "stf",
        "lei",
        "processo",
        "investigacao",
    ),
    "golpes_plataformas": (
        "whatsapp",
        "facebook",
        "instagram",
        "pix",
        "golpe",
        "link",
        "cadastro",
        "premio",
        "celular",
        "aplicativo",
    ),
    "meio_ambiente": (
        "amazonia",
        "queimada",
        "desmatamento",
        "clima",
        "chuva",
        "enchente",
        "ambiental",
        "meio ambiente",
    ),
    "internacional": (
        "eua",
        "estados unidos",
        "donald trump",
        "biden",
        "china",
        "russia",
        "ucrania",
        "israel",
        "argentina",
    ),
    "economia": (
        "dinheiro",
        "imposto",
        "preco",
        "salario",
        "auxilio",
        "beneficio",
        "banco",
        "economia",
        "inflacao",
    ),
}

SILVER_PAYLOAD_FIELDS = (
    "layer",
    "preprocessing_version",
    "normalized_text_hash",
    "duplicate_group_id",
    "is_duplicate",
    "duplicate_of_url",
    "injection_risk",
    "injection_matches",
    "topic",
    "topic_score",
    "topic_keywords",
    "entity_candidates",
    "document_quality_score",
)


@dataclass(frozen=True)
class InjectionAssessment:
    risk: str
    matches: tuple[str, ...]


@dataclass(frozen=True)
class TopicAssessment:
    topic: str
    score: int
    keywords: tuple[str, ...]


def normalize_text_for_hash(text: str) -> str:
    """Normaliza texto para hash/deduplicacao exata robusta a espacos/pontuacao."""
    lowered = (text or "").casefold()
    lowered = "".join(
        char
        for char in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(char)
    )
    without_punct = _PUNCT_RE.sub(" ", lowered)
    return _SPACE_RE.sub(" ", without_punct).strip()


def stable_text_hash(text: str) -> str:
    normalized = normalize_text_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def assess_prompt_injection(text: str) -> InjectionAssessment:
    matches = tuple(name for name, pattern in PROMPT_INJECTION_PATTERNS if pattern.search(text or ""))
    if len(matches) >= 2:
        risk = "high"
    elif matches:
        risk = "medium"
    else:
        risk = "low"
    return InjectionAssessment(risk=risk, matches=matches)


def infer_topic(title: str, text: str) -> TopicAssessment:
    corpus = normalize_text_for_hash(f"{title} {text}")
    scores: dict[str, Counter[str]] = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        counts = Counter()
        for keyword in keywords:
            normalized_keyword = normalize_text_for_hash(keyword)
            if " " in normalized_keyword:
                count = corpus.count(normalized_keyword)
            else:
                count = len(re.findall(rf"\b{re.escape(normalized_keyword)}\b", corpus))
            if count:
                counts[keyword] = count
        if counts:
            scores[topic] = counts

    if not scores:
        return TopicAssessment(topic="outros", score=0, keywords=())

    topic, counts = max(scores.items(), key=lambda item: (sum(item[1].values()), item[0]))
    return TopicAssessment(
        topic=topic,
        score=sum(counts.values()),
        keywords=tuple(keyword for keyword, _ in counts.most_common(8)),
    )


def extract_entity_candidates(title: str, text: str, limit: int = 24) -> list[str]:
    """Extrai entidades candidatas simples ate GLiNER entrar no pipeline."""
    sample = f"{title}. {text[:4000]}"
    candidates: Counter[str] = Counter()

    for match in re.finditer(r"\b[A-ZÁÉÍÓÚÃÕÇ]{2,8}\b", sample):
        candidates[match.group(0)] += 2

    phrase_re = re.compile(
        r"\b(?:[A-ZÁÉÍÓÚÃÕÇ][a-záéíóúâêôãõç]+)(?:\s+(?:d[aeo]s?|e|[A-ZÁÉÍÓÚÃÕÇ][a-záéíóúâêôãõç]+)){0,4}"
    )
    for match in phrase_re.finditer(sample):
        value = _SPACE_RE.sub(" ", match.group(0)).strip(" .,:;")
        if len(value) >= 4 and value.casefold() not in {"foto", "veja", "outros"}:
            candidates[value] += 1

    return [value for value, _ in candidates.most_common(limit)]


def document_quality_score(record: dict) -> float:
    text = record.get("texto") or ""
    title = record.get("titulo") or ""
    score = 0.0
    if record.get("url"):
        score += 0.2
    if title:
        score += 0.2
    if len(text) >= 500:
        score += 0.3
    elif len(text) >= 160:
        score += 0.15
    if record.get("data_publicacao"):
        score += 0.15
    if record.get("dominio"):
        score += 0.15
    return round(min(score, 1.0), 3)


def build_silver_records(records: Iterable[dict]) -> tuple[list[dict], dict]:
    """Enriquece registros crus/processados e marca duplicatas exatas."""
    silver_records: list[dict] = []
    first_seen_by_hash: dict[str, dict] = {}
    topic_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    duplicate_count = 0

    for record in records:
        title = (record.get("titulo") or "").strip()
        text = record.get("texto") or ""
        url = (record.get("url") or "").strip()
        normalized_hash = stable_text_hash(f"{title}\n{text}")
        duplicate_of = first_seen_by_hash.get(normalized_hash)
        is_duplicate = duplicate_of is not None
        if is_duplicate:
            duplicate_count += 1
        else:
            first_seen_by_hash[normalized_hash] = record

        injection = assess_prompt_injection(f"{title}\n{text}")
        topic = infer_topic(title, text)
        topic_counts[topic.topic] += 1
        risk_counts[injection.risk] += 1

        silver = {
            **record,
            "document_id": document_id_for(url),
            "layer": "silver",
            "preprocessing_version": PREPROCESSING_VERSION,
            "normalized_text_hash": normalized_hash,
            "duplicate_group_id": normalized_hash,
            "is_duplicate": is_duplicate,
            "duplicate_of_url": (duplicate_of or {}).get("url", "") if is_duplicate else "",
            "injection_risk": injection.risk,
            "injection_matches": list(injection.matches),
            "topic": topic.topic,
            "topic_score": topic.score,
            "topic_keywords": list(topic.keywords),
            "entity_candidates": extract_entity_candidates(title, text),
            "document_quality_score": document_quality_score(record),
        }
        silver_records.append(silver)

    report = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "records": len(silver_records),
        "duplicate_records": duplicate_count,
        "unique_duplicate_groups": len(first_seen_by_hash),
        "topics": dict(sorted(topic_counts.items())),
        "injection_risk": dict(sorted(risk_counts.items())),
    }
    return silver_records, report
