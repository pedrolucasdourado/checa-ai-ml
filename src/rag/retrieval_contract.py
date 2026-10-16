"""Read-only production contract for canonical local Qdrant retrievals.

The module is intentionally independent of LLM generation: it verifies the
same embedding, collection and similarity threshold used by the service before
a deployment is allowed to replace the running application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from src.config import RETRIEVAL_OVERFETCH, RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD
from src.fact_check_service import FactCheckService
from src.rag.retriever import evidence_from_hit


@dataclass(frozen=True)
class RetrievalContractCase:
    query: str
    expected_url: str


@dataclass(frozen=True)
class RetrievalContractResult:
    query: str
    expected_url: str
    candidate_count: int
    best_score: float | None
    expected_score: float | None

    @property
    def matched(self) -> bool:
        return self.expected_score is not None


CANONICAL_CASES = (
    RetrievalContractCase(
        query="Mascaras reduzem a oxigenacao segundo Bolsonaro",
        expected_url=(
            "https://noticias.uol.com.br/confere/ultimas-noticias/2021/06/17/"
            "nao-e-verdade-que-mascaras-reduzem-oxigenacao-como-disse-bolsonaro-em-live.htm"
        ),
    ),
    RetrievalContractCase(
        query="Foto mostra Schwarzenegger usando camiseta de apoio a Bolsonaro",
        expected_url=(
            "https://noticias.uol.com.br/comprova/ultimas-noticias/2021/07/08/"
            "e-falsa-foto-de-schwarzenegger-com-camiseta-de-apoio-a-bolsonaro.htm"
        ),
    ),
)


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))


def evaluate_contract(
    search,
    *,
    cases: tuple[RetrievalContractCase, ...] = CANONICAL_CASES,
    limit: int = RETRIEVAL_TOP_K * RETRIEVAL_OVERFETCH,
) -> list[RetrievalContractResult]:
    """Return raw-retrieval diagnostics without calling generation or Tavily."""
    results: list[RetrievalContractResult] = []
    for case in cases:
        evidences = [evidence_from_hit(hit) for hit in search(case.query, limit)]
        matching = [
            evidence.score
            for evidence in evidences
            if _canonical_url(evidence.url) == _canonical_url(case.expected_url)
        ]
        results.append(
            RetrievalContractResult(
                query=case.query,
                expected_url=case.expected_url,
                candidate_count=len(evidences),
                best_score=max((item.score for item in evidences), default=None),
                expected_score=max(matching, default=None),
            )
        )
    return results


def main() -> int:
    service = FactCheckService.get_instance()
    results = evaluate_contract(service._search)
    payload = [
        {
            "query": result.query,
            "candidate_count": result.candidate_count,
            "best_score": result.best_score,
            "expected_score": result.expected_score,
            "matched": result.matched,
        }
        for result in results
    ]
    print(json.dumps(payload, ensure_ascii=False))
    failed = [
        result
        for result in results
        if result.expected_score is None or result.expected_score < SIMILARITY_THRESHOLD
    ]
    if failed:
        details = "; ".join(
            f"{result.query!r}: expected_score={result.expected_score}, "
            f"best_score={result.best_score}, candidates={result.candidate_count}"
            for result in failed
        )
        raise SystemExit(f"Contrato de recuperação local falhou: {details}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
