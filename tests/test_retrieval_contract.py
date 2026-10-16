from types import SimpleNamespace

from src.rag.retrieval_contract import RetrievalContractCase, evaluate_contract


EXPECTED_URL = "https://checamos.afp.com/checagem"


def _hit(url: str, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        score=score,
        payload={
            "document_id": url,
            "chunk_index": 0,
            "texto_chunk": "Trecho de uma checagem.",
            "titulo": "Checagem",
            "dominio": "checamos.afp.com",
            "url": url,
        },
    )


def test_retrieval_contract_reports_expected_local_source_and_score():
    case = RetrievalContractCase("alegação conhecida", EXPECTED_URL)

    results = evaluate_contract(
        lambda query, limit: [_hit("https://outra.example/a", 0.91), _hit(EXPECTED_URL, 0.72)],
        cases=(case,),
    )

    assert results[0].candidate_count == 2
    assert results[0].best_score == 0.91
    assert results[0].expected_score == 0.72
    assert results[0].matched is True


def test_retrieval_contract_exposes_missing_expected_source():
    case = RetrievalContractCase("alegação conhecida", EXPECTED_URL)

    result = evaluate_contract(
        lambda query, limit: [_hit("https://outra.example/a", 0.91)],
        cases=(case,),
    )[0]

    assert result.candidate_count == 1
    assert result.best_score == 0.91
    assert result.expected_score is None
    assert result.matched is False
