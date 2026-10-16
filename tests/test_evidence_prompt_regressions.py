import json
from pathlib import Path

import pytest

from src.rag.prompts import EVIDENCE_PROMPT_VERSION, SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parent.parent
REGRESSION_GOLD = {
    "gc-004": "contradita",
    "gc-005": "contradita",
    "gc-007": "contradita",
    "gc-036": "inconclusiva",
    "gc-038": "sustentada",
    "gc-060": "sustentada",
    "gc-071": "contradita",
    "gc-076": "inconclusiva",
    "gc-080": "inconclusiva",
    "gc-096": "inconclusiva",
    "gc-098": "inconclusiva",
}


def _golden_by_id() -> dict[str, dict]:
    rows = (
        json.loads(line)
        for line in (ROOT / "evals" / "golden_claims.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return {row["id"]: row for row in rows}


def test_evidence_prompt_v7_encodes_inconclusive_boundary() -> None:
    normalized = " ".join(SYSTEM_PROMPT.casefold().split())

    assert EVIDENCE_PROMPT_VERSION == "evidence-flow-v7"
    assert "ausência de confirmação" in normalized
    assert "silêncio da fonte" in normalized
    assert "entidade/atribuição" in normalized
    assert "números diferentes" in normalized
    assert "negações simples ou duplas" in normalized
    assert "todos" in normalized and "contraexemplo explícito" in normalized
    assert 'pessoa disse/afirmou p' in normalized
    assert "não use a falsidade de p para contradizer que a pessoa disse p" in normalized
    assert "máscaras reduzem oxigenação" in normalized
    assert "aprovação em comissão ou em uma casa" in normalized
    assert "entrada em vigor e implementação prática são fatos distintos" in normalized
    assert "senado aprovou uma pec sobre a escala 6x1" in normalized


@pytest.mark.parametrize(("case_id", "expected"), REGRESSION_GOLD.items())
def test_reviewed_regression_cases_remain_in_golden_set(case_id: str, expected: str) -> None:
    item = _golden_by_id()[case_id]

    assert item["gold_conclusion"] == expected
    assert item["claim"].strip()
    assert item["evidence"]["texto"].strip()
