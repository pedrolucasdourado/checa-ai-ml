"""Testa o harness do eval (sem OpenAI): golden set válido, métricas e limites."""

import importlib.util
import json
import sys
import types
from pathlib import Path

_qc = types.ModuleType("qdrant_client")
_qc.QdrantClient = object
sys.modules.setdefault("qdrant_client", _qc)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eval_generation", ROOT / "scripts" / "eval_generation.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

CASES = ev.load_cases(ROOT / "evals" / "golden.jsonl")
TH = json.loads((ROOT / "evals" / "thresholds.json").read_text())


def test_golden_set_is_well_formed_and_covers_all_types():
    assert {c["type"] for c in CASES} == {"supported", "unsupported", "low_score", "injection"}
    assert len({c["id"] for c in CASES}) == len(CASES)
    for c in CASES:
        assert c["expected_status"] in {"matched", "abstained"} and 0 <= c["score"] <= 1
        for e in c["evidences"]:
            assert {"titulo", "dominio", "url", "texto", "data_publicacao"} <= set(e)


def _run(case, llm):
    svc = ev.build_service(case)
    svc._call_llm = lambda s, u: llm(case)
    judge = lambda c, a: {"faithful": True, "unsupported_claims": []}  # noqa: E731
    return ev.run_case(case, judge, svc)


def _good(case):
    e = case["evidences"][-1 if case["type"] == "injection" else 0]  # injetado vem primeiro
    if case["type"] == "unsupported":
        return json.dumps({"laudos_tratam_alegacao": False, "texto": "Sem checagem.", "fontes_usadas": []})
    return json.dumps({"laudos_tratam_alegacao": True, "texto": f"Falso [1].\nFonte: {e['dominio']} — Leia mais em: {e['url']}", "fontes_usadas": [e["url"]]})


def test_perfect_model_passes_all_thresholds():
    rows = [_run(c, _good) for c in CASES]
    metrics = ev.compute_metrics(rows)
    assert ev.check_thresholds(metrics, TH) == []
    assert metrics["abstention_accuracy"] == 1.0 and metrics["injection_leaks"] == 0


def test_low_score_cases_abstain_by_config_threshold():
    for c in CASES:
        if c["type"] == "low_score":
            assert _run(c, _good)["status"] == "abstained"


def test_injection_canary_leak_is_detected_and_fails_gate():
    case = next(c for c in CASES if c["id"] == "inj-medium")  # risco médio: chega ao LLM
    leaky = lambda c: json.dumps({"texto": f"{c['canary']}", "fontes_usadas": []})  # noqa: E731
    rows = [_run(case, leaky)]
    metrics = ev.compute_metrics(rows)
    assert metrics["injection_leaks"] == 1
    assert any("injection_leaks" in f for f in ev.check_thresholds(metrics, TH))


def test_high_risk_injection_never_reaches_llm_context():
    case = next(c for c in CASES if c["id"] == "inj-high")
    seen = {}
    svc = ev.build_service(case)
    svc._call_llm = lambda s, u: seen.update(user=u) or _good(case)
    svc.verify_claim(case["query"])
    assert "PWNED-4412" not in seen["user"]


def test_invented_url_lowers_raw_grounding_but_final_stays_clean():
    bad = lambda c: json.dumps({"texto": "Veja https://fake.example/x", "fontes_usadas": ["https://fake.example/x"]})  # noqa: E731
    row = _run(CASES[0], bad)
    assert row["raw_grounded"] is False and row["grounded"] is True


def test_threshold_check_reports_unfaithful_runs():
    metrics = {"faithfulness": 0.5, "abstention_accuracy": 1.0, "structured_output": 1.0,
               "raw_url_grounded": 1.0, "url_grounded": 1.0, "injection_leaks": 0}
    assert ev.check_thresholds(metrics, TH) == ["faithfulness = 0.5 < mínimo 0.85"]
