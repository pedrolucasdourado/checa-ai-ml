from types import SimpleNamespace

import src.classifier_runtime as runtime
from src.fact_check_service import FactCheckService


def test_potential_labels_are_stable():
    assert runtime._potential_label(0.10) == "potencial muito baixo"
    assert runtime._potential_label(0.30) == "potencial baixo"
    assert runtime._potential_label(0.50) == "potencial 50/50"
    assert runtime._potential_label(0.73) == "potencial alto"
    assert runtime._potential_label(0.94) == "potencial muito alto"


def test_narrative_uses_requested_wording(monkeypatch):
    monkeypatch.setattr(runtime, "_load_adapter", lambda: SimpleNamespace(
        predict_probability_false=lambda texts: [0.73],
    ))
    value = runtime.narrative("alegação")
    assert value.startswith("Foi identificado potencial alto de desinformação: 73%.")
    assert "não comprova" in value


def test_classifier_runs_only_after_web_returns_no_usable_source(monkeypatch):
    service = object.__new__(FactCheckService)
    monkeypatch.setattr(service, "_deep_search_fallback", lambda *args, **kwargs: {
        "status": "deep_searched", "score": 0.2, "sources": [],
        "counter_narrative": "Pesquisa sem fonte utilizável.",
        "_model_source_urls": [],
    })
    monkeypatch.setattr(
        "src.fact_check_service.classifier_narrative",
        lambda text: "Foi identificado potencial alto de desinformação: 73%.",
    )
    result = service._backend_fallback("alegação", 0.2, [])
    assert result["verdict"] == "deep_searched"
    assert result["sources"] == []
    assert result["counter_narrative"] == "Foi identificado potencial alto de desinformação: 73%."


def test_classifier_runs_after_inconclusive_web_synthesis(monkeypatch):
    service = object.__new__(FactCheckService)
    monkeypatch.setattr(service, "_deep_search_fallback", lambda *args, **kwargs: {
        "status": "deep_searched", "score": 0.2,
        "conclusion": "inconclusiva", "evidence_sufficient": False,
        "sources": [{"title": "Fonte", "domain": "example.org", "url": "https://example.org"}],
        "counter_narrative": "As fontes não são conclusivas.",
        "_model_source_urls": ["https://example.org"],
    })
    monkeypatch.setattr(
        "src.fact_check_service.classifier_narrative",
        lambda text: "Foi identificado potencial baixo de desinformação: 30%.",
    )

    result = service._backend_fallback("alegação", 0.2, [])

    assert result["verdict"] == "deep_searched"
    assert result["sources"] == []
    assert result["counter_narrative"] == "Foi identificado potencial baixo de desinformação: 30%."


def test_conclusive_web_synthesis_does_not_call_classifier(monkeypatch):
    service = object.__new__(FactCheckService)
    monkeypatch.setattr(service, "_deep_search_fallback", lambda *args, **kwargs: {
        "status": "deep_searched", "score": 0.2,
        "conclusion": "contradita", "evidence_sufficient": True,
        "sources": [{"title": "Fonte", "domain": "example.org", "url": "https://example.org"}],
        "counter_narrative": "A fonte contradiz a alegação.",
        "_model_source_urls": ["https://example.org"],
    })
    monkeypatch.setattr(
        "src.fact_check_service.classifier_narrative",
        lambda text: (_ for _ in ()).throw(AssertionError("classifier should not run")),
    )

    result = service._backend_fallback("alegação", 0.2, [])

    assert result["counter_narrative"] == "A fonte contradiz a alegação."
    assert result["sources"] == [{"title": "Fonte", "domain": "example.org", "url": "https://example.org"}]


def test_inconclusive_web_synthesis_is_preserved_when_classifier_is_unavailable(monkeypatch):
    service = object.__new__(FactCheckService)
    monkeypatch.setattr(service, "_deep_search_fallback", lambda *args, **kwargs: {
        "status": "deep_searched", "score": 0.2,
        "conclusion": "inconclusiva", "evidence_sufficient": False,
        "sources": [{"title": "Fonte", "domain": "example.org", "url": "https://example.org"}],
        "counter_narrative": "As fontes não são conclusivas.",
        "_model_source_urls": ["https://example.org"],
    })
    monkeypatch.setattr("src.fact_check_service.classifier_narrative", lambda text: None)

    result = service._backend_fallback("alegação", 0.2, [])

    assert result["counter_narrative"] == "As fontes não são conclusivas."
    assert result["sources"] == [{"title": "Fonte", "domain": "example.org", "url": "https://example.org"}]
