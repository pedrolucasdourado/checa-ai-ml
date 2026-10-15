import os
import json

# Testes comuns nunca enviam traces ao Langfuse real. O smoke explicitamente
# habilitado preserva a configuração do ambiente para validar tracing opcional.
if not os.environ.get("RUN_E2E"):
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["LANGFUSE_CAPTURE_CONTENT"] = "true"

import pytest


@pytest.fixture(autouse=True)
def offline_prompt_catalog(monkeypatch):
    """Versioned synthetic prompts keep unit tests independent from Langfuse."""
    from types import SimpleNamespace

    from src import observability

    templates = {
        'evidence-flow': '{{query}}\n{{contexto}}',
        'claim-analysis': ('--- ENTRADA ---\n{{message}}\n\n'
                           '--- CONTEXTO DA URL, SE DISPONÍVEL ---\n{{url_context}}'),
        'ml-v1-verification': '{{query}}\n{{contexto}}\n{{max_chars}}',
        'ml-v1-condensation': '{{query}}\n{{contexto}}\n{{max_chars}}\n{{narrative}}',
    }

    class Client:
        def get_prompt(self, name, **kwargs):
            return SimpleNamespace(name=name, version=1, type='chat', is_fallback=False,
                prompt=[{'role': 'system', 'content': 'System'},
                        {'role': 'user', 'content': templates[name]}])

    monkeypatch.setattr(observability, '_client', Client())
    monkeypatch.setattr(observability, '_client_failed', False)


@pytest.fixture(autouse=True)
def offline_safety_and_web(monkeypatch, request):
    """Moderação (OpenAI) e DeepSearch (Tavily) são rede: nos testes ficam inertes."""
    if request.node.get_closest_marker("live_e2e"):
        return

    import src.fact_check_service as fcs
    from src.safety.moderation import ModerationResult

    monkeypatch.setattr(fcs, "check_moderation", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "validate_output", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "perform_web_search", lambda query, max_results=5: [])
    def analyze_claim(_self, _system, user):
        message = user.split("--- ENTRADA ---\n", 1)[-1].split(
            "\n\n--- CONTEXTO DA URL", 1
        )[0].strip()
        return json.dumps(
            {
                "has_verifiable_claim": True,
                "claim": message,
                "missing_context": [],
                "explanation": "Alegação verificável para o teste.",
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(fcs.FactCheckService, "_call_claim_analyzer", analyze_claim)
    # Síntese do DeepSearch nunca chama a OpenAI nos testes, mas preserva o
    # contrato estrito para não mascarar falhas de parsing.
    monkeypatch.setattr(
        fcs.FactCheckService,
        "_call_llm_text",
        lambda self, s, u: json.dumps(
            {
                "conclusion": "contradita",
                "evidence_sufficient": True,
                "evidence_level": "forte",
                "explanation": "A evidência externa resolve a alegação [1].",
                "source_urls": [],
            },
            ensure_ascii=False,
        ),
    )
