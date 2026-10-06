import os

# Testes nunca enviam traces ao Langfuse real, mesmo com chaves no .env.
# (load_dotenv não sobrescreve variáveis já definidas.)
os.environ["LANGFUSE_ENABLED"] = "false"

import pytest


@pytest.fixture(autouse=True)
def offline_safety_and_web(monkeypatch):
    """Moderação (OpenAI) e DeepSearch (Tavily) são rede: nos testes ficam inertes."""
    import src.fact_check_service as fcs
    from src.safety.moderation import ModerationResult

    monkeypatch.setattr(fcs, "check_moderation", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "validate_output", lambda text: ModerationResult(is_safe=True))
    monkeypatch.setattr(fcs, "perform_web_search", lambda query, max_results=5: [])
