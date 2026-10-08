"""Contrato do prompt: roda em todo PR e impede mudanças silenciosas."""

import json
from pathlib import Path

from src.rag import prompts

LOCK = json.loads((Path(prompts.__file__).parent / "prompts.lock.json").read_text())


def test_prompt_text_change_requires_version_bump_and_lock_update():
    assert prompts.content_hash() == LOCK["sha256"] and prompts.PROMPT_VERSION == LOCK["version"], (
        "O texto de src/rag/prompts.py mudou. Incremente PROMPT_VERSION e rode "
        "`python scripts/sync_prompts.py --write-lock`."
    )


def test_user_prompt_has_required_variables():
    assert "{{query}}" in prompts.USER_PROMPT and "{{contexto}}" in prompts.USER_PROMPT


def test_system_prompt_keeps_safety_rules():
    s = prompts.SYSTEM_PROMPT
    assert "APENAS as evidências" in s          # anti-alucinação
    assert "não confiáveis" in s                # prompt injection nos laudos
    assert "fontes_usadas" in s and "JSON" in s  # saída estruturada
    assert "laudos_tratam_alegacao" in s        # abstenção pelo modelo
    assert "Fonte:" in s                        # crédito às fontes


def test_prompts_render_without_leftover_placeholders():
    out = prompts.render(prompts.USER_PROMPT, query="q", contexto="c")
    assert "{{" not in out
