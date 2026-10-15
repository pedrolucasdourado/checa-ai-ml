"""Contrato do prompt: roda em todo PR e impede mudanças silenciosas."""

import json
from pathlib import Path

from src.rag import prompts


def test_named_loader_uses_each_remote_version_and_validates_roles(monkeypatch):
    from types import SimpleNamespace

    from src import observability

    variables = {
        'evidence-flow': '{{query}} {{contexto}}',
        'claim-analysis': '{{message}} {{url_context}}',
        'ml-v1-verification': '{{query}} {{contexto}} {{max_chars}}',
        'ml-v1-condensation': '{{query}} {{contexto}} {{max_chars}} {{narrative}}',
    }

    class Client:
        def get_prompt(self, name, **kwargs):
            assert kwargs['label'] == 'staging' and kwargs['type'] == 'chat'
            return SimpleNamespace(name=name, version=list(variables).index(name) + 1,
                type='chat', is_fallback=False, prompt=[
                    {'role': 'system', 'content': 'System'},
                    {'role': 'user', 'content': variables[name]},
                ])

    monkeypatch.setattr(observability, '_client', Client())
    got = [prompts.load_named_prompt(name, label='staging') for name in variables]
    assert [item.version for item in got] == [
        'evidence-flow@1', 'claim-analysis@2', 'ml-v1-verification@3',
        'ml-v1-condensation@4']
    assert all(item.source == 'langfuse' for item in got)


def test_named_loader_rejects_invalid_remote_prompts(monkeypatch):
    from types import SimpleNamespace

    import pytest

    from src import observability

    bad_messages = [
        [{'role': 'system', 'content': 'System'}],
        [{'role': 'system', 'content': 'A'}, {'role': 'system', 'content': 'B'},
         {'role': 'user', 'content': '{{query}} {{contexto}}'}],
        [{'role': 'system', 'content': 'A'}, {'role': 'user', 'content': 'No variables'}],
    ]
    for messages in bad_messages:
        class Client:
            def get_prompt(self, name, **kwargs):
                return SimpleNamespace(name=name, version=1, type='chat',
                                       is_fallback=False, prompt=messages)
        monkeypatch.setattr(observability, '_client', Client())
        with pytest.raises(prompts.PromptUnavailable):
            prompts.load_named_prompt('evidence-flow', label='staging')

    for wrong in ('text', 'fallback'):
        class Client:
            def get_prompt(self, name, **kwargs):
                return SimpleNamespace(name=name, version=1, type=wrong,
                    is_fallback=wrong == 'fallback', prompt=[
                        {'role': 'system', 'content': 'System'},
                        {'role': 'user', 'content': '{{query}} {{contexto}}'},
                    ])
        monkeypatch.setattr(observability, '_client', Client())
        with pytest.raises(prompts.PromptUnavailable):
            prompts.load_named_prompt('evidence-flow', label='staging')

    class Missing:
        def get_prompt(self, name, **kwargs):
            raise RuntimeError('label missing')
    monkeypatch.setattr(observability, '_client', Missing())
    with pytest.raises(prompts.PromptUnavailable):
        prompts.load_named_prompt('evidence-flow', label='production')


def test_render_does_not_expand_placeholders_inside_user_data():
    assert prompts.render('{{query}} / {{contexto}}', query='{{contexto}}',
                          contexto='evidence') == '{{contexto}} / evidence'

LOCK = json.loads((Path(prompts.__file__).parent / "prompts.lock.json").read_text())


def test_prompt_text_change_requires_version_bump_and_lock_update():
    assert (
        LOCK["schema_version"] == 2
        and prompts.content_hash() == LOCK["sha256"]
        and prompts.prompt_manifest() == LOCK["prompts"]
    ), (
        "O texto de src/rag/prompts.py mudou. Incremente PROMPT_VERSION e rode "
        "`python scripts/sync_prompts.py --write-lock`."
    )


def test_lock_hash_covers_claim_analysis_prompts(monkeypatch):
    original_hash = prompts.content_hash()
    monkeypatch.setattr(
        prompts,
        "CLAIM_ANALYSIS_SYSTEM_PROMPT",
        prompts.CLAIM_ANALYSIS_SYSTEM_PROMPT + "\nMudança de teste.",
    )

    assert prompts.content_hash() != original_hash


def test_each_prompt_has_an_independent_name_version_and_hash():
    manifest = prompts.prompt_manifest()

    assert set(manifest) == {"evidence-flow", "claim-analysis"}
    assert all(item["version"] and len(item["sha256"]) == 64 for item in manifest.values())


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
