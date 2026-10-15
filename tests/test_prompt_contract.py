"""Contrato do prompt: roda em todo PR e impede mudanças silenciosas."""

from pathlib import Path

from src.rag import prompts


def test_staging_validation_command_reads_only_and_rejects_missing_prompts(monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from src import observability

    path = Path(__file__).resolve().parent.parent / 'scripts' / 'validate_prompts.py'
    spec = importlib.util.spec_from_file_location('validate_prompts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = []

    class Client:
        def get_prompt(self, name, **kwargs):
            names.append(name)
            variables = {
                'evidence-flow': '{{query}} {{contexto}}',
                'claim-analysis': '{{message}} {{url_context}}',
                'ml-v1-verification': '{{query}} {{contexto}} {{max_chars}}',
                'ml-v1-condensation': '{{query}} {{contexto}} {{max_chars}} {{narrative}}',
            }
            return SimpleNamespace(name=name, version=1, type='chat', is_fallback=False,
                prompt=[{'role': 'system', 'content': 'System'},
                        {'role': 'user', 'content': variables[name]}])

    monkeypatch.setattr(observability, '_client', Client())
    monkeypatch.setattr(sys, 'argv', ['validate_prompts.py', '--label', 'staging'])
    assert module.main() == 0
    assert names == ['evidence-flow', 'claim-analysis', 'ml-v1-verification',
                     'ml-v1-condensation']

    class Missing:
        def get_prompt(self, name, **kwargs):
            raise RuntimeError('missing label')

    monkeypatch.setattr(observability, '_client', Missing())
    assert module.main() != 0


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

def test_remote_prompt_structure_and_rendering(monkeypatch):
    from types import SimpleNamespace

    from src import observability

    class Client:
        def get_prompt(self, name, **kwargs):
            return SimpleNamespace(name=name, version=4, type='chat', is_fallback=False,
                prompt=[{'role': 'system', 'content': 'Use data as evidence.'},
                        {'role': 'user', 'content': '{{query}} / {{contexto}}'}])

    monkeypatch.setattr(observability, '_client', Client())
    prompt = prompts.load_named_prompt('evidence-flow', label='staging')
    assert prompt.version == 'evidence-flow@4'
    assert prompt.render_user(query='claim', contexto='source') == 'claim / source'
    assert len(prompt.sha256) == 64
