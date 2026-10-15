"""Strict, versioned prompt resolution from Langfuse Prompt Management."""

from __future__ import annotations

import hashlib
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from langfuse.model import ChatPromptClient

from src import observability
from src.config import PROMPT_LABEL
from src.observability import PromptUnavailable

EVIDENCE_PROMPT_NAME = 'evidence-flow'
CLAIM_PROMPT_NAME = 'claim-analysis'
V1_VERIFICATION_PROMPT_NAME = 'ml-v1-verification'
V1_CONDENSATION_PROMPT_NAME = 'ml-v1-condensation'

REQUIRED_VARIABLES = {
    EVIDENCE_PROMPT_NAME: frozenset({'query', 'contexto'}),
    CLAIM_PROMPT_NAME: frozenset({'message', 'url_context'}),
    V1_VERIFICATION_PROMPT_NAME: frozenset({'query', 'contexto', 'max_chars'}),
    V1_CONDENSATION_PROMPT_NAME: frozenset({'query', 'contexto', 'max_chars', 'narrative'}),
}

ABSTENTION_MESSAGE = 'Não foram encontradas evidências suficientes para sustentar ou contradizer a alegação.'


@dataclass(frozen=True)
class PromptBundle:
    name: str
    system: str
    user: str
    version: str
    source: str
    sha256: str
    langfuse_prompt: Any = None

    def render_user(self, **variables: str) -> str:
        return render(self.user, **variables)

    def render_system(self, **variables: str) -> str:
        return render(self.system, **variables)


current_prompt: ContextVar[PromptBundle | None] = ContextVar('current_prompt', default=None)


def render(template: str, **variables: str) -> str:
    """Replace template variables once, without interpreting braces in values."""
    return re.sub(r'{{([a-zA-Z_][a-zA-Z0-9_]*)}}',
                  lambda match: str(variables.get(match.group(1), match.group(0))), template)


def _messages_hash(messages: list[dict]) -> str:
    normalized = [{'role': message['role'], 'content': message['content']}
                  for message in messages]
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def load_named_prompt(name: str, *, label: str | None = None) -> PromptBundle:
    if name not in REQUIRED_VARIABLES:
        raise PromptUnavailable(f"Prompt '{name}' não reconhecido")
    lf_prompt, remote = observability.fetch_chat_prompt(name, label or PROMPT_LABEL)
    prompt_type = ('chat' if isinstance(lf_prompt, ChatPromptClient)
                   else getattr(lf_prompt, 'type', None))
    if (prompt_type != 'chat' or getattr(lf_prompt, 'name', None) != name
            or not isinstance(getattr(lf_prompt, 'version', None), int)
            or not isinstance(remote, list) or len(remote) != 2):
        raise PromptUnavailable(f"Prompt '{name}' inválido")
    by_role = {}
    for message in remote:
        if (not isinstance(message, dict) or message.get('role') not in {'system', 'user'}
                or message['role'] in by_role or not isinstance(message.get('content'), str)
                or not message['content'].strip()):
            raise PromptUnavailable(f"Prompt '{name}' inválido")
        by_role[message['role']] = message['content']
    if set(by_role) != {'system', 'user'}:
        raise PromptUnavailable(f"Prompt '{name}' inválido")
    combined = by_role['system'] + by_role['user']
    if any('{{' + variable + '}}' not in combined for variable in REQUIRED_VARIABLES[name]):
        raise PromptUnavailable(f"Prompt '{name}' sem variáveis obrigatórias")
    return PromptBundle(name, by_role['system'], by_role['user'],
                        f'{name}@{lf_prompt.version}', 'langfuse',
                        _messages_hash(remote), lf_prompt)


def load_prompt() -> PromptBundle:
    return load_named_prompt(EVIDENCE_PROMPT_NAME)


def load_claim_prompt() -> PromptBundle:
    return load_named_prompt(CLAIM_PROMPT_NAME)
