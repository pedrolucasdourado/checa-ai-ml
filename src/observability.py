"""
src/observability.py
─────────────────────────────────────────────────────────────────────
Camada fina sobre o Langfuse (tracing, tokens, custo, latência, feedback).

Princípios:
  - Observabilidade nunca derruba o pipeline: toda falha do SDK é logada
    e engolida; exceções do código instrumentado sempre propagam.
  - Sem LANGFUSE_PUBLIC_KEY/SECRET_KEY (ou LANGFUSE_ENABLED=false) tudo
    vira no-op, então testes e dev local não precisam do Langfuse.
  - Custo: não é calculado aqui. O Langfuse infere custo a partir do
    `model` + `usage_details` (gpt-4o-mini e text-embedding-3-* já têm
    preço cadastrado). Modelo novo/custom → cadastre em Settings > Models.
"""

from __future__ import annotations

import logging
import os
import re
import ssl
import sys
from contextlib import contextmanager
from typing import Any, Iterator

from src.config import LANGFUSE_CAPTURE_CONTENT, LANGFUSE_ENABLED

log = logging.getLogger("checa-ai.observability")

TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_REDACTED = "[conteúdo omitido: LANGFUSE_CAPTURE_CONTENT=false]"

_client: Any = None
_client_failed = False


def _ensure_ca_bundle() -> None:
    """
    Python sem CA padrão (ex.: 3.14 do python.org no macOS sem "Install Certificates")
    faz o exportador do Langfuse falhar com CERTIFICATE_VERIFY_FAILED. Nesse caso
    aponta o SSL para o bundle do certifi, sem desligar a verificação do certificado.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return
    paths = ssl.get_default_verify_paths()
    if paths.cafile and os.path.exists(paths.cafile):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        log.warning("Sem CA padrão e sem certifi: o envio ao Langfuse pode falhar por SSL.")


def _get_client():
    """Cliente Langfuse singleton, ou None se desabilitado/indisponível."""
    global _client, _client_failed
    if _client is not None or _client_failed:
        return _client
    if not LANGFUSE_ENABLED:
        _client_failed = True
        return None
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        log.info("Langfuse desabilitado: LANGFUSE_PUBLIC_KEY/SECRET_KEY ausentes.")
        _client_failed = True
        return None
    try:
        from langfuse import get_client

        _ensure_ca_bundle()
        _client = get_client()
    except Exception:  # noqa: BLE001 - observabilidade nunca derruba o serviço
        log.exception("Falha ao inicializar o Langfuse; tracing desabilitado.")
        _client_failed = True
    return _client


def is_enabled() -> bool:
    return _get_client() is not None


def _content(value: Any) -> Any:
    """Aplica a política de privacidade ao texto que vai para o Langfuse."""
    if LANGFUSE_CAPTURE_CONTENT or value is None:
        return value
    return f"{_REDACTED} ({len(str(value))} caracteres)"


class Observation:
    """Handle de um span/generation; update() nunca levanta exceção."""

    def __init__(self, obs: Any = None, trace_id: str | None = None) -> None:
        self._obs = obs
        self.trace_id = trace_id

    def update(self, **kwargs: Any) -> None:
        if self._obs is None:
            return
        for key in ("input", "output"):
            if key in kwargs:
                kwargs[key] = _content(kwargs[key])
        try:
            self._obs.update(**kwargs)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao atualizar observação do Langfuse.")


_NOOP = Observation()


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Observation]:
    """Abre um span/generation/retriever/embedding aninhado no trace atual."""
    client = _get_client()
    if client is None:
        yield _NOOP
        return

    for key in ("input", "output"):
        if key in kwargs:
            kwargs[key] = _content(kwargs[key])
    try:
        cm = client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
        obs = cm.__enter__()
        handle = Observation(obs, client.get_current_trace_id())
    except Exception:  # noqa: BLE001
        log.exception("Falha ao abrir observação '%s' no Langfuse.", name)
        yield _NOOP
        return

    try:
        yield handle
    except BaseException as exc:
        handle.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}"[:500])
        try:
            cm.__exit__(*sys.exc_info())
        except Exception:  # noqa: BLE001
            log.exception("Falha ao fechar observação '%s'.", name)
        raise
    else:
        try:
            cm.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            log.exception("Falha ao fechar observação '%s'.", name)


@contextmanager
def request_trace(
    name: str,
    *,
    input: Any = None,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    version: str | None = None,
) -> Iterator[Observation]:
    """Abre o trace raiz de uma requisição; os spans filhos herdam session/tags."""
    client = _get_client()
    if client is None:
        yield _NOOP
        return
    try:
        from langfuse import propagate_attributes

        attrs: dict[str, Any] = {"trace_name": name}
        if session_id:
            attrs["session_id"] = session_id
        if user_id:
            attrs["user_id"] = user_id
        if tags:
            attrs["tags"] = tags
        if version:
            attrs["version"] = version
        if metadata:
            attrs["metadata"] = {k: str(v)[:200] for k, v in metadata.items()}
        propagate = propagate_attributes(**attrs)
        propagate.__enter__()
    except Exception:  # noqa: BLE001
        log.exception("Falha ao propagar atributos do trace '%s'.", name)
        propagate = None

    try:
        with observation(name, input=input) as root:
            yield root
    finally:
        if propagate is not None:
            try:
                propagate.__exit__(*sys.exc_info())
            except Exception:  # noqa: BLE001
                log.exception("Falha ao fechar atributos do trace '%s'.", name)


def score_trace(
    trace_id: str,
    name: str,
    value: float | str,
    *,
    data_type: str = "NUMERIC",
    comment: str | None = None,
    idempotent: bool = False,
) -> bool:
    """
    Anexa um score a um trace. Retorna False se o Langfuse estiver desligado
    ou se o envio falhar. `idempotent=True` faz reenvios sobrescreverem o
    score anterior (ex.: usuário trocando 👍 por 👎) em vez de duplicar.
    """
    client = _get_client()
    if client is None:
        return False
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            data_type=data_type,
            comment=comment,
            score_id=f"{trace_id}-{name}" if idempotent else None,
        )
        return True
    except Exception:  # noqa: BLE001
        log.exception("Falha ao registrar score '%s' no trace %s.", name, trace_id)
        return False


def flush() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception:  # noqa: BLE001
            log.exception("Falha no flush do Langfuse.")


def shutdown() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.shutdown()
        except Exception:  # noqa: BLE001
            log.exception("Falha no shutdown do Langfuse.")
