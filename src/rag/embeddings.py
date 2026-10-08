"""
src/rag/embeddings.py
─────────────────────────────────────────────────────────────────────
Abstração de provedores de embeddings.

  - OpenAIEmbeddingProvider: via API (sem torch, sem GPU).
  - LocalSentenceTransformerProvider: sentence-transformers local
    (import preguiçoso: torch só é carregado se este provider for usado).

Indexação e consulta DEVEM usar o mesmo provider/modelo. Trocar de
provider muda a dimensão do vetor, então exige uma coleção própria.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

from src.config import EMBED_MODEL, EMBEDDING_PROVIDER, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS


class EmbeddingProvider(Protocol):
    name: str
    model: str

    @property
    def dimension(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    """Embeddings via OpenAI. Os vetores já vêm normalizados (norma L2 = 1)."""

    name = "openai"
    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100) -> None:
        self.model = model
        self._batch_size = batch_size
        self._client = None
        # Tokens consumidos na última chamada (lido pelo tracing para estimar custo).
        self.last_usage_tokens: int | None = None

    @property
    def dimension(self) -> int:
        if self.model not in self._DIMENSIONS:
            raise ValueError(f"Dimensão desconhecida para o modelo '{self.model}'")
        return self._DIMENSIONS[self.model]

    def _get_client(self):
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY não configurada. "
                    "Defina a variável de ambiente ou adicione ao arquivo .env."
                )
            self._client = openai.OpenAI(
                api_key=api_key, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES
            )
        return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        client = self._get_client()
        tokens = 0
        for start in range(0, len(texts), self._batch_size):
            # A API rejeita string vazia; substitui por um espaço.
            batch = [t if t.strip() else " " for t in texts[start : start + self._batch_size]]
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
            tokens += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
        self.last_usage_tokens = tokens
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class LocalSentenceTransformerProvider:
    """Embeddings locais com sentence-transformers (normalizados)."""

    name = "local"

    def __init__(self, model: str = "paraphrase-multilingual-mpnet-base-v2") -> None:
        self.model = model
        self._model = None

    def _get(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model, device=device)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get().get_sentence_embedding_dimension()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return (
            self._get()
            .encode(
                list(texts),
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .tolist()
        )

    def embed_query(self, text: str) -> list[float]:
        return (
            self._get()
            .encode(text, normalize_embeddings=True, convert_to_numpy=True)
            .tolist()
        )


def get_embedding_provider(
    provider: str = EMBEDDING_PROVIDER, model: str = EMBED_MODEL
) -> EmbeddingProvider:
    """Cria o provider configurado (`EMBEDDING_PROVIDER` = 'openai' | 'local')."""
    if provider == "openai":
        return OpenAIEmbeddingProvider(model)
    if provider == "local":
        return LocalSentenceTransformerProvider(model)
    raise ValueError(f"EMBEDDING_PROVIDER inválido: '{provider}' (use 'openai' ou 'local')")
