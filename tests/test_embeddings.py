from types import SimpleNamespace

import pytest

from src.rag.embeddings import (
    LocalSentenceTransformerProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append((model, list(input)))
        # devolve fora de ordem para validar a reordenação por index
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), 0.0]) for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)))


def make_provider(batch_size=2):
    p = OpenAIEmbeddingProvider("text-embedding-3-small", batch_size=batch_size)
    p._client = SimpleNamespace(embeddings=FakeEmbeddings())
    return p


def test_openai_batches_and_keeps_order():
    p = make_provider(batch_size=2)
    out = p.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])
    assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [len(c[1]) for c in p._client.embeddings.calls] == [2, 2, 1]


def test_openai_replaces_empty_strings():
    p = make_provider()
    p.embed_documents(["", "  "])
    assert p._client.embeddings.calls[0][1] == [" ", " "]


def test_openai_embed_query_returns_single_vector():
    assert make_provider().embed_query("abc") == [3.0, 0.0]


def test_openai_dimension_and_unknown_model():
    assert OpenAIEmbeddingProvider("text-embedding-3-small").dimension == 1536
    with pytest.raises(ValueError):
        OpenAIEmbeddingProvider("modelo-desconhecido").dimension


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAIEmbeddingProvider().embed_query("x")


def test_factory():
    assert isinstance(get_embedding_provider("openai", "text-embedding-3-small"), OpenAIEmbeddingProvider)
    local = get_embedding_provider("local", "m")
    assert isinstance(local, LocalSentenceTransformerProvider)
    assert local._model is None  # import preguiçoso: não carrega torch na criação
    with pytest.raises(ValueError):
        get_embedding_provider("foo", "m")
