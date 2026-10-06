from src.rag.chunking import (
    build_chunks,
    chunk_text,
    document_id_for,
    point_id_for,
)


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    assert chunk_text(None) == []


def test_short_text_is_single_chunk():
    assert chunk_text("Texto curto.", chunk_size=100, overlap=10) == ["Texto curto."]


def test_chunks_respect_size_and_cover_all_text():
    words = [f"palavra{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    joined = " ".join(chunks)
    for w in words:
        assert w in joined


def test_chunks_overlap_and_do_not_cut_words():
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), chunk_size=200, overlap=40)

    valid = set(words)
    for c in chunks:
        assert set(c.split()) <= valid  # nenhuma palavra cortada ao meio
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.split()) & set(nxt.split())  # há sobreposição


def test_prefers_sentence_boundary():
    text = ("Primeira frase longa o bastante para ocupar espaço. " * 3) + "Final sem fim " * 20
    first = chunk_text(text, chunk_size=160, overlap=20)[0]
    assert first.endswith(".")


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_ids_are_stable_and_distinct():
    assert document_id_for("https://x.com/a") == document_id_for(" https://x.com/a ")
    doc = document_id_for("https://x.com/a")
    assert point_id_for(doc, 0) == point_id_for(doc, 0)
    assert point_id_for(doc, 0) != point_id_for(doc, 1)


def test_build_chunks_metadata():
    rec = {
        "titulo": "É falso que X",
        "texto": "Frase um. " * 200,
        "url": "https://g1.globo.com/fato-ou-fake/1",
        "dominio": "g1.globo.com",
        "data_publicacao": "2024-01-01",
    }
    chunks = build_chunks(rec, chunk_size=300, overlap=50)

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert len({c.document_id for c in chunks}) == 1
    assert len({c.content_hash for c in chunks}) == 1
    first = chunks[0]
    assert first.idioma == "pt"
    assert first.embed_text.startswith("É falso que X. ")
    p = first.payload()
    assert p["url"] == rec["url"] and p["dominio"] == "g1.globo.com"
    assert p["texto_chunk"] == first.text


def test_build_chunks_is_deterministic():
    rec = {"titulo": "t", "texto": "abc " * 500, "url": "https://x.com/1"}
    a = [c.chunk_id for c in build_chunks(rec, 200, 30)]
    b = [c.chunk_id for c in build_chunks(rec, 200, 30)]
    assert a == b
