from src.rag import corpus_version as cv


def test_reads_md5_from_dvc_pointer(tmp_path):
    corpus = tmp_path / "fact_checks_all.jsonl"
    (tmp_path / "fact_checks_all.jsonl.dvc").write_text(
        "outs:\n- md5: aefa9232c62aa855a650063026d7c1f7\n  size: 10\n  hash: md5\n  path: fact_checks_all.jsonl\n"
    )
    assert cv.corpus_version(corpus) == "aefa9232c62a"


def test_unknown_when_pointer_missing_or_malformed(tmp_path):
    assert cv.corpus_version(tmp_path / "nada.jsonl") == cv.UNKNOWN
    (tmp_path / "x.jsonl.dvc").write_text("outs: []\n")
    assert cv.corpus_version(tmp_path / "x.jsonl") == cv.UNKNOWN


def test_default_corpus_uses_processed_directory_pointer(monkeypatch, tmp_path):
    corpus = tmp_path / "data" / "processed" / "fact_checks_all.jsonl"
    pointer = tmp_path / "data" / "processed.dvc"
    pointer.parent.mkdir(parents=True)
    pointer.write_text("outs:\n- md5: 1234567890abcdef1234567890abcdef\n", encoding="utf-8")
    monkeypatch.setattr(cv, "DEFAULT_CORPUS_FILE", corpus)
    monkeypatch.setattr(cv, "PROCESSED_DIRECTORY_POINTER", pointer)

    assert cv.corpus_version() == "1234567890ab"


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_VERSION", "v2-manual")
    assert cv.corpus_version(tmp_path / "qualquer.jsonl") == "v2-manual"
