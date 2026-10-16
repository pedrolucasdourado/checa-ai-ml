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


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("CORPUS_VERSION", "v2-manual")
    assert cv.corpus_version(tmp_path / "qualquer.jsonl") == "v2-manual"
