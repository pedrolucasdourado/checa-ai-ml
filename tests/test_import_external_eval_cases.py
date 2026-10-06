import json

from scripts.import_external_eval_cases import parse_row, split_rows, to_eval_case


def test_split_and_parse_pasted_table_row():
    text = (
        "shares \ttext \tmisinformation \tsource \trevision\n"
        "0 \t27 \tMensagem com\nmais de uma linha\nhttps://example.com/x \t1 \t"
        "https://factcheck.test/a \tNaN\n"
        "1 \t3 \tOutra mensagem \t0 \tNaN \t1.0"
    )

    rows = [parse_row(raw) for raw in split_rows(text)]

    assert len(rows) == 2
    assert rows[0]["idx"] == 0
    assert rows[0]["misinformation"] is True
    assert rows[0]["source"] == "https://factcheck.test/a"
    assert "mais de uma linha" in rows[0]["query"]
    assert rows[1]["source"] is None


def test_eval_case_matches_only_when_reference_url_is_in_corpus():
    row = {
        "idx": 7,
        "shares": 24,
        "query": "Mensagem de teste",
        "misinformation": True,
        "source": "https://factcheck.test/a",
        "revision": None,
    }

    matched = to_eval_case(row, {"https://factcheck.test/a"}, "external_news")
    abstained = to_eval_case(row, set(), "external_news")

    assert matched["expected_status"] == "matched"
    assert matched["expected_url"] == "https://factcheck.test/a"
    assert abstained["expected_status"] == "review"
    assert abstained["expected_url"] is None
    assert abstained["needs_manual_review"] is True


def test_eval_case_is_json_serializable():
    row = {
        "idx": 1,
        "shares": 2,
        "query": "Mensagem",
        "misinformation": False,
        "source": None,
        "revision": "1.0",
    }

    json.dumps(to_eval_case(row, set(), "external_news"), ensure_ascii=False)
