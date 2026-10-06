"""
import_external_eval_cases.py
─────────────────────────────────────────────────────────────────────
Converte lotes tabulares colados de datasets externos em casos JSONL
para avaliacao de retrieval.

O script foi feito para datasets de mensagens (ex.: WhatsApp) com colunas
parecidas com:

    shares text misinformation source revision

Quando a URL de `source` existe na base Silver, o caso vira `matched`.
Quando a fonte nao existe na base ou e ausente, o caso vira `review`,
pois pode haver uma checagem equivalente em outro veiculo.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_INPUT = Path("data/eval/external_news.tsv")
DEFAULT_CORPUS = Path("data/processed/fact_checks_silver.jsonl")
DEFAULT_OUTPUT = Path("data/eval/retrieval_candidates_external_news.jsonl")


def slugify(value: str, max_len: int = 56) -> str:
    value = value.lower()
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return (value or "case")[:max_len].strip("_")


def normalize_url(url: str | None) -> str | None:
    if not url or url.lower() == "nan":
        return None
    return url.strip()


def load_corpus_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("url"):
                urls.add(row["url"])
    return urls


def split_rows(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").strip()
    if "\n0 \t" in text or text.startswith("0 \t"):
        body = re.sub(r"^shares\s+\ttext\s+\tmisinformation\s+\tsource\s+\trevision\s*\n", "", text)
        return re.split(r"\n(?=\d+\s+\t\d+\s+\t)", body)
    return []


def parse_row(raw: str) -> dict | None:
    parts = [part.strip() for part in raw.split("\t")]
    if len(parts) < 6:
        return None
    idx = parts[0]
    shares = parts[1]
    misinformation = parts[-3]
    source = normalize_url(parts[-2])
    revision = None if parts[-1].lower() == "nan" else parts[-1]
    message = "\t".join(parts[2:-3]).strip()
    if not idx.isdigit() or not message:
        return None
    return {
        "idx": int(idx),
        "shares": int(shares) if shares.isdigit() else None,
        "query": message,
        "misinformation": misinformation == "1",
        "source": source,
        "revision": revision,
    }


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line).get("id", ""))
    return ids


def to_eval_case(row: dict, corpus_urls: set[str], dataset_name: str) -> dict:
    source = row["source"]
    source_in_corpus = source in corpus_urls if source else False
    status = "matched" if source_in_corpus else "review"
    domain = urlparse(source).netloc if source else "sem_fonte"
    prefix = f"{dataset_name}_{row['idx']:03d}_{slugify(domain)}"
    return {
        "id": prefix,
        "query": row["query"],
        "expected_url": source if source_in_corpus else None,
        "expected_status": status,
        "source_dataset": dataset_name,
        "dataset_label": "fake" if row["misinformation"] else "true_or_uncertain",
        "external_reference_url": source,
        "external_reference_in_corpus": source_in_corpus,
        "needs_manual_review": not source_in_corpus,
        "shares": row["shares"],
    }


def run(args: argparse.Namespace) -> None:
    raw_text = args.input.read_text(encoding="utf-8")
    corpus_urls = load_corpus_urls(args.corpus)
    seen_ids = existing_ids(args.output)

    rows = [parse_row(raw) for raw in split_rows(raw_text)]
    cases = [
        to_eval_case(row, corpus_urls, args.dataset_name)
        for row in rows
        if row is not None
    ]
    new_cases = [case for case in cases if case["id"] not in seen_ids]

    if not args.dry_run and new_cases:
        with open(args.output, "a", encoding="utf-8") as f:
            for case in new_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    matched = sum(1 for case in new_cases if case["expected_status"] == "matched")
    abstained = sum(1 for case in new_cases if case["expected_status"] == "abstained")
    print(
        f"Parsed={len(cases)} | new={len(new_cases)} | matched={matched} | "
        f"abstained={abstained} | output={args.output}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Importa casos externos para data/eval/retrieval_eval.jsonl")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-name", default="external_news")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())
