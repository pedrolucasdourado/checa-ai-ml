"""
Gera a camada Silver a partir do JSONL processado dos fact-checks.

Exemplo:
    python scripts/build_silver_dataset.py
    python scripts/build_silver_dataset.py --input data/processed/fact_checks_all.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.preprocessing import build_silver_records


DEFAULT_INPUT = Path("data/processed/fact_checks_clean.jsonl")
DEFAULT_OUTPUT = Path("data/processed/fact_checks_silver.jsonl")
DEFAULT_REPORT = Path("reports/silver/fact_checks_silver_report.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Linha %d invalida ignorada em %s", line_no, path)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    log.info("Carregando registros de %s", args.input)
    records = load_jsonl(args.input)
    log.info("%d registros carregados", len(records))

    silver_records, report = build_silver_records(records)
    write_jsonl(args.output, silver_records)
    write_report(args.report, report)

    log.info("Silver salvo em %s", args.output)
    log.info("Relatorio salvo em %s", args.report)
    log.info(
        "%d registros | %d duplicados | riscos=%s | topicos=%s",
        report["records"],
        report["duplicate_records"],
        report["injection_risk"],
        report["topics"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera JSONL Silver com metadados de pre-processamento")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
