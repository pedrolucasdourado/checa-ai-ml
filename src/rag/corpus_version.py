"""
src/rag/corpus_version.py
─────────────────────────────────────────────────────────────────────
Versão do corpus de laudos = hash (md5) do ponteiro DVC do arquivo indexado.

Grava-se a mesma versão no payload dos pontos do Qdrant (ingest) e nos traces
do Langfuse (serviço), ligando cada resposta ao conjunto de laudos usado.
`CORPUS_VERSION` no ambiente sobrescreve (ex.: coleção indexada em outra máquina).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from src.config import PROJECT_ROOT

DEFAULT_CORPUS_FILE = PROJECT_ROOT / "data" / "processed" / "fact_checks_all.jsonl"
UNKNOWN = "unknown"
_MD5_RE = re.compile(r"^\s*-?\s*md5:\s*([0-9a-f]{32})", re.MULTILINE)


def corpus_version(corpus_file: Path | str | None = None) -> str:
    override = os.environ.get("CORPUS_VERSION", "").strip()
    if override:
        return override
    pointer = Path(str(corpus_file or DEFAULT_CORPUS_FILE) + ".dvc")
    try:
        match = _MD5_RE.search(pointer.read_text(encoding="utf-8"))
    except OSError:
        return UNKNOWN
    return match.group(1)[:12] if match else UNKNOWN
