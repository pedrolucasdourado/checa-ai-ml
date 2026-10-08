"""
sync_prompts.py
────────────────────────────────────────────────────────────────────────
Publica os prompts de src/rag/prompts.py no Langfuse Prompt Management
como uma NOVA versão (o histórico é preservado).

Fluxo recomendado:
    python scripts/sync_prompts.py                    # nova versão com label "staging"
    # testar com PROMPT_LABEL=staging; satisfeito, no Langfuse mova o label
    # "production" para a versão (ou rode com --label production)
    python scripts/sync_prompts.py --label production

Rollback: no Langfuse, mova o label "production" para a versão anterior.
Exige LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import observability  # noqa: E402
from src.rag.prompts import PROMPT_NAME, PROMPT_VERSION, sync_to_langfuse  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Publica os prompts locais no Langfuse")
    parser.add_argument("--label", action="append", help="label(s) da nova versão (padrão: staging)")
    args = parser.parse_args()
    labels = args.label or ["staging"]

    version = sync_to_langfuse(labels)
    observability.flush()
    if version is None:
        print("Não foi possível publicar (Langfuse desligado, sem chaves ou erro; veja os logs).", file=sys.stderr)
        return 1
    print(f"Prompt '{PROMPT_NAME}' versão {version} publicado (origem local {PROMPT_VERSION}) com labels {labels}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
