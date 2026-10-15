"""Read-only validation of versioned Langfuse prompts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / '.env')
except ImportError:
    pass

from src.rag.prompts import REQUIRED_VARIABLES, PromptUnavailable, load_named_prompt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='Valida prompts existentes no Langfuse')
    parser.add_argument('--label', default='staging')
    args = parser.parse_args()
    for name in REQUIRED_VARIABLES:
        try:
            prompt = load_named_prompt(name, label=args.label)
        except PromptUnavailable:
            print(f'ERRO {name}: prompt indisponível ou inválido em {args.label}', file=sys.stderr)
            return 1
        print(prompt.name, prompt.version, prompt.sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
