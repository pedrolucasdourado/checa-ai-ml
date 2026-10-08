"""
sync_prompts.py
────────────────────────────────────────────────────────────────────────
Mantém o Langfuse Prompt Management alinhado com src/rag/prompts.py.
Idempotente: só cria versão nova quando o texto realmente mudou.

Modos:
    --check              (PR) compara local x Langfuse e imprime o que aconteceria; não escreve.
    (padrão)             (merge na main) publica o texto local com o label "staging".
    --promote            (release/deploy) move o label "production" para a versão
                         que contém EXATAMENTE o texto local. Falha se ela não existir.
    --write-lock         grava src/rag/prompts.lock.json (versão + hash) após editar o prompt.

Rollback: no Langfuse, mova o label "production" para a versão anterior
(ou rode o workflow de release de novo a partir da tag anterior).

Exige LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

sys.path.insert(0, str(ROOT))
from src import observability  # noqa: E402
from src.rag import prompts  # noqa: E402

LOCK_PATH = ROOT / "src" / "rag" / "prompts.lock.json"
NAME = prompts.PROMPT_NAME


def find_matching_version() -> tuple[int, list[str]] | None:
    """Versão do Langfuse com texto idêntico ao local (procura em latest/staging/production)."""
    local = prompts.local_messages()
    for label in ("latest", "staging", "production"):
        found = observability.get_prompt_version(NAME, label)
        if found and found[1] == local:
            return found[0], [label]
    return None


def cmd_write_lock() -> int:
    LOCK_PATH.write_text(
        json.dumps({"version": prompts.PROMPT_VERSION, "sha256": prompts.content_hash()}, indent=2) + "\n"
    )
    print(f"🔒 {LOCK_PATH.relative_to(ROOT)} atualizado ({prompts.PROMPT_VERSION}).")
    return 0


def cmd_sync(label: str, check: bool) -> int:
    if not observability.is_enabled():
        print("❌ Langfuse desligado ou sem chaves (LANGFUSE_PUBLIC_KEY/SECRET_KEY).", file=sys.stderr)
        return 1
    match = find_matching_version()
    if match:
        version, _ = match
        has_label = (observability.get_prompt_version(NAME, label) or (None,))[0] == version
        if has_label:
            print(f"✅ Sem mudanças: '{NAME}' v{version} já está em '{label}' ({prompts.PROMPT_VERSION}).")
        elif check:
            print(f"ℹ️  Texto igual à v{version}; o label '{label}' seria movido para ela.")
        else:
            if not observability.set_prompt_labels(NAME, version, [label]):
                return 1
            print(f"✅ Label '{label}' movido para '{NAME}' v{version} (texto inalterado).")
        return 0

    if check:
        print(f"📝 O texto local ({prompts.PROMPT_VERSION}) difere do Langfuse: uma nova versão seria criada em '{label}'.")
        return 0
    version = prompts.sync_to_langfuse([label])
    observability.flush()
    if version is None:
        print("❌ Falha ao publicar o prompt (veja os logs).", file=sys.stderr)
        return 1
    print(f"✅ Prompt '{NAME}' v{version} publicado em '{label}' (origem {prompts.PROMPT_VERSION}).")
    return 0


def cmd_promote(check: bool) -> int:
    if not observability.is_enabled():
        print("❌ Langfuse desligado ou sem chaves.", file=sys.stderr)
        return 1
    match = find_matching_version()
    if not match:
        print(
            f"❌ Nenhuma versão no Langfuse com o texto local ({prompts.PROMPT_VERSION}). "
            "Rode o sync (merge na main) antes de promover.",
            file=sys.stderr,
        )
        return 1
    version, _ = match
    current = observability.get_prompt_version(NAME, "production")
    if current and current[0] == version:
        print(f"✅ 'production' já aponta para '{NAME}' v{version}.")
        return 0
    if check:
        print(f"ℹ️  'production' passaria de v{current[0] if current else '-'} para v{version}.")
        return 0
    if not observability.set_prompt_labels(NAME, version, ["production"]):
        return 1
    print(f"🚀 '{NAME}' v{version} promovida para 'production' (antes: v{current[0] if current else '-'}).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sincroniza/promove prompts no Langfuse")
    parser.add_argument("--label", default="staging", help="label do sync (padrão: staging)")
    parser.add_argument("--promote", action="store_true", help="move o label production para a versão local")
    parser.add_argument("--check", "--dry-run", dest="check", action="store_true", help="não escreve nada")
    parser.add_argument("--write-lock", action="store_true", help="atualiza src/rag/prompts.lock.json")
    args = parser.parse_args()

    if args.write_lock:
        return cmd_write_lock()
    if args.promote:
        return cmd_promote(args.check)
    return cmd_sync(args.label, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
