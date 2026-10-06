"""
generate_debunk.py
────────────────────────────────────────────────────────────────────────
Pipeline RAG ponta a ponta para geração de contranarrativas:

  Boato (query) ──► Qdrant (retrieval) ──► [limiar de score] ──► GPT-4o-mini ──► Contranarrativa

Técnica de prompt: Truth Sandwich
  1. Afirma o FATO verdadeiro primeiro
  2. Explica o boato brevemente (sem amplificá-lo)
  3. Destrói o boato com as evidências jornalísticas recuperadas
  4. Credita a fonte

Uso:
    python scripts/generate_debunk.py
    python scripts/generate_debunk.py --query "Vacina causa morte em idosos"
    python scripts/generate_debunk.py --query "..." --threshold 0.55 --model gpt-4o-mini

Variáveis de ambiente:
    OPENAI_API_KEY  → obrigatória (ou via arquivo .env na raiz do projeto)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# ── Carrega .env se existir ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass   # python-dotenv opcional

import openai

# ──────────────────────────── Configurações ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    QDRANT_PATH,
    QDRANT_URL,
    SIMILARITY_THRESHOLD as DEFAULT_THRESHOLD,   # limiar de abstinência
)
from src.rag.embeddings import EmbeddingProvider, get_embedding_provider  # noqa: E402
from src.rag.qdrant import get_qdrant_client  # noqa: E402
DEFAULT_QUERY = (
    "Hackers invadiram o TSE e transformaram as justificativas em votos válidos"
)

# ─────────────────────── Prompt Template ─────────────────────────────
SYSTEM_PROMPT = """\
Você é um verificador de fatos especializado em combate à desinformação no Brasil.
Sua missão é redigir contranarrativas curtas, didáticas e embasadas em evidências \
jornalísticas reais para serem compartilhadas via WhatsApp ou redes sociais.

Regras estritas:
1. COMECE sempre afirmando o FATO verídico logo na primeira frase (Truth Sandwich).
2. Em seguida, mencione brevemente a alegação falsa circulando — sem amplificá-la.
3. Use APENAS as evidências contidas no "Laudo de Checagem" fornecido. \
PROIBIDO inventar dados, estatísticas ou declarações não presentes no laudo.
4. Tom: cortês, claro, direto e acessível — adequado para leigos.
5. Comprimento: 3 a 5 parágrafos curtos. Sem markdown, sem bullets, só texto corrido.
6. Finalize com a linha exata:
   "Fonte: {dominio} — Leia mais em: {url}"
   (substitua pelas variáveis reais do laudo)
"""

USER_PROMPT_TEMPLATE = """\
--- ALEGAÇÃO RECEBIDA ---
{query}

--- LAUDO DE CHECAGEM RECUPERADO ---
Título da matéria: {titulo}
Agência checadora: {dominio}
Data de publicação: {data_publicacao}
URL da checagem: {url}

Texto integral do laudo:
{texto_completo}

--- TAREFA ---
Com base EXCLUSIVAMENTE no laudo acima, redija uma contranarrativa clara e \
fundamentada para desmentir a alegação recebida. Siga as regras do sistema.
"""


# ──────────────────────────── Funções ─────────────────────────────────
def load_embed_model() -> EmbeddingProvider:
    print("🔄 Carregando provider de embeddings…", flush=True)
    return get_embedding_provider()


def retrieve(
    query: str,
    model: EmbeddingProvider,
    client: QdrantClient,
    top_k: int = 1,
) -> list:
    vec = model.embed_query(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k,
        with_payload=True,
    ).points
    return results


def build_prompt(query: str, hit) -> tuple[str, str]:
    """Retorna (system_prompt, user_prompt) com os dados do hit injetados."""
    p = hit.payload
    user = USER_PROMPT_TEMPLATE.format(
        query=query,
        titulo=p.get("titulo", ""),
        dominio=p.get("dominio", ""),
        data_publicacao=p.get("data_publicacao", ""),
        url=p.get("url", ""),
        texto_completo=p.get("texto_chunk") or p.get("texto_completo", ""),
    )
    system = SYSTEM_PROMPT.format(
        dominio=p.get("dominio", ""),
        url=p.get("url", ""),
    )
    return system, user


def call_llm(system: str, user: str, model: str, temperature: float) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n❌  OPENAI_API_KEY não encontrada.", file=sys.stderr)
        print("    Defina a variável de ambiente ou crie um arquivo .env com:", file=sys.stderr)
        print("    OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def print_separator(char: str = "═", width: int = 70) -> None:
    print(char * width)


def print_hit_summary(hit, query: str) -> None:
    p = hit.payload
    print_separator()
    print("📋 CONTEXTO RECUPERADO DO QDRANT")
    print_separator("─")
    print(f"  Score de similaridade : {hit.score:.4f}")
    print(f"  Título                : {p.get('titulo', '')}")
    print(f"  Agência               : {p.get('dominio', '')}")
    print(f"  Data                  : {p.get('data_publicacao', '')}")
    print(f"  URL                   : {p.get('url', '')}")
    print(f"\n  Trecho do laudo:")
    snippet = " ".join((p.get("texto_chunk") or p.get("texto_completo", "")).split()[:60]) + "…"
    print(textwrap.fill(snippet, width=70, initial_indent="  ", subsequent_indent="  "))
    print_separator()


# ──────────────────────────── Main ────────────────────────────────────
def run(args: argparse.Namespace) -> None:
    print()
    print_separator("═")
    print("🔎 BOATO / ALEGAÇÃO RECEBIDA:")
    print(f'   "{args.query}"')
    print_separator("═")

    # 1. Carrega modelo e cliente Qdrant
    embed_model = load_embed_model()
    qdrant = get_qdrant_client(
        url=None if args.local else args.qdrant_url,
        path=args.qdrant_path,
        local_fallback=args.local,
    )

    # 2. Recupera o top-1 da base vetorial
    print("\n🔍 Buscando checagem relevante no índice Qdrant…")
    hits = retrieve(args.query, embed_model, qdrant, top_k=1)

    if not hits:
        print("\n⚠️  Nenhum resultado encontrado no índice. Base pode estar vazia.")
        return

    top_hit = hits[0]
    print_hit_summary(top_hit, args.query)

    # 3. Lógica de abstinência
    if top_hit.score < args.threshold:
        print(f"\n🔕 ABSTINÊNCIA ATIVADA")
        print(f"   Score ({top_hit.score:.4f}) abaixo do limiar ({args.threshold}).")
        print(f"\n   Retorno: Abstenção: Nenhuma checagem oficial encontrada para essa alegação.")
        return

    print(f"\n✅ Score ({top_hit.score:.4f}) acima do limiar ({args.threshold}). Gerando contranarrativa…\n")

    # 4. Constrói prompt e chama a LLM
    system_prompt, user_prompt = build_prompt(args.query, top_hit)
    contranarrativa = call_llm(system_prompt, user_prompt, args.model, args.temperature)

    # 5. Exibe resultado
    print_separator("═")
    print("🧾 CONTRANARRATIVA GERADA — PRONTA PARA COMPARTILHAR")
    print_separator("═")
    print()
    # Formata com quebras de linha para melhor leitura no terminal
    for paragraph in contranarrativa.split("\n"):
        if paragraph.strip():
            print(textwrap.fill(paragraph.strip(), width=70))
            print()
    print_separator("═")


# ──────────────────────────── Entry point ─────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG pipeline: Boato → Qdrant → GPT-4o-mini → Contranarrativa"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="Boato ou alegação a ser verificada",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Score mínimo de similaridade para acionar a LLM (padrão: 0.55)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_MODEL,
        help="Modelo OpenAI a usar (padrão: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=LLM_TEMPERATURE,
        help="Temperatura do LLM (padrão: 0.2)",
    )
    parser.add_argument("--qdrant-url", type=str, default=QDRANT_URL, help="URL do Qdrant server")
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH, help="Diretório Qdrant local")
    parser.add_argument("--local", action="store_true", help="Usa Qdrant local em vez do server")
    args = parser.parse_args()
    run(args)
