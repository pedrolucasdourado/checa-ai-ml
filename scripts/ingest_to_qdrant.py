"""
ingest_to_qdrant.py
────────────────────────────────────────────────────────────────────────
Lê o arquivo 'data/processed/fact_checks_all.jsonl' (ou outro via --input),
gera embeddings dos textos de checagem usando um modelo multilíngue/português
via sentence-transformers e indexa no Qdrant em modo LOCAL (qdrant_db/) ou
Docker (localhost:6333) se disponível.

Ao final executa uma consulta de teste de busca semântica.

Uso:
    python scripts/ingest_to_qdrant.py [--input data/processed/fact_checks_all.jsonl]
                                        [--collection fact_checks_pt]
                                        [--query "estão jogando fora cédulas de votação"]
                                        [--top-k 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Qdrant ──────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# ─── Configurações padrão ─────────────────────────────────────────────
DEFAULT_INPUT   = Path("data/processed/fact_checks_all.jsonl")
DEFAULT_DB_PATH = Path("data/qdrant_db")
COLLECTION_NAME = "fact_checks_pt"
BATCH_SIZE      = 32
MAX_TEXT_CHARS  = 3_000   # truncamento do texto para o payload (contexto da LLM)

# Modelo: multilíngue de alta qualidade (suporta PT de forma excelente),
# 768 dimensões — sem precisar baixar BERTimbau separadamente.
# Troca por "neuralmind/bert-base-portuguese-cased" se quiser PT puro
# (mas exigiria pool manual via transformers).
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────── Helpers ────────────────────────────────
def load_records(path: Path) -> list[dict]:
    """Carrega o JSONL garantindo campos mínimos."""
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # Garante campos mínimos
                if rec.get("texto") and rec.get("url"):
                    records.append(rec)
            except json.JSONDecodeError:
                log.warning("Linha %d inválida — ignorada", i + 1)
    return records


def make_embed_text(rec: dict) -> str:
    """
    Texto usado para gerar o embedding.
    Combina título + primeiros 512 chars do texto para capturar
    o contexto da checagem com eficiência.
    """
    titulo = (rec.get("titulo") or "").strip()
    texto  = (rec.get("texto")  or "").strip()[:512]
    return f"{titulo}. {texto}" if titulo else texto


def truncate_text(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"


def get_qdrant_client(db_path: Path) -> QdrantClient:
    """
    Tenta conectar ao Docker (localhost:6333); se falhar, usa modo local.
    """
    try:
        client = QdrantClient(host="localhost", port=6333, timeout=3)
        client.get_collections()          # testa a conexão
        log.info("✅ Conectado ao Qdrant Docker em localhost:6333")
        return client
    except Exception:
        log.info("Docker não disponível — usando modo LOCAL em '%s'", db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(db_path))


def recreate_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        log.info("Coleção '%s' já existe — recriando...", name)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("Coleção '%s' criada (%d dims, COSINE)", name, vector_size)


def upsert_in_batches(
    client: QdrantClient,
    collection: str,
    records: list[dict],
    embeddings: list[list[float]],
) -> None:
    points = []
    for idx, (rec, vec) in enumerate(zip(records, embeddings)):
        points.append(
            PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "titulo":          rec.get("titulo", ""),
                    "texto_completo":  truncate_text(rec.get("texto", "")),
                    "url":             rec.get("url", ""),
                    "dominio":         rec.get("dominio", ""),
                    "data_publicacao": rec.get("data_publicacao", ""),
                },
            )
        )

    total = len(points)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Upsert", unit="batch"):
        batch = points[start : start + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)

    log.info("✅ %d pontos indexados na coleção '%s'", total, collection)


# ──────────────────────────── Main ───────────────────────────────────
def run(args: argparse.Namespace) -> None:
    # 1. Carrega registros
    log.info("Carregando dados de '%s'...", args.input)
    records = load_records(args.input)
    if not records:
        log.error("Nenhum registro válido encontrado em '%s'", args.input)
        sys.exit(1)
    log.info("%d registros carregados", len(records))

    # 2. Carrega o modelo de embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Carregando modelo '%s' em %s...", MODEL_NAME, device)
    model = SentenceTransformer(MODEL_NAME, device=device)
    vector_size = model.get_sentence_embedding_dimension()
    log.info("Dimensão do embedding: %d", vector_size)

    # 3. Gera embeddings
    texts = [make_embed_text(r) for r in records]
    log.info("Gerando embeddings para %d textos...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # normalização L2 → COSINE = dot-product
        convert_to_numpy=True,
    ).tolist()

    # 4. Conecta ao Qdrant e prepara a coleção
    client = get_qdrant_client(Path("data/qdrant_db"))
    recreate_collection(client, args.collection, vector_size)

    # 5. Indexa em lotes
    upsert_in_batches(client, args.collection, records, embeddings)

    # 6. Consulta de teste
    query_semantic_search(client, args.collection, model, args.query, args.top_k)


def query_semantic_search(
    client: QdrantClient,
    collection: str,
    model: SentenceTransformer,
    query: str,
    top_k: int,
) -> None:
    log.info("\n%s", "─" * 70)
    log.info("🔍 CONSULTA DE TESTE: \"%s\"", query)

    query_vec = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    if not results:
        print("\n⚠️  Nenhum resultado encontrado.")
        return

    print(f"\n{'═'*70}")
    print(f"RESULTADO DA BUSCA SEMÂNTICA — Top-{top_k}")
    print(f"{'═'*70}")
    print(f"🔎 Boato consultado: \"{query}\"\n")

    for rank, hit in enumerate(results, 1):
        p = hit.payload
        titulo   = p.get("titulo", "(sem título)")
        dominio  = p.get("dominio", "")
        url      = p.get("url", "")
        data     = p.get("data_publicacao", "")
        texto    = p.get("texto_completo", "")
        score    = hit.score

        # Exibe trecho inicial do laudo (primeiros 400 chars)
        trecho = textwrap.fill(texto[:400], width=70)

        print(f"{'─'*70}")
        print(f"#{rank}  Score de Similaridade: {score:.4f}")
        print(f"    Título:   {titulo}")
        print(f"    Agência:  {dominio}")
        print(f"    Data:     {data}")
        print(f"    URL:      {url}")
        print(f"\n    Trecho do laudo de desmentido:")
        print(f"    {trecho}")
        print()

    print("═" * 70)


# ──────────────────────────── Entry point ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa fact-checks no Qdrant e faz busca semântica")
    parser.add_argument("--input",      type=Path,  default=DEFAULT_INPUT,   help="Arquivo JSONL de entrada")
    parser.add_argument("--collection", type=str,   default=COLLECTION_NAME, help="Nome da coleção Qdrant")
    parser.add_argument("--top-k",      type=int,   default=3,               help="Número de resultados da busca de teste")
    parser.add_argument(
        "--query", type=str,
        default="estão jogando fora cédulas de votação",
        help="Boato/query para o teste de busca semântica",
    )
    args = parser.parse_args()
    run(args)
