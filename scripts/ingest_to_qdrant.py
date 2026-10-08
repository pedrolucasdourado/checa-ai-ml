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
                                        [--collection fact_checks_pt_v2] [--recreate]
                                        [--query "estão jogando fora cédulas de votação"]
                                        [--top-k 3]
"""

from __future__ import annotations

import argparse
from array import array
import json
import logging
import os
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

from tqdm import tqdm

# ─── Qdrant ──────────────────────────────────────────────────────────
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import COLLECTION_NAME, EMBED_MODEL, EMBEDDING_PROVIDER, QDRANT_PATH, QDRANT_URL
from src.rag.chunking import Chunk, build_chunks
from src.rag.embeddings import EmbeddingProvider, get_embedding_provider
from src.rag.qdrant import get_qdrant_client

# ─── Configurações padrão ─────────────────────────────────────────────
DEFAULT_INPUT   = Path("data/processed/fact_checks_silver.jsonl")
DEFAULT_DB_PATH = QDRANT_PATH
BATCH_SIZE      = 32
EMBED_BATCH_SIZE = 100
CACHE_DIR = Path("data/embedding_cache")

# Modelo: multilíngue de alta qualidade (suporta PT de forma excelente),
# 768 dimensões — sem precisar baixar BERTimbau separadamente.
# Troca por "neuralmind/bert-base-portuguese-cased" se quiser PT puro
# (mas exigiria pool manual via transformers).
# Definido centralmente em src/config.py.
MODEL_NAME = EMBED_MODEL

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


def build_all_chunks(records: list[dict]) -> list[Chunk]:
    """Divide cada laudo em chunks (1 vetor por chunk)."""
    chunks: list[Chunk] = []
    for rec in records:
        chunks.extend(build_chunks(rec))
    return chunks


def _safe_filename(value: str) -> str:
    """Nome de arquivo estável para provider/modelo/collection."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def default_cache_path(collection: str, provider: EmbeddingProvider) -> Path:
    name = _safe_filename(f"{collection}_{provider.name}_{provider.model}_{provider.dimension}d")
    return CACHE_DIR / f"{name}.sqlite3"


class EmbeddingCache:
    """Cache SQLite de embeddings por chunk/modelo, com escrita incremental."""

    def __init__(self, path: Path, provider: EmbeddingProvider, dimension: int) -> None:
        self.path = path
        self.provider = provider
        self.dimension = dimension
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_embeddings_model
            ON embeddings(provider, model, dimension)
            """
        )
        self.conn.commit()

    def get(self, chunk: Chunk) -> list[float] | None:
        row = self.conn.execute(
            """
            SELECT vector
            FROM embeddings
            WHERE chunk_id = ?
              AND content_hash = ?
              AND provider = ?
              AND model = ?
              AND dimension = ?
            """,
            (
                chunk.chunk_id,
                chunk.content_hash,
                self.provider.name,
                self.provider.model,
                self.dimension,
            ),
        ).fetchone()
        if row is None:
            return None

        vec = array("f")
        vec.frombytes(row[0])
        if len(vec) != self.dimension:
            return None
        return vec.tolist()

    def put(self, chunk: Chunk, vector: list[float]) -> None:
        vec = array("f", (float(x) for x in vector))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                chunk_id, content_hash, provider, model, dimension, vector, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                chunk.chunk_id,
                chunk.content_hash,
                self.provider.name,
                self.provider.model,
                self.dimension,
                sqlite3.Binary(vec.tobytes()),
            ),
        )

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def embed_chunks_with_cache(
    chunks: list[Chunk],
    provider: EmbeddingProvider,
    cache: EmbeddingCache | None,
    batch_size: int,
) -> list[list[float]]:
    """Gera embeddings reutilizando cache e salvando checkpoints por batch."""
    embeddings: list[list[float] | None] = [None] * len(chunks)
    missing: list[tuple[int, Chunk]] = []

    if cache is not None:
        for i, chunk in enumerate(tqdm(chunks, desc="Cache", unit="chunk")):
            cached = cache.get(chunk)
            if cached is None:
                missing.append((i, chunk))
            else:
                embeddings[i] = cached
        log.info("Cache de embeddings: %d hits, %d miss", len(chunks) - len(missing), len(missing))
    else:
        missing = list(enumerate(chunks))
        log.info("Cache de embeddings desativado: %d chunks serão gerados", len(missing))

    if missing:
        log.info("Gerando embeddings para %d chunks faltantes...", len(missing))
    for start in tqdm(range(0, len(missing), batch_size), desc="Embeddings", unit="batch"):
        batch = missing[start : start + batch_size]
        texts = [chunk.embed_text for _, chunk in batch]
        vectors = provider.embed_documents(texts)
        for (index, chunk), vector in zip(batch, vectors):
            embeddings[index] = vector
            if cache is not None:
                cache.put(chunk, vector)
        if cache is not None:
            cache.commit()

    return [vec for vec in embeddings if vec is not None]


def prepare_collection(
    client: QdrantClient, name: str, vector_size: int, recreate: bool = False
) -> None:
    """Cria a coleção se não existir; só apaga e recria com `recreate=True`."""
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        if not recreate:
            log.info("Coleção '%s' já existe — upsert idempotente (IDs estáveis).", name)
            return
        log.info("Coleção '%s' já existe — recriando (--recreate)...", name)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    log.info("Coleção '%s' criada (%d dims, COSINE)", name, vector_size)


def upsert_in_batches(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vec,
            payload={**chunk.payload(), "embedding_model": MODEL_NAME},
        )
        for chunk, vec in zip(chunks, embeddings)
    ]

    total = len(points)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Upsert", unit="batch"):
        batch = points[start : start + BATCH_SIZE]
        client.upsert(collection_name=collection, points=batch)

    log.info("%d pontos indexados na coleção '%s'", total, collection)


# ──────────────────────────── Main ───────────────────────────────────
def run(args: argparse.Namespace) -> None:
    # 1. Carrega registros
    log.info("Carregando dados de '%s'...", args.input)
    records = load_records(args.input)
    if not records:
        log.error("Nenhum registro válido encontrado em '%s'", args.input)
        sys.exit(1)
    log.info("%d registros carregados", len(records))

    # 2. Provider de embeddings (openai | local, via EMBEDDING_PROVIDER)
    provider = get_embedding_provider()
    vector_size = provider.dimension
    log.info("Provider '%s', modelo '%s', %d dims", provider.name, provider.model, vector_size)

    # 3. Chunking + embeddings
    chunks = build_all_chunks(records)
    log.info("%d laudos → %d chunks", len(records), len(chunks))

    cache: EmbeddingCache | None = None
    if args.no_embedding_cache:
        log.info("Cache de embeddings desativado por --no-embedding-cache")
    else:
        cache_path = args.embedding_cache or default_cache_path(args.collection, provider)
        cache = EmbeddingCache(cache_path, provider, vector_size)
        log.info("Cache de embeddings: %s", cache.path)

    try:
        embeddings = embed_chunks_with_cache(
            chunks=chunks,
            provider=provider,
            cache=cache,
            batch_size=args.embedding_batch_size,
        )
    finally:
        if cache is not None:
            cache.close()

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Falha ao gerar embeddings: {len(embeddings)} vetores para {len(chunks)} chunks"
        )

    # 4. Conecta ao Qdrant e prepara a coleção
    client = get_qdrant_client(
        url=args.qdrant_url,
        path=args.qdrant_path,
        local_fallback=not args.require_docker,
    )
    prepare_collection(client, args.collection, vector_size, recreate=args.recreate)

    # 5. Indexa em lotes
    upsert_in_batches(client, args.collection, chunks, embeddings)

    # 6. Consulta de teste
    if args.skip_query:
        log.info("Consulta de teste ignorada por --skip-query")
    else:
        query_semantic_search(client, args.collection, provider, args.query, args.top_k)


def query_semantic_search(
    client: QdrantClient,
    collection: str,
    provider: EmbeddingProvider,
    query: str,
    top_k: int,
) -> None:
    log.info("\n%s", "-" * 70)
    log.info("CONSULTA DE TESTE: \"%s\"", query)

    query_vec = provider.embed_query(query)

    results = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    ).points

    if not results:
        print("\nNenhum resultado encontrado.")
        return

    print(f"\n{'='*70}")
    print(f"RESULTADO DA BUSCA SEMANTICA - Top-{top_k}")
    print(f"{'='*70}")
    print(f"Boato consultado: \"{query}\"\n")

    for rank, hit in enumerate(results, 1):
        p = hit.payload
        titulo   = p.get("titulo", "(sem título)")
        dominio  = p.get("dominio", "")
        url      = p.get("url", "")
        data     = p.get("data_publicacao", "")
        texto    = p.get("texto_chunk") or p.get("texto_completo", "")
        score    = hit.score

        # Exibe trecho inicial do laudo (primeiros 400 chars)
        trecho = textwrap.fill(texto[:400], width=70)

        print(f"{'-'*70}")
        print(f"#{rank}  Score de Similaridade: {score:.4f}")
        print(f"    Título:   {titulo}")
        print(f"    Agência:  {dominio}")
        print(f"    Data:     {data}")
        print(f"    URL:      {url}")
        print(f"\n    Trecho do laudo de desmentido:")
        print(f"    {trecho}")
        print()

    print("=" * 70)


# ──────────────────────────── Entry point ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexa fact-checks no Qdrant e faz busca semântica")
    parser.add_argument("--input",      type=Path,  default=DEFAULT_INPUT,   help="Arquivo JSONL de entrada")
    parser.add_argument("--collection", type=str,   default=COLLECTION_NAME, help="Nome da coleção Qdrant")
    parser.add_argument("--qdrant-url", type=str,   default=QDRANT_URL,       help="URL do Qdrant server")
    parser.add_argument("--qdrant-path", type=Path, default=DEFAULT_DB_PATH,  help="Diretório Qdrant local para fallback")
    parser.add_argument("--require-docker", action="store_true",             help="Falha se Qdrant server não estiver disponível")
    parser.add_argument("--recreate",   action="store_true",                  help="Apaga e recria a coleção alvo antes de indexar")
    parser.add_argument("--top-k",      type=int,   default=3,               help="Número de resultados da busca de teste")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Arquivo SQLite para cache/checkpoint dos embeddings",
    )
    parser.add_argument(
        "--no-embedding-cache",
        action="store_true",
        help="Desativa o cache de embeddings",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help="Número de chunks por batch enviado ao provider de embeddings",
    )
    parser.add_argument(
        "--query", type=str,
        default="estão jogando fora cédulas de votação",
        help="Boato/query para o teste de busca semântica",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Não executa a consulta de teste ao final da indexação",
    )
    args = parser.parse_args()
    run(args)
