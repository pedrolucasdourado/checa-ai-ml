"""
src/config.py
─────────────────────────────────────────────────────────────────────
Configuração centralizada do pipeline RAG.

Todos os valores podem ser sobrescritos por variáveis de ambiente.
Indexador, serviço e scripts devem importar daqui para garantir que
usam exatamente o mesmo modelo, coleção e parâmetros.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ─── Embeddings ──────────────────────────────────────────────────────
# 'openai' (API, sem torch/GPU) ou 'local' (sentence-transformers).
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "openai").lower()
_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "local": "paraphrase-multilingual-mpnet-base-v2",
}
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", _DEFAULT_MODELS.get(EMBEDDING_PROVIDER, ""))

# ─── Qdrant ──────────────────────────────────────────────────────────
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_PATH = Path(os.environ.get("QDRANT_PATH", PROJECT_ROOT / "data" / "qdrant_db"))
# Coleção legada (documento inteiro, 1 vetor, MPNet) mantida para rollback.
LEGACY_COLLECTION_NAME = "fact_checks_pt"
BRONZE_COLLECTION_NAME = os.environ.get("QDRANT_BRONZE_COLLECTION", "fact_checks_bronze")
SILVER_COLLECTION_NAME = os.environ.get("QDRANT_SILVER_COLLECTION", "fact_checks_silver")
# Coleção com chunking (1 vetor por chunk). Uma por provider: a dimensão difere.
COLLECTION_NAME = os.environ.get(
    "QDRANT_COLLECTION",
    "fact_checks_pt_v2" if EMBEDDING_PROVIDER == "local" else f"fact_checks_pt_v2_{EMBEDDING_PROVIDER}",
)

# ─── Chunking (em caracteres; o MPNet trunca em ~128 tokens) ─────────
CHUNK_SIZE = _env_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "pt")

# ─── Retrieval ───────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.60)
# Máximo de chunks que chegam ao LLM, e de documentos distintos entre eles.
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 5)
MAX_DOCUMENTS = _env_int("MAX_DOCUMENTS", 3)
# Candidatos buscados no Qdrant antes da deduplicação = TOP_K * fator.
RETRIEVAL_OVERFETCH = _env_int("RETRIEVAL_OVERFETCH", 3)
# Hits abaixo de (limiar - margem) são descartados do contexto.
CONTEXT_SCORE_MARGIN = _env_float("CONTEXT_SCORE_MARGIN", 0.10)
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 4_000)

# ─── Cluster-aware retrieval ─────────────────────────────────────────
CLUSTER_RERANK_ENABLED = os.environ.get("CLUSTER_RERANK_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CLUSTER_APPROVED_BONUS = _env_float("CLUSTER_APPROVED_BONUS", 0.015)
CLUSTER_NEEDS_SUBCLUSTER_PENALTY = _env_float("CLUSTER_NEEDS_SUBCLUSTER_PENALTY", 0.020)
CLUSTER_LOW_CONFIDENCE_PENALTY = _env_float("CLUSTER_LOW_CONFIDENCE_PENALTY", 0.005)
CLUSTER_HIGH_CONFIDENCE_BONUS = _env_float("CLUSTER_HIGH_CONFIDENCE_BONUS", 0.005)

# ─── LLM ─────────────────────────────────────────────────────────────
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
