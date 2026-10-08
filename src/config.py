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
# Timeout (s) e retries do cliente OpenAI; sem isso uma chamada pendurada trava o worker.
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 30.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ─── Observabilidade (Langfuse) ──────────────────────────────────────
# O SDK lê LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL do ambiente.
# Sem as chaves, o tracing vira no-op e o pipeline funciona normalmente.
LANGFUSE_ENABLED = _env_bool("LANGFUSE_ENABLED", True)
# false → traces guardam só tamanho/metadados, nunca o texto do usuário (LGPD).
LANGFUSE_CAPTURE_CONTENT = _env_bool("LANGFUSE_CAPTURE_CONTENT", True)

# ─── Safety Guardrails ───────────────────────────────────────────────
SAFE_REFUSAL_MESSAGE = "Desculpe, mas não posso processar esta solicitação porque ela viola nossas diretrizes de segurança e moderação."
SAFE_INJECTION_REFUSAL_MESSAGE = "Detectamos uma tentativa de contornar as instruções do sistema. Por favor, envie a alegação que deseja verificar."

# ─── Deep Search / Fallback Prompts ───────────────────────────────────
DEEP_SEARCH_SYSTEM_PROMPT = """\
Você é um assistente de pesquisa web neutro. 
O sistema de checagem oficial não encontrou laudos conclusivos para a alegação do usuário.
Sua tarefa é resumir as informações encontradas na internet sobre o assunto.

REGRAS CRÍTICAS:
1. NÃO diga se a alegação é "Fato" ou "Fake". Você não tem um laudo oficial para isso.
2. Use frases como: "Encontramos menções a...", "Alguns sites relatam que...", "Não há consenso claro, mas...".
3. Apresente os pontos principais encontrados nos resultados da busca.
4. FINALIZE SEMPRE com um aviso de pensamento crítico: 
   "Atenção: Não encontramos checagens oficiais para este caso. Recomendamos cautela e que você verifique a credibilidade das fontes citadas antes de acreditar ou compartilhar."
5. Mantenha um tom informativo, neutro e cauteloso.
"""

DEEP_SEARCH_USER_PROMPT = """\
ALEGAÇÃO: {query}

RESULTADOS DA BUSCA WEB:
{web_results}

TAREFA: Com base nos resultados acima, forneça um resumo informativo e neutro. Lembre-se de não dar um veredito final.
"""
PROMPT_LABEL = os.environ.get("PROMPT_LABEL", "production")
