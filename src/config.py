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
def _calibrated_threshold(default: float) -> float:
    """Limiar calibrado para `provider/modelo` (scripts/calibrate_threshold.py).

    Cosseno de modelos de embedding diferentes tem distribuições diferentes,
    então o mesmo valor fixo não é comparável entre provedores.
    """
    path = PROJECT_ROOT / "evals" / "calibrated_thresholds.json"
    try:
        import json

        from src.evaluation.calibration import artifact_is_eligible

        entry = json.loads(path.read_text(encoding="utf-8")).get(
            f"{EMBEDDING_PROVIDER}/{EMBED_MODEL}"
        )
        tier = os.environ.get("RETRIEVAL_CALIBRATION_TIER", "production").strip().lower()
        return (
            float(entry["threshold"])
            if entry and artifact_is_eligible(entry, runtime_tier=tier)
            else default
        )
    except (OSError, ValueError, KeyError, TypeError):
        return default


SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", _calibrated_threshold(0.60))
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


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# ─── Híbrido (BM25 + denso, RRF) e reranker cross-encoder ────────────
# Desligado por padrão: no A/B pareado (benchmark de 120 casos parafraseados) não
# houve ganho (4 melhoras x 5 pioras, teste de sinal p=1,0). Ver docs/RETRIEVAL_BENCHMARK.md.
HYBRID_ENABLED = _env_flag("HYBRID_ENABLED", False)
HYBRID_LEXICAL_WEIGHT = _env_float("HYBRID_LEXICAL_WEIGHT", 1.2)  # >1 quebra empates do RRF a favor do lexical; tunar com evaluate_retrieval
RERANKER_ENABLED = _env_flag("RERANKER_ENABLED", False)
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_TOP_N = _env_int("RERANKER_TOP_N", 15)

# ─── NLI auxiliar ────────────────────────────────────────────────────
NLI_ENABLED = _env_flag("NLI_ENABLED", False)
NLI_MODEL = os.environ.get("NLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
NLI_MODEL_REVISION = os.environ.get("NLI_MODEL_REVISION", "b5113eb")
NLI_DEVICE = os.environ.get("NLI_DEVICE", "auto").strip().lower()
NLI_ABSTAIN_ON_DISAGREEMENT = _env_flag("NLI_ABSTAIN_ON_DISAGREEMENT", False)

# ─── Named-entity recognition (experimental) ─────────────────────────
# "auto" uses spaCy when its optional model is installed and otherwise keeps
# the existing heuristic behaviour. Production can remain on "heuristic".
NER_BACKEND = os.environ.get("NER_BACKEND", "heuristic").strip().lower()
NER_SPACY_MODEL = os.environ.get("NER_SPACY_MODEL", "pt_core_news_lg").strip()

# ─── LLM ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.environ.get("LLM_API_KEY", "").strip()
# Tarefa de classificação/grounding: 0 por padrão para reprodutibilidade.
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.0)
# Timeout (s) e retries do cliente OpenAI; sem isso uma chamada pendurada trava o worker.
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 30.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)

# ─── Classificador auxiliar (após RAG e busca web inconclusivos) ──────
CLASSIFIER_ENABLED = _env_flag("CLASSIFIER_ENABLED", False)
CLASSIFIER_TFIDF_MODEL_PATH = os.environ.get("CLASSIFIER_TFIDF_MODEL_PATH", "").strip()
CLASSIFIER_BERTIMBAU_MODEL_PATH = os.environ.get("CLASSIFIER_BERTIMBAU_MODEL_PATH", "").strip()
CLASSIFIER_BACKEND = os.environ.get("CLASSIFIER_BACKEND", "bertimbau").strip().lower()
CLASSIFIER_BERTIMBAU_DEVICE = os.environ.get("CLASSIFIER_BERTIMBAU_DEVICE", "cpu").strip()
CLASSIFIER_BERTIMBAU_BATCH_SIZE = _env_int("CLASSIFIER_BERTIMBAU_BATCH_SIZE", 16)


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
