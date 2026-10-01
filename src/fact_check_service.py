"""
src/fact_check_service.py
─────────────────────────────────────────────────────────────────────
Serviço de verificação de fatos via pipeline RAG:
  1. Converte a mensagem do usuário em embedding (sentence-transformers)
  2. Busca no Qdrant a checagem mais similar
  3. Se score >= limiar, gera contranarrativa via GPT-4o-mini
  4. Caso contrário, retorna abstinência

Singleton por processo: o modelo e o cliente Qdrant são carregados
apenas uma vez no startup do servidor.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_DB_PATH  = Path(__file__).parent.parent / "data" / "qdrant_db"
COLLECTION_NAME = "fact_checks_pt"
EMBED_MODEL     = "paraphrase-multilingual-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.55
LLM_MODEL       = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2

# ─── Prompt Templates ────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
Você é um verificador de fatos especializado em combate à desinformação no Brasil.
Sua missão é redigir contranarrativas curtas, didáticas e embasadas em evidências \
jornalísticas reais para serem compartilhadas via WhatsApp ou redes sociais.

Regras estritas:
1. COMECE sempre afirmando o FATO verídico logo na primeira frase (Truth Sandwich).
2. Mencione brevemente a alegação falsa circulando — sem amplificá-la.
3. Use APENAS as evidências contidas no "Laudo de Checagem" fornecido. \
PROIBIDO inventar dados, estatísticas ou declarações não presentes no laudo.
4. Tom: cortês, claro, direto e acessível — adequado para leigos.
5. Comprimento: 3 a 5 parágrafos curtos. Sem markdown, sem bullets, só texto corrido.
6. Finalize com a linha exata (substituindo pelos valores reais):
   "Fonte: {dominio} — Leia mais em: {url}"
"""

_USER_PROMPT = """\
--- ALEGAÇÃO RECEBIDA ---
{query}

--- LAUDO DE CHECAGEM RECUPERADO ---
Título: {titulo}
Agência: {dominio}
Data de publicação: {data_publicacao}
URL: {url}

Texto integral do laudo:
{texto_completo}

--- TAREFA ---
Com base EXCLUSIVAMENTE no laudo acima, redija uma contranarrativa clara e \
fundamentada para desmentir a alegação recebida. Siga as regras do sistema.
"""


class FactCheckService:
    """
    Singleton que carrega modelo e cliente Qdrant uma única vez.
    Use `FactCheckService.get_instance()` em vez de instanciar diretamente.
    """

    _instance: "FactCheckService | None" = None

    def __init__(self) -> None:
        self._embed_model = SentenceTransformer(EMBED_MODEL)
        self._qdrant = QdrantClient(path=str(QDRANT_DB_PATH))

    @classmethod
    def get_instance(cls) -> "FactCheckService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── Embedding ────────────────────────────────────────────────────
    def _embed(self, text: str) -> list[float]:
        return (
            self._embed_model
            .encode(text, normalize_embeddings=True, convert_to_numpy=True)
            .tolist()
        )

    # ─── Retrieval ────────────────────────────────────────────────────
    def _retrieve(self, query: str, top_k: int = 1):
        vec = self._embed(query)
        return self._qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=top_k,
            with_payload=True,
        ).points

    # ─── LLM ─────────────────────────────────────────────────────────
    def _call_llm(self, system: str, user: str) -> str:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY não configurada. "
                "Defina a variável de ambiente ou adicione ao arquivo .env."
            )

        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    # ─── Public API ───────────────────────────────────────────────────
    def verify_claim(self, user_message: str) -> dict:
        """
        Executa o pipeline RAG completo.

        Retorno:
            {
                "status":            "matched" | "abstained",
                "score":             float,
                "evidence":          dict | None,
                "counter_narrative": str,
            }
        """
        hits = self._retrieve(user_message, top_k=1)

        if not hits:
            return {
                "status": "abstained",
                "score": 0.0,
                "evidence": None,
                "counter_narrative": (
                    "Abstenção: Nenhuma checagem oficial encontrada para essa alegação."
                ),
            }

        hit = hits[0]
        score = hit.score
        payload = hit.payload

        evidence = {
            "titulo":          payload.get("titulo", ""),
            "texto_completo":  payload.get("texto_completo", ""),
            "dominio":         payload.get("dominio", ""),
            "url":             payload.get("url", ""),
            "data_publicacao": payload.get("data_publicacao", ""),
        }

        if score < SIMILARITY_THRESHOLD:
            return {
                "status": "abstained",
                "score": round(score, 4),
                "evidence": evidence,
                "counter_narrative": (
                    "Abstenção: Nenhuma checagem oficial encontrada para essa alegação."
                ),
            }

        # Monta prompts e chama a LLM
        system_prompt = _SYSTEM_PROMPT.format(
            dominio=evidence["dominio"],
            url=evidence["url"],
        )
        user_prompt = _USER_PROMPT.format(
            query=user_message,
            titulo=evidence["titulo"],
            dominio=evidence["dominio"],
            data_publicacao=evidence["data_publicacao"],
            url=evidence["url"],
            texto_completo=evidence["texto_completo"],
        )

        counter_narrative = self._call_llm(system_prompt, user_prompt)

        return {
            "status": "matched",
            "score": round(score, 4),
            "evidence": evidence,
            "counter_narrative": counter_narrative,
        }
