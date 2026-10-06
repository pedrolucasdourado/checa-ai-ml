"""
app.py — Checa-AI FastAPI Application
──────────────────────────────────────────────────────────────────────
Servidor principal. Expõe:
  GET  /            → página home (redireciona para /debunk)
  GET  /debunk      → interface de verificação de boatos
  POST /api/debunk  → endpoint JSON do pipeline RAG

Inicialização do FactCheckService é feita no lifespan do FastAPI,
garantindo que o modelo seja carregado UMA ÚNICA VEZ, no startup.

Uso:
    ./.venv/bin/uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Garante que src/ está no path quando rodando da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent))
from src.fact_check_service import FactCheckService

log = logging.getLogger("checa-ai")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

BASE_DIR   = Path(__file__).parent
TEMPLATES  = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ─── Lifespan: carrega modelos pesados UMA VEZ no startup ─────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Checa-AI startup — carregando FactCheckService…")
    try:
        FactCheckService.get_instance()
        log.info("✅ FactCheckService pronto.")
    except Exception as exc:
        log.error("❌ Falha ao inicializar FactCheckService: %s", exc)
    yield
    log.info("🛑 Checa-AI shutdown.")


# ─── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Checa-AI",
    description="Pipeline RAG para verificação de fatos e geração de contranarrativas",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ─── Schemas ──────────────────────────────────────────────────────────
class DebunkRequest(BaseModel):
    message: str = Field(..., min_length=10, description="Texto suspeito a ser verificado")


class EvidenceOut(BaseModel):
    titulo: str
    dominio: str
    url: str
    data_publicacao: str | None


class SourceOut(BaseModel):
    title: str
    domain: str
    url: str
    published_at: str | None
    score: float
    rerank_score: float | None = None
    cluster_id: int | None = None
    cluster_label: str = ""
    cluster_type: str = ""
    cluster_confidence: str = ""
    cluster_review_status: str = ""


class DebunkResponse(BaseModel):
    status: str           # "matched" | "abstained"
    score: float
    evidence: EvidenceOut | None
    sources: list[SourceOut] = []
    counter_narrative: str


# ─── Rotas ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/debunk")


@app.get("/debunk", response_class=HTMLResponse, include_in_schema=False)
async def debunk_page(request: Request):
    return TEMPLATES.TemplateResponse(
        request=request,
        name="pages/debunk.html",
        context={"title": "Verificar Boato — Checa-AI"},
    )


@app.post("/api/debunk", response_model=DebunkResponse)
async def api_debunk(body: DebunkRequest):
    """
    Recebe uma mensagem suspeita e retorna:
    - status: matched (contranarrativa gerada) ou abstained (sem checagem)
    - score: similaridade com a checagem mais próxima
    - evidence: metadados da checagem recuperada
    - counter_narrative: texto da contranarrativa ou mensagem de abstinência
    """
    try:
        service = FactCheckService.get_instance()
        result  = service.verify_claim(body.message)
    except RuntimeError as exc:
        # OPENAI_API_KEY ausente ou outro erro de configuração
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("Erro inesperado em /api/debunk: %s", exc)
        raise HTTPException(status_code=500, detail="Erro interno no servidor.")

    evidence_out = None
    if result.get("evidence"):
        ev = result["evidence"]
        evidence_out = EvidenceOut(
            titulo=ev.get("titulo", ""),
            dominio=ev.get("dominio", ""),
            url=ev.get("url", ""),
            data_publicacao=ev.get("data_publicacao"),
        )

    return DebunkResponse(
        status=result["status"],
        score=result["score"],
        evidence=evidence_out,
        sources=[SourceOut(**s) for s in result.get("sources", [])],
        counter_narrative=result["counter_narrative"],
    )


# ─── Healthcheck ──────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "checa-ai"}
