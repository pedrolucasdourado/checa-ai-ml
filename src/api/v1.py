"""Versioned verification contract for backend clients."""

from __future__ import annotations

import logging
from typing import Literal

from anyio import CapacityLimiter, to_thread
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.config import COLLECTION_NAME
from src.fact_check_service import FactCheckService, VerificationGenerationError

log = logging.getLogger(__name__)
router = APIRouter()
_verification_limiter = CapacityLimiter(4)


class VerifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain a nonblank character")
        return value


class VerificationSource(BaseModel):
    title: str
    domain: str
    url: str


class VerifyResponse(BaseModel):
    schema_version: Literal[1]
    verdict: Literal["matched", "insufficient_evidence"]
    similarity_score: float
    counter_narrative: str | None
    sources: list[VerificationSource]


@router.post("/api/v1/verify", response_model=VerifyResponse)
async def verify(body: VerifyRequest) -> VerifyResponse:
    try:
        service = await to_thread.run_sync(FactCheckService.get_instance)
        result = await to_thread.run_sync(
            service.verify_claim_for_backend,
            body.text,
            limiter=_verification_limiter,
        )
        return VerifyResponse.model_validate(result)
    except VerificationGenerationError as exc:
        log.warning("Verification generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Verification generation failed") from exc
    except Exception as exc:
        log.exception("Verification request failed")
        raise HTTPException(status_code=503, detail="Verification unavailable") from exc


@router.get("/ready")
async def ready() -> dict[str, str]:
    try:
        service = await to_thread.run_sync(FactCheckService.get_instance)
        collection = await to_thread.run_sync(service._qdrant.get_collection, COLLECTION_NAME)
        vectors = collection.config.params.vectors
        size = vectors.get("size") if isinstance(vectors, dict) else vectors.size
        if size != service._embedder.dimension:
            raise ValueError("Collection vector dimension does not match embedding provider")
    except Exception as exc:
        log.warning("Verification service not ready: %s", exc)
        raise HTTPException(status_code=503, detail="Verification index unavailable") from exc
    return {"status": "ready"}
