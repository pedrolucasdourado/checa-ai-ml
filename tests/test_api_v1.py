import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from app import app
from src.fact_check_service import FactCheckService, VerificationGenerationError


@pytest.fixture
def fake_service(monkeypatch):
    result = {
        "schema_version": 1,
        "verdict": "insufficient_evidence",
        "similarity_score": 0.1,
        "counter_narrative": None,
        "sources": [],
    }
    service = SimpleNamespace(
        inputs=[],
        _embedder=SimpleNamespace(dimension=1536),
        _qdrant=SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1536))))),
    )

    def verify(text):
        service.inputs.append(text)
        return result.copy()

    service.verify_claim_for_backend = verify
    service.verify_claim = lambda text: {
        "status": "abstained", "score": 0.1, "evidence": None,
        "sources": [], "counter_narrative": "Sem checagem.",
    }
    monkeypatch.setattr(FactCheckService, "get_instance", lambda: service)
    return service, result


@pytest_asyncio.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_short_and_unicode_inputs_reach_service(client, fake_service):
    service, _ = fake_service
    for text in ("x", "é", "界"):
        response = await client.post("/api/v1/verify", json={"text": text})
        assert response.status_code == 200
    assert service.inputs == ["x", "é", "界"]


@pytest.mark.asyncio
async def test_invalid_inputs_return_422(client, fake_service):
    for text in ("", " \n ", "x" * 10001):
        assert (await client.post("/api/v1/verify", json={"text": text})).status_code == 422


@pytest.mark.asyncio
async def test_versioned_results_serialize_exactly(client, fake_service):
    _, result = fake_service
    assert (await client.post("/api/v1/verify", json={"text": "x"})).json() == result
    result.update(
        verdict="matched", similarity_score=0.8,
        counter_narrative="Fato checado [1].",
        sources=[{"title": "Título", "domain": "example.org", "url": "https://example.org/a"}],
    )
    assert (await client.post("/api/v1/verify", json={"text": "x"})).json() == result


@pytest.mark.asyncio
async def test_generation_error_is_not_a_verdict(client, fake_service):
    service, _ = fake_service
    service.verify_claim_for_backend = lambda text: (_ for _ in ()).throw(VerificationGenerationError("citação inválida"))
    response = await client.post("/api/v1/verify", json={"text": "x"})
    assert response.status_code >= 500
    assert "verdict" not in response.json()


@pytest.mark.asyncio
async def test_readiness_checks_collection_and_dimension(client, fake_service):
    service, _ = fake_service
    assert (await client.get("/ready")).status_code == 200
    service._qdrant.get_collection = lambda name: (_ for _ in ()).throw(RuntimeError("missing"))
    assert (await client.get("/ready")).status_code == 503
    service._qdrant.get_collection = lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=384))))
    assert (await client.get("/ready")).status_code == 503


@pytest.mark.asyncio
async def test_legacy_route_shape_is_preserved(client, fake_service):
    response = await client.post("/api/debunk", json={"message": "alegação longa"})
    assert response.status_code == 200
    assert set(response.json()) == {"status", "score", "evidence", "sources", "counter_narrative"}


@pytest.mark.asyncio
async def test_slow_verification_does_not_block_health(client, fake_service):
    service, _ = fake_service
    entered = threading.Event()
    release = threading.Event()

    def slow(text):
        entered.set()
        release.wait(timeout=2)
        return {"schema_version": 1, "verdict": "insufficient_evidence", "similarity_score": 0.0,
                "counter_narrative": None, "sources": []}

    service.verify_claim_for_backend = slow
    pending = asyncio.create_task(client.post("/api/v1/verify", json={"text": "x"}))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert (await asyncio.wait_for(client.get("/health"), 0.5)).status_code == 200
    finally:
        release.set()
        await pending
