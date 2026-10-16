import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from app import app
from src.fact_check_service import FactCheckService, VerificationGenerationError


@pytest.fixture
def fake_service(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    result = {
        "schema_version": 1,
        "verdict": "insufficient_evidence",
        "similarity_score": 0.1,
        "counter_narrative": None,
        "sources": [],
    }
    service = SimpleNamespace(
        inputs=[],
        tracking=[],
        _embedder=SimpleNamespace(dimension=1536),
        _qdrant=SimpleNamespace(get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=1536))))),
    )

    def verify(text, **tracking):
        service.inputs.append(text)
        service.tracking.append(tracking)
        return result.copy()

    service.verify_claim_for_backend = verify
    service.verify_claim = lambda text, *, session_id=None: {
        "status": "abstained",
        "score": 0.1,
        "evidence": None,
        "counter_narrative": "Sem checagem.",
        "claim": text,
        "conclusion": "inconclusiva",
        "evidence_sufficient": False,
        "evidence_level": "ausente",
        "explanation": "Sem checagem.",
        "sources": [],
        "probabilistic_assessment": None,
        "trace_id": None,
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
    result.update(verdict="deep_searched", similarity_score=0.1,
                  counter_narrative="Pesquisa web cautelosa.", sources=[],
                  conclusion="inconclusiva", evidence_sufficient=False)
    response = await client.post("/api/v1/verify", json={"text": "x"})
    assert response.status_code == 200
    assert response.json() == result
    assert response.json()["sources"] == []
    result.update(verdict="blocked", similarity_score=None,
                  counter_narrative="Solicitação bloqueada.")
    response = await client.post("/api/v1/verify", json={"text": "x"})
    assert response.status_code == 200
    assert response.json() == result
    assert response.json()["similarity_score"] is None
    result.update(
        verdict="matched", similarity_score=0.8,
        counter_narrative="Fato checado [1].",
        sources=[{"title": "Título", "domain": "example.org", "url": "https://example.org/a"}],
    )
    assert (await client.post("/api/v1/verify", json={"text": "x"})).json() == result


def test_openapi_verification_schema_has_four_verdicts_and_nullable_score():
    schema = app.openapi()["components"]["schemas"]["VerifyResponse"]
    assert set(schema["properties"]["verdict"]["enum"]) == {
        "matched", "deep_searched", "insufficient_evidence", "blocked"
    }
    score = schema["properties"]["similarity_score"]
    assert {item["type"] for item in score["anyOf"]} == {"number", "null"}
    conclusion = schema["properties"]["conclusion"]
    assert set(conclusion["anyOf"][0]["enum"]) == {
        "sustentada", "contradita", "inconclusiva"
    }


@pytest.mark.asyncio
async def test_generation_error_is_not_a_verdict(client, fake_service):
    service, _ = fake_service
    service.verify_claim_for_backend = lambda text: (_ for _ in ()).throw(VerificationGenerationError("citação inválida"))
    response = await client.post("/api/v1/verify", json={"text": "x"})
    assert response.status_code >= 500
    assert "verdict" not in response.json()


@pytest.mark.asyncio
async def test_upstream_error_does_not_log_private_content(client, fake_service, caplog):
    service, _ = fake_service
    secret = "alegação privada ou chave secreta"

    def fail(text):
        raise RuntimeError(secret)

    service.verify_claim_for_backend = fail
    response = await client.post("/api/v1/verify", json={"text": secret})

    assert response.status_code == 503
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_readiness_error_does_not_log_private_content(client, fake_service, caplog):
    service, _ = fake_service
    secret = "private-key-in-upstream-error"
    service._qdrant.get_collection = lambda name: (_ for _ in ()).throw(RuntimeError(secret))

    response = await client.get("/ready")

    assert response.status_code == 503
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_readiness_checks_collection_and_dimension(client, fake_service):
    service, _ = fake_service
    assert (await client.get("/ready")).status_code == 200
    service._qdrant.get_collection = lambda name: (_ for _ in ()).throw(RuntimeError("missing"))
    assert (await client.get("/ready")).status_code == 503
    service._qdrant.get_collection = lambda name: SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=SimpleNamespace(size=384))))
    assert (await client.get("/ready")).status_code == 503


@pytest.mark.asyncio
async def test_readiness_requires_generation_key(client, fake_service, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (await client.get("/ready")).status_code == 503


@pytest.mark.asyncio
async def test_slow_embedding_dimension_does_not_block_health(client, fake_service):
    service, _ = fake_service
    entered = threading.Event()
    release = threading.Event()

    class SlowEmbedder:
        @property
        def dimension(self):
            entered.set()
            release.wait(timeout=1)
            return 1536

    service._embedder = SlowEmbedder()
    started = time.monotonic()
    pending = asyncio.create_task(client.get("/ready"))
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        assert time.monotonic() - started < 0.2
        assert (await asyncio.wait_for(client.get("/health"), 0.1)).status_code == 200
    finally:
        release.set()
        await pending


@pytest.mark.asyncio
async def test_legacy_debunk_route_shape_is_preserved(client, fake_service):
    response = await client.post("/api/debunk", json={"message": "alegação longa"})
    assert response.status_code == 200
    assert set(response.json()) == {
        "status", "score", "evidence", "sources", "counter_narrative", "trace_id"
    }


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


@pytest.mark.asyncio
async def test_internal_token_is_enforced_when_configured(client, fake_service, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-token")
    url = "/api/v1/verify"
    assert (await client.post(url, json={"text": "x"})).status_code == 401
    wrong = await client.post(url, json={"text": "x"}, headers={"X-Internal-Token": "nope"})
    assert wrong.status_code == 401
    ok = await client.post(url, json={"text": "x"}, headers={"X-Internal-Token": "secret-token"})
    assert ok.status_code == 200


@pytest.mark.asyncio
async def test_health_and_ready_do_not_require_internal_token(client, fake_service, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-token")
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/ready")).status_code == 200


PSEUDONYM = "u_" + "ab12cd34ef56ab78cd90"


@pytest.mark.asyncio
async def test_pseudonym_and_session_reach_the_service(client, fake_service):
    service, _ = fake_service

    response = await client.post(
        "/api/v1/verify",
        json={"text": "alegação", "user_id": PSEUDONYM, "session_id": "0b0e4a0e-5f8d-4a06"},
    )

    assert response.status_code == 200
    assert service.tracking == [{"user_id": PSEUDONYM, "session_id": "0b0e4a0e-5f8d-4a06"}]


@pytest.mark.asyncio
async def test_request_without_tracking_is_unchanged(client, fake_service):
    service, _ = fake_service

    response = await client.post("/api/v1/verify", json={"text": "alegação"})

    assert response.status_code == 200
    assert service.tracking == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_id",
    ["5563984267474", "5563984267474@s.whatsapp.net", "+55 63 98426-7474", "u_xyz", "u_" + "a" * 8],
)
async def test_raw_phone_or_malformed_user_id_is_rejected(client, fake_service, user_id):
    service, _ = fake_service

    response = await client.post("/api/v1/verify", json={"text": "alegação", "user_id": user_id})

    assert response.status_code == 422
    assert service.inputs == []


@pytest.mark.asyncio
async def test_malformed_session_id_is_rejected(client, fake_service):
    response = await client.post(
        "/api/v1/verify", json={"text": "alegação", "session_id": "tem espaço"}
    )
    assert response.status_code == 422
