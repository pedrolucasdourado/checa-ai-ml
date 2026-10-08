# ML–Backend Verification Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the backend WhatsApp worker to a versioned ML verification API and send a short, sourced, non-simulated answer.

**Architecture:** The ML repository serves `POST /api/v1/verify` from a persistent FastAPI container using its existing Qdrant index. The backend worker calls that API through an outbound HTTP adapter, checkpoints a versioned result, and formats one WhatsApp message through its existing outbox and Evolution delivery flow.

**Tech Stack:** Python, FastAPI, Pydantic, Qdrant, OpenAI or local embeddings, HTTPX, PostgreSQL, Redis Streams, Docker Compose, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-ml-backend-integration-design.md` in `checa-ai-ml`.

## Global Constraints

- Repositories: `checa-ai-ml` owns retrieval, generation, `/api/v1/verify`, and its container; sibling `checa-ai` owns WhatsApp orchestration and the outbound HTTP adapter.
- Verdicts are only `matched` and `insufficient_evidence`; similarity is not factual confidence and must not appear as such to the user.
- One WhatsApp response per request, at most 1,000 characters; `matched` includes a short narrative and at least one cited source URL.
- An insufficient result contains no narrative or sources treated as proof; an API failure is never converted into insufficient evidence or silent simulation.
- Use the existing indexed Qdrant collection. Do not add DVC pull, indexing, or a true/false classifier to this feature.
- Keep `POST /api/debunk` and the backend's webhook, leases, checkpoints, outbox, Redis, and Evolution delivery behavior.
- The backend's current `feat/new-link-reader` working tree contains unrelated uncommitted changes. Before executing backend tasks, establish an isolated branch/worktree whose baseline includes the link flow reviewed for this plan; do not overwrite those changes.

## Review Focus

1. A one-character or Unicode-only nonblank message must receive a valid verdict rather than a backend retry caused by the ML route's old 10-character minimum (Task 2 test).
2. Fabricated citation numbers, non-HTTP source URLs, or duplicate source URLs must be rejected or removed before a matched response is returned (Task 1 test).
3. A long source URL or linked-article title must never produce a WhatsApp message over 1,000 characters or silently cut a URL (Task 5 test).
4. A worker restart after a real-result checkpoint must restore the result and skip a second ML call (Task 6 test).
5. One slow verification must not block `/health` or another accepted verification indefinitely (Task 2 test).

---

## File map and task order

| Task | Repository | Main files | Deliverable |
|---|---|---|---|
| 1 | `checa-ai-ml` | `src/fact_check_service.py`, `tests/test_rag_pipeline.py` | Versioned result construction from existing retrieval |
| 2 | `checa-ai-ml` | `src/api/v1.py`, `app.py`, `tests/test_api_v1.py` | HTTP contract, readiness, and nonblocking request handling |
| 3 | `checa-ai-ml` | `Dockerfile`, `.dockerignore`, `requirements-api.txt`, `compose.yaml` | Persistent API container beside Qdrant |
| 4 | `checa-ai` | `app/domain/verification.py`, `app/application/ports/verifier.py`, `app/application/use_cases/verify_content.py`, `app/adapters/outbound/ai/ml_verifier.py` | Validated domain result and outbound HTTP adapter |
| 5 | `checa-ai` | `app/adapters/outbound/ai/verification_response_formatter.py` | Deterministic 1,000-character WhatsApp reply |
| 6 | `checa-ai` | `app/config.py`, `app/bootstrap.py`, `app/application/use_cases/orchestrate_message.py` | Real-mode wiring, checkpoint restore, and failure policy |
| 7 | `checa-ai` | `docker-compose.ml.yml`, `.env.example`, `README.md`, `tests/integration/test_ml_api_pipeline.py` | Network wiring and complete mocked round trip |

### Task 1: Build the ML verification result

**Files:** Modify `checa-ai-ml/src/fact_check_service.py`, `checa-ai-ml/tests/test_rag_pipeline.py`, and `checa-ai-ml/src/config.py`.

**Interfaces:** Produce `FactCheckService.verify_claim_for_backend(text: str) -> dict` with keys `schema_version`, `verdict`, `similarity_score`, `counter_narrative`, and `sources`; define `VerificationGenerationError`. Task 2 exposes this method over HTTP.

- [ ] **Step 1: Write failing tests.** Add `test_v1_insufficient_hides_weak_sources`, `test_v1_matched_returns_only_cited_sources`, and `test_v1_rejects_fabricated_or_unsafe_citations`. Assert version `1`, the exact two verdict values, null narrative/empty sources on abstention, at most two unique HTTP(S) sources, and no returned URL absent from retrieved evidence; reject a cited unsafe URL rather than silently dropping its reference. Add `test_v1_condenses_or_rejects_overlong_narrative` with a stub LLM that ignores the first size request, and `test_llm_client_reused_across_requests`.
- [ ] **Step 2: Verify red.** Run from `checa-ai-ml`: `python -m pytest tests/test_rag_pipeline.py -q`. Expected: the new tests fail because `verify_claim_for_backend` is absent.
- [ ] **Step 3: Implement the smallest service change.** Reuse `_retrieve`, `build_context`, and `_call_llm`; add a short narrative prompt without source footer and a 500-character narrative target. Validate numbered citations against selected evidence, build source URLs from those evidence objects only, reject unsafe cited URLs, deduplicate, attempt one condensation if needed, and raise `VerificationGenerationError` if grounding or length still fails. Cache the OpenAI generation client per service instance. Preserve `verify_claim` for `/api/debunk`.
- [ ] **Step 4: Verify green.** Run the same test file; expected: all cases pass, including existing RAG cases.
- [ ] **Step 5: Commit in `checa-ai-ml`.** `git add src/fact_check_service.py src/config.py tests/test_rag_pipeline.py` then `git commit -m "feat: return grounded verification result for backend"`.

### Task 2: Expose the versioned ML API safely

**Files:** Create `checa-ai-ml/src/api/__init__.py`, `checa-ai-ml/src/api/v1.py`, and `checa-ai-ml/tests/test_api_v1.py`; modify `checa-ai-ml/app.py`.

**Interfaces:** `src.api.v1.router` mounts `POST /api/v1/verify`; input is `{"text": str}` with 1–10,000 nonblank characters. Output uses Task 1's keys; `GET /ready` checks the configured Qdrant collection and vector dimension. Existing `GET /health` remains liveness.

- [ ] **Step 1: Write failing API tests.** Assert a one-character claim and a Unicode-only nonblank claim reach the service; whitespace-only/over-10,000 input returns 422; `matched` and `insufficient_evidence` serialize exactly as v1; `VerificationGenerationError` yields a non-verdict 5xx; missing/wrong-dimension collection yields 503. Assert `/api/debunk` still has its existing response shape. Add `test_slow_verification_does_not_block_health` with concurrent ASGI requests and a controlled blocking stub.
- [ ] **Step 2: Verify red.** Run from `checa-ai-ml`: `python -m pytest tests/test_api_v1.py -q`. Expected: missing route or failing assertions.
- [ ] **Step 3: Implement schemas, route, and readiness.** Use a Pydantic request/response model in `src/api/v1.py`; call `verify_claim_for_backend` outside the event loop with bounded concurrency. Initialize/reuse clients in the app lifespan, and verify Qdrant collection/dimension before reporting ready. Keep `/api/debunk` behavior intact.
- [ ] **Step 4: Verify green.** Run `python -m pytest tests/test_api_v1.py tests/test_rag_pipeline.py -q`; expected: pass.
- [ ] **Step 5: Commit in `checa-ai-ml`.** Stage the files above and commit `feat: expose versioned verification API`.

### Task 3: Containerize the ML API

**Files:** Create `checa-ai-ml/Dockerfile`, `checa-ai-ml/.dockerignore`, and `checa-ai-ml/requirements-api.txt`; modify `checa-ai-ml/compose.yaml` and `checa-ai-ml/README.md`.

**Interfaces:** Compose service `ml-api` serves port 8000 inside network `checa-ai-ml-shared`, connects to Qdrant at `http://qdrant:6333`, and uses the configured embedding provider/collection. Backend Task 7 joins the same network. It does not run ingestion.

- [ ] **Step 1: Add a deployment check.** Record in the README the commands that must succeed: `docker compose config -q`, `docker compose build ml-api`, and `GET /ready` after starting with an already indexed collection. Specify that readiness is 503 when the collection is absent; it is not auto-created.
- [ ] **Step 2: Verify the current configuration lacks the service.** Run from `checa-ai-ml`: `docker compose config --services`; expected: `qdrant` only, no `ml-api`.
- [ ] **Step 3: Implement the image and Compose service.** Use a production dependency list separate from notebooks, exclude datasets and secrets from the build context, reuse the Qdrant volume, and give both services the named network. Pass embedding/Qdrant/OpenAI settings through environment variables without hard-coded secrets.
- [ ] **Step 4: Verify configuration and build.** Run `docker compose config -q` and `docker compose build ml-api`; expected: both exit 0. With a populated index available, start the services and confirm `/ready` returns 200; with an empty index, confirm 503.
- [ ] **Step 5: Commit in `checa-ai-ml`.** Stage the five files and commit `build: run verification API beside Qdrant`.

### Task 4: Add the backend verification port and HTTP adapter

**Files:** Create `checa-ai/app/domain/verification.py`, `checa-ai/app/application/ports/verifier.py`, `checa-ai/app/application/use_cases/verify_content.py`, `checa-ai/app/adapters/outbound/ai/ml_verifier.py`, `checa-ai/tests/unit/test_verification_contracts.py`, and `checa-ai/tests/contract/test_ml_verifier.py`.

**Interfaces:** `VerificationSource(title: str, domain: str, url: str)` and `VerificationResult(schema_version: int, verdict: str, similarity_score: float, counter_narrative: str | None, sources: tuple[VerificationSource, ...], result_type: str = "verification")`. `VerifierPort.verify(content: AnalysisInput) -> VerificationResult`; `VerifyContentUseCase.execute(content)` validates both sides; `MlVerifier.verify(content)` posts to `/api/v1/verify` using an injected `httpx.AsyncClient`.

- [ ] **Step 1: Write failing contract tests.** Assert the two verdict invariants, finite score in `[0, 1]`, at least one HTTP(S) source for matched, no source/narrative for insufficient, and rejection of unknown schema versions. Mock HTTP to assert exact request JSON and map 422 to `PermanentStageError`, network/503/5xx to `TransientStageError`, and malformed JSON to `InvalidContract`; no case returns simulation or logs the claim text.
- [ ] **Step 2: Verify red.** Run from `checa-ai`: `.venv/bin/python -m pytest tests/unit/test_verification_contracts.py tests/contract/test_ml_verifier.py -q`; expected: import or assertion failures.
- [ ] **Step 3: Implement the domain contract and adapter.** Keep HTTPX/Pydantic parsing in the adapter, domain validation in `app/domain/verification.py`, and request timeout injectable. Do not import ML repository code into the backend.
- [ ] **Step 4: Verify green.** Run the same two test files; expected: pass.
- [ ] **Step 5: Commit in the backend branch.** Stage these six files and commit `feat: add versioned ML verification adapter`.

### Task 5: Format real verification results for WhatsApp

**Files:** Create `checa-ai/app/adapters/outbound/ai/verification_response_formatter.py` and `checa-ai/tests/contract/test_verification_formatter.py`; extend `checa-ai/app/domain/verification.py` with `VerificationResponseContext(result: VerificationResult, link: LinkContext | None)`.

**Interfaces:** `VerificationResponseFormatter.format(context: VerificationResponseContext) -> FormattedResponse` is async to match the existing formatter call site. It emits `Checagem encontrada` plus narrative and source URLs for `matched`; it emits a fixed `Evidência insuficiente` message for abstention. It never invokes a second LLM.

- [ ] **Step 1: Write failing formatter tests.** Assert a matched result contains the verdict, narrative, each cited URL, and no similarity percentage or demo notice; a link result distinguishes the analyzed article's title/domain from fact-check URLs; abstention has no evidence links. Assert a long source URL or article title cannot cause output over 1,000 characters or a silently truncated URL.
- [ ] **Step 2: Verify red.** Run `.venv/bin/python -m pytest tests/contract/test_verification_formatter.py -q`; expected: import or assertion failures.
- [ ] **Step 3: Implement deterministic formatting.** Keep source URLs verbatim, bound compact article attribution, and reject an impossible layout with `InvalidContract` so the orchestrator uses its non-verdict error path. Use the existing `FormattedResponse` type.
- [ ] **Step 4: Verify green.** Run the same test file; expected: pass.
- [ ] **Step 5: Commit in the backend branch.** Stage the formatter, domain, and test files; commit `feat: format sourced verification replies`.

### Task 6: Wire real mode into the backend worker

**Files:** Modify `checa-ai/app/config.py`, `checa-ai/app/bootstrap.py`, and `checa-ai/app/application/use_cases/orchestrate_message.py`; extend `checa-ai/tests/unit/test_worker.py` and `checa-ai/tests/integration/test_orchestrator_postgres.py`.

**Interfaces:** `CLASSIFIER_BACKEND=ml_api` selects Tasks 4–5; `ML_API_URL` and `ML_API_TIMEOUT_SECONDS` configure the client. The worker's analysis step accepts either the existing `AnalysisResult` or Task 4's `VerificationResult`; checkpoint restoration discriminates by `result_type` and `schema_version`.

- [ ] **Step 1: Write failing wiring and recovery tests.** Assert `ml_api` mode injects `MlVerifier` and `VerificationResponseFormatter`, simulation stays explicitly selectable, a saved real result resumes at response formatting without a second ML call, and `analysis_outcome`/`responses.kind` no longer say simulation. Assert exhausted 503/timeout produces the existing error reply without an insufficient verdict.
- [ ] **Step 2: Verify red.** Run `.venv/bin/python -m pytest tests/unit/test_worker.py tests/integration/test_orchestrator_postgres.py -q`; expected: the new real-mode cases fail.
- [ ] **Step 3: Implement configuration and orchestration.** Reuse the worker's shared `httpx.AsyncClient`, align HTTP/analysis timeouts with its renewable lease, choose context/formatter by result type, restore versioned JSONB safely, and preserve simulated behavior. Require Gemini input settings in both modes; require Gemini response settings only for simulation.
- [ ] **Step 4: Verify green.** Run those files with PostgreSQL test service available; expected: pass. Run `.venv/bin/python -m ruff check app tests`; expected: no findings.
- [ ] **Step 5: Commit in the backend branch.** Stage modified files and tests; commit `feat: use ML verification in WhatsApp worker`.

### Task 7: Connect the stacks and prove the round trip

**Files:** Create `checa-ai/docker-compose.ml.yml` and `checa-ai/tests/integration/test_ml_api_pipeline.py`; modify `checa-ai/.env.example` and `checa-ai/README.md`.

**Interfaces:** The optional Compose override joins only the backend worker to Task 3's `checa-ai-ml-shared` network and sets `CLASSIFIER_BACKEND=ml_api`, `ML_API_URL=http://ml-api:8000`. The base backend Compose remains usable in simulation mode.

- [ ] **Step 1: Write failing integration tests.** With an HTTPX mock for `/api/v1/verify`, run webhook → persistence/outbox → Redis → worker → checkpoint → delivery outbox → Evolution mock for matched and insufficient results. Assert one WhatsApp send after a duplicate webhook, linked-article attribution, quoted-text selection, and no source leakage on abstention.
- [ ] **Step 2: Verify red.** Run `.venv/bin/python -m pytest tests/integration/test_ml_api_pipeline.py -q` with test PostgreSQL/Redis available; expected: missing override/real wiring or assertion failures.
- [ ] **Step 3: Add override and operating instructions.** Document ML Compose startup before the backend override, index/provider compatibility, URL configuration, readiness checks, and the fact that data ingestion is outside this feature. Keep the two repositories independently buildable.
- [ ] **Step 4: Verify green.** From `checa-ai`, run `.venv/bin/python -m ruff check app tests` and `.venv/bin/python -m pytest tests/unit tests/contract tests/integration -q` with the dedicated PostgreSQL/Redis test services; run `docker compose -f docker-compose.yml -f docker-compose.ml.yml config -q`. From `checa-ai-ml`, run `python -m pytest tests -q`. Perform one manual smoke test against an already indexed Qdrant collection: matched and insufficient, each within 1,000 characters.
- [ ] **Step 5: Commit in the backend branch.** Stage the four files and commit `test: verify WhatsApp to ML API round trip`.

## Final verification

- Check the exact API schema against the backend adapter's parser, including names, nullability, and enum values.
- Confirm both repositories have clean diffs and that no user-owned backend changes were overwritten.
- Report automated results and the limits of any manual smoke test. Do not claim the model's factual quality was validated by contract tests.
