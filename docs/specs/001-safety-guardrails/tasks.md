# Tasks: Safety Guardrails & Moderation Layer

## Phase 1: Setup & Foundational
- [ ] T001 Create directory structure `src/safety/` and `__init__.py`
- [ ] T002 [P] Implement OpenAI Moderation wrapper in `src/safety/moderation.py`
- [ ] T003 [P] Implement prompt injection detection logic in `src/safety/injection.py`
- [ ] T004 [P] Implement output validation logic in `src/safety/validator.py`

## Phase 2: Input Guardrails [US1: Toxic Input Blocking]
- [ ] T005 Integrate `moderation.py` check at the start of `FactCheckService.verify_claim` in `src/fact_check_service.py`
- [ ] T006 Implement the "Safe-Refusal" return path in `FactCheckService.verify_claim` to bypass RAG when toxic
- [ ] T007 Add unit tests for toxic input blocking in `tests/test_safety_input.py`

## Phase 3: Injection Guardrails [US2: Prompt Injection Mitigation]
- [ ] T008 Integrate `injection.py` check after moderation in `src/fact_check_service.py`
- [ ] T009 Define standardized refusal message for jailbreak attempts in `src/config.py`
- [ ] T010 Add unit tests for prompt injection detection in `tests/test_safety_injection.py`

## Phase 4: Output Guardrails [US3: Output Validation]
- [ ] T011 Integrate `validator.py` scan after LLM generation in `src/fact_check_service.py`
- [ ] T012 Implement fallback logic if generated counter-narrative is flagged as unsafe in `src/fact_check_service.py`
- [ ] T013 Add unit tests for output safety validation in `tests/test_safety_output.py`

## Phase 5: Observability & Polish
- [ ] T014 Implement logging for all blocked events in `src/fact_check_service.py`
- [ ] T015 Add safety-related error types in `src/config.py` or a new `src/safety/exceptions.py`
- [ ] T016 Perform end-to-end integration test with a set of safe, toxic, and injection prompts in `tests/test_safety_e2e.py`

## Dependencies
- Phase 1 $\rightarrow$ Phase 2 $\rightarrow$ Phase 3 $\rightarrow$ Phase 4 $\rightarrow$ Phase 5

## Implementation Strategy
MVP Scope: Phase 1 + Phase 2 (Input Moderation). This provides the most immediate security value.
