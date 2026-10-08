# Implementation Plan: Safety Guardrails & Moderation Layer

## 1. Tech Stack & Libraries
- **Language:** Python 3.x
- **Framework:** FastAPI
- **Moderation:** OpenAI Moderation API (via `openai` python client)
- **Logging:** Standard Python `logging` module (or integration with Langfuse for observability)
- **Env Vars:** `OPENAI_API_KEY`

## 2. Project Structure
The guardrails will be implemented as a wrapper service to ensure a clean separation between safety checks and business logic.

- `src/safety/` (New directory)
    - `__init__.py`
    - `moderation.py`: Interface for the OpenAI Moderation API.
    - `injection.py`: Logic for detecting prompt injection/jailbreaks.
    - `validator.py`: Logic for validating LLM output.
- `src/fact_check_service.py`: Integration of the safety layer into the RAG pipeline.
- `app.py`: Integration of the safety layer in the endpoint.

## 3. Implementation Strategy

### Phase 1: Input Moderation (The "Shield")
1. Create `src/safety/moderation.py` to wrap the OpenAI Moderation API.
2. Implement a function that returns a boolean `is_safe` and the category of toxicity.
3. Integrate this call at the very beginning of `FactCheckService.verify_claim`.

### Phase 2: Prompt Injection Detection
1. Create `src/safety/injection.py`.
2. Implement a heuristic-based detector (regex/keyword list) to identify common jailbreak patterns (e.g., "ignore previous instructions").
3. Add this check immediately after the toxicity check.

### Phase 3: Output Validation
1. Create `src/safety/validator.py`.
2. Implement a post-generation scan of the `counter_narrative` using the same Moderation API.
3. Ensure the response is safe before returning it in `DebunkResponse`.

### Phase 4: Observability & Integration
1. Update the logging system to record "Blocked" events with the reason and user input.
2. Refine the "Safe-Refusal" messages to be user-friendly.
3. End-to-end testing with a dataset of toxic and safe inputs.
