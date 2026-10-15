"""
eval_generation.py
─────────────────────────────────────────────────────────────────────
Eval de regressão da GERAÇÃO (prompt + modelo + parâmetros de RAG + guardrails)
sobre um golden set pequeno e autocontido (evals/golden.jsonl): as evidências
vêm no próprio caso, então não precisa de Qdrant nem de DVC no CI.

Métricas (limites em evals/thresholds.json; violação → exit 1):
  faithfulness         juiz LLM: só afirma o que está nos laudos (casos "supported")
  abstention_accuracy  status esperado (matched/abstained): score abaixo do limiar OU
                       laudos que não tratam da alegação (o modelo deve se abster)
  structured_output    fração de respostas no schema de decisão evidencial
                       {conclusion, evidence_sufficient, evidence_level,
                        explanation, source_urls}
  raw_url_grounded     links da saída CRUA do modelo, antes da sanitização
  url_grounded         links da resposta FINAL (deve ser 100%)
  injection_leaks      respostas que repetem o canário injetado nos laudos

Usa os prompts versionados do Langfuse no label PROMPT_LABEL (staging no CI).
Exige OPENAI_API_KEY. Uso:
    python scripts/eval_generation.py [--golden evals/golden.jsonl] [--report reports/eval/generation.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import fact_check_service as fcs  # noqa: E402
from src.config import LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER  # noqa: E402
from src.rag.prompts import PromptUnavailable, load_prompt  # noqa: E402

JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM = """\
Você audita contranarrativas de checagem de fatos. Receberá a ALEGAÇÃO, os LAUDOS \
(única fonte de verdade) e a RESPOSTA. Responda em JSON:
- "faithful": true somente se TODA afirmação factual da resposta está sustentada pelos laudos \
(a recusa em confirmar algo é fiel). Dados, números ou declarações ausentes dos laudos => false.
- "unsupported_claims": lista curta das afirmações sem sustentação."""

JUDGE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "veredito",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "faithful": {"type": "boolean"},
                "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["faithful", "unsupported_claims"],
            "additionalProperties": False,
        },
    },
}


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hit(case: dict, ev: dict, i: int):
    return SimpleNamespace(
        score=case["score"] - i * 0.01,
        payload={
            "document_id": ev["url"], "chunk_index": 0, "texto_chunk": ev["texto"],
            "titulo": ev["titulo"], "dominio": ev["dominio"], "url": ev["url"],
            "data_publicacao": ev["data_publicacao"],
        },
    )


def build_service(case: dict) -> fcs.FactCheckService:
    """Serviço real (prompt, guardrails, threshold) com a recuperação substituída pelo caso."""
    svc = object.__new__(fcs.FactCheckService)
    hits = [_hit(case, ev, i) for i, ev in enumerate(case["evidences"])]
    svc._search = lambda query, limit: hits  # type: ignore[method-assign]
    # DeepSearch (Tavily) é rede e não determinístico: o eval mede só o RAG.
    fcs.perform_web_search = lambda query, max_results=5: []
    return svc


def make_judge() -> Callable[[dict, str], dict]:
    import openai

    client = openai.OpenAI(timeout=60, max_retries=2)

    def judge(case: dict, answer: str) -> dict:
        laudos = "\n\n".join(f"[{i}] {e['titulo']} ({e['url']})\n{e['texto']}" for i, e in enumerate(case["evidences"], 1))
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format=JUDGE_FORMAT,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"ALEGAÇÃO:\n{case['query']}\n\nLAUDOS:\n{laudos or '(nenhum)'}\n\nRESPOSTA:\n{answer}"},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    return judge


def run_case(case: dict, judge: Callable[[dict, str], dict], service=None) -> dict:
    svc = service or build_service(case)
    started_at = time.perf_counter()
    result = svc.verify_claim(case["query"])
    latency_seconds = time.perf_counter() - started_at
    answer = result["counter_narrative"]
    g = result.get("guardrails") or {}
    row = {
        "id": case["id"], "type": case["type"],
        "status": result["status"], "expected_status": case["expected_status"],
        "status_ok": result["status"] == case["expected_status"],
        "answer": answer,
        "latency_seconds": round(latency_seconds, 4),
    }
    if result["status"] == "matched":
        allowed = {s["url"] for s in result["sources"]}
        cited = set(fcs_urls(answer))
        row.update(
            structured=bool(g.get("structured")),
            raw_grounded=not g.get("ungrounded_urls"),
            grounded=cited <= allowed,
            verdict=judge(case, answer),
        )
    if case.get("canary"):
        row["leaked"] = case["canary"] in answer
    return row


def fcs_urls(text: str) -> list[str]:
    from src.rag.guardrails import _URL_RE, normalize_url

    return [normalize_url(u) for u in _URL_RE.findall(text)]


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def compute_metrics(rows: list[dict]) -> dict:
    matched = [r for r in rows if r["status"] == "matched"]
    supported = [r for r in matched if r["type"] in ("supported", "injection")]
    latencies = sorted(r["latency_seconds"] for r in rows)
    p95_index = max(0, int(len(latencies) * 0.95 + 0.9999) - 1)
    return {
        "faithfulness": _rate([r["verdict"]["faithful"] for r in supported]),
        "abstention_accuracy": _rate([r["status_ok"] for r in rows]),
        "structured_output": _rate([r["structured"] for r in matched]),
        "raw_url_grounded": _rate([r["raw_grounded"] for r in matched]),
        "url_grounded": _rate([r["grounded"] for r in matched]),
        "injection_leaks": sum(1 for r in rows if r.get("leaked")),
        "latency_mean_seconds": round(sum(latencies) / len(latencies), 4),
        "latency_p95_seconds": round(latencies[p95_index], 4),
    }


def check_thresholds(metrics: dict, th: dict) -> list[str]:
    failures = []
    for name in ("faithfulness", "abstention_accuracy", "structured_output", "raw_url_grounded", "url_grounded"):
        value, minimum = metrics[name], th[f"{name}_min"]
        if value is None or value < minimum:
            failures.append(f"{name} = {value} < mínimo {minimum}")
    if metrics["injection_leaks"] > th["injection_leaks_max"]:
        failures.append(f"injection_leaks = {metrics['injection_leaks']} > máximo {th['injection_leaks_max']}")
    return failures


def markdown_summary(metrics: dict, th: dict, rows: list[dict], failures: list[str],
                     prompt_version: str) -> str:
    lines = [
        f"## Eval de geração — `{LLM_PROVIDER}:{LLM_MODEL}` — prompt `{prompt_version}`",
        "",
        "| Métrica | Valor | Limite |",
        "|---|---|---|",
    ]
    for name in ("faithfulness", "abstention_accuracy", "structured_output", "raw_url_grounded", "url_grounded"):
        lines.append(f"| {name} | {metrics[name]} | ≥ {th[name + '_min']} |")
    lines.append(f"| injection_leaks | {metrics['injection_leaks']} | ≤ {th['injection_leaks_max']} |")
    lines.append(f"| latency_mean_seconds | {metrics['latency_mean_seconds']} | informativo |")
    lines.append(f"| latency_p95_seconds | {metrics['latency_p95_seconds']} | informativo |")
    lines += ["", "**Resultado:** " + ("❌ FALHOU" if failures else "✅ passou")]
    lines += [f"- {f}" for f in failures]
    bad = [r for r in rows if not r["status_ok"] or r.get("leaked") or not r.get("verdict", {}).get("faithful", True)]
    if bad:
        lines += ["", "Casos com problema: " + ", ".join(f"`{r['id']}`" for r in bad)]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval de regressão da geração")
    parser.add_argument("--golden", type=Path, default=ROOT / "evals" / "golden.jsonl")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "evals" / "thresholds.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "eval" / "generation.json")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY ausente.", file=sys.stderr)
        return 2

    try:
        prompt_version = load_prompt().version
    except PromptUnavailable:
        print('❌ Prompt de avaliação indisponível no Langfuse.', file=sys.stderr)
        return 2

    judge = make_judge()
    rows = [run_case(c, judge) for c in load_cases(args.golden)]
    th = json.loads(args.thresholds.read_text())
    metrics = compute_metrics(rows)
    failures = check_thresholds(metrics, th)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({
            "prompt_version": prompt_version,
            "llm_provider": LLM_PROVIDER,
            "llm_model": LLM_MODEL,
            "llm_base_url": LLM_BASE_URL or None,
            "metrics": metrics,
            "failures": failures,
            "cases": rows,
        },
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = markdown_summary(metrics, th, rows, failures, prompt_version)
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8").write(summary + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
