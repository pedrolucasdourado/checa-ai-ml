"""
curate_silver_clusters.py
─────────────────────────────────────────────────────────────────────
Gera uma camada de curadoria para os clusters Silver.

Entrada:
    reports/silver_clusters/silver_cluster_summary.json

Saidas:
    reports/silver_clusters/silver_cluster_curation.json
    reports/silver_clusters/silver_cluster_curation.html

O objetivo e transformar clusters numericos (C0, C1...) em objetos
revisaveis: label sugerido, tipo, confianca, problema provavel e
recomendacao de proxima acao.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path("reports") / "silver_clusters" / "silver_cluster_summary.json"
DEFAULT_OUTPUT_DIR = Path("reports") / "silver_clusters"


def as_dict(items: list[list[Any]] | list[tuple[Any, Any]]) -> dict[str, int]:
    return {str(key): int(value) for key, value in items}


def keys(items: dict[str, int]) -> set[str]:
    return {key.casefold() for key in items}


def has_any(values: set[str], *needles: str) -> bool:
    return any(needle.casefold() in values for needle in needles)


def contains_any(values: set[str], *needles: str) -> bool:
    return any(any(needle.casefold() in value for value in values) for needle in needles)


def slug_label(label: str) -> str:
    normalized = label.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


def dominant_share(counts: dict[str, int], total: int) -> float:
    if not counts or total <= 0:
        return 0.0
    return max(counts.values()) / total


def count_any(counts: dict[str, int], *needles: str) -> int:
    total = 0
    for key, value in counts.items():
        lowered = key.casefold()
        if any(needle.casefold() in lowered for needle in needles):
            total += value
    return total


def suggest_label(cluster: dict[str, Any]) -> tuple[str, str]:
    topics = as_dict(cluster.get("top_topics", []))
    domains = as_dict(cluster.get("top_domains", []))
    keyword_counts = as_dict(cluster.get("top_keywords", []))
    term_counts = as_dict(cluster.get("top_title_terms", []))
    keywords = keys(keyword_counts)
    terms = keys(term_counts)
    signals = keywords | terms

    top_domain = max(domains, key=domains.get) if domains else ""
    if top_domain == "g1.globo.com" and dominant_share(domains, cluster["document_count"]) >= 0.75:
        return "fato_ou_fake_g1_imagens_videos", "fonte_formato"

    video_terms = count_any(term_counts, "video", "vídeo", "foto", "mostra", "imagem")
    if video_terms >= 120:
        return "imagens_videos_fora_de_contexto", "formato"

    lula_terms = count_any(term_counts, "lula", "pt")
    if lula_terms >= 80:
        return "politica_lula_pt", "tema"

    bolsonaro_terms = count_any(term_counts, "bolsonaro", "jair")
    if bolsonaro_terms >= 120:
        return "politica_bolsonaro_governo", "tema"

    if has_any(signals, "whatsapp", "golpe", "link") and has_any(signals, "gratis", "grátis", "site", "cadastro", "premio"):
        return "golpes_whatsapp_links_cadastros", "tema"

    election_terms = count_any(term_counts, "urna", "urnas", "voto", "votos", "fraude", "eleição", "eleições", "eleicao", "eleicoes")
    if election_terms >= 120:
        return "urnas_eleicoes_fraude", "tema"

    justice_terms = count_any(term_counts, "stf", "justica", "justiça", "policia", "polícia", "crime", "lei", "moro")
    if justice_terms >= 60 or (topics.get("seguranca_justica", 0) >= 150 and topics.get("politica_eleicoes", 0) >= 150):
        return "justica_stf_crimes_politica", "misto"

    vaccine_terms = count_any(term_counts, "vacina", "vacinas", "coronavac", "imunizacao", "imunização")
    if vaccine_terms >= 20 and count_any(term_counts, "vacina", "vacinas", "coronavac") >= 20:
        return "vacinas_covid_imunizacao", "tema"

    treatment_terms = count_any(term_counts, "cloroquina", "hidroxicloroquina", "ivermectina", "tratamento")
    if treatment_terms >= 40:
        return "tratamento_precoce_cloroquina_covid", "tema"

    if contains_any(signals, "covid", "coronavirus", "coronavírus", "pandemia", "hospital", "mortes"):
        return "covid_pandemia_saude_publica", "tema"

    if contains_any(signals, "bolsonaro", "governo", "presidente"):
        return "politica_bolsonaro_governo", "tema"

    if contains_any(signals, "video", "vídeo", "foto", "mostra", "imagem"):
        return "imagens_videos_fora_de_contexto", "formato"

    top_topic = max(topics, key=topics.get) if topics else "outros"
    return f"cluster_{top_topic}", "misto"


def confidence_for(cluster: dict[str, Any], cluster_type: str) -> str:
    total = int(cluster.get("document_count") or 0)
    topic_share = dominant_share(as_dict(cluster.get("top_topics", [])), total)
    domain_share = dominant_share(as_dict(cluster.get("top_domains", [])), total)

    if cluster_type == "fonte_formato":
        return "alta" if domain_share >= 0.75 else "media"
    if topic_share >= 0.80:
        return "alta"
    if topic_share >= 0.60:
        return "media"
    return "baixa"


def issue_for(cluster: dict[str, Any], cluster_type: str, confidence: str) -> str:
    total = int(cluster.get("document_count") or 0)
    domain_share = dominant_share(as_dict(cluster.get("top_domains", [])), total)
    topic_share = dominant_share(as_dict(cluster.get("top_topics", [])), total)

    if cluster_type == "fonte_formato":
        return "Cluster parece dominado por fonte/formato, nao apenas por tema."
    if confidence == "baixa":
        return "Cluster mistura temas demais; revisar antes de gravar no Qdrant."
    if topic_share < 0.70:
        return "Tema principal existe, mas ha subtemas misturados."
    if domain_share >= 0.70:
        return "Tema coerente, mas com possivel vies de fonte."
    return "Tema parece coerente para primeira versao."


def recommendation_for(cluster: dict[str, Any], cluster_type: str, confidence: str) -> str:
    if cluster_type == "fonte_formato":
        return "Nao usar como tema final; separar campos de fonte/formato dos campos tematicos."
    if confidence == "baixa":
        return "Rodar subclusterizacao interna ou dividir por topico heuristico antes de publicar."
    if cluster["document_count"] >= 600:
        return "Cluster grande; considerar subclusters para melhorar agentes especialistas."
    return "Pode virar candidato a cluster_label apos revisao dos exemplos."


def curate_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    label, cluster_type = suggest_label(cluster)
    confidence = confidence_for(cluster, cluster_type)
    return {
        "cluster_id": cluster["cluster_id"],
        "suggested_label": label,
        "suggested_slug": slug_label(label),
        "cluster_type": cluster_type,
        "confidence": confidence,
        "document_count": cluster["document_count"],
        "issue": issue_for(cluster, cluster_type, confidence),
        "recommendation": recommendation_for(cluster, cluster_type, confidence),
        "top_topics": cluster.get("top_topics", []),
        "top_domains": cluster.get("top_domains", []),
        "top_keywords": cluster.get("top_keywords", []),
        "top_title_terms": cluster.get("top_title_terms", []),
        "examples": cluster.get("examples", []),
    }


def build_curation(summary: dict[str, Any]) -> dict[str, Any]:
    clusters = [curate_cluster(cluster) for cluster in summary.get("clusters", [])]
    return {
        "metadata": {
            **summary.get("metadata", {}),
            "curation_version": "silver-cluster-curation-v1",
            "status": "draft_for_human_review",
        },
        "clusters": clusters,
    }


def fmt_pairs(items: list[list[Any]] | list[tuple[Any, Any]], limit: int = 8) -> str:
    return ", ".join(f"{html.escape(str(key))} ({value})" for key, value in items[:limit])


def save_html(path: Path, curation: dict[str, Any]) -> None:
    cards = []
    for cluster in curation["clusters"]:
        examples = "".join(
            (
                f'<li><a href="{html.escape(example.get("url", ""))}">'
                f'{html.escape(example.get("title", ""))}</a></li>'
            )
            for example in cluster["examples"][:8]
        )
        cards.append(
            f"""
      <article class="card confidence-{html.escape(cluster['confidence'])}">
        <header>
          <span class="id">C{cluster['cluster_id']}</span>
          <h2>{html.escape(cluster['suggested_label'])}</h2>
          <span class="pill">{html.escape(cluster['cluster_type'])}</span>
          <span class="pill">{html.escape(cluster['confidence'])}</span>
          <span class="count">{cluster['document_count']} docs</span>
        </header>
        <p class="issue">{html.escape(cluster['issue'])}</p>
        <p class="recommendation">{html.escape(cluster['recommendation'])}</p>
        <dl>
          <dt>Tópicos</dt><dd>{fmt_pairs(cluster['top_topics'], 6)}</dd>
          <dt>Palavras</dt><dd>{fmt_pairs(cluster['top_keywords'], 8)}</dd>
          <dt>Termos nos títulos</dt><dd>{fmt_pairs(cluster['top_title_terms'], 8)}</dd>
          <dt>Fontes</dt><dd>{fmt_pairs(cluster['top_domains'], 6)}</dd>
        </dl>
        <h3>Exemplos</h3>
        <ol>{examples}</ol>
      </article>
"""
        )

    metadata = curation["metadata"]
    content = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Curadoria de clusters Silver</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .intro {{ color: #4b5563; margin: 0 0 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    .card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 8px; padding: 16px; }}
    .card header {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
    .id {{ font-weight: 700; color: #374151; }}
    h2 {{ font-size: 18px; margin: 0; flex-basis: 100%; }}
    h3 {{ font-size: 14px; margin: 12px 0 6px; }}
    .pill {{ font-size: 12px; background: #eef2ff; color: #3730a3; padding: 3px 7px; border-radius: 999px; }}
    .count {{ font-size: 12px; color: #4b5563; }}
    .issue {{ color: #991b1b; margin: 8px 0; }}
    .recommendation {{ color: #065f46; margin: 8px 0 12px; }}
    dl {{ margin: 0; }}
    dt {{ font-weight: 700; margin-top: 8px; }}
    dd {{ margin: 2px 0 0; color: #374151; }}
    ol {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 4px 0; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <main>
    <h1>Curadoria de clusters Silver</h1>
    <p class="intro">{metadata.get('documents')} documentos, {metadata.get('clusters')} clusters. Rótulos automáticos em versão rascunho para revisão humana.</p>
    <section class="grid">
      {''.join(cards)}
    </section>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary = json.loads(args.input.read_text(encoding="utf-8"))
    curation = build_curation(summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "silver_cluster_curation.json"
    html_path = args.output_dir / "silver_cluster_curation.html"
    json_path.write_text(json.dumps(curation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_html(html_path, curation)

    print(f"Clusters: {len(curation['clusters'])}")
    for cluster in curation["clusters"]:
        print(
            f"C{cluster['cluster_id']}: {cluster['suggested_label']} | "
            f"{cluster['cluster_type']} | {cluster['confidence']} | "
            f"{cluster['document_count']} docs"
        )
    print("\nSaved:")
    print(f"  json: {json_path}")
    print(f"  html: {html_path}")
    return curation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera relatorio de curadoria dos clusters Silver")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    run(parser.parse_args())
