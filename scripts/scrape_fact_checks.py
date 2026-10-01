"""
scrape_fact_checks.py
─────────────────────
Prova de conceito: extrai o texto jornalístico real (laudo de desmentido)
das URLs de Classe == 0 do FakeRecogna, priorizando as agências com
matérias estruturadas (UOL Comprova, G1 Fato ou Fake, Projeto Comprova).

Saída: data/processed/fact_checks_clean.jsonl
       data/processed/fact_checks_errors.jsonl  (URLs que falharam)

Uso:
    python scripts/scrape_fact_checks.py [--batch 50] [--output data/processed/fact_checks_clean.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import trafilatura
from tqdm import tqdm

# ─────────────────────────── Configurações ───────────────────────────
EXCEL_PATH = Path("data/raw/FakeRecogna.xlsx")
OUTPUT_JSONL = Path("data/processed/fact_checks_clean.jsonl")
ERRORS_JSONL = Path("data/processed/fact_checks_errors.jsonl")

# Agências prioritárias (melhor estrutura de laudo jornalístico)
PRIORITY_DOMAINS = [
    "noticias.uol.com.br",  # UOL Confere / Comprova
    "g1.globo.com",          # G1 Fato ou Fake
    "projetocomprova.com.br",
]

# Parâmetros HTTP
REQUEST_TIMEOUT = 15          # segundos por requisição
SLEEP_MIN = 1.0               # pausa mínima entre requisições (segundos)
SLEEP_MAX = 2.5               # pausa máxima (respeita robots.txt e rate-limit)
MAX_RETRIES = 2               # tentativas em caso de erro transitório

# Headers realistas para evitar bloqueio 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────── Funções auxiliares ───────────────────────────
def extract_domain(url: str) -> str:
    try:
        return urlparse(str(url)).netloc.replace("www.", "")
    except Exception:
        return ""


def fetch_html(url: str, session: requests.Session) -> str | None:
    """Baixa o HTML da URL com retries e timeout."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.HTTPError as e:
            log.warning("HTTP %s → %s (tentativa %d/%d)", e.response.status_code, url, attempt, MAX_RETRIES)
            if e.response.status_code in (403, 410, 404):
                return None          # sem retry para erros permanentes
        except requests.exceptions.Timeout:
            log.warning("Timeout → %s (tentativa %d/%d)", url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            log.warning("Erro de rede → %s | %s (tentativa %d/%d)", url, e, attempt, MAX_RETRIES)

        if attempt < MAX_RETRIES:
            time.sleep(SLEEP_MIN * attempt)
    return None


def extract_article(html: str, url: str) -> dict | None:
    """
    Usa trafilatura com fallback manual via BeautifulSoup para extrair
    título + corpo limpo do artigo.
    """
    # ── Trafilatura (extrator principal) ──────────────────────────────
    metadata = trafilatura.extract_metadata(html, default_url=url)
    body = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,   # prioriza precisão → menos ruído
        no_fallback=False,
    )

    if not body or len(body.split()) < 30:
        # ── Fallback: BeautifulSoup ────────────────────────────────────
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")

        # Remove scripts/estilos
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Tenta encontrar o container principal
        article = (
            soup.find("article")
            or soup.find("div", class_=lambda c: c and any(k in c for k in ["content", "body", "texto", "article"]))
            or soup.find("main")
        )
        body = article.get_text(separator="\n", strip=True) if article else None

    if not body or len(body.split()) < 30:
        return None

    title = (
        (metadata.title if metadata and metadata.title else None)
        or _extract_title_from_html(html)
        or ""
    )

    pub_date = (
        str(metadata.date) if metadata and metadata.date else None
    )

    return {
        "titulo": title.strip(),
        "texto": body.strip(),
        "url": url,
        "dominio": extract_domain(url),
        "data_publicacao": pub_date,
        "scraped_at": datetime.utcnow().isoformat() + "Z",
        "palavras": len(body.split()),
    }


def _extract_title_from_html(html: str) -> str | None:
    """Extrai <title> via BeautifulSoup como último recurso."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else None
    except Exception:
        return None


# ─────────────────────────── Pipeline principal ───────────────────────────
def load_priority_urls(batch: int) -> list[dict]:
    """Carrega e prioriza URLs de Classe == 0 para as agências selecionadas."""
    df = pd.read_excel(EXCEL_PATH)
    fake_df = df[df["Classe"] == 0].copy()
    fake_df["dominio"] = fake_df["URL"].apply(extract_domain)

    priority_df = fake_df[fake_df["dominio"].isin(PRIORITY_DOMAINS)].copy()

    # Embaralha para variar as amostras entre agências
    priority_df = priority_df.sample(frac=1, random_state=42).reset_index(drop=True)

    urls = []
    for _, row in priority_df.head(batch).iterrows():
        urls.append({
            "url": str(row["URL"]),
            "dominio": row["dominio"],
            "titulo_original": str(row.get("Titulo", "")),
            "classe": int(row["Classe"]),
        })

    log.info("URLs carregadas: %d (de %d disponíveis)", len(urls), len(priority_df))
    _counts = priority_df.head(batch)["dominio"].value_counts().to_dict()
    log.info("Distribuição por agência: %s", _counts)
    return urls


def run(batch: int, output: Path, errors_output: Path) -> None:
    urls = load_priority_urls(batch)

    output.parent.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    session = requests.Session()

    with open(output, "w", encoding="utf-8") as f_out, \
         open(errors_output, "w", encoding="utf-8") as f_err:

        for item in tqdm(urls, desc="Scraping", unit="url", colour="cyan"):
            url = item["url"]
            log.debug("Processando → %s", url)

            html = fetch_html(url, session)

            if html is None:
                failed += 1
                error_record = {**item, "erro": "fetch_failed", "scraped_at": datetime.utcnow().isoformat() + "Z"}
                f_err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                time.sleep(SLEEP_MIN)
                continue

            article = extract_article(html, url)

            if article is None:
                failed += 1
                error_record = {**item, "erro": "extraction_failed", "scraped_at": datetime.utcnow().isoformat() + "Z"}
                f_err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            else:
                # Enriquece com metadados originais da planilha
                article["titulo_original_planilha"] = item["titulo_original"]
                f_out.write(json.dumps(article, ensure_ascii=False) + "\n")
                success += 1

            # Pausa respeitosa entre requests
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

    # ── Resumo final ─────────────────────────────────────────────────
    total = success + failed
    log.info("─" * 60)
    log.info("Concluído! Processadas: %d URLs", total)
    log.info("  ✅ Sucesso:  %d (%.1f%%)", success, success / total * 100 if total else 0)
    log.info("  ❌ Falhas:   %d (%.1f%%)", failed, failed / total * 100 if total else 0)
    log.info("  📄 Saída:   %s", output)

    # ── Exibe 1 exemplo completo extraído ────────────────────────────
    if success > 0:
        with open(output, encoding="utf-8") as f:
            first = json.loads(f.readline())
        print("\n" + "═" * 70)
        print("EXEMPLO EXTRAÍDO — PROVA DE CONCEITO")
        print("═" * 70)
        print(f"Domínio:     {first['dominio']}")
        print(f"URL:         {first['url']}")
        print(f"Data:        {first['data_publicacao']}")
        print(f"Título:      {first['titulo']}")
        print(f"Palavras:    {first['palavras']}")
        print("\nTexto extraído (primeiras 800 palavras):\n")
        words = first["texto"].split()
        preview = " ".join(words[:800])
        print(preview)
        if len(words) > 800:
            print(f"\n[... +{len(words) - 800} palavras omitidas na visualização ...]")
        print("═" * 70)


# ─────────────────────────── Entry point ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper de fact-checks do FakeRecogna")
    parser.add_argument("--batch", type=int, default=50, help="Número de URLs a processar (padrão: 50)")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL, help="Caminho do arquivo JSONL de saída")
    args = parser.parse_args()

    run(
        batch=args.batch,
        output=args.output,
        errors_output=args.output.parent / "fact_checks_errors.jsonl",
    )
