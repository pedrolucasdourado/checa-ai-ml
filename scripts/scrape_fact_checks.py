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
from collections import Counter
from datetime import UTC, datetime
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
MAX_RETRIES = 3               # tentativas em caso de erro transitório

# Headers realistas para evitar bloqueios simples de 403. UOL tende a
# rejeitar clients muito "requests puro", então tentamos alguns perfis.
BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "DNT": "1",
    "Pragma": "no-cache",
    "Upgrade-Insecure-Requests": "1",
}

BROWSER_PROFILES = [
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Sec-CH-UA": '"Google Chrome";v="129", "Chromium";v="129", "Not=A?Brand";v="24"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.6 Safari/605.1.15"
        ),
        "Referer": "https://www.google.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
    },
    {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/129.0.0.0 Safari/537.36"
        ),
        "Referer": "https://noticias.uol.com.br/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
    },
]

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


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def build_headers(url: str, attempt: int) -> dict[str, str]:
    """Monta headers de navegador variando perfil e referer por tentativa."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    profile = BROWSER_PROFILES[(attempt - 1) % len(BROWSER_PROFILES)]
    headers = {**BASE_HEADERS, **profile}

    if domain == "noticias.uol.com.br":
        headers.setdefault("Referer", "https://www.google.com/")
        headers["Origin"] = "https://noticias.uol.com.br"

    return headers


def make_session() -> requests.Session:
    """Cria sessão HTTP; usa cloudscraper se estiver disponível."""
    try:
        import cloudscraper  # type: ignore

        log.info("cloudscraper disponível — usando sessão compatível com Cloudflare")
        return cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "desktop": True}
        )
    except ImportError:
        return requests.Session()


def fetch_html(url: str, session: requests.Session) -> tuple[str | None, int | None, str | None]:
    """Baixa o HTML da URL com retries, timeout e headers browser-like."""
    last_status = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=build_headers(url, attempt), timeout=REQUEST_TIMEOUT)
            last_status = resp.status_code
            resp.raise_for_status()
            return resp.text, resp.status_code, None
        except requests.exceptions.HTTPError as e:
            last_status = e.response.status_code if e.response is not None else None
            last_error = f"http_{last_status}"
            log.warning("HTTP %s -> %s (tentativa %d/%d)", last_status, url, attempt, MAX_RETRIES)
            if last_status in (410, 404):
                return None, last_status, last_error
        except requests.exceptions.Timeout:
            last_error = "timeout"
            log.warning("Timeout -> %s (tentativa %d/%d)", url, attempt, MAX_RETRIES)
        except requests.exceptions.RequestException as e:
            last_error = type(e).__name__
            log.warning("Erro de rede -> %s | %s (tentativa %d/%d)", url, e, attempt, MAX_RETRIES)

        if attempt < MAX_RETRIES:
            time.sleep(SLEEP_MIN * attempt)
    return None, last_status, last_error


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
        "scraped_at": utc_now_iso(),
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
def row_to_item(row) -> dict:
    return {
        "url": str(row["URL"]),
        "dominio": row["dominio"],
        "titulo_original": str(row.get("Titulo", "")),
        "classe": int(row["Classe"]),
    }


def load_dataset_urls(
    batch: int | None,
    domains: list[str] | None = None,
    shuffle: bool = True,
) -> list[dict]:
    """Carrega URLs de Classe == 0, opcionalmente filtradas por domínio."""
    df = pd.read_excel(EXCEL_PATH)
    fake_df = df[df["Classe"] == 0].copy()
    fake_df["dominio"] = fake_df["URL"].apply(extract_domain)

    selected_df = fake_df[fake_df["dominio"].isin(domains)].copy() if domains else fake_df

    if shuffle:
        # Embaralha para variar as amostras entre agências.
        selected_df = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)

    limited_df = selected_df.head(batch) if batch else selected_df
    urls = [row_to_item(row) for _, row in limited_df.iterrows()]

    log.info("URLs carregadas: %d (de %d disponíveis)", len(urls), len(selected_df))
    _counts = limited_df["dominio"].value_counts().to_dict()
    log.info("Distribuição por domínio: %s", _counts)
    return urls


def load_priority_urls(batch: int | None) -> list[dict]:
    """Carrega URLs de Classe == 0 para as agências prioritárias."""
    return load_dataset_urls(batch=batch, domains=PRIORITY_DOMAINS)


def load_urls_from_errors(path: Path, batch: int | None = None) -> list[dict]:
    """Carrega URLs de um JSONL de erros para retentativa."""
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            urls.append({
                "url": item["url"],
                "dominio": item.get("dominio") or extract_domain(item["url"]),
                "titulo_original": item.get("titulo_original", ""),
                "classe": int(item.get("classe", 0)),
            })

    if batch:
        urls = urls[:batch]

    log.info("URLs carregadas de erros: %d", len(urls))
    return urls


def load_existing_urls(path: Path) -> set[str]:
    """Lê URLs já salvas para evitar duplicação em modo append."""
    if not path.exists():
        return set()

    urls = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("url"):
                urls.add(rec["url"])
    return urls


def run(
    batch: int | None,
    output: Path,
    errors_output: Path,
    from_errors: Path | None = None,
    append: bool = False,
    all_domains: bool = False,
    domains: list[str] | None = None,
    sleep_min: float = SLEEP_MIN,
    sleep_max: float = SLEEP_MAX,
) -> None:
    if from_errors:
        urls = load_urls_from_errors(from_errors, batch=batch)
    elif all_domains:
        urls = load_dataset_urls(batch=batch, domains=domains)
    else:
        urls = load_dataset_urls(batch=batch, domains=domains or PRIORITY_DOMAINS)

    if append:
        existing_urls = load_existing_urls(output)
        before = len(urls)
        urls = [item for item in urls if item["url"] not in existing_urls]
        log.info("Modo append: %d URLs ja presentes ignoradas", before - len(urls))

    output.parent.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    session = make_session()
    output_mode = "a" if append else "w"
    success_by_domain: Counter[str] = Counter()
    failed_by_domain: Counter[str] = Counter()
    last_success: dict | None = None

    with open(output, output_mode, encoding="utf-8") as f_out, \
         open(errors_output, "w", encoding="utf-8") as f_err:

        for item in tqdm(urls, desc="Scraping", unit="url", colour="cyan"):
            url = item["url"]
            log.debug("Processando → %s", url)

            html, http_status, fetch_error = fetch_html(url, session)

            if html is None:
                failed += 1
                failed_by_domain[item["dominio"]] += 1
                error_record = {
                    **item,
                    "erro": "fetch_failed",
                    "http_status": http_status,
                    "fetch_error": fetch_error,
                    "scraped_at": utc_now_iso(),
                }
                f_err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                time.sleep(sleep_min)
                continue

            article = extract_article(html, url)

            if article is None:
                failed += 1
                failed_by_domain[item["dominio"]] += 1
                error_record = {**item, "erro": "extraction_failed", "scraped_at": utc_now_iso()}
                f_err.write(json.dumps(error_record, ensure_ascii=False) + "\n")
            else:
                # Enriquece com metadados originais da planilha
                article["titulo_original_planilha"] = item["titulo_original"]
                f_out.write(json.dumps(article, ensure_ascii=False) + "\n")
                success += 1
                success_by_domain[item["dominio"]] += 1
                last_success = article

            # Pausa respeitosa entre requests
            time.sleep(random.uniform(sleep_min, sleep_max))

    # ── Resumo final ─────────────────────────────────────────────────
    total = success + failed
    log.info("-" * 60)
    log.info("Concluído! Processadas: %d URLs", total)
    log.info("  Sucesso:  %d (%.1f%%)", success, success / total * 100 if total else 0)
    log.info("  Falhas:   %d (%.1f%%)", failed, failed / total * 100 if total else 0)
    log.info("  Saida:    %s", output)
    log.info("  Sucesso por domínio: %s", dict(success_by_domain))
    if failed_by_domain:
        log.info("  Falhas por domínio: %s", dict(failed_by_domain))

    # ── Exibe 1 exemplo completo extraído ────────────────────────────
    if last_success:
        first = last_success
        print("\n" + "=" * 70)
        print("EXEMPLO EXTRAIDO - PROVA DE CONCEITO")
        print("=" * 70)
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
            print(f"\n[... +{len(words) - 800} palavras omitidas na visualizacao ...]")
        print("=" * 70)


# ─────────────────────────── Entry point ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper de fact-checks do FakeRecogna")
    parser.add_argument("--batch", type=int, default=50, help="Número de URLs a processar (padrão: 50). Use 0 com --all para todas.")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSONL, help="Caminho do arquivo JSONL de saída")
    parser.add_argument("--from-errors", type=Path, help="JSONL de erros para retentativa")
    parser.add_argument("--append", action="store_true", help="Anexa ao arquivo de saída sem duplicar URLs")
    parser.add_argument("--all", action="store_true", help="Processa todos os domínios de Classe == 0")
    parser.add_argument("--sleep-min", type=float, default=SLEEP_MIN, help="Pausa mínima entre requests")
    parser.add_argument("--sleep-max", type=float, default=SLEEP_MAX, help="Pausa máxima entre requests")
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Filtra domínios específicos, ex: --domains boatos.org e-farsas.com",
    )
    args = parser.parse_args()

    run(
        batch=args.batch or None,
        output=args.output,
        errors_output=args.output.parent / "fact_checks_errors.jsonl",
        from_errors=args.from_errors,
        append=args.append,
        all_domains=args.all,
        domains=args.domains,
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
    )
