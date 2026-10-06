# Data Curation & Scraping Workflow (Checa-AI)

Este documento descreve a metodologia e o fluxo técnico utilizado para criar a base de dados vetorial de alta qualidade que alimenta o sistema RAG (Retrieval-Augmented Generation) do Checa-AI.

## 1. O Problema da Base de Dados Original
O projeto utilizou inicialmente a base de dados acadêmica **FakeRecogna** (`FakeRecogna.xlsx`). Durante a auditoria inicial, identificamos duas limitações críticas para o uso dessa base em uma arquitetura RAG:

1. **Contexto Truncado:** A coluna que armazenava o texto original das notícias no dataset passou por um processo rigoroso de pré-processamento focado em treinamento de classificadores tradicionais (Machine Learning clássico). Foi aplicada remoção de *stop words*, *stemming* e *lematização*. Isso destruiu a sintaxe e a legibilidade do texto, tornando-o inviável para ser fornecido como contexto para um LLM (ex: GPT-4o-mini).
2. **Escopo do Modelo:** A base original contém 50% de notícias verdadeiras (Classe `1`) e 50% de fake news (Classe `0`). Como nosso sistema RAG tem o objetivo de gerar **contranarrativas explicativas** (desmentidos), apenas as notícias da Classe `0` (que apontam para laudos jornalísticos de agências de checagem) têm valor semântico.

A solução de engenharia foi ignorar os textos truncados da planilha e utilizar apenas a coluna de **URLs** (que possui 100% de integridade) para refazer a raspagem do conteúdo jornalístico real e limpo.

## 2. A Decisão Metodológica (Filtro de Agências)
Ao analisarmos as URLs de Classe `0`, filtramos os domínios para focar exclusivamente nas três agências de maior rigor metodológico:

1. `g1.globo.com` (Fato ou Fake)
2. `noticias.uol.com.br` (UOL Confere)
3. `projetocomprova.com.br` (Projeto Comprova)

### Por que essa escolha?
- **Padrão Jornalístico (Grounding):** Estas três agências seguem os preceitos metodológicos da IFCN (*International Fact-Checking Network*). Os laudos possuem explicações detalhadas, consulta a especialistas oficiais e dados estatísticos, o que fornece um *Grounding Factual* perfeito para a LLM refutar o boato sem alucinar.
- **Estrutura HTML (Previsibilidade):** Portais maiores possuem uma árvore DOM (HTML) com tags `<article>` muito bem estruturadas, evitando a captura de lixo (memes, comentários de usuários, banners), diferente de blogs de desmentido menores.

## 3. O Fluxo de Extração (Pipeline)
O script de extração (`scripts/scrape_fact_checks.py`) segue o seguinte fluxo de execução:

1. **Ingestão das URLs:** O script carrega as 2.160 URLs priorizadas.
2. **Requisição HTTP Responsável:** Utiliza a biblioteca `requests` com cabeçalhos (`User-Agent`) realistas e implementa um sistema de *Rate Limiting* aleatório (pausa de 1 a 2,5 segundos entre requisições) para evitar bloqueios via IP (Erros 403/429). Implementa até 2 retentativas (retries) em caso de timeout.
3. **Limpeza e Parsing (Trafilatura):** O HTML bruto baixado é enviado para a biblioteca `trafilatura` (ferramenta acadêmica projetada para extrair apenas o núcleo textual jornalístico).
4. **Fallback (BeautifulSoup):** Caso o `trafilatura` não consiga identificar o corpo da matéria, o sistema faz um fallback para `BeautifulSoup`, isolando manualmente tags `<article>` ou `divs` principais e decompondo elementos de menu, footer e scripts.
5. **Estruturação JSONL:** Se a extração for bem-sucedida, o artigo é salvo em formato `JSON Lines`.

## 4. Estrutura do Dado Final
O resultado do scraping é salvo em `data/processed/fact_checks_full.jsonl`. Cada linha representa um artefato pronto para ser vetorizado pelo Qdrant.

**Exemplo de Objeto:**
```json
{
  "titulo": "Medidas de proteção contra a covid-19 ainda deverão ser mantidas... : Projeto Comprova",
  "texto": "Medidas de proteção contra a covid-19 ainda deverão ser mantidas... Especialistas alertam que esses cuidados devem ser tomados por todos os vacinados até que o país alcance a imunização coletiva...",
  "url": "https://projetocomprova.com.br/...",
  "dominio": "projetocomprova.com.br",
  "data_publicacao": "2021-04-16",
  "scraped_at": "2026-10-01T12:42:31Z",
  "palavras": 1855,
  "titulo_original_planilha": "Título original no FakeRecogna"
}
```

## 5. Resultados
Das 2.160 URLs originais da planilha que pertenciam a esses 3 domínios:
- **1.784 URLs (82,6%)** foram raspadas com sucesso absoluto, resultando em textos puros de alta densidade semântica.
- **376 URLs (17,4%)** falharam. Destas falhas, a grande maioria corresponde a *Link Rot* (Páginas apagadas pelas agências retornando erro 404 ao longo dos anos) ou paywalls rigorosos.

Este artefato final (`fact_checks_full.jsonl`) é a fundação para o modelo `mpnet-base-v2` gerar os embeddings da Busca Semântica do projeto.
