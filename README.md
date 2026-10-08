<div align="center">
  <img src="assets/logo.png" alt="Logo do checa-ai" width="320" />

  # Checa AI - Machine Learning Model

  Este repositório contém a inteligência por trás do **Checa AI**. Aqui é onde desenvolvemos, treinamos e versionamos os modelos de Machine Learning que são consumidos pelo [checa-ai-backend](https://github.com/pedrolucasdourado/checa-ai-backend).
</div>

---

## Objetivo

O objetivo deste módulo é fornecer um sistema inteligente de **verificação automática de fatos e geração de contranarrativas** para o combate à desinformação no Brasil. O Checa-AI atua como o "cérebro" da aplicação, combinando recuperação de informação vetorial (Retrieval-Augmented Generation — RAG) com modelos de linguagem generativos para produzir respostas fundamentadas em evidências jornalísticas reais.

## Fundamentação Científica

### Retrieval-Augmented Generation (RAG)

O sistema implementa a arquitetura **RAG** (Lewis et al., 2020), que combina um componente de **recuperação** (*retriever*) com um componente de **geração** (*generator*). Essa abordagem resolve duas limitações críticas dos LLMs puros:

1. **Alucinação factual**: LLMs podem gerar informações plausíveis mas falsas. Ao ancorar a geração em documentos recuperados de fontes jornalísticas verificadas, o sistema garante **grounding factual**.
2. **Conhecimento desatualizado**: O corpus de checagens é atualizado independentemente do modelo, permitindo cobertura de desinformação recente sem re-treinamento.

### Sentence Embeddings e Busca Semântica

O módulo de recuperação utiliza o modelo **paraphrase-multilingual-mpnet-base-v2** (Reimers & Gurevych, 2019), um Sentence-BERT (SBERT) treinado com aprendizado contrastivo para gerar representações vetoriais densas (768 dimensões) de sentenças. Diferentemente de abordagens baseadas em TF-IDF ou BM25, os embeddings capturam **similaridade semântica** — permitindo que a query *"estão jogando fora cédulas de votação"* recupere checagens sobre fraude eleitoral mesmo sem correspondência lexical exata.

Os vetores são normalizados via **L2-normalization**, o que transforma a distância cosseno em produto escalar, otimizando a busca por similaridade no Qdrant.

### Clusterização Temática via BERTopic

Para análise exploratória e identificação de narrativas dominantes, empregamos o **BERTopic** (Grootendorst, 2022), que combina:

- **UMAP** para redução de dimensionalidade dos embeddings
- **HDBSCAN** para clusterização hierárquica baseada em densidade
- **c-TF-IDF** (class-based TF-IDF) para extração de descritores temáticos por cluster

Essa pipeline permitiu a descoberta automática de eixos temáticos como **Saúde/Covid-19**, **Sistema Eleitoral**, **Política Nacional** e **Meio Ambiente** — insumos para a futura implementação de agentes especializados por domínio.

### Técnica de Prompt: Truth Sandwich

A geração de contranarrativas segue o framework **Truth Sandwich** (Lakoff, 2018), uma técnica cognitiva recomendada por pesquisadores em comunicação e desinformação:

1. **Afirma o fato verídico** logo na primeira frase (ancoragem positiva)
2. **Menciona o boato brevemente** sem amplificá-lo
3. **Apresenta as evidências** que refutam a alegação
4. **Reafirma o fato** com a fonte oficial

Essa estrutura evita o *efeito de familiaridade* (illusory truth effect), onde a repetição excessiva do boato acaba por reforçá-lo na memória do leitor.

## Arquitetura do Pipeline

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG — CHECA-AI                       │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────────┐   │
│  │  FakeRecogna │    │   Scraper    │    │   fact_checks_*.jsonl  │   │
│  │   (.xlsx)    │───▶│ trafilatura  │───▶│  Laudos jornalísticos  │   │
│  │  12k linhas  │    │ + bs4        │    │  limpos + metadados    │   │
│  └─────────────┘    └──────────────┘    └───────────┬────────────┘   │
│                                                     │                │
│                                          ┌──────────▼──────────┐     │
│                                          │  Sentence-BERT      │     │
│                                          │  mpnet-base-v2      │     │
│                                          │  768d, norm. L2     │     │
│                                          └──────────┬──────────┘     │
│                                                     │                │
│                                          ┌──────────▼──────────┐     │
│                                          │  Qdrant (local)     │     │
│                                          │  fact_checks_pt     │     │
│                                          │  COSINE similarity  │     │
│                                          └──────────┬──────────┘     │
│                                                     │                │
│  ┌──────────────┐              ┌────────────────────▼──────────┐     │
│  │  Boato       │  embedding   │   Busca Semântica (Top-K)     │     │
│  │  (WhatsApp)  │─────────────▶│   score ≥ 0.55 → gerar       │     │
│  └──────────────┘              │   score < 0.55 → abstinência  │     │
│                                └────────────────────┬──────────┘     │
│                                                     │                │
│                                          ┌──────────▼──────────┐     │
│                                          │  GPT-4o-mini        │     │
│                                          │  Truth Sandwich      │     │
│                                          │  temp=0.2, grounded  │     │
│                                          └──────────┬──────────┘     │
│                                                     │                │
│                                          ┌──────────▼──────────┐     │
│                                          │  Contranarrativa    │     │
│                                          │  + Fonte oficial    │     │
│                                          └─────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
```

## Análise de Dados (EDA)
Realizamos uma auditoria rigorosa nos datasets para garantir a qualidade do treinamento e evitar vieses (*shortcut learning*).

**Resumo do Corpus:**
- **Volume Total:** 36.773 registros.
- **Amostras Únicas:** 22.410 registros (após deduplicação).
- **Bases Utilizadas:** `FakeRecogna`, `FakeTrueBr` e `fakeWhatsApp`.
- **Compatibilidade BERT:** Mediana de **105 tokens** por texto; apenas **8.24%** dos registros excedem o limite de 512 tokens do BERTimbau.

**Principais Achados:**
- **Deduplicação:** Identificada alta taxa de duplicatas na base de WhatsApp (71%), tratadas para evitar *overfitting*.
- **Leakage:** Detecção de "assinaturas" de sites de checagem (ex: G1, Lupa), essenciais para a limpeza dos dados e evitar que o modelo aprenda a fonte em vez do conteúdo.
- **Estilometria:** Notou-se maior densidade de exclamações e uso de Caps Lock em notícias falsas.

## Estrutura do Projeto

```text
├── app.py                  # Servidor FastAPI (rotas HTML + API REST)
├── data/
│   ├── raw/                # Datasets originais (FakeRecogna.xlsx)
│   ├── processed/          # Laudos extraídos (fact_checks_*.jsonl)
│   └── qdrant_db/          # Índice vetorial Qdrant persistente (local)
├── scripts/
│   ├── scrape_fact_checks.py   # Scraper de laudos via trafilatura/bs4
│   ├── ingest_to_qdrant.py     # Ingestão e indexação vetorial no Qdrant
│   ├── generate_debunk.py      # Pipeline RAG standalone (CLI)
│   └── cluster_topics.py       # Clusterização temática (BERTopic)
├── src/
│   ├── fact_check_service.py   # Serviço RAG (Singleton: Qdrant + LLM)
│   ├── data/                   # Scripts de ingestão e limpeza
│   ├── features/               # Engenharia de features
│   ├── models/                 # Scripts de treinamento e inferência
│   └── visualization/          # Visualização de dados
├── static/
│   ├── css/debunk.css          # Design system (dark glassmorphism)
│   └── js/debunk.js            # Frontend: POST → resultado renderizado
├── templates/
│   ├── components/sidebar.html # Sidebar de navegação
│   └── pages/debunk.html       # Interface de verificação de boatos
├── notebooks/              # Jupyter Notebooks (EDA, prototipagem)
├── exploration/            # Análises exploratórias comparativas
├── docs/                   # Documentação do modelo e experimentos
├── models/                 # Modelos treinados e serializados
├── reports/                # Relatórios de performance e métricas
├── references/             # Dicionários de dados e referências bibliográficas
├── requirements.txt        # Dependências do projeto
└── .env                    # Variáveis de ambiente (OPENAI_API_KEY)
```

## Como Executar

### 1. Instalação
```bash
# Clone e configure o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
pip install trafilatura beautifulsoup4 sentence-transformers qdrant-client openai python-dotenv bertopic
```

### 2. Configuração
Crie o arquivo `.env` na raiz do projeto:
```bash
OPENAI_API_KEY=sk-...
```

### 3. Pipeline de Dados
```bash
# Etapa 1 — Extrair laudos jornalísticos das URLs do FakeRecogna
python scripts/scrape_fact_checks.py --batch 2000

# Etapa 2 — Indexar os laudos no Qdrant (vetorização com Sentence-BERT)
python scripts/ingest_to_qdrant.py --input data/processed/fact_checks_full.jsonl

# Etapa 3 — (Opcional) Análise de clusters temáticos
python scripts/cluster_topics.py --input data/processed/fact_checks_full.jsonl
```

### 4. Servidor Web
```bash
# Iniciar o servidor FastAPI
uvicorn app:app --reload --port 8000

# Acesse: http://localhost:8000/debunk
```

### API de verificação em contêiner

O `ml-api` atende `POST /api/v1/verify` na porta local `18001` e conversa com o
Qdrant na rede Docker `checa-ai-ml-shared`. A coleção vetorial precisa estar
indexada previamente com o mesmo provedor, modelo e dimensão de embeddings.
O contêiner não executa ingestão nem cria a coleção automaticamente.

Configure `OPENAI_API_KEY` no `.env` quando usar embeddings OpenAI ou geração
via OpenAI. Ajuste `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` e
`QDRANT_COLLECTION` para a coleção realmente indexada. Se usar embeddings
locais, defina explicitamente as três variáveis correspondentes e
`INSTALL_LOCAL_EMBEDDINGS=true` antes do build. Essa opção instala
`sentence-transformers` na imagem; a imagem padrão usa embeddings OpenAI.

```bash
docker compose config -q
docker compose build ml-api
docker compose up -d
curl -i http://localhost:18001/ready
```

`/health` indica que o servidor está vivo. `/ready` retorna `200` somente se
a coleção configurada existir e sua dimensão corresponder ao modelo de
embeddings; retorna `503` quando o índice estiver ausente ou incompatível.

Exemplo de chamada:

```bash
curl -X POST http://localhost:18001/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"text":"Mensagem a verificar"}'
```

### 5. Teste via CLI (sem servidor)
```bash
python scripts/generate_debunk.py --query "Hackers invadiram o TSE e transformaram justificativas em votos"
```

## Tecnologias Utilizadas

| Camada | Tecnologia | Papel |
|---|---|---|
| **Linguagem** | Python 3.14 | Core |
| **Web Framework** | FastAPI + Uvicorn | API REST + SSR com Jinja2 |
| **Embeddings** | Sentence-Transformers (mpnet-base-v2) | Representação vetorial semântica |
| **Vector Store** | Qdrant (modo local) | Indexação e busca por similaridade cosseno |
| **LLM** | OpenAI GPT-4o-mini | Geração de contranarrativas grounded |
| **Scraping** | Trafilatura + BeautifulSoup | Extração de texto jornalístico limpo |
| **Clusterização** | BERTopic (UMAP + HDBSCAN + c-TF-IDF) | Descoberta de eixos temáticos |
| **Dados** | Pandas, NumPy, Scikit-learn | Manipulação, EDA e pré-processamento |
| **Frontend** | HTML + CSS + JavaScript (Vanilla) | Interface dark glassmorphism |
| **Ambiente** | Jupyter Notebooks | Exploração e prototipagem |

## Referências

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS.
- Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*. arXiv:2203.05794.
- Lakoff, G. (2018). *The Truth Sandwich: A Strategy for Responsible News Coverage of Lies*.

## Equipe
**Grupo 09 — Residência de IA, Instituto Eldorado.**

| | Integrante | Papel |
|---|---|---|
| <img src="https://github.com/AmandaElisa.png" width="60" alt="Amanda Elisa" /> | [Amanda Elisa de Oliveira Carvalho](https://github.com/AmandaElisa) | Scrum Master (Líder) & AI Builder |
| <img src="https://github.com/pedrolucasdourado.png" width="60" alt="Pedro Lucas" /> | [Pedro Lucas Dourado Santos](https://github.com/pedrolucasdourado) | Product Owner & Full-Stack/AI Builder |
| <img src="https://github.com/jhsribeiro.png" width="60" alt="Jhiovana Ribeiro" /> | [Jhiovana Ribeiro](https://github.com/jhsribeiro) | Data Specialist / Data Engineering (ETL) |
| <img src="https://github.com/LeoAlec.png" width="60" alt="Leo Alec" /> | [Leo Alec](https://github.com/LeoAlec) | Data Specialist / Data Science |
| <img src="https://github.com/NasserCaixeta.png" width="60" alt="Nasser Camêllo Caixeta" /> | [Nasser Camêllo Caixeta](https://github.com/NasserCaixeta) | Backend Specialist / AI Orchestrator |

---
Desenvolvido para o projeto Checa AI.
