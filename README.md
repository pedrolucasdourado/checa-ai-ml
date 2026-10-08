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

### Embeddings e Busca Semântica

O módulo de recuperação suporta múltiplos provedores de embeddings para flexibilidade entre performance e privacidade:

- **OpenAI (Padrão)**: Utiliza o modelo `text-embedding-3-small` (1536 dimensões), oferecendo alta performance sem a necessidade de infraestrutura de GPU local.
- **Sentence-BERT (Local)**: Utiliza o modelo `paraphrase-multilingual-mpnet-base-v2` (768 dimensões), permitindo a execução totalmente local e privada.

Os vetores são normalizados via **L2-normalization**, o que transforma a distância cosseno em produto escalar, otimizando a busca por similaridade no Qdrant.

### Chunking e Estratégia de Recuperação

Para lidar com documentos extensos e melhorar a precisão da recuperação, implementamos:

- **Multi-language Chunking**: Divisão de textos em fragmentos menores com *overlap*, garantindo que o contexto semântico não seja perdido nas bordas dos chunks.
- **Cluster-aware Retrieval**: Um sistema de re-ranking que utiliza a clusterização temática para ajustar o score de similaridade. Documentos pertencentes a clusters "aprovados" ou de "alta confiança" recebem bônus, enquanto clusters ruidosos são penalizados.

### Silver Dataset Clustering

Para análise exploratória e melhoria da recuperação, empregamos o **BERTopic** (Grootendorst, 2022), que combina UMAP, HDBSCAN e c-TF-IDF. Desenvolvemos um fluxo de dados em camadas:

1. **Bronze Dataset**: O corpus bruto de laudos extraídos e indexados.
2. **Silver Dataset**: Um subconjunto refinado onde clusters temáticos são identificados e curados humanamente, permitindo que o sistema priorize narrativas validadas.

### Técnica de Prompt: Truth Sandwich

A geração de contranarrativas segue o framework **Truth Sandwich** (Lakoff, 2018), uma técnica cognitiva recomendada por pesquisadores em comunicação e desinformação:

1. **Afirma o fato verídico** logo na primeira frase (ancoragem positiva)
2. **Menciona o boato brevemente** sem amplificá-lo
3. **Apresenta as evidências** que refutam a alegação
4. **Reafirma o fato** com a fonte oficial

### Segurança e DeepSearch Fallback

Para garantir a robustez e a utilidade do sistema, implementamos camadas de segurança e de busca expandida:

- **Safety Guardrails**:
    - **Filtros de Entrada**: Integração com APIs de moderação para bloquear prompts tóxicos e detecção de *Prompt Injection* (Suporte a PT/EN).
    - **Validação de Saída**: Scanner de segurança que valida a resposta do LLM antes de exibi-la ao usuário.
    - **Respostas de Recusa**: Fluxos de fallback para queries que violam as diretrizes de segurança.
- **DeepSearch Fallback**: Quando o sistema não encontra laudos oficiais no banco (Abstenção), ele dispara automaticamente uma busca na web via **Tavily AI**. O resultado é sintetizado de forma neutra, sem dar veredito de fato/fake, e finaliza com um aviso de pensamento crítico.

## Arquitetura do Pipeline

```text
┌──────────────────────────────────────────────────────────────────────┐
│                        PIPELINE RAG — CHECA-AI                       │
│                                                                      │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────┐│
│  │  Datasets    │───▶│  Scraper/    │───▶│  Bronze DB    │───▶│ Silver││
│  │  (Raw/XLSX)  │    │  Cleaning    │    │ (Qdrant/JSON) │    │ Dataset││
│  └─────────────┘    └──────────────┘    └───────┬────────┘    └───┬───┘│
│                                                  │                │   │
│                                    ┌─────────────▼──────────┐     │   │
│                                    │  Chunking & Embedding   │◀────┘   │
│                                    │  (OpenAI / Local SBERT)│         │
│                                    └─────────────┬──────────┘       │
│                                                  │                   │
│                                    ┌─────────────▼──────────┐        │
│                                    │  Qdrant Vector Store   │        │
│                                    │  (Similarity Search)    │        │
│                                    └─────────────┬──────────┘       │
│                                                  │                   │
│  ┌──────────────┐              ┌───────────────▼───────────┐           │
│  │  Boato       │  embedding   │   Cluster-aware Re-rank  │           │
│  │  (Query)     │─────────────▶│   (Top-K + Cluster Bonus)│           │
│  └──────────────┘              └───────────────┬────────────┘           │
│                                                │                      │
│                                    ┌───────────▼────────────┐        │
│                                    │  GPT-4o-mini (Generator)│        │
│                                    │  Truth Sandwich Prompt  │        │
│                                    └───────────┬────────────┘         │
│                                                │                      │
│                                    ┌───────────▼────────────┐        │
│                                    │  Contranarrativa Final  │        │
│                                    │  + Fontes Verificadas  │        │
│                                    └─────────────────────────┘        │
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
│   ├── interim/            # Dados em processamento (Silver Dataset)
│   ├── processed/          # Laudos extraídos e limpos
│   └── qdrant_db/          # Índice vetorial Qdrant persistente
├── scripts/
│   ├── scrape_fact_checks.py   # Scraper de laudos jornalísticos
│   ├── ingest_to_qdrant.py     # Indexação vetorial no Qdrant
│   ├── build_silver_dataset.py # Pipeline de criação do Silver Dataset
│   ├── cluster_topics.py       # Clusterização temática (BERTopic)
│   ├── curate_silver_clusters.py # Ferramenta de curadoria de clusters
│   ├── evaluate_retrieval.py   # Avaliação de performance do Retriever
│   ├── agent_cli.py            # CLI Interativo para testes do agente
│   └── generate_debunk.py      # Pipeline RAG standalone (CLI)
├── src/
│   ├── config.py               # Configuração centralizada do projeto
│   ├── fact_check_service.py   # Orquestrador do serviço RAG
│   ├── rag/                    # Core do sistema de recuperação
│   │   ├── chunking.py         # Lógica de fragmentação de texto
│   │   ├── embeddings.py       # Abstração de provedores (OpenAI/Local)
│   │   ├── qdrant.py           # Interface de comunicação com Qdrant
│   │   ├── retriever.py         # Lógica de busca e re-ranking
│   │   ├── web_search.py       # Busca web via Tavily AI (Fallback)
│   │   └── preprocessing.py    # Limpeza e normalização de texto
│   ├── data/                   # Utilitários de manipulação de dados
│   ├── features/               # Engenharia de features
│   ├── models/                 # Modelos de ML e inferência
│   ├── safety/                 # Camadas de Guardrails (Moderação/Injeção)
│   └── visualization/          # Ferramentas de visualização
├── static/                     # Assets de frontend (CSS/JS)
├── templates/                  # Templates Jinja2 (HTML)
├── notebooks/                  # Jupyter Notebooks (EDA e Prototipagem)
├── specs/                      # Especificações técnicas (ex: Safety Guardrails)
├── docs/                       # Documentação técnica e fluxos
├── models/                     # Artefatos de modelos serializados
├── references/                 # Referências bibliográficas e dicionários
├── requirements.txt            # Dependências do projeto
└── .env                        # Variáveis de ambiente
```

## Como Executar

### 1. Instalação
```bash
# Clone e configure o ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Configuração
Crie o arquivo `.env` na raiz do projeto:
```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
EMBEDDING_PROVIDER=openai  # Opções: 'openai' ou 'local'
```

### 3. Pipeline de Dados (Fluxo Recomendado)
```bash
# Etapa 1 — Extrair laudos jornalísticos das URLs do FakeRecogna
python scripts/scrape_fact_checks.py --batch 2000

# Etapa 2 — Construir o Silver Dataset (Limpeza e Refinamento)
python scripts/build_silver_dataset.py

# Etapa 3 — Indexar no Qdrant (Bronze Dataset)
python scripts/ingest_to_qdrant.py --input data/processed/fact_checks_silver.jsonl

# Etapa 4 — (Opcional) Curar clusters e avaliar a recuperação
python scripts/curate_silver_clusters.py
python scripts/evaluate_retrieval.py
```

### 4. Servidor Web
```bash
# Iniciar o servidor FastAPI
uvicorn app:app --reload --port 8000

# Acesse: http://localhost:8000/debunk
```

### 5. Teste via CLI (sem servidor)
```bash
# Para testar a orquestração completa (RAG + DeepSearch + Guardrails)
python scripts/agent_cli.py
```

## Observabilidade e LLMOps (Langfuse)

Cada chamada a `POST /api/debunk` gera um **trace** no Langfuse:

```text
verify-claim            (trace: latência total, session, tags, prompt_version)
├─ retrieval            (retriever: query → chunks/scores, candidatos vs. selecionados)
│  └─ embed-query       (embedding: tokens + custo)
└─ llm-generation       (generation: prompt, resposta, tokens in/out, custo, finish_reason)
```

| Necessidade | Onde aparece |
|---|---|
| Traces por requisição | Langfuse > Tracing; `trace_id` volta no JSON da API |
| Tokens e custo | `usage_details` por generation/embedding; custo inferido pelo Langfuse a partir do modelo |
| Latência | duração de cada span (embedding, busca, LLM) e do trace |
| Feedback do usuário | botões 👍/👎 → `POST /api/feedback` → score `user_feedback` no trace |
| Qualidade online | scores automáticos: `top_similarity`, `abstained`, `sources_grounded` (URL citada que não veio dos laudos = alucinação de link) |
| Versão do prompt | Langfuse Prompt Management (`debunk@<n>`); cada generation fica ligada à versão; filtre por tag `prompt:<versão>` |

**Configuração:** copie `.env.example` para `.env` e preencha `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY` e `LANGFUSE_BASE_URL`. Sem as chaves o tracing é desligado e nada muda no pipeline.
Com `LANGFUSE_CAPTURE_CONTENT=false` o texto dos usuários não é enviado (só métricas e metadados).

### Guardrails (`src/rag/guardrails.py`)

- **Saída estruturada:** o LLM responde `{texto, fontes_usadas}` (JSON Schema estrito + Pydantic). Se vier fora do schema, o texto cru é aproveitado.
- **Anti-link-inventado:** toda URL citada precisa estar entre as evidências recuperadas; URLs fora do contexto são removidas da resposta e uma fonte real é creditada. O score `sources_grounded` mede a saída crua do modelo (taxa de alucinação de link) e `structured_output` mede a aderência ao schema.
- **Prompt injection nos laudos:** chunks com risco `high` (payload da ingestão ou reavaliação em runtime) são excluídos do contexto; o prompt também instrui o modelo a tratar laudos como dados não confiáveis.

### CI/CD (GitHub Actions)

| Quando | Workflow | O que faz |
|---|---|---|
| Todo PR / push na main | `ci.yml` › `tests` | `pytest` (inclui contrato do prompt, guardrails, harness do eval) |
| PR que muda prompt, guardrails, retriever, serviço, `config.py` ou `evals/` | `ci.yml` › `eval` | Eval de regressão (`scripts/eval_generation.py`) no golden set `evals/golden.jsonl`; **falha o PR** se faithfulness, abstenção correta, saída estruturada, links ou injection ficarem fora de `evals/thresholds.json` |
| Merge na main com `prompts.py` alterado | `prompts-staging.yml` | Publica a nova versão no Langfuse com label `staging` (idempotente) |
| Release publicado (ou manual) | `release.yml` | Eval → aprovação do environment `production` → move o label `production` para a versão do release |

- **Check obrigatório da branch:** `ci-ok` (verde = testes ok e eval ok ou desnecessário).
- **Prompt alterado?** Incremente `PROMPT_VERSION` e rode `python scripts/sync_prompts.py --write-lock`; sem isso o teste de contrato falha. `python scripts/sync_prompts.py --check` mostra o que mudaria no Langfuse sem escrever.
- **Eval local:** `python scripts/eval_generation.py` (usa `OPENAI_API_KEY`, ~11 casos, centavos). O golden set é autocontido (evidências no próprio caso), sem Qdrant/DVC. Ao encontrar uma falha real em produção, adicione o caso a `evals/golden.jsonl`.
- **Rollback:** republique o release anterior ou mova o label `production` no Langfuse.
- **Setup único no GitHub:** secrets `OPENAI_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (+ variável `LANGFUSE_BASE_URL`); environment `production` com *Required reviewers*; proteger a `main` exigindo o check `ci-ok`. PRs de forks não recebem secrets (o eval é pulado com aviso).

### Gestão de prompts

`src/rag/prompts.py` é o módulo único de prompts (usado pela API e por `scripts/generate_debunk.py`).
Em runtime o prompt `debunk` é lido do **Langfuse Prompt Management** pelo label `PROMPT_LABEL`
(`production` por padrão; use `staging` para testar), com cache do SDK. Se o Langfuse ou o prompt
estiverem indisponíveis, o texto local do módulo é usado como fallback.

```bash
python scripts/sync_prompts.py            # texto local → Langfuse com label staging (só cria versão se mudou)
python scripts/sync_prompts.py --promote  # move production para a versão local (normalmente feito pelo release.yml)
```

Editar o prompt no Langfuse (ou rodar o sync) cria uma nova versão sem redeploy; **rollback** = mover
o label `production` para a versão anterior. Variáveis usam `{{query}}` e `{{contexto}}`.
Se mudar o texto local, incremente `PROMPT_VERSION` e atualize o lock (ver CI/CD).

> Ao trocar `LLM_MODEL`, confira
> em Langfuse > Settings > Models se o preço do modelo está cadastrado (senão o custo fica vazio).

## Tecnologias Utilizadas

| Camada | Tecnologia | Papel |
|---|---|---|
| **Linguagem** | Python 3.14 | Core |
| **Web Framework** | FastAPI + Uvicorn | API REST + SSR com Jinja2 |
| **Embeddings** | Sentence-Transformers (mpnet-base-v2) | Representação vetorial semântica |
| **Vector Store** | Qdrant (modo local) | Indexação e busca por similaridade cosseno |
| **LLM** | OpenAI GPT-4o-mini | Geração de contranarrativas grounded |
| **Scraping** | Trafilatura + BeautifulSoup | Extração de texto jornalístico limpo |
| **Web Search** | Tavily AI | Busca web para fallback de abstenção |
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
