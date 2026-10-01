<div align="center">
  <img src="assets/logo.png" alt="Logo do checa-ai" width="320" />

  # Checa AI - Machine Learning Model

  Este repositório contém a inteligência por trás do **Checa AI**. Aqui é onde desenvolvemos, treinamos e versionamos os modelos de Machine Learning que são consumidos pelo [checa-ai-backend](https://github.com/pedrolucasdourado/checa-ai-backend).
</div>

---

## Objetivo
O objetivo deste módulo é fornecer previsões e classificações precisas para a verificação de fake news e desinformação, servindo como o "cérebro" da aplicação e permitindo classificações granulares para lidar com a ambiguidade da desinformação.

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
Seguimos o padrão **Cookiecutter Data Science** para garantir reprodutibilidade e organização:

```text
├── data/               # Dados (raw, processed, interim, external)
├── docs/               # Documentação do modelo e experimentos
├── models/             # Modelos treinados e serializados (.pkl, .joblib, etc)
├── notebooks/           # Jupyter Notebooks para exploração e prototipagem
├── references/          # Dicionários de dados e referências bibliográficas
├── reports/            # Relatórios de performance e métricas
└── src/                # Código fonte modularizado
    ├── data/           # Scripts de ingestão e limpeza
    ├── features/       # Engenharia de features (feature engineering)
    ├── models/         # Scripts de treinamento e inferência
    └── visualization/  # Scripts de visualização de dados
```

## Como Executar

### 1. Instalação
Instale as dependências necessárias:
```bash
pip install -r requirements.txt
```

### 2. Acesso aos dados (DVC)
Os dados são versionados com [DVC](https://dvc.org) e ficam no DagsHub: `https://dagshub.com/pedrolucasdourado/checa-ai-ml.dvc`. O acesso exige um token do DagsHub, mesmo com o repositório público.

1. Crie uma conta no [DagsHub](https://dagshub.com) e gere um token em **Settings → Tokens**.
2. Dentro do repositório clonado, configure as credenciais **localmente** (o `--local` grava em `.dvc/config.local`, que não é versionado; nunca coloque o token no `.dvc/config`):
   ```bash
   dvc remote modify --local origin auth basic
   dvc remote modify --local origin user SEU_USUARIO_DAGSHUB
   dvc remote modify --local origin password SEU_TOKEN
   ```
3. Baixe os dados:
   ```bash
   dvc pull data/raw/*.dvc data/processed/dataset.csv
   ```
   Prefira esse comando ao `dvc pull` sem argumentos, que pode remover os arquivos de `data/raw` após o download.

Problemas comuns:
- `Checkout failed ... data/processed/dataset.csv`: o remote respondeu 401 (credenciais ausentes ou incorretas). Refaça o passo 2.
- `SSLCertVerificationError ... self-signed certificate`: a rede (proxy, VPN ou antivírus) intercepta o HTTPS. Troque de rede ou configure `dvc remote modify --local origin ssl_verify /caminho/ca.pem`.
- Para enviar dados novos, rode `dvc repro` e depois `dvc push`, e commite o `dvc.lock`. A conta precisa ter permissão de escrita no repositório.

### 3. Fluxo de Trabalho
1. **Exploração**: Comece pelos arquivos em `notebooks/`.
2. **Processamento**: Use os scripts em `src/data/` para preparar os dados.
3. **Treinamento**: Execute os scripts em `src/models/` para gerar o modelo final.
4. **Exportação**: O modelo final será salvo na pasta `models/` para ser consumido pelo backend.

## Tecnologias Utilizadas
- **Linguagem:** Python 3.x
- **Manipulação de Dados:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Transformers (BERTimbau)
- **API/Serviço:** FastAPI (se aplicável)
- **Ambiente:** Jupyter Notebooks

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
