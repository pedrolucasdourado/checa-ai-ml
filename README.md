<div align="center">
  <img src="assets/logo.png" alt="Logo do checa-ai" width="320" />

  # Checa AI - Machine Learning Model

  Este repositório contém a inteligência por trás do **Checa AI**. Aqui é onde desenvolvemos, treinamos e versionamos os modelos de Machine Learning que são consumidos pelo [checa-ai-backend](https://github.com/pedrolucasdourado/checa-ai-backend).
</div>


---

## Objetivo
O objetivo deste módulo é fornecer previsões e classificações precisas para a verificação de fake news e desinformação, servindo como o "cérebro" da aplicação e permitindo classificações granulares para lidar com a ambiguidade da desinformação.

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

### 2. Fluxo de Trabalho
1. **Exploração**: Comece pelos arquivos em `notebooks/`.
2. **Processamento**: Use os scripts em `src/data/` para preparar os dados.
3. **Treinamento**: Execute os scripts em `src/models/` para gerar o modelo final.
4. **Exportação**: O modelo final será salvo na pasta `models/` para ser consumido pelo backend.

## Tecnologias Utilizadas
- **Linguagem:** Python 3.x
- **Manipulação de Dados:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, [OUTRAS LIBS]
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