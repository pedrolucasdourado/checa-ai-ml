# Versionamento de dados (DVC + DagsHub)

Os laudos processados não ficam no git: o git guarda só ponteiros `.dvc` e os arquivos
vão para o remote `origin` (DagsHub, `https://dagshub.com/pedrolucasdourado/checa-ai-ml.dvc`).

Arquivos versionados em `data/processed/`:

| Arquivo | Conteúdo |
|---|---|
| `fact_checks_all.jsonl` | todos os laudos extraídos |
| `fact_checks_clean.jsonl` | laudos válidos (saída do `scrape_fact_checks.py`) |
| `fact_checks_errors.jsonl` | URLs que falharam no scraping |
| `fact_checks_full.jsonl` | conjunto completo usado na indexação |

## Credenciais (uma vez por máquina)

Use o token do DagsHub (Settings > Tokens). Ele fica em `.dvc/config.local`, que o git ignora:

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <usuario_dagshub>
dvc remote modify origin --local password <token_dagshub>
```

## Baixar os dados

```bash
dvc pull
python scripts/ingest_to_qdrant.py --input data/processed/fact_checks_all.jsonl
```

## Publicar dados novos ou atualizados

```bash
dvc add data/processed/fact_checks_{all,clean,errors,full}.jsonl
git add data/processed/*.dvc data/processed/.gitignore
git commit -m "data: atualiza laudos"
dvc push
git push
```

O `.dvc` commitado identifica a versão exata dos dados, então cada commit do git
reproduz o mesmo conjunto de laudos.
