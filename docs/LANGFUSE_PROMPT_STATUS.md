# Estado de validação dos prompts no Langfuse

O serviço mantém dois prompts independentes e auditáveis:

| Prompt | Versão local | Hash local |
| --- | --- | --- |
| `evidence-flow` | `evidence-flow-v4` (candidato; não promover sem held-out independente) | registrado em `src/rag/prompts.lock.json` |
| `claim-analysis` | `claim-analysis-v1` | registrado em `src/rag/prompts.lock.json` |

Os textos locais são a semente do `sync_prompts.py` e só são usados em runtime com o Langfuse desligado (dev/testes). Com o Langfuse habilitado, prompt/label ausente gera `PromptUnavailableError` (sem fallback silencioso). Ambos são sincronizados pelo mesmo `scripts/sync_prompts.py`
e aparecem nos metadados do trace (versão e hash). A extração de alegação usa seu
próprio bundle durante a chamada à LLM; ela não herda o prompt de síntese.

## Publicar o candidato v4 quando as credenciais estiverem disponíveis

O `evidence-flow-v4` foi validado localmente e no held-out, mas não foi enviado
ao Langfuse. O procedimento abaixo publica primeiro em `staging`; ele não muda
o label `production` até a etapa final.

### 1. Configurar credenciais sem versionar segredos

No `.env` local (que deve permanecer fora do Git), preencha:

```dotenv
LANGFUSE_ENABLED=true
LANGFUSE_CAPTURE_CONTENT=false
LANGFUSE_PUBLIC_KEY=<sua chave pública do projeto Langfuse>
LANGFUSE_SECRET_KEY=<sua chave secreta do projeto Langfuse>
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_TRACING_ENVIRONMENT=staging
PROMPT_LABEL=staging
```

Não passe chaves na linha de comando, não as cole em issues/logs e não altere
`.env.example` com valores reais. Mantenha `LANGFUSE_CAPTURE_CONTENT=false`
para não enviar alegações e evidências em texto aberto aos traces.

### 2. Conferir a mudança remotamente (somente leitura)

Com o ambiente virtual ativo, execute:

```powershell
python scripts/sync_prompts.py --check --label staging
```

O resultado esperado é `INFO` informando que `evidence-flow-v4` será criado em
`staging`, ou `OK` se uma versão idêntica já estiver disponível. Esse comando
não cria versão nem move labels.

### 3. Publicar em staging

Depois de conferir o resultado do passo anterior:

```powershell
python scripts/sync_prompts.py --label staging
```

O script publica o texto local somente se ainda não houver uma versão idêntica e
atribui o label `staging`. Guarde o número da versão retornado pelo comando.

### 4. Validar o ambiente de staging

Mantenha `PROMPT_LABEL=staging` no ambiente que aponta para a instância de
staging e execute o smoke E2E descrito em [LIVE_E2E.md](LIVE_E2E.md). Verifique
nos traces que o prompt resolvido é `evidence-flow-v4` e que `source=langfuse`.
Compare taxa de abstenção, latência, erros de schema e amostras revisadas com a
baseline. O script offline `eval_golden_claims.py` deliberadamente desliga o
Langfuse, portanto ele valida o fallback local e não substitui essa etapa.

### 5. Promover produção e rollback

Somente após a validação de staging e uma autorização explícita para alterar
produção:

```powershell
python scripts/sync_prompts.py --promote
```

O comando move `production` para a versão cujo texto é exatamente igual ao
arquivo local. Se houver regressão, faça rollback no Langfuse movendo o label
`production` para a versão anterior conhecida; não edite a chave nem reescreva
o histórico do prompt.

## Última verificação remota

A verificação somente leitura executada nesta máquina não conseguiu consultar o
Langfuse: `LANGFUSE_ENABLED` está desabilitado e as chaves Langfuse não estavam
disponíveis ao processo. Portanto, a versão remota ativa **não foi confirmada** e
nenhum label ou prompt foi alterado.

Quando as chaves forem disponibilizadas, siga o procedimento de `staging` acima.
Não use `--promote` apenas para testar conectividade: ele altera o label de produção.
