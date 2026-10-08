# Integração da verificação RAG com o backend do WhatsApp

**Data:** 2026-10-08
**Estado:** desenho aprovado em conversa; documento aguardando revisão do usuário

## Objetivo

Conectar o serviço de verificação do repositório `checa-ai-ml` ao fluxo real do
repositório irmão `checa-ai`. Uma mensagem de texto ou notícia obtida de um link
deve chegar ao serviço de ML pelo worker do backend. O usuário receberá uma única
mensagem de WhatsApp, com até 1.000 caracteres, contendo o veredito **Checagem
encontrada** ou **Evidência insuficiente**. Quando houver checagem, a mensagem
também conterá contranarrativa curta e links das fontes utilizadas.

Esses vereditos indicam se o acervo contém evidência adequada para responder à
alegação. O score de busca mede similaridade com checagens, não a probabilidade
de uma alegação ser verdadeira ou falsa. Esta entrega não cria os rótulos
"falso" e "verdadeiro".

## Estado atual e limites

- `checa-ai-ml/app.py` já expõe `POST /api/debunk` para sua interface web. Seu
  `FactCheckService` consulta Qdrant e usa um LLM para gerar contranarrativa.
- O Compose de ML sobe somente Qdrant. A API FastAPI ainda não possui imagem ou
  serviço próprios no Compose. A rota atual executa trabalho síncrono dentro de
  uma função `async` e cria o cliente OpenAI de geração a cada chamada.
- No backend, `POST /webhooks/evolution` autentica e persiste o evento. O worker
  consome o trabalho via Redis Streams, prepara texto ou notícia de link, chama
  `AnalyzeContentUseCase`, salva checkpoints no PostgreSQL e envia a resposta
  via outbox e Evolution API. Hoje o worker injeta `SimulatedClassifier`, o
  formatador exige aviso de demonstração e a conclusão grava `simulation`.
- O Qdrant já indexado será usado nesta etapa. Baixar laudos do DVC, gerar ou
  atualizar o índice, classificar fatos como verdadeiros/falsos e mudar a
  ingestão de dados ficam fora do escopo.

## Arquitetura

```text
WhatsApp → Evolution → POST /webhooks/evolution (backend)
                     → PostgreSQL + outbox → Redis → worker
                     → adaptador HTTP → POST /api/v1/verify (ML)
                                      → Qdrant + geração fundamentada
                     ← resultado versionado
                     → checkpoint → formato WhatsApp → outbox → Evolution
```

O webhook do backend continua retornando 202 após persistir o evento; não
espera pelo ML. O worker faz uma requisição HTTP de saída e recebe o resultado
na mesma conexão. Portanto não é necessária uma nova rota de entrada no
backend para o fluxo de WhatsApp. A rota diagnóstica de desenvolvimento
`POST /webhooks/messages` não participa da integração.

O repositório de ML ganhará um container para sua API FastAPI no seu próprio
Compose, ligado ao Qdrant já existente. O worker do backend alcançará esse
container por uma rede Docker compartilhada, usando uma URL configurável.
A configuração também permite apontar para outra implantação da mesma API
versionada sem alterar o domínio do backend. A API de ML ficará acessível
somente pela rede de serviços no ambiente integrado.

## Contrato HTTP v1 do serviço de ML

`POST /api/v1/verify` é o contrato entre serviços. A rota existente
`POST /api/debunk` permanece para a interface web e ambas compartilham a
lógica de busca; a nova rota terá apresentação e validação próprias.

Entrada JSON:

```json
{"text":"texto da alegação ou trechos literais da notícia"}
```

`text` deve ser não vazio e ter no máximo 10.000 caracteres, alinhado ao
`AnalysisInput` do backend. A API não recebe URL para buscar páginas: o
backend já extrai e prepara o texto de links antes dessa etapa.

Resposta JSON de checagem encontrada:

```json
{
  "schema_version": 1,
  "verdict": "matched",
  "similarity_score": 0.78,
  "counter_narrative": "Texto curto fundamentado nos laudos recuperados.",
  "sources": [
    {"title": "Título da checagem", "domain": "exemplo.org", "url": "https://exemplo.org/checagem"}
  ]
}
```

`verdict` aceita apenas `matched` ou `insufficient_evidence`.
`similarity_score` fica entre 0 e 1 e é usado para auditoria, não como
confiança factual exibida ao usuário. `matched` exige contranarrativa não
vazia e ao menos uma fonte HTTP(S) efetivamente usada. A nova rota seleciona
no máximo duas fontes que caibam na resposta final; a contranarrativa não
repete linhas de fontes, pois o backend renderiza os links estruturados.
As URLs vêm exclusivamente dos laudos recuperados, e a nova rota só expõe
fontes cuja referência na contranarrativa foi validada. Para entradas de link,
o orçamento da contranarrativa reserva espaço para identificar também a notícia
analisada no texto final do backend. `insufficient_evidence` devolve
`counter_narrative: null` e `sources: []`,
mesmo que a busca tenha encontrado candidatos abaixo do limiar. O backend não
trata candidatos fracos como prova.

Entrada inválida retorna 422. Dependências não prontas retornam 503. Falhas
internas retornam 5xx; nenhuma dessas respostas é um veredito.

## Contrato e fluxo no backend

Um adaptador HTTP de saída fará a chamada à API de ML, com cliente e conexões
reutilizáveis, URL e prazos configuráveis. A camada de aplicação valida a
versão, o veredito, o score, a contranarrativa e as fontes antes de aceitar o
resultado. O resultado real tem um contrato de domínio versionado e é salvo no
checkpoint `analysis_result` existente; a retomada distingue esse contrato do
resultado simulado já persistido. O adaptador HTTP não entra no domínio nem no
webhook.

O worker escolhe o adaptador real por configuração. O simulado permanece
disponível explicitamente para desenvolvimento e reversão; uma falha da API
real nunca ativa o simulado de forma silenciosa. No modo real, o backend deixa
de impor o aviso de demonstração e grava o resultado efetivo em
`analysis_outcome` e `responses.kind`, em vez dos valores fixos
`simulation`/`simulated`. O fluxo de deduplicação, lease, checkpoint, outbox,
Redis Streams e envio quoted permanece o mesmo.

## Resposta ao usuário

O backend formata o resultado real sem uma segunda chamada a LLM. Para
`matched`, a mensagem contém o título **Checagem encontrada**, uma
contranarrativa curta e os links das fontes usadas. Se a entrada veio de um
link, mantém identificação compacta da notícia analisada, sem confundir essa
notícia com as fontes de checagem. Para `insufficient_evidence`, envia uma
frase fixa que informa que não há evidência suficiente no acervo para verificar
aquela alegação; não apresenta fontes ou contranarrativa como prova.

A mensagem final tem no máximo 1.000 caracteres. O serviço de ML gera uma
contranarrativa com orçamento de tamanho considerando as fontes selecionadas;
o backend valida o texto montado antes de enviá-lo. O formatador não corta
silenciosamente URL, frase ou ressalva factual. Caso uma resposta fundamentada
não caiba após uma tentativa de condensação no serviço de ML, o serviço retorna
erro de geração e o backend segue o tratamento de falha, sem inventar um
veredito. Pelo menos uma fonte usada deve estar visível em qualquer resposta
`matched` enviada.

## Disponibilidade, desempenho e falhas

- A API de ML expõe liveness e readiness distintos. Readiness só passa quando
  Qdrant responde, a coleção configurada existe, a dimensão dos vetores é
  compatível com o provedor de embeddings e a configuração exigida está
  presente. Ela não indexa dados automaticamente.
- A API é um processo persistente. Inicializa/reutiliza clientes e modelo
  quando aplicável, afasta chamadas síncronas do event loop FastAPI e limita
  a concorrência de inferências para não esgotar seus recursos. O backend usa
  conexões HTTP reutilizáveis; não faz polling do serviço.
- Prazos da chamada HTTP e da etapa de análise são configurados de forma
  coerente com o lease renovado do worker. Latência e erros serão medidos por
  etapa, sem registrar textos de usuários, laudos ou chaves em logs.
- Falha de rede, timeout e indisponibilidade 5xx/503 são falhas temporárias e
  seguem as tentativas limitadas já existentes no backend. Resposta inválida
  ou incompatível com a versão é rejeitada; entrada 422 não é repetida como se
  fosse falha de rede. Após esgotar tentativas, o usuário recebe a mensagem
  de erro do fluxo existente, sem veredito factual ou indicação de evidência
  insuficiente.
- Uma resposta válida `insufficient_evidence` termina a análise sem nova
  tentativa. Checkpoints impedem repetir uma chamada ao ML após resultado
  persistido; uma queda antes do checkpoint ainda pode repetir a inferência,
  conforme a semântica atual do worker.

## Verificação e critérios de aceite

1. Testes de contrato do ML cobrem entrada válida/inválida, os dois vereditos,
   limites e campos obrigatórios, fontes HTTP(S), 503 de readiness e erros de
   geração sem veredito.
2. Testes do adaptador do backend usam HTTP simulado para verificar mapeamento,
   versão incompatível, timeout, 422, 503/5xx e ausência de fallback simulado.
3. Testes do formatador cobrem 1.000 caracteres, pelo menos uma fonte usada,
   distinção entre URL analisada e fonte de checagem, e resposta fixa de
   evidência insuficiente.
4. Teste de integração percorre webhook → PostgreSQL/outbox → Redis → worker
   → API de ML simulada → checkpoint → outbox de entrega → Evolution simulada.
   Cobre texto, link, citação, reentrega do webhook e retomada sem duplicar a
   resposta.
5. Teste operacional manual usa uma coleção Qdrant já carregada e confirma a
   rede entre os containers, readiness, uma checagem encontrada e uma
   abstenção. Esse teste não mede qualidade factual geral do corpus.

A implementação está concluída quando o backend responde pelo WhatsApp usando
o resultado real da API de ML nos dois vereditos, sem aviso de demonstração no
modo real, mantendo deduplicação e recuperação de falhas do fluxo atual.
