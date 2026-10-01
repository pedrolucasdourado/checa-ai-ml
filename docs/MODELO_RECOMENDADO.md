# Diretriz Metodológica: Soluções e Limitações para Fake News com BERTimbau no FakeRecogna

Este documento sumariza as decisões de modelagem, soluções viáveis de engenharia de dados e arquitetura, bem como as limitações intrínsecas a serem consideradas na implementação do classificador de Fake News usando **BERTimbau** sobre o corpus **FakeRecogna**.

---

## 1. Tratamento da Entrada e a Limitação dos 512 Tokens

O checkpoint pré-treinado do BERTimbau possui um limite nativo de 512 subwords no seu tensor de posições absolutas (`max_position_embeddings = 512`). As amostras completas da coluna `News` frequentemente excedem esse limite.

### Solução A: Classificação Focada na Alegação (`Title` ou `Title + Sub-title`)
* **Abordagem:** Alimentar o modelo unicamente com a coluna `Title` ou com a concatenação `Title + " [SEP] " + Sub-title`. No FakeRecogna, o título das amostras falsas atua como a **alegação central (*claim*)** resumida pelas agências.
* **Vantagens:**
  * Comprimento raramente excede 60 tokens, eliminando truncamento de informação.
  * Treinamento computacionalmente mais leve (menor consumo de VRAM e menor tempo por época).
  * Isola o texto dos ruídos estruturais e assinaturas editoriais presentes no corpo das matérias.
* **Limitações:**
  * Desconsidera o corpo da notícia. Em cenários reais onde a manchete é neutra e o conteúdo enganoso reside no corpo do texto, o modelo perde o poder discriminativo.

### Solução B: Truncamento Estratégico (*Head + Tail*) no Corpo (`News`)
* **Abordagem:** Preservar a janela de 512 tokens concatenando os primeiros 128 tokens (*lead* jornalístico) com os últimos 382 tokens (conclusão/apelo), reservando os 2 slots para `[CLS]` e `[SEP]`.
* **Vantagens:**
  * Captura os pontos de maior densidade retórica de notícias falsas (o gancho inicial e a chamada à ação final) sem alterar a dimensão de entrada do Transformer.
* **Limitações:**
  * O desenvolvimento intermediário do texto é descartado. Se a inconsistência ou falsidade estiver exclusivamente no miolo da matéria, o modelo sofrerá sub-representação.

### Solução C: Segmentação com Agregação Diferenciável (*Sliding Window Chunking*)
* **Abordagem:** Dividir o texto `News` em blocos contínuos de até 512 tokens com sobreposição (passo de 64 tokens). Extrair as representações de cada bloco e agregá-las via camada de *Attention Pooling* ou *Max Pooling*.
* **Vantagens:**
  * Cobertura de 100% da matéria, sem perda de parágrafos.
  * Padrão metodológico do estado da arte para documentos extensos.
* **Limitações:**
  * Multiplica o número de *forward passes* pela quantidade de chunks gerados, aumentando significativamente a latência e o tempo de treinamento.

---

## 2. Higienização Textual e Mitigação de Vazamento (*Data Leakage*)

As amostras falsas do FakeRecogna derivam de 6 agências de checagem (*Boatos.org*, *Fato ou Fake*, *E-farsas*, *UOL Confere*, *AFP Checamos*, *Projeto Comprova*), enquanto as legítimas derivam de portais como *G1*, *UOL* e *Ministério da Saúde*.

* **Soluções Recomendadas:**
  * Utilizar como base o arquivo **`FakeRecogna_no_removal_words.xlsx`** para preservar a coesão léxica e a sintaxe natural exigidas pela autoatenção do BERTimbau.
  * Aplicar pipeline de sanitização via expressões regulares (*regex*) voltado especificamente a:
    * Remover jargões e chamadas dos checadores (*"Boatos.org"*, *"segundo apurou o E-farsas"*, *"Verdadeiro ou Falso?"*, *"Conclusão: é boato"*).
    * Remover metadados e créditos de portais tradicionais (*"Por G1 — Brasília"*, *"Foto: TV Globo"*, links e botões de compartilhamento social).
* **Limitações:**
  * A limpeza via regras determinísticas não elimina 100% dos resíduos estilísticos editoriais das redações, permanecendo um risco residual de o modelo memorizar padrões de escrita específicos em vez de traços gerais de desinformação.

---

## 3. Delimitação Metodológica: Classificação Estilística vs. Fact-Checking

O FakeRecogna é um dataset desemparelhado: não há correspondência direta 1-para-1 entre o boato da classe falsa e uma matéria desmentindo aquele fato específico na classe verdadeira.

* **Diretriz de Escopo:**
  * O escopo do modelo deve ser delimitado formalmente como **Classificação Semântico-Discursiva / Estilométrica** (identificação de linguagem sensacionalista, apelos retóricos e padrões sintáticos de desinformação), e não como verificação factual em tempo real (*Grounding/Fact-Checking*).
* **Validação Fora de Domínio (*Out-of-Domain Evaluation*):**
  * Utilizar a coluna `Category` (*Health*, *Politics*, *Entertainment*, *Brazil*, *Science*, *World*) para testar generalização cruzada.
  * Estratégia: treinar em subconjuntos temáticos específicos (ex.: Política e Entretenimento) e avaliar a métrica F1 em categorias retidas (ex.: Saúde), atestando que o classificador não depende de vocabulário conjuntural para separar as classes.
* **Limitações:**
  * Textos fraudulentos com alta sofisticação formal (que copiam a sobriedade jornalística neutra em terceira pessoa) tendem a gerar falsos negativos, pois o modelo não acessa bases factuais externas de validação.

---

## Matriz Resumo de Decisões Técnicas

| Frente de Decisão | Abordagem Proposta | Principal Vantagem | Principal Ponto de Atenção |
| :--- | :--- | :--- | :--- |
| **Entrada Rápida** | `Title` (Alegação) | Elimina o gargalo dos 512 tokens e acelera o treino | Não analisa matérias com manchetes neutras |
| **Entrada Completa** | `News` com Head+Tail ou Chunking | Avalia o conteúdo denso da matéria | Maior custo de processamento / perda do miolo (se Head+Tail) |
| **Sanitização** | Regex sobre `FakeRecogna_no_removal_words` | Mantém stop-words preservando a atenção do Transformer | Risco de vazamento por jargões remanescentes |
| **Validação** | Estratificada por `Category` (*Out-of-Domain*) | Prova capacidade de generalização temática | Queda esperada nas métricas frente ao split puramente aleatório |