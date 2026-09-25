# Guia do GitHub Spec Kit no checa-ai

Este repositório foi configurado com o [GitHub Spec Kit](https://github.com/github/spec-kit), permitindo que todo o time pratique **Spec-Driven Development (SDD)** com seus agentes de IA favoritos.

Como o nosso time utiliza ferramentas variadas, configuramos o repositório com suporte **multi-agente** compartilhado sobre uma única fonte de verdade.

---

## 🎯 Fonte de Verdade Unificada (`.specify/`)

Independentemente do assistente que cada pessoa utiliza, toda a inteligência e padronização do projeto fica centralizada em:

- **`.specify/memory/constitution.md`**: Princípios inegociáveis do projeto (Arquitetura Hexagonal, Ports & Adapters, Test-First com pytest, desinformação granular e explicabilidade híbrida ML+LLM).
- **`.specify/templates/`**: Modelos padronizados de especificações (`spec.md`), planos técnicos (`plan.md`), listas de tarefas (`tasks.md`) e checklists de validação.
- **`.specify/scripts/`**: Scripts de suporte e automação compartilhados.

---

## 🤖 Como Usar no seu Agente de IA Favorito

O Spec Kit foi configurado com comandos e skills prontas para os principais agentes:

### 1. GitHub Copilot (VS Code / JetBrains)
- As skills estão em `.github/skills/speckit-*`.
- No chat do Copilot, basta digitar:
  - `/speckit-constitution`
  - `/speckit-specify`
  - `/speckit-plan`
  - `/speckit-tasks`
  - `/speckit-implement`
  - `/speckit-converge`

### 2. Cursor
- As skills estão disponíveis em `.cursor/skills/speckit-*`.
- No chat do Cursor (ou Composer), invoque diretamente as skills como `/speckit-<comando>` (ex: `/speckit-specify`).

### 3. Claude Code
- As skills estão disponíveis em `.claude/skills/speckit-*`.
- No terminal do Claude Code, execute `/speckit-<comando>` para guiar o agente através das fases de SDD.

### 4. Antigravity (Google / Gemini)
- As skills estão disponíveis em `.agents/skills/speckit-*`.
- No chat do Antigravity, acione as skills normalmente via `/speckit-<comando>`.

### 5. Qualquer outro Agente ou Manualmente
- Se você ou algum colega usar outro agente (ex: Windsurf, Devin, Aider, Codex, ChatGPT web) ou preferir usar o terminal:
  - O arquivo `.specify/memory/constitution.md` e os templates em `.specify/templates/` podem ser passados diretamente como contexto para o modelo.
  - Você pode usar o CLI oficial via terminal:
    ```bash
    specify check
    specify integration list
    ```

---

## 🔄 Fluxo de Trabalho Recomendado (SDD)

Ao iniciar uma nova funcionalidade, refatoração ou pipeline:

```text
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. /speckit-specify   ──> Descreve o "o quê" e o "porquê"   │
  │ 2. /speckit-clarify   ──> (Opcional) Tira dúvidas e riscos  │
  │ 3. /speckit-plan      ──> Cria o plano técnico e contratos  │
  │ 4. /speckit-checklist ──> (Opcional) Valida critérios       │
  │ 5. /speckit-tasks     ──> Decompõe em passos incrementais   │
  │ 6. /speckit-analyze   ──> (Opcional) Verifica consistência  │
  │ 7. /speckit-implement ──> O agente implementa cada task     │
  │ 8. /speckit-converge  ──> Valida o código contra a spec     │
  └─────────────────────────────────────────────────────────────┘
```

1. **`/speckit-specify`**: Cria a especificação inicial em linguagem clara, definindo requisitos e casos de uso sem se perder em código prematuro.
2. **`/speckit-plan`**: Converte a especificação em um plano de arquitetura técnica (respeitando as portas e adaptadores do `checa-ai`).
3. **`/speckit-tasks`**: Gera uma lista ordenada e atômica de tarefas executáveis.
4. **`/speckit-implement`**: Instrui o agente a codificar as tarefas passo a passo, acompanhado de testes automatizados (`pytest`).
5. **`/speckit-converge`**: Compara o código final implementado com os requisitos iniciais para garantir que nada ficou para trás.

---

## 🛠️ Manutenção do CLI (`specify`)

Se precisar atualizar integrações ou rodar comandos do Spec Kit via linha de comando:

```bash
# Verificar status das integrações instaladas
specify integration list

# Adicionar suporte a um novo agente (ex: zed, trae, gemini)
specify integration install <nome-do-agente> --force

# Verificar saúde das ferramentas
specify check
```

