# checa-ai Constitution

## Core Principles

### I. Arquitetura Hexagonal & Isolamento de Domínio (Ports and Adapters)
- As regras de negócio e casos de uso residem exclusivamente em `app/application/use_cases/`.
- Comunicações externas (FastAPI, Evolution API / WhatsApp, PostgreSQL, Redis, modelos de ML e APIs de LLM) são tratadas estritamente em `app/adapters/` (inbound/outbound).
- O domínio e a aplicação nunca devem depender diretamente de detalhes de infraestrutura ou bibliotecas de transporte; interações ocorrem sempre através de portas (interfaces) em `app/application/ports/`.

### II. Desenvolvimento Orientado por Especificação (Spec-Driven Development - SDD)
- Nenhuma funcionalidade complexa ou alteração de contrato de pipeline é implementada sem especificação prévia (`.specify/` e `docs/`).
- O ciclo padrão de desenvolvimento segue: **Especificar** (`/speckit-specify`) → **Planejar** (`/speckit-plan`) → **Tarefas** (`/speckit-tasks`) → **Implementar** (`/speckit-implement`) → **Convergência** (`/speckit-converge`).
- Mudanças arquiteturais relevantes devem ser registradas em Architecture Decision Records (ADRs) em `docs/decisions/`.

### III. Classificação Granular & Explicabilidade Híbrida (ML + LLM)
- O `checa-ai` não se limita a vereditos binários ("verdadeiro" vs. "falso"). A desinformação mistura fatos e distorções, exigindo categorização nuanceada e explicabilidade clara para o usuário final.
- Conforme ADR 002, adota-se estratégia híbrida onde modelos de ML/classificadores estruturados e LLMs cooperam para gerar respostas precisas, didáticas e acessíveis via WhatsApp.

### IV. Testes Automatizados e Confiabilidade (Test-First)
- Toda funcionalidade ou correção deve ser acompanhada de testes em `tests/` (`pytest`).
- Manter testes unitários para a camada de aplicação/regras de negócio e testes de integração/contrato para adaptadores (APIs, webhooks, banco).
- A suíte de testes deve passar com sucesso antes de submissão de PRs.

### V. Segurança de Dados, Observabilidade e Boas Práticas Python
- Dados brutos (`data/raw/`) e datasets sensíveis nunca devem ser versionados no Git.
- Segredos, tokens e chaves de API devem ser carregados estritamente via variáveis de ambiente (`.env` / Pydantic Settings).
- Tratamento resiliente de falhas em webhooks assíncronos e chamadas externas, com logs estruturados para facilitar rastreamento e depuração.
- Aderência ao Python 3.11+, type hints explícitos, formatação consistente e boas práticas de código limpo.

## Governança & Uso com Agentes de IA

- Esta constituição estabelece os princípios invioláveis para desenvolvedores humanos e agentes de IA que atuam no repositório.
- Agentes de IA (Copilot, Cursor, Claude Code, Antigravity, etc.) devem consultar esta constituição antes de planejar e implementar qualquer código.
- Qualquer proposta que viole estes princípios deve ser explicitamente justificada e aprovada pelo time.

**Versão**: 1.0.0 | **Ratificado**: 2026-09-24 | **Última Atualização**: 2026-09-24
