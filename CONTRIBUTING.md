# Contribuindo para o Checa AI - ML

Bem-vindo ao repositório de Machine Learning do Checa AI! Para mantermos a qualidade e a reprodutibilidade do modelo, seguimos um fluxo rigoroso de contribuição.

## 🛠️ Fluxo de Trabalho (Git Flow)

**A branch `main` é protegida.** Não realize pushes diretos para a `main`.

### Passo a Passo para Contribuições:

1. **Crie uma Branch de Trabalho:**
   Sempre crie uma branch descritiva para sua alteração:
   - `feat/nome-da-funcionalidade` (Novas funcionalidades)
   - `fix/correcao-do-bug` (Correções)
   - `docs/atualizacao-documentacao` (Mudanças em docs ou README)
   - `chore/ajustes-de-ambiente` (Dependências, .gitignore, etc)

2. **Siga o Ciclo SDD (Spec-Driven Development):**
   Se estiver implementando algo novo, use o **Spec Kit** configurado no projeto:
   `Specify \rightarrow Plan \rightarrow Tasks \rightarrow Implement \rightarrow Converge`

3. **Valide suas Alterações:**
   - Execute os testes unitários: `pytest tests/`
- Certifique-se de que as alterações no código estão seguindo as diretrizes da [Constituição do Projeto](.specify/memory/constitution.md).

4. **Abra um Pull Request (PR):**
   - Faça o push da sua branch: `git push origin nome-da-sua-branch`
   - Abra a PR no GitHub detalhando o que foi feito.
   - Aguarde a revisão de ao menos um integrante do time.

5. **Merge:**
   Após a aprovação, a alteração será mergeada na `main` e a branch de feature será excluída.

---

## 📚 Padrões de Código
- **Python 3.11+** com type hints.
- **Padrão de Pastas:** Siga rigorosamente a estrutura do Cookiecutter Data Science.
- **Dados:** Nunca suba arquivos da pasta `data/raw` para o Git.
