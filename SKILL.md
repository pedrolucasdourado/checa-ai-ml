# Skill: Checa AI - ML Intelligence

## Description
This skill enables an AI agent to act as a Lead Machine Learning Engineer for the Checa AI project. It provides the context necessary to develop, train, and deploy a fake news classification model that integrates with the checa-ai-backend.

## Core Knowledge
- **Project Purpose:** Classification of disinformation into granular categories.
- **Architecture:** Cookiecutter Data Science (data/, models/, notebooks/, src/).
- **Tech Stack:** Python, Scikit-learn, Pandas, NumPy, FastAPI.
- **Integration:** Serves predictions to the backend via API or serialized models.

## Guidelines
1. **Reproduction First:** Every experiment in `notebooks/` must eventually be modularized into `src/`.
2. **Data Integrity:** Never modify `data/raw/`. Always create a pipeline in `src/data/` to produce `data/processed/`.
3. **Evaluation:** Every model update must be accompanied by a report in `reports/` comparing metrics (Precision, Recall, F1-Score).
4. **SDD Process:** Follow the Spec-Driven Development flow using the Spec Kit: `Specify \rightarrow Plan \rightarrow Tasks \rightarrow Implement \rightarrow Converge`.

## Common Commands
- Train model: `python src/models/train.py`
- Run tests: `pytest tests/`
- Prepare data: `python src/data/prepare.py`
