"""Optional textual-pattern signal used only after evidence retrieval fails."""

from __future__ import annotations

import logging
from functools import lru_cache

from src.config import (
    CLASSIFIER_BACKEND,
    CLASSIFIER_BERTIMBAU_BATCH_SIZE,
    CLASSIFIER_BERTIMBAU_DEVICE,
    CLASSIFIER_BERTIMBAU_MODEL_PATH,
    CLASSIFIER_ENABLED,
    CLASSIFIER_TFIDF_MODEL_PATH,
)

logger = logging.getLogger("checa-ai.classifier")


def _potential_label(probability_false: float) -> str:
    if probability_false <= 0.20:
        return "potencial muito baixo"
    if probability_false <= 0.40:
        return "potencial baixo"
    if probability_false < 0.60:
        return "potencial 50/50"
    if probability_false <= 0.80:
        return "potencial alto"
    return "potencial muito alto"


@lru_cache(maxsize=1)
def _load_adapter():
    if not CLASSIFIER_ENABLED:
        return None
    try:
        from src.classifier_evaluation.adapters import BertimbauAdapter, TfidfAdapter

        if CLASSIFIER_BACKEND == "tfidf" and CLASSIFIER_TFIDF_MODEL_PATH:
            return TfidfAdapter(CLASSIFIER_TFIDF_MODEL_PATH, "configured")
        if CLASSIFIER_BACKEND == "bertimbau" and CLASSIFIER_BERTIMBAU_MODEL_PATH:
            return BertimbauAdapter(
                CLASSIFIER_BERTIMBAU_MODEL_PATH,
                "configured",
                device=CLASSIFIER_BERTIMBAU_DEVICE,
                batch_size=CLASSIFIER_BERTIMBAU_BATCH_SIZE,
            )
    except Exception:
        logger.exception("Classifier artifact could not be loaded")
    return None


def narrative(text: str) -> str | None:
    """Return a safe display message, or None when classification is unavailable."""
    adapter = _load_adapter()
    if adapter is None:
        return None
    try:
        probability = float(adapter.predict_probability_false([text])[0])
    except Exception:
        logger.exception("Classifier inference failed")
        return None
    percentage = round(probability * 100)
    return (
        f"Foi identificado {_potential_label(probability)} de desinformação: {percentage}%.\n\n"
        "Esse resultado é apenas um sinal auxiliar e não comprova que a alegação seja falsa."
    )
