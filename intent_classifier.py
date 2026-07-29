import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


MODEL_DIR = Path("ml_model")
MODEL_FILE = MODEL_DIR / "intent_model.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

MINIMUM_CONFIDENCE = 0.25


@lru_cache(maxsize=1)
def load_intent_model():
    """
    Loads the trained intent model once and caches it.
    """
    if not MODEL_FILE.exists():
        return None

    return joblib.load(MODEL_FILE)


def clear_model_cache():
    """
    Clears the cached model after retraining.
    """
    load_intent_model.cache_clear()

def get_model_status():
    """
    Returns model availability and training metrics for the dashboard.
    """

    status = {
        "available": MODEL_FILE.exists(),
        "status": "Ready" if MODEL_FILE.exists() else "Not trained",
        "accuracy": None,
        "cross_validation_accuracy": None,
        "cross_validation_std": None,
        "samples": 0,
        "trained_at": None,
        "classes": [],
    }

    if METRICS_FILE.exists():
        try:
            with open(
                METRICS_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                metrics = json.load(file)

            status["accuracy"] = metrics.get(
                "accuracy"
            )

            status["cross_validation_accuracy"] = metrics.get(
                "cross_validation_accuracy"
            )

            status["cross_validation_std"] = metrics.get(
                "cross_validation_std"
            )

            status["samples"] = metrics.get(
                "samples",
                0,
            )

            status["trained_at"] = metrics.get(
                "trained_at"
            )

            status["classes"] = metrics.get(
                "classes",
                [],
            )

        except (OSError, json.JSONDecodeError):
            pass

    return status
    
    
def classify_question(question):
    """
    Predicts a question intent using a trained TF-IDF
    and Logistic Regression model.
    """

    question = question.strip()

    if not question:
        return {
            "intent": "general",
            "confidence": 0.0,
            "model_available": MODEL_FILE.exists(),
            "low_confidence": True,
        }

    try:
        model = load_intent_model()
    except Exception:
        model = None

    if model is None:
        return {
            "intent": "general",
            "confidence": 0.0,
            "model_available": False,
            "low_confidence": True,
        }

    try:
        probabilities = model.predict_proba([question])[0]
        best_index = int(np.argmax(probabilities))

        predicted_intent = str(model.classes_[best_index])
        confidence = float(probabilities[best_index])

        low_confidence = confidence < MINIMUM_CONFIDENCE

        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "model_available": True,
            "low_confidence": low_confidence,
        }

    except Exception:
        return {
            "intent": "general",
            "confidence": 0.0,
            "model_available": False,
            "low_confidence": True,
        }

    