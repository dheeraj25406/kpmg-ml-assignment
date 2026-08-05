"""
Inference for the doubt triage model.

Loads the trained TF-IDF vectorizer (models/tfidf_vectorizer.pkl) and
Logistic Regression classifier (models/doubt_triage_model.pkl) with
joblib, and classifies a student doubt's text quality/urgency category.
"""

import os

import joblib

from src.triage.preprocess import clean_text
from src.triage.routing import route_by_confidence

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "doubt_triage_model.pkl")

_vectorizer = None
_model = None

LABEL_MAP = {
    "HQ": "High Quality Doubt",
    "LQ_EDIT": "Needs More Details",
    "LQ_CLOSE": "Low Quality / Close"
}

def _load_artifacts():
    """Load the TF-IDF vectorizer and classifier once and cache them."""
    global _vectorizer, _model
    if _vectorizer is None:
        try:
            _vectorizer = joblib.load(VECTORIZER_PATH)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at '{VECTORIZER_PATH}'."
            ) from exc
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Triage model not found at '{MODEL_PATH}'."
            ) from exc
    return _vectorizer, _model


def predict_triage(text: str) -> dict:
    """
    Classify a student doubt's quality/urgency category.

    Args:
        text: raw doubt text (like question title + body).

    Returns:
        dict with:
            - prediction: predicted class label, one of
              "HQ" (high quality), "LQ_EDIT" (low quality, needs edit),
              "LQ_CLOSE" (low quality, likely to be closed)
            - confidence: max class probability, float in [0, 1]
            - probabilities: dict mapping class -> probability
            - routing: "auto-approved" or "teacher-review", based on
              confidence (see src/triage/routing.py)

    Raises ValueError if text is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input 'text' must be a non-empty string.")

    vectorizer, model = _load_artifacts()

    cleaned = clean_text(text)
    tfidf_features = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(tfidf_features)[0]
    class_labels = model.classes_

    prob_by_class = {
        str(cls): float(prob) for cls, prob in zip(class_labels, probabilities)
    }
    prediction = str(class_labels[probabilities.argmax()])
    confidence = float(probabilities.max())

    return {
        "prediction": prediction,
        "category": LABEL_MAP.get(prediction, prediction),
        "confidence": round(confidence, 4),
        "probabilities": prob_by_class,
        "routing": route_by_confidence(confidence),
    }
