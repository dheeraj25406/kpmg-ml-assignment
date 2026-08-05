"""
Inference for the code quality grading model.

Loads the trained LightGBM classifier (models/code_quality_model.pkl) with
joblib and predicts whether a code submission is likely defective, along
with a confidence score for downstream routing.
"""

import os

import joblib
import pandas as pd

from src.grading.preprocess import FEATURE_COLUMNS, validate_and_order_features
from src.triage.routing import route_by_confidence

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "code_quality_model.pkl"
)

_model = None


def _load_model():
    """Load the trained LightGBM model once and cache it."""
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Grading model not found at '{MODEL_PATH}'. "
                "Ensure models/code_quality_model.pkl exists."
            ) from exc
    return _model


def predict_quality(features: dict) -> dict:
    """
    Predict code quality for a single submission.

    Args:
        features: dict with the 21 raw code metrics the model was trained
            on (see src/grading/preprocess.py FEATURE_COLUMNS), e.g.
            {"loc": 45, "v(g)": 6, ..., "branchCount": 10}.

    Returns:
        dict with:
            - prediction: 0 (not defective) or 1 (defective)
            - label: "defective" or "not_defective"
            - confidence: max class probability, float in [0, 1]
            - probabilities: dict mapping class -> probability
            - routing: "auto-approved" or "teacher-review", based on
              confidence (see src/triage/routing.py)

    Raises ValueError if features are missing or non-numeric.
    """
    model = _load_model()
    ordered_values = validate_and_order_features(features)

    # LightGBM was trained on raw (unscaled) features, so no scaling here.
    # Wrapped in a DataFrame with the original column names so the model
    # sees matching feature names (avoids a sklearn feature-name warning
    # and keeps input/training preprocessing consistent).
    input_df = pd.DataFrame([ordered_values], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(input_df)[0]
    class_labels = model.classes_

    prob_by_class = {
        str(cls): float(prob) for cls, prob in zip(class_labels, probabilities)
    }
    prediction = int(class_labels[probabilities.argmax()])
    confidence = float(probabilities.max())

    defect_probability = prob_by_class["1"]

    quality_score = round((1 - defect_probability) * 100, 2)

    return {
        "prediction": prediction,
        "label": "defective" if prediction == 1 else "not_defective",
        "quality_score": quality_score,
        "defect_probability": round(defect_probability, 4),
        "confidence": round(confidence, 4),
        "probabilities": prob_by_class,
        "routing": route_by_confidence(confidence),
    }
