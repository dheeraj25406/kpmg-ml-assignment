"""
Confidence-based routing for model predictions.

Applies a single, simple rule to any prediction that carries a confidence
score: if the model is confident enough, auto-approve its decision;
otherwise send it to a teacher for manual review. Used by both the
grading and triage inference paths.
"""

# minimum confidence required to auto-approve a prediction.
# Kept as a single configurable constant so both grading and triage
# routing use the same, easily-tunable threshold.
# here we are using confidence threshold of 0.75, so if the model's confidence
# is 75% or higher, we auto-approve the prediction; otherwise, we send it for
# teacher review.

# Confidence threshold value can be adjusted based on experimentation and 
# model performance to get the desired trade-off

CONFIDENCE_THRESHOLD = 0.75


def route_by_confidence(confidence: float) -> str:
    """
    Decide whether a prediction can be auto-approved or needs review.

    Args:
        confidence: the model's max class probability, float in [0, 1].

    Returns:
        "auto-approved" if confidence >= CONFIDENCE_THRESHOLD,
        otherwise "teacher-review".

    Raises:
        ValueError: if confidence is not within [0, 1].
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be between 0 and 1, got: {confidence}")

    if confidence >= CONFIDENCE_THRESHOLD:
        return "auto-approved"
    return "teacher-review"
