"""
FastAPI application exposing the code quality grading and doubt triage
models for inference.

Run with:
    uvicorn app.main:app --reload

Endpoints:
    POST /predict/grading  - predict code submission quality
    POST /predict/triage   - classify a student doubt's quality/urgency
"""

import sys
from pathlib import Path

# Allow running "uvicorn app.main:app" from the project root without
# installing the project as a package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.grading.predict import predict_quality
from src.triage.predict import predict_triage

app = FastAPI(
    title="LMS ML Pipeline API",
    description="Code quality grading and student doubt triage inference.",
    version="1.0.0",
)


class TriageRequest(BaseModel):
    """Raw doubt text submitted by a student."""

    text: str


@app.get("/")
def root():
    """Basic health check."""
    return {"status": "ok", "service": "LMS ML Pipeline API"}


@app.post("/predict/grading")
def predict_grading(payload: dict):
    """
    Predict code submission quality (defect risk) from raw code metrics.

    Request body must be a JSON object with all 21 feature keys used
    during training:
        "loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e",
        "b", "t", "lOCode", "lOComment", "lOBlank", "locCodeAndComment",
        "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"

    A plain dict is used here (instead of a typed Pydantic model) because
    several feature names, e.g. "v(g)", are not valid Python identifiers.
    """
    try:
        result = predict_quality(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


@app.post("/predict/triage")
def predict_triage_endpoint(payload: TriageRequest):
    """
    Classify a student doubt's quality/urgency category from its text.
    """
    try:
        result = predict_triage(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result
