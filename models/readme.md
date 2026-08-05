# Models

This directory contains the trained machine learning models and preprocessing artifacts generated during training.

These files are loaded directly by the inference pipeline in `src/` and are **not retrained** during prediction.

## Files

| File | Description |
|------|-------------|
| `code_quality_model.pkl` | Trained LightGBM classifier used for software defect prediction. |
| `doubt_triage_model.pkl` | Trained Logistic Regression classifier used for doubt quality classification. |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer fitted on the training dataset. Used to convert incoming text into numerical feature vectors before classification. |
| `scaler.pkl` | StandardScaler fitted during experimentation for the baseline Logistic Regression model in the code quality notebook. Included for reproducibility but not used by the deployed LightGBM inference pipeline. |

---

## Model Loading

All artifacts are serialized using **Joblib** and should be loaded using:

```python
import joblib

model = joblib.load("models/code_quality_model.pkl")
```

---

## Model Summary

### Code Quality Model

- Model: **LightGBM Classifier**
- Task: Binary classification
- Output:
  - `0` → Not Defective
  - `1` → Defective
- Features: 21 software code metrics

---

### Doubt Triage Model

- Model: **Logistic Regression**
- Feature Extractor: **TF-IDF Vectorizer**
- Task: Multi-class text classification
- Classes:
  - `HQ` → High Quality
  - `LQ_EDIT` → Needs More Details
  - `LQ_CLOSE` → Low Quality / Should Be Closed

---

## Notes

- These models were trained in the notebooks located in the `notebooks/` directory.
- The inference code in `src/` only loads these artifacts and performs prediction.
- No model retraining occurs during inference.
