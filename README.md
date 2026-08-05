# LMS ML-Based Grading & Doubt Triage Pipeline

## Overview

This is an implementation of a ML pipeline for an LMS to automate two important tasks:

1. **Code Submission Quality Prediction**
   - Predicts whether a submitted code sample is likely to contain defects using standard complexity metrics.
   - Provides a confidence score and decides whether the submission can be automatically approved or needs manual review.

2. **Student Doubt Triage**
   - Classifies student questions based on their quality and urgency.
   - Uses NLP techniques to identify whether a doubt is high quality or requires improvement.
   - Uses prediction confidence to simulate automatic routing versus teacher review.

The objective was to design a complete ML workflow including preprocessing, feature engineering, model evaluation, confidence-based routing, and deployment through an API.

---

## Project Structure

```text
kpmg-ml-assignment/
│
├── app/
│   └── main.py                  # FastAPI application
│
├── datasets/
│   ├── code-quality/
│   └── doubt-triage/
│
├── models/
│   ├── code_quality_model.pkl
│   ├── doubt_triage_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── code-quality.ipynb
│   └── doubt-triage.ipynb
│
├── src/
│   ├── grading/
│   │   ├── preprocess.py     # feature validation and preprocessing
│   │   ├── predict.py        # code quality prediction pipeline
│   │
│   └── triage/
│       ├── preprocess.py.   # text cleaning pipeline
│       ├── predict.py       # doubt classification pipeline
│       ├── routing.py       # confidence based routing logic
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 1. Code Quality Prediction

## Dataset

The grading model uses metrics from the NASA software defect prediction dataset.
The dataset was taken from Kaggle.

Features include:

- Lines of code (`loc`)
- Cyclomatic complexity (`v(g)`)
- Halstead metrics
- Operator and operand statistics
- Branch count
- Code/comment statistics

The target variable:

```
defects
```

represents whether a code submission is likely to contain defects.

---

## Approach

### Data Processing

- Removed unnecessary identifiers
- Checked missing values and duplicates
- Performed stratified train-test split
- Handled class imbalance using class weights

### Models Tested

Baseline:

- Logistic Regression

Final Model:

- LightGBM Classifier

LightGBM was selected because it performs well on structured tabular data and captures non-linear relationships between software metrics and defect probability.

---

## Prediction Output

The model returns:

- Defect prediction
- Label, either defective or not defective
- Quality score out of 100
- Probability of defect
- Confidence score
- Probabilities
- Automated routing decision


Example:

```json
{
  "prediction": 0,
  "label": "not_defective",
  "quality_score": 99.9,
  "defect_probability": 0.001,
  "confidence": 0.999,
  "probabilities": {
    "0": 0.9989879420894656,
    "1": 0.0010120579105344103
  },
  "routing": "auto-approved"
}
```

---

# 2. Student Doubt Triage

## Dataset

The NLP classifier uses Stack Overflow question quality data.
The dataset was taken from Kaggle.

Input fields:

- Question title
- Question body
- Tags

Classes:

| Label | Meaning |
|---|---|
| HQ | High Quality |
| LQ_EDIT | Low Quality but can be improved |
| LQ_CLOSE | Low Quality and likely should be closed |

---

## Text Processing

The pipeline performs:

- HTML removal
- URL removal
- Lowercasing
- Punctuation cleaning
- Whitespace normalization

Feature extraction:

- TF-IDF Vectorization
- Maximum vocabulary size: 20,000
- Unigrams and bigrams

---

## Model

Models evaluated:

- Multinomial Naive Bayes
- Logistic Regression

Final model:

```
Logistic Regression + TF-IDF
```

Logistic Regression was selected due to better classification performance on sparse text features.
Logistic Regression outperformed Multinomial Naive Bayes in all the standard metrics.

---

## Prediction Output

Example:

```json
{
  "prediction": "LQ_EDIT",
  "category": "Needs More Details",
  "confidence": 0.8671,
  "probabilities": {
    "HQ": 0.037429976260330414,
    "LQ_CLOSE": 0.09543542622737622,
    "LQ_EDIT": 0.8671345975122933
  },
  "routing": "auto-approved"
}
```

---

# Confidence-Based Routing

The system uses prediction confidence to decide whether a prediction can be automated.

Current threshold:

```
confidence >= 0.75
```

I felt that a threshold of 0.75 would be optimal, but it can be experimented to get the right trade-off.

Routing:

| Confidence | Decision |
|---|---|
| >= 0.75 | Auto-approved |
| < 0.75 | Teacher review |

The threshold was chosen to balance automation and reliability by avoiding automatic decisions when the model is uncertain.

---

# API Deployment

The project exposes predictions through FastAPI.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```
Make sure to download the Doubt-Triage dataset, following the instructions mentioned in `datasets/doubt-triage/readme.md` before starting the server.

Start server:

```bash
uvicorn app.main:app --reload
```

API available at:

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Health Check

```
GET /
```

Response:

```json
{
  "status": "ok",
  "service": "LMS ML Pipeline API"
}
```

---

## Code Quality Prediction

```
POST /predict/grading
```

Input:

JSON object containing the 21 software metric features.

```json
{
  "loc": 50,
  "v(g)": 5,
  "ev(g)": 3,
  "iv(g)": 2,
  "n": 100,
  "v": 400,
  "l": 5,
  "d": 10,
  "i": 30,
  "e": 4000,
  "b": 0.02,
  "t": 200,
  "lOCode": 40,
  "lOComment": 5,
  "lOBlank": 5,
  "locCodeAndComment": 45,
  "uniq_Op": 10,
  "uniq_Opnd": 15,
  "total_Op": 50,
  "total_Opnd": 50,
  "branchCount": 5
}
```
<img width="1456" height="724" alt="Screenshot 2026-08-05 at 10 48 15 AM" src="https://github.com/user-attachments/assets/abeafa7b-38f1-41d5-a5d3-9e0627edfc80" />

Output:

```json
{
  "prediction": 0,
  "label": "not_defective",
  "quality_score": 99.9,
  "defect_probability": 0.001,
  "confidence": 0.999,
  "probabilities": {
    "0": 0.9989879420894656,
    "1": 0.0010120579105344103
  },
  "routing": "auto-approved"
}
```
<img width="1392" height="478" alt="Screenshot 2026-08-05 at 10 48 48 AM" src="https://github.com/user-attachments/assets/bd8fa331-86cf-407f-a5b2-91e710bb8bf1" />

---

## Doubt Triage Prediction

```
POST /predict/triage
```

Input:

```json
{
  "text": "Python segmentation error while running code"
}
```

<img width="1434" height="716" alt="Screenshot 2026-08-05 at 10 49 05 AM" src="https://github.com/user-attachments/assets/cce90928-0adb-4637-a242-3a50210eeaf4" />


Output:

```json
{
  "prediction": "LQ_EDIT",
  "category": "Needs More Details",
  "confidence": 0.8671,
  "probabilities": {
    "HQ": 0.037429976260330414,
    "LQ_CLOSE": 0.09543542622737622,
    "LQ_EDIT": 0.8671345975122933
  },
  "routing": "auto-approved"
}
```

<img width="1447" height="663" alt="Screenshot 2026-08-05 at 10 49 17 AM" src="https://github.com/user-attachments/assets/4d7a6436-b14a-471f-a199-3abe52f82074" />


---

# Reproducibility

The trained models are saved using `joblib`:

- LightGBM model
- Logistic Regression classifier
- TF-IDF vectorizer

Inference uses the same preprocessing logic used during training to ensure consistency.

---

# Key ML Practices Followed

- Stratified train-test splitting
- Leakage prevention by fitting transformations only on training data
- Class imbalance handling
- Baseline comparison
- Confidence calibration and routing logic
- Separation of training and inference pipelines
- API-based model serving

---

# Future Improvements

Possible improvements:

- Model monitoring after deployment
- Automated threshold tuning using validation data
- More advanced NLP models (BERT/LLMs)
- Explainable AI using SHAP for grading predictions
- Continuous retraining pipeline
