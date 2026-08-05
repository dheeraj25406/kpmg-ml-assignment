# Notebooks

This directory contains the Jupyter notebooks used during model development and experimentation.

The notebooks include the complete machine learning workflow, from exploratory data analysis to model training, evaluation, comparison, and model serialization.

## Contents

### `code-quality.ipynb`

Implements the software defect prediction pipeline.

**Workflow**
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Train/Test split
- Baseline Logistic Regression model
- LightGBM model
- Model comparison
- Hyperparameter tuning experiments
- Model evaluation
- Save trained model (`code_quality_model.pkl`)

---

### `doubt-triage.ipynb`

Implements the NLP-based doubt triage pipeline.

**Workflow**
- Exploratory Data Analysis (EDA)
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Baseline Multinomial Naive Bayes model
- Logistic Regression classifier
- Model comparison
- Evaluation on the held-out test set
- Save trained artifacts (`doubt_triage_model.pkl` and `tfidf_vectorizer.pkl`)

---

## Purpose

The notebooks were used only during the experimentation and training phase.

The production inference pipeline is implemented separately under the `src/` directory, where the saved models are loaded to make predictions without retraining.
