## Dataset

The doubt triage model is trained on the **60k Stack Overflow Questions with Quality Rating** dataset from Kaggle. This dataset contains real-world programming questions along with manually curated quality labels, making it suitable for building an automated doubt triage system for an LMS.

### Dataset Source

**Kaggle:** 60k Stack Overflow Questions with Quality Rating

https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate

> **Note:** The dataset is not included in this repository because GitHub's Push Protection flagged one of the text entries as a potential secret. To reproduce the project, download the dataset from the above link and place `train.csv` inside:
>
> ```
> datasets/doubt-triage/
> ```

---

## Why This Dataset?

Several datasets were evaluated before selecting this one:

| Dataset | Reason Not Selected |
|---------|---------------------|
| EdNet | Extremely large (130M+ interactions), unsuitable for a 24-hour assignment |
| Complete Stack Overflow Dump | Requires Google BigQuery and extensive preprocessing |
| Moodle Forum Datasets | Limited availability and inconsistent labeling |

The **60k Stack Overflow Questions with Quality Rating** dataset was chosen because it:

- Contains real programming-related questions.
- Provides pre-labeled quality classes.
- Closely resembles student programming doubts submitted on an LMS.
- Requires minimal preprocessing while remaining representative of real-world queries.
- Is well-suited for demonstrating an end-to-end NLP classification pipeline.

---

## Input Features

Each record contains the following information:

| Column | Description | Data Type |
|--------|-------------|----------|
| `Id` | Unique identifier | `int64` |
| `Title` | Question title | `string` |
| `Body` | Detailed question description | `string` |
| `Tags` | Associated Stack Overflow tags | `string` |
| `CreationDate` | Date of question creation | `string` |
| `Y` | Target quality label | `string` |

---

## Target Classes

| Label | Meaning |
|------|---------|
| **HQ** | High-quality question |
| **LQ_EDIT** | Low-quality question that can be improved with additional details |
| **LQ_CLOSE** | Low-quality question that is unlikely to be useful and is suitable for closure |

---

## Dataset Summary

| Property | Value |
|----------|-------|
| Original Dataset Size | ~60,000 questions |
| Records Used | 45,000 |
| Class Distribution | 15,000 per class (balanced) |
| Number of Input Features | 5 |
| Target Column | `Y` |
| Missing Values | None |
| Duplicate Records | None |

---

## Preprocessing

The following preprocessing steps were applied before model training:

- Combined **Title**, **Body**, and **Tags** into a single text field.
- Removed HTML tags and URLs.
- Converted text to lowercase.
- Removed punctuation and numeric characters.
- Normalized whitespace.
- Generated TF-IDF features using unigrams and bigrams (`max_features=20,000`).
