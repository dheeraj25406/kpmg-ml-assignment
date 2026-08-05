Dataset could not be uploaded to github as it threw secrets exposed error.
It might have considered some of the data in the dataset as secret exposure.

So, I have ran it locally, and uploading the dataset link here.
Dataset used is 60k Stack Overflow Questions with Quality Rating from Kaggle.

Dataset Link : https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate

To run it locally, please download the dataset and place it datasets/doubt-triage


Other datasets studied:

- EdNet dataset - contains over 131 million interactions, too huge
- Stack Overflow complete dataset - needed to be dowloaded from google Bigquery as it is also too huge
- Moodle forums - unable to find


Hence, 60k Stack Overflow Questions with Quality Rating was considered. 

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

### Dataset Overview

The doubt triage model is trained on a dataset containing **45,000 Stack Overflow questions**. Each record consists of the question title, body, tags, creation date, and a quality label used for classification.

| Column | Description | Data Type |
|--------|-------------|----------|
| `Id` | Unique identifier for each question | `int64` |
| `Title` | Title of the question | `string` |
| `Body` | Detailed question content | `string` |
| `Tags` | Associated Stack Overflow tags | `string` |
| `CreationDate` | Date when the question was posted | `string` |
| `Y` | Target quality label (`HQ`, `LQ_EDIT`, `LQ_CLOSE`) | `string` |

**Dataset Summary**

- **Total Records:** 45,000, 15,000 each of HQ, LQ_EDIT and LQ_CLOSE
- **Features:** 5 input columns
- **Target Column:** `Y`
- **Missing Values:** None
- **Repeated records:** None
