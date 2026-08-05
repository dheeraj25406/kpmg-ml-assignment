## Dataset

The grading model is trained on the **NASA CM1 Software Defect Prediction Dataset**, which contains static software metrics extracted from source code modules. The objective is to predict whether a software module is **defective** based on its code complexity and maintainability metrics. jm1 was also a part of the dataset folder used, but cm1 was preferred as it was smaller and experimentation-friendly.

Dataset was taken from Kaggle.
Dataset link : https://www.kaggle.com/datasets/radowanulhaque/software-defect

### Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | NASA CM1 Software Defect Dataset |
| Total Records | 505 |
| Input Features | 21 |
| Target Column | `defects` |
| Task | Binary Classification |

### Feature Description

The dataset consists of software engineering metrics, including:

| Feature | Description |
|---------|-------------|
| `loc` | Lines of Code |
| `v(g)` | Cyclomatic Complexity |
| `ev(g)` | Essential Complexity |
| `iv(g)` | Design Complexity |
| `n` | Halstead Program Length |
| `v` | Halstead Volume |
| `l` | Halstead Program Level |
| `d` | Halstead Difficulty |
| `i` | Halstead Intelligence |
| `e` | Halstead Effort |
| `b` | Estimated Number of Bugs |
| `t` | Halstead Time Estimate |
| `lOCode` | Logical Lines of Code |
| `lOComment` | Logical Comment Lines |
| `lOBlank` | Blank Lines |
| `locCodeAndComment` | Lines containing both code and comments |
| `uniq_Op` | Unique Operators |
| `uniq_Opnd` | Unique Operands |
| `total_Op` | Total Operators |
| `total_Opnd` | Total Operands |
| `branchCount` | Number of Branches |

### Target Labels

The target variable is **`defects`**.

| Value | Meaning |
|------|---------|
| `0` | Non-defective code |
| `1` | Defective code |

### Data Preprocessing

The following preprocessing steps were performed before training:

- Removed the `id` column.
- Checked for missing values and duplicates.
- Converted the target column (`defects`) to integer labels.
- Performed a stratified train-test split.
- Applied class balancing using `class_weight="balanced"`.

The final model was trained using the processed feature set without feature scaling, as the selected LightGBM model is insensitive to feature scaling.
