"""
Preprocessing for the code quality grading model.

The saved LightGBM model (models/code_quality_model.pkl) was trained on the
21 raw (unscaled) numeric code-metric columns from datasets/code-quality/cm1.csv,
in a fixed column order. See notebooks/code-quality.ipynb for the training
pipeline. This module only validates and orders incoming feature dicts so
they match what the model expects. It does not fit or change any values.
"""

# Exact feature order the model was trained on (from cm1.csv, minus id/defects).
FEATURE_COLUMNS = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]


def validate_and_order_features(features: dict) -> list:
    """
    Validate an incoming feature dict and return values in the exact
    column order the trained model expects.

    Args:
        features: dict mapping feature name -> numeric value. Must contain
            all keys listed in FEATURE_COLUMNS.

    Returns:
        List of feature values in FEATURE_COLUMNS order, ready to be
        passed to the model as a single-row input.

    Raises: ValueError if any required feature is missing, or any value is
            not a number.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in features]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")

    ordered_values = []
    for col in FEATURE_COLUMNS:
        value = features[col]
        if value is None:
            raise ValueError(f"Feature '{col}' is missing a value (None).")
        try:
            ordered_values.append(float(value))
        except (TypeError, ValueError):
            raise ValueError(
                f"Feature '{col}' must be numeric, got: {value!r}"
            )

    return ordered_values
