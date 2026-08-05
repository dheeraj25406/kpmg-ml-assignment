"""
Preprocessing for the doubt triage model.

Replicates the exact text-cleaning function used in
notebooks/doubt-triage.ipynb when the TF-IDF vectorizer and classifier
were fit. This function must stay identical to the notebook version,
otherwise the fitted vocabulary/IDF weights in models/tfidf_vectorizer.pkl
will no longer match the text seen at inference time.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean raw doubt text the same way training data was cleaned.

    Steps: strip HTML tags, strip URLs, lowercase, remove punctuation,
    remove digits, collapse extra whitespace.

    Args:
        text: raw input text (title + body + tags, or any free text).

    Returns Cleaned text string
    """
    text = str(text)

    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
