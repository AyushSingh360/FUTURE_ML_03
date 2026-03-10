import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_vectorizer(texts: pd.Series) -> TfidfVectorizer:
    """Fit a TF‑IDF vectorizer on a series of pre‑processed texts.
    Returns the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(texts)
    return vectorizer


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series):
    """Transform texts into TF‑IDF vectors using a fitted vectorizer.
    Returns a sparse matrix.
    """
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: Union[str, Path]):
    """Persist the fitted vectorizer to disk using pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(filepath: Union[str, Path]) -> TfidfVectorizer:
    """Load a persisted TF‑IDF vectorizer from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
