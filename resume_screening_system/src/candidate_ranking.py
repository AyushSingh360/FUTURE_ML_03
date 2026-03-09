import pandas as pd
import numpy as np
from .similarity_scoring import compute_similarity


def rank_candidates(resume_df: pd.DataFrame, job_vector, vectorizer) -> pd.DataFrame:
    """Rank candidates based on similarity scores.
    Args:
        resume_df: DataFrame with at least a 'resume_id' column and pre‑processed text column.
        job_vector: TF‑IDF vector for the job description (1‑D array).
        vectorizer: fitted TF‑IDF vectorizer used to transform resume texts.
    Returns:
        DataFrame with columns ['resume_id', 'score'] sorted descending by score.
    """
    # Transform resume texts
    resume_vectors = vectorizer.transform(resume_df['preprocessed_text'])
    scores = compute_similarity(resume_vectors, job_vector)
    ranking = pd.DataFrame({
        'resume_id': resume_df['resume_id'],
        'score': scores
    })
    ranking = ranking.sort_values('score', ascending=False).reset_index(drop=True)
    return ranking
