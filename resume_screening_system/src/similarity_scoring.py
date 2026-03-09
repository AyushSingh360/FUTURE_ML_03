from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def compute_similarity(resume_vectors, job_vector) -> np.ndarray:
    """Compute cosine similarity between each resume vector and a single job description vector.
    Returns a 1‑D array of similarity scores.
    """
    # Ensure job_vector is 2‑D for sklearn
    job_vec_2d = job_vector.reshape(1, -1)
    similarities = cosine_similarity(resume_vectors, job_vec_2d).flatten()
    return similarities
