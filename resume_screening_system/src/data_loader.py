"""
data_loader.py
--------------
Functions to load resume and job description datasets.

Supports two resume formats:
  1. Kaggle `snehaanbhawal/resume-dataset`
     Columns: ID, Resume_str, Resume_html, Category
  2. Custom placeholder CSV
     Columns: resume_id, resume_text, job_role, skills, experience
"""

from pathlib import Path
import pandas as pd


def load_kaggle_resumes(dataset_slug: str = "snehaanbhawal/resume-dataset") -> pd.DataFrame:
    """Download and load the Kaggle Resume Dataset using kagglehub.
    Returns a normalised DataFrame with columns:
        resume_id, resume_text, job_role, skills, experience
    """
    import kagglehub
    path = kagglehub.dataset_download(dataset_slug)
    print(f"Dataset downloaded to: {path}")
    # The dataset contains a single CSV file
    csv_files = list(Path(path).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}")
    df = pd.read_csv(csv_files[0])
    # Rename / normalise columns
    df = df.rename(columns={
        "ID": "resume_id",
        "Resume_str": "resume_text",
        "Category": "job_role",
    })
    # Add placeholder columns so the rest of the pipeline works unchanged
    if "skills" not in df.columns:
        df["skills"] = ""
    if "experience" not in df.columns:
        df["experience"] = ""
    return df[["resume_id", "resume_text", "job_role", "skills", "experience"]]


def load_resumes(csv_path: str | Path) -> pd.DataFrame:
    """Load resumes CSV into a DataFrame.
    Expected columns: resume_id, resume_text, job_role, skills, experience
    """
    return pd.read_csv(csv_path)


def load_job_descriptions(csv_path: str | Path) -> pd.DataFrame:
    """Load job descriptions CSV into a DataFrame.
    Expected columns: job_title, job_description, required_skills
    """
    return pd.read_csv(csv_path)
