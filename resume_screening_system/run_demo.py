"""
run_demo.py
-----------
End-to-end demo of the AI Resume Screening pipeline.
Downloads the Kaggle snehaanbhawal/resume-dataset automatically via kagglehub.

Usage:
    python run_demo.py
"""

import os
import sys

# Allow importing from src/ without installing as a package
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")      # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_kaggle_resumes, load_job_descriptions
from text_preprocessing import preprocess
from skill_extraction import extract_skills_from_dataframe, SKILL_LIST
from vectorization import fit_vectorizer, transform_texts, save_vectorizer
from similarity_scoring import compute_similarity
from skill_gap_analysis import identify_missing_skills


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
JOB_DESC_CSV = os.path.join(BASE_DIR, "data", "job_descriptions.csv")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

# How many resumes to process (set to None to use the full dataset)
MAX_RESUMES = 100


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1 – Loading resume dataset from Kaggle …")
resumes = load_kaggle_resumes("snehaanbhawal/resume-dataset")
if MAX_RESUMES:
    resumes = resumes.head(MAX_RESUMES).copy()
resumes["resume_id"] = resumes["resume_id"].astype(str)
print(f"  Loaded {len(resumes)} resumes. Roles found: {resumes['job_role'].nunique()}")

print("Step 1b – Loading job descriptions …")
jobs = load_job_descriptions(JOB_DESC_CSV)
print(f"  Loaded {len(jobs)} job descriptions.")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 2 – Preprocessing resume text …")
resumes["preprocessed_text"] = resumes["resume_text"].fillna("").apply(preprocess)
jobs["preprocessed_text"] = jobs["job_description"].fillna("").apply(preprocess)
print("  Done.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. SKILL EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 3 – Extracting skills …")
resume_skill_sets = extract_skills_from_dataframe(resumes, text_column="resume_text")
resumes["extracted_skills"] = [";".join(sorted(s)) for s in resume_skill_sets]
print("  Sample skills:", resume_skill_sets[:3])


# ──────────────────────────────────────────────────────────────────────────────
# 4. VECTORIZATION (TF-IDF)
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 4 – Fitting TF-IDF vectorizer …")
corpus = pd.concat([resumes["preprocessed_text"], jobs["preprocessed_text"]])
vectorizer = fit_vectorizer(corpus)
save_vectorizer(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
resume_vectors = transform_texts(vectorizer, resumes["preprocessed_text"])
print(f"  Vectorizer trained. Vocabulary size: {len(vectorizer.vocabulary_)}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. SIMILARITY SCORING & RANKING (per job)
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 5 – Computing cosine similarity scores …")
all_rankings = []

for _, job in jobs.iterrows():
    job_vec = transform_texts(vectorizer, pd.Series([job["preprocessed_text"]]))
    scores = compute_similarity(resume_vectors, job_vec[0])
    ranking = pd.DataFrame({
        "resume_id": resumes["resume_id"].values,
        "score": scores,
        "job_title": job["job_title"],
    }).sort_values("score", ascending=False).reset_index(drop=True)
    ranking["rank"] = ranking.index + 1
    all_rankings.append(ranking)

    top_n = ranking.head(10)
    print(f"\n  Job: {job['job_title']}")
    print(top_n[["rank", "resume_id", "score"]].to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────────
# 6. SKILL GAP ANALYSIS (for first job, top 5 candidates)
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 6 – Skill gap analysis …")
first_job = jobs.iloc[0]
required_skills = set(s.strip().lower() for s in first_job["required_skills"].split(";"))
top_candidates = all_rankings[0].head(5)

gap_rows = []
for _, row in top_candidates.iterrows():
    res_row = resumes[resumes["resume_id"] == row["resume_id"]].iloc[0]
    cand_skills = set(s.strip().lower() for s in res_row["extracted_skills"].split(";") if s)
    missing = identify_missing_skills(cand_skills, required_skills)
    gap_rows.append({
        "rank": row["rank"],
        "resume_id": row["resume_id"],
        "score": round(row["score"], 4),
        "candidate_skills": ";".join(sorted(cand_skills)) or "—",
        "missing_skills": ";".join(sorted(missing)) or "None",
    })
    print(f"  Rank {row['rank']} | Resume {row['resume_id']} | Missing: {missing or 'None'}")

gap_df = pd.DataFrame(gap_rows)
gap_csv = os.path.join(VIZ_DIR, "skill_gaps.csv")
gap_df.to_csv(gap_csv, index=False)
print(f"  Skill gaps saved to {gap_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
print("\nStep 7 – Generating visualizations …")

# 7a. Candidate scores bar chart (top 20 for first job)
top20 = all_rankings[0].head(20)
plt.figure(figsize=(12, 5))
sns.barplot(data=top20, x="resume_id", y="score", palette="viridis", order=top20["resume_id"])
plt.title(f"Top-20 Candidate Scores — {first_job['job_title']}", fontsize=14)
plt.xlabel("Resume ID")
plt.ylabel("Cosine Similarity Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "candidate_scores.png"), dpi=150)
plt.close()
print("  candidate_scores.png saved.")

# 7b. Skill distribution across all resumes
all_skills = []
for skill_set in resume_skill_sets:
    all_skills.extend(s.lower() for s in skill_set)
skill_counts = pd.Series(all_skills).value_counts().head(15)
plt.figure(figsize=(10, 5))
sns.barplot(x=skill_counts.index, y=skill_counts.values, palette="magma")
plt.title("Top-15 Skill Frequency in Resumes", fontsize=14)
plt.xlabel("Skill")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "skill_distribution.png"), dpi=150)
plt.close()
print("  skill_distribution.png saved.")

# 7c. Ranking chart (score distribution histogram)
all_scores = all_rankings[0]["score"]
plt.figure(figsize=(8, 4))
sns.histplot(all_scores, bins=20, kde=True, color="steelblue")
plt.title("Resume Score Distribution", fontsize=14)
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Number of Resumes")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "ranking_chart.png"), dpi=150)
plt.close()
print("  ranking_chart.png saved.")

print("\n✅ Demo completed. All outputs are in the visualizations/ folder.")
