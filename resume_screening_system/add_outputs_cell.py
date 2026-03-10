import json

NOTEBOOK = "notebooks/resume_analysis.ipynb"

cell_source = """\
# ── 9. Save All Outputs ───────────────────────────────────────────────────────
import os, json, shutil
from datetime import datetime

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# 1. Full ranking table
ranking.to_csv(f"{OUT}/ranking_table.csv", index=False)
print(f"Saved ranking_table.csv  ({len(ranking)} rows)")

# 2. Preprocessed resumes
resumes[["resume_id", "job_role", "extracted_skills", "preprocessed_text"]].to_csv(
    f"{OUT}/resumes_processed.csv", index=False)
print(f"Saved resumes_processed.csv  ({len(resumes)} rows)")

# 3. Skill gap report (top-20 candidates)
req = set(s.strip().lower() for s in job["required_skills"].split(";"))
gap_rows = []
for _, row in ranking.head(20).iterrows():
    r = resumes[resumes["resume_id"] == row["resume_id"]].iloc[0]
    got = set(s.strip().lower() for s in r["extracted_skills"].split(";") if s)
    missing = identify_missing_skills(got, req)
    gap_rows.append({
        "rank": row["rank"],
        "resume_id": row["resume_id"],
        "similarity_score": round(row["score"], 4),
        "matched_skills": "; ".join(sorted(got & req)) or "None",
        "missing_skills": "; ".join(sorted(missing)) or "None",
    })
pd.DataFrame(gap_rows).to_csv(f"{OUT}/skill_gap_report.csv", index=False)
print(f"Saved skill_gap_report.csv  ({len(gap_rows)} rows)")

# 4. Skill frequency table
all_skills = [s.lower() for ss in skill_sets for s in ss]
skill_freq = pd.Series(all_skills).value_counts().reset_index()
skill_freq.columns = ["skill", "frequency"]
skill_freq.to_csv(f"{OUT}/skill_frequency.csv", index=False)
print(f"Saved skill_frequency.csv  ({len(skill_freq)} unique skills)")

# 5. Summary stats JSON
stats = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "job_title": job["job_title"],
    "total_resumes": int(len(resumes)),
    "vocabulary_size": int(len(vectorizer.vocabulary_)),
    "top_candidate_id": str(ranking.iloc[0]["resume_id"]),
    "top_score": round(float(ranking.iloc[0]["score"]), 4),
    "mean_score": round(float(ranking["score"].mean()), 4),
    "median_score": round(float(ranking["score"].median()), 4),
    "std_score": round(float(ranking["score"].std()), 4),
    "unique_skills_found": int(len(skill_freq)),
    "required_skills": job["required_skills"],
}
with open(f"{OUT}/summary_stats.json", "w") as fh:
    json.dump(stats, fh, indent=2)

print("\\n===== Summary Stats =====")
for k, v in stats.items():
    print(f"  {k:<25} {v}")

# 6. Copy visualisation PNGs into outputs/
for png in ["candidate_scores.png", "skill_distribution.png", "ranking_chart.png"]:
    src = f"visualizations/{png}"
    if os.path.exists(src):
        shutil.copy(src, f"{OUT}/{png}")
        print(f"Copied  {png}  →  {OUT}/")

print("\\nAll outputs saved to:", os.path.abspath(OUT))
"""

with open(NOTEBOOK, encoding="utf-8") as f:
    nb = json.load(f)

# Remove any previously injected cell with this header to avoid duplicates
nb["cells"] = [
    c for c in nb["cells"]
    if "Save All Outputs" not in "".join(c.get("source", []))
]

nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_source,
})

with open(NOTEBOOK, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Cell injected successfully.")
