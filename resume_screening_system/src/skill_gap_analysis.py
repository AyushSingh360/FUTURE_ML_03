from typing import Set, List, Dict
import pandas as pd


def identify_missing_skills(resume_skills: Set[str], required_skills: Set[str]) -> Set[str]:
    """Return the set of required skills that are missing from the resume skills."""
    return required_skills - resume_skills


def skill_gap_for_dataframe(df_resumes, df_jobs, resume_skill_col: str = "skills", job_skill_col: str = "required_skills") -> List[Dict]:
    """Compute skill gaps for each resume against each job description.
    Returns a list of dictionaries with resume_id, job_title, missing_skills.
    """
    results = []
    for _, resume in df_resumes.iterrows():
        resume_id = resume["resume_id"]
        resume_skills = set(resume[resume_skill_col]) if isinstance(resume[resume_skill_col], str) else set()
        for _, job in df_jobs.iterrows():
            job_title = job["job_title"]
            required = set(job[job_skill_col]) if isinstance(job[job_skill_col], str) else set()
            missing = identify_missing_skills(resume_skills, required)
            results.append({
                "resume_id": resume_id,
                "job_title": job_title,
                "missing_skills": ",".join(sorted(missing))
            })
    return results
