# AI Resume Screening and Candidate Ranking System

## Overview
This project implements an end‑to‑end NLP pipeline that reads resumes and job descriptions, extracts skills, vectorizes texts, computes similarity scores, ranks candidates, and identifies skill gaps. It is designed as a showcase for an internship portfolio.

## Project Structure
```
resume_screening_system/
│
├── data/
│   ├── resumes_dataset.csv          # Sample resumes
│   └── job_descriptions.csv         # Sample job descriptions
│
├── notebooks/
│   └── resume_analysis.ipynb       # Demonstrates the full pipeline
│
├── src/
│   ├── data_loader.py
│   ├── text_preprocessing.py
│   ├── skill_extraction.py
│   ├── vectorization.py
│   ├── similarity_scoring.py
│   ├── candidate_ranking.py
│   └── skill_gap_analysis.py
│
├── models/
│   └── vectorizer.pkl               # Saved TF‑IDF vectorizer (generated on first run)
│
├── visualizations/
│   ├── candidate_scores.png        # Bar chart of ranking scores
│   ├── skill_distribution.png      # Histogram of skill frequencies
│   └── ranking_chart.png           # Ranked bar plot
│
├── requirements.txt
└── README.md
```

## Setup
1. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - pandas, numpy, spacy, nltk, scikit-learn, matplotlib, seaborn, sentence-transformers (optional)
3. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Download NLTK stopwords** (run once)
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Running the Demo Notebook
Open `notebooks/resume_analysis.ipynb` in Jupyter Lab/Notebook. The notebook walks through:
- Loading the sample CSVs
- Pre‑processing resume text
- Extracting skills
- Vectorizing texts (TF‑IDF)
- Computing similarity scores for each resume against a selected job description
- Ranking candidates
- Visualising results
- Identifying missing skills for each candidate

## Visualizations
The notebook (or the `visualizations/visualize.py` script) generates three PNG files in the `visualizations/` folder:
- `candidate_scores.png` – bar chart of similarity scores
- `skill_distribution.png` – histogram of how often each skill appears in the resume set
- `ranking_chart.png` – ranked bar plot of candidates

## Extending the Project
- Replace the placeholder CSVs with real data.
- Swap the TF‑IDF vectorizer for a semantic model (e.g., `sentence-transformers` BERT) for richer similarity.
- Add unit tests in a `tests/` directory.
- Deploy as a small Flask/FastAPI service for interactive querying.

## License
This project is provided for educational purposes. Feel free to adapt and share.
