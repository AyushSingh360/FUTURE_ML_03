import spacy
from spacy.matcher import PhraseMatcher
from typing import List, Set

# Predefined skill dictionary (can be extended)
SKILL_LIST = [
    "python",
    "java",
    "machine learning",
    "ml",
    "deep learning",
    "dl",
    "sql",
    "data analysis",
    "data analytics",
    "cloud",
    "aws",
    "azure",
    "gcp",
    "communication",
    "nlp",
    "natural language processing",
    "statistics",
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
]

nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in SKILL_LIST]
matcher.add("SKILL", patterns)


def extract_skills(text: str) -> Set[str]:
    """Extract skills from the given text using spaCy phrase matcher.
    Returns a set of matched skill strings in their original form.
    """
    doc = nlp(text)
    matches = matcher(doc)
    skills = {doc[start:end].text for _, start, end in matches}
    return skills


def extract_skills_from_dataframe(df, text_column: str = "resume_text") -> List[Set[str]]:
    """Apply skill extraction to a pandas DataFrame column.
    Returns a list of skill sets corresponding to each row.
    """
    return [extract_skills(text) for text in df[text_column].fillna("")]
