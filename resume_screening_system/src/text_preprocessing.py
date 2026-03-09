import re
import string
from typing import List
import spacy
from nltk.corpus import stopwords

# Load spaCy model (ensure it's downloaded)
nlp = spacy.load('en_core_web_sm')

# Ensure NLTK stopwords are downloaded (user may need to run nltk.download('stopwords') beforehand)
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Apply basic cleaning: lowercasing, remove punctuation, extra whitespace."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize using spaCy, removing stopwords and lemmatizing."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in STOP_WORDS]
    return tokens


def preprocess(text: str) -> str:
    """Full preprocessing pipeline returning a cleaned string of lemmatized tokens."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    return " ".join(tokens)
