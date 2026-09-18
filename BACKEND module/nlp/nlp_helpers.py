"""
CARIVIX AI - NLP Engine
Reusable helper functions for text classification, sentiment analysis,
entity recognition, and topic modeling.

Design note: functions here rely on scikit-learn (already a common
dependency) and pure regex so they work without extra installs. Where a
richer model helps (e.g. spaCy NER, transformer sentiment), a drop-in
override hook is provided so the LLM Intelligence Layer / a heavier
model can be swapped in later without changing call sites.
"""

import re
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Callable

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("carivix.nlp")


def clean_text(text: str, lowercase: bool = True, remove_urls: bool = True) -> str:
    """Basic text normalization shared by all NLP functions below."""
    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def train_text_classifier(texts: List[str], labels: List[str]) -> Pipeline:
    """
    Train a lightweight TF-IDF + Naive Bayes text classifier.

    Use for: Political / Economic / Technology News classification, or any
    custom category set. Returns a fitted sklearn Pipeline with .predict().
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    pipeline.fit(texts, labels)
    logger.info("Trained text classifier on %s documents, %s classes",
                len(texts), len(set(labels)))
    return pipeline


def classify_texts(pipeline: Pipeline, texts: List[str]) -> List[str]:
    """Apply a trained classifier (from train_text_classifier) to new texts."""
    return list(pipeline.predict(texts))


# --- Sentiment analysis -----------------------------------------------------

_POSITIVE_WORDS = {
    "good", "great", "excellent", "positive", "growth", "gain", "profit",
    "strong", "improve", "improved", "success", "successful", "boost",
    "record", "surge", "opportunity", "beneficial", "upbeat", "recovery",
}
_NEGATIVE_WORDS = {
    "bad", "poor", "negative", "decline", "loss", "weak", "fall", "fell",
    "crisis", "risk", "risky", "recession", "downturn", "concern", "fraud",
    "cut", "layoff", "layoffs", "shortage", "delay", "failure", "collapse",
}


def analyze_sentiment(
    text: str,
    scorer: Optional[Callable[[str], float]] = None,
) -> Dict[str, Any]:
    """
    Classify sentiment as Positive / Negative / Neutral.

    By default uses a simple lexicon-based score. Pass a custom `scorer`
    (e.g. a transformer or VADER-based function returning a float in
    [-1, 1]) to swap in a stronger model without changing call sites.
    """
    if scorer:
        score = scorer(text)
    else:
        tokens = clean_text(text).split()
        pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
        neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
        total = pos + neg
        score = 0.0 if total == 0 else (pos - neg) / total

    if score > 0.15:
        label = "Positive"
    elif score < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "score": round(score, 3)}


def batch_analyze_sentiment(texts: List[str], scorer: Optional[Callable[[str], float]] = None) -> List[Dict[str, Any]]:
    """Vectorized convenience wrapper around analyze_sentiment for a list of documents."""
    return [analyze_sentiment(t, scorer=scorer) for t in texts]


# --- Entity recognition ------------------------------------------------------

_ENTITY_PATTERNS = {
    "PERSON_OR_ORG": re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b"),
    "MONEY": re.compile(r"[₹$€£]\s?\d[\d,]*(?:\.\d+)?\s?(?:crore|lakh|million|billion|trillion)?", re.IGNORECASE),
    "PERCENT": re.compile(r"\d+(?:\.\d+)?\s?%"),
    "DATE": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:19|20)\d{2}\b"),
}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Lightweight, regex-based entity extraction covering capitalized
    name/company spans, monetary amounts, percentages, and dates.

    For production-grade Person/Org/Location tagging, swap in a spaCy or
    transformer NER pipeline behind this same function signature.
    """
    entities: Dict[str, List[str]] = {}
    for label, pattern in _ENTITY_PATTERNS.items():
        matches = sorted(set(m.strip() for m in pattern.findall(text) if m.strip()))
        if matches:
            entities[label] = matches
    return entities


# --- Topic modeling -----------------------------------------------------------

def extract_topics(
    texts: List[str],
    num_topics: int = 5,
    num_words_per_topic: int = 8,
) -> List[List[str]]:
    """
    Identify hidden themes across a document collection using LDA topic
    modeling. Returns a list of topics, each a list of top keywords.
    """
    vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    topics = []
    for topic in lda.components_:
        top_indices = topic.argsort()[-num_words_per_topic:][::-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


def top_keywords(text: str, n: int = 10) -> List[Tuple[str, int]]:
    """Simple frequency-based keyword extraction for quick summarization previews."""
    tokens = clean_text(text).split()
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
                 "is", "are", "was", "were", "be", "with", "as", "by", "at", "this", "that"}
    filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
    return Counter(filtered).most_common(n)
