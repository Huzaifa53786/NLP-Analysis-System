"""Data validation and reusable scikit-learn NLP workflows."""
from __future__ import annotations
import io
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

MAX_TEXT_CHARS = 20_000
MAX_ROWS = 50_000
MAX_CSV_BYTES = 10 * 1024 * 1024
MODEL_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": lambda: LinearSVC(class_weight="balanced"),
    "Naive Bayes": lambda: MultinomialNB(),
}


def clean_text(text: str) -> str:
    """Normalize text conservatively while retaining punctuation and accents."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", text, flags=re.IGNORECASE)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAILTOKEN ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Return Unicode word tokens without language-specific stop-word removal."""
    return re.findall(r"(?u)\b[\w]+(?:['’][\w]+)?\b", clean_text(text).lower())


def analyze_text(texts: Iterable[str], top_n: int = 10) -> dict[str, Any]:
    texts = list(texts)
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Provide at least one non-empty text entry.")
    if any(len(t) > MAX_TEXT_CHARS for t in texts):
        raise ValueError(f"Each text must be {MAX_TEXT_CHARS:,} characters or fewer.")
    tokens = [token for text in texts for token in tokenize(text)]
    if not tokens:
        raise ValueError("No words were found. Add text containing at least one word.")
    frequency = Counter(tokens)
    return {
        "document_count": len(texts), "character_count": sum(map(len, texts)),
        "word_count": len(tokens), "unique_word_count": len(frequency),
        "average_words_per_document": round(len(tokens) / len(texts), 2),
        "top_terms": [{"term": term, "count": count} for term, count in frequency.most_common(max(1, min(top_n, 50)))],
        "normalized_preview": [clean_text(t)[:500] for t in texts[:5]],
    }


def read_labeled_csv(csv_bytes: bytes, text_column: str, label_column: str) -> tuple[list[str], list[str]]:
    if not csv_bytes:
        raise ValueError("Upload a non-empty CSV file.")
    if len(csv_bytes) > MAX_CSV_BYTES:
        raise ValueError("CSV must be no larger than 10 MB.")
    try:
        frame = pd.read_csv(io.BytesIO(csv_bytes), dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc
    if text_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(f"CSV must contain the selected columns: {text_column!r} and {label_column!r}.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"CSV must contain no more than {MAX_ROWS:,} data rows.")
    frame = frame[[text_column, label_column]].copy()
    frame[text_column] = frame[text_column].map(clean_text)
    frame[label_column] = frame[label_column].map(lambda value: str(value).strip())
    frame = frame[(frame[text_column] != "") & (frame[label_column] != "")]
    if frame.empty:
        raise ValueError("No usable rows found. Each row needs both text and label values.")
    if (frame[text_column].str.len() > MAX_TEXT_CHARS).any():
        raise ValueError(f"Each text must be {MAX_TEXT_CHARS:,} characters or fewer.")
    return frame[text_column].tolist(), frame[label_column].tolist()


def train_classifier(texts: list[str], labels: list[str], model_name: str = "Logistic Regression", test_size: float = 0.2, random_state: int = 42) -> tuple[Pipeline, dict[str, Any]]:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Choose a supported model: {', '.join(MODEL_FACTORIES)}.")
    if len(texts) != len(labels) or len(texts) < 10:
        raise ValueError("Provide matching text and label lists with at least 10 usable examples.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("Test size must be between 0.1 and 0.4.")
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Training requires at least two different labels.")
    if min(counts.values()) < 2:
        raise ValueError("Each label needs at least 2 examples so the model can be evaluated.")
    n_test = max(len(counts), round(len(texts) * test_size))
    if len(texts) - n_test < len(counts):
        raise ValueError("Not enough rows for a stratified train/test split. Add more examples per label.")
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=n_test, random_state=random_state, stratify=labels)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)),
        ("classifier", MODEL_FACTORIES[model_name]()),
    ])
    try:
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
    except ValueError as exc:
        raise ValueError(f"Could not train this dataset: {exc}") from exc
    classes = [str(label) for label in model.classes_]
    metrics = {
        "model": model_name, "train_examples": len(x_train), "test_examples": len(x_test),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4), "labels": classes,
        "classification_report": classification_report(y_test, predictions, labels=model.classes_, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
    }
    return model, metrics


def predict_text(model: Pipeline, texts: list[str]) -> list[dict[str, Any]]:
    if not texts or len(texts) > 100:
        raise ValueError("Provide between 1 and 100 texts per prediction request.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each prediction needs non-empty text.")
    if any(len(text) > MAX_TEXT_CHARS for text in texts):
        raise ValueError(f"Each text must be {MAX_TEXT_CHARS:,} characters or fewer.")
    labels = model.predict(texts)
    classifier = model.named_steps["classifier"]
    results = []
    probabilities = model.predict_proba(texts) if hasattr(classifier, "predict_proba") else None
    for i, (text, label) in enumerate(zip(texts, labels)):
        result: dict[str, Any] = {"text": text, "label": str(label)}
        if probabilities is not None:
            result["confidence"] = round(float(max(probabilities[i])), 4)
        results.append(result)
    return results
