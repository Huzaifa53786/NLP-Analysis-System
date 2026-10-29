import csv
import io
import pytest
from nlp_text_analysis.core import analyze_text, clean_text, read_labeled_csv, train_classifier, predict_text


def csv_content():
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["message", "category", "unused"])
    for label, examples in {
        "positive": ["great and useful product", "excellent support today", "very happy with service", "worked perfectly for me", "quick delivery and great quality", "helpful team solved this"],
        "negative": ["bad and broken product", "awful support experience", "very unhappy with service", "never worked for me", "late delivery and poor quality", "rude team ignored this"],
    }.items():
        for example in examples:
            writer.writerow([example, label, "drop"])
    return out.getvalue().encode()


def test_clean_text_normalizes_unicode_and_masks_identifiers():
    assert clean_text("  Ｈello\nhttps://example.org/a  person@example.org ") == "Hello URLTOKEN EMAILTOKEN"


def test_analyze_text_returns_counts_and_terms():
    result = analyze_text(["Green green tree", "Tree by river"], top_n=2)
    assert result["document_count"] == 2
    assert result["word_count"] == 6
    assert result["top_terms"] == [{"term": "green", "count": 2}, {"term": "tree", "count": 2}]


def test_analyze_text_rejects_empty_input():
    with pytest.raises(ValueError, match="non-empty"):
        analyze_text(["  "])


def test_csv_loader_selects_columns_and_drops_incomplete_rows():
    raw = b"body,tag\nHello,yes\n,yes\nBye,no\n"
    texts, labels = read_labeled_csv(raw, "body", "tag")
    assert texts == ["Hello", "Bye"]
    assert labels == ["yes", "no"]


def test_csv_loader_rejects_missing_columns():
    with pytest.raises(ValueError, match="selected columns"):
        read_labeled_csv(b"body,tag\nHello,yes", "text", "tag")


def test_train_evaluate_predict_end_to_end():
    texts, labels = read_labeled_csv(csv_content(), "message", "category")
    model, metrics = train_classifier(texts, labels, random_state=3)
    assert set(metrics["labels"]) == {"positive", "negative"}
    assert metrics["train_examples"] + metrics["test_examples"] == len(texts)
    result = predict_text(model, ["excellent useful product", "broken awful support"])
    assert result[0]["label"] == "positive"
    assert result[1]["label"] == "negative"
    assert "confidence" in result[0]


def test_train_rejects_undersized_or_single_class_data():
    with pytest.raises(ValueError, match="at least 10"):
        train_classifier(["one"] * 4, ["yes"] * 4)
    with pytest.raises(ValueError, match="two different labels"):
        train_classifier(["same word"] * 10, ["yes"] * 10)


def test_prediction_rejects_blank_text():
    texts, labels = read_labeled_csv(csv_content(), "message", "category")
    model, _ = train_classifier(texts, labels)
    with pytest.raises(ValueError, match="non-empty"):
        predict_text(model, [" "])

