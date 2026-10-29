import pytest
from fastapi import HTTPException
from nlp_text_analysis.api import AnalyzeRequest, PredictRequest, TrainRequest, analyze, health, home, predict, train, _models

CSV = "text,label\nGreat product,good\nExcellent service,good\nLove it,good\nVery helpful,good\nWorks perfectly,good\nHappy with it,good\nBad product,bad\nBroken on arrival,bad\nAwful service,bad\nDo not like it,bad\nPoor experience,bad\nVery disappointing,bad\n"

def test_health_and_static_ui():
    assert health()["status"] == "ok"
    assert home().path.name == "index.html"

def test_analyze_api_and_validation():
    result = analyze(AnalyzeRequest(texts=["green tree", "blue river"]))
    assert result["unique_word_count"] == 4
    with pytest.raises(HTTPException) as error:
        analyze(AnalyzeRequest(texts=[""]))
    assert error.value.status_code == 422

def test_train_and_predict_api():
    _models.clear()
    result = train(TrainRequest(csv=CSV, text_column="text", label_column="label"))
    predictions = predict(result["model_id"], PredictRequest(texts=["great product"]))
    assert predictions["predictions"][0]["label"] == "good"
    assert result["metrics"]["accuracy"] >= 0

def test_predict_unknown_model_returns_not_found():
    with pytest.raises(HTTPException) as error:
        predict("missing", PredictRequest(texts=["hello"]))
    assert error.value.status_code == 404

