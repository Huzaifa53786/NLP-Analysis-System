"""FastAPI service for the browser UI and JSON API."""
from __future__ import annotations
from collections import OrderedDict
from threading import Lock
from uuid import uuid4
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .core import analyze_text, read_labeled_csv, train_classifier, predict_text

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Open Text Lab", version="1.0.0", description="Text exploration and custom text classification API")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
_models: OrderedDict[str, Any] = OrderedDict()
_models_lock = Lock()
MAX_ACTIVE_MODELS = 5


class RequestSizeLimitMiddleware:
    """Reject oversized declared bodies before JSON parsing or model work."""
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            raw_length = next((v for k, v in scope["headers"] if k.lower() == b"content-length"), None)
            if raw_length:
                try:
                    too_large = int(raw_length) > self.max_bytes
                except ValueError:
                    too_large = False
                if too_large:
                    response = JSONResponse(status_code=413, content={"detail": "Request body exceeds 11 MB."})
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)


app.add_middleware(RequestSizeLimitMiddleware, max_bytes=11 * 1024 * 1024)

class AnalyzeRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    top_n: int = Field(default=10, ge=1, le=50)

class TrainRequest(BaseModel):
    csv: str = Field(min_length=1, max_length=10 * 1024 * 1024)
    text_column: str = Field(min_length=1, max_length=200)
    label_column: str = Field(min_length=1, max_length=200)
    model: str = "Logistic Regression"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)

class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    try:
        return analyze_text(request.texts, request.top_n)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/api/train")
def train(request: TrainRequest):
    if request.text_column == request.label_column:
        raise HTTPException(status_code=422, detail="Choose different text and label columns.")
    try:
        texts, labels = read_labeled_csv(request.csv.encode("utf-8"), request.text_column, request.label_column)
        model, metrics = train_classifier(texts, labels, request.model, request.test_size)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    model_id = uuid4().hex
    with _models_lock:
        _models[model_id] = model
        while len(_models) > MAX_ACTIVE_MODELS:
            _models.popitem(last=False)
    return {"model_id": model_id, "metrics": metrics, "message": "Model trained in memory for this server process."}

@app.post("/api/predict/{model_id}")
def predict(model_id: str, request: PredictRequest):
    with _models_lock:
        model = _models.get(model_id)
        if model is not None:
            _models.move_to_end(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found. Train a model again; models are temporary and process-local.")
    try:
        return {"predictions": predict_text(model, request.texts)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
