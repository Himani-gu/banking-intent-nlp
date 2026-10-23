"""FastAPI service for intent prediction.

    uvicorn intent_nlp.api:app --port 8000
    MODEL_PATH=models/distilbert_base_uncased_frac1 uvicorn intent_nlp.api:app --port 8000

``MODEL_PATH`` may point to a baseline ``model.joblib`` (default) or to a fine-tuned
transformer directory. Interactive docs are served at ``/docs``.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from . import __version__
from .predict import load_predictor

DEFAULT_MODEL_PATH = "models/tfidf_lr_frac1/model.joblib"


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=1000, examples=["My card still has not arrived"])
    top_k: int = Field(3, ge=1, le=10)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class IntentScore(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    intent: str
    confidence: float
    top_k: List[IntentScore]
    latency_ms: float
    served_model: str


def create_app(model_path: Optional[str] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        app.state.predictor = load_predictor(path)
        yield

    app = FastAPI(
        title="Banking Intent Classifier",
        description="Classifies a customer's banking query into one of 77 intents.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        predictor = app.state.predictor
        return {"status": "ok", "model": predictor.name, "num_intents": len(predictor.labels)}

    @app.get("/intents")
    def intents():
        return {"intents": app.state.predictor.labels}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        predictor = app.state.predictor
        start = time.perf_counter()
        prediction = predictor.predict(request.text, request.top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return PredictResponse(
            intent=prediction.intent,
            confidence=prediction.confidence,
            top_k=[IntentScore(intent=i, confidence=c) for i, c in prediction.top_k],
            latency_ms=round(latency_ms, 2),
            served_model=predictor.name,
        )

    return app


app = create_app()
