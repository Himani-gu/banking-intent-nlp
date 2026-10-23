"""Unified inference interface for the baseline and the fine-tuned transformer.

    python -m intent_nlp.predict --model models/tfidf_lr_frac1/model.joblib "my card is lost"
    python -m intent_nlp.predict --model models/distilbert_base_uncased_frac1 "my card is lost"
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Prediction:
    intent: str
    confidence: float
    top_k: list  # list of (intent, confidence) tuples, best first


class BasePredictor:
    name: str
    labels: list

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        proba = self.predict_proba([text])[0]
        order = np.argsort(-proba)[: max(1, min(top_k, len(self.labels)))]
        ranked = [(self.labels[i], float(proba[i])) for i in order]
        return Prediction(intent=ranked[0][0], confidence=ranked[0][1], top_k=ranked)


class TfidfPredictor(BasePredictor):
    def __init__(self, path):
        import joblib

        bundle = joblib.load(path)
        self.pipeline = bundle["pipeline"]
        self.labels = list(bundle["labels"])
        self.name = bundle.get("name", "tfidf_lr")

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(texts))


class TransformerPredictor(BasePredictor):
    """Loads a model saved with ``save_pretrained`` (config.json + weights + tokenizer)."""

    def __init__(self, model_dir, device=None, max_length: int = 64, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = torch.device(device) if device else _pick_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        n = self.model.config.num_labels
        self.labels = [self.model.config.id2label[i] for i in range(n)]
        self.name = Path(model_dir).name

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        texts = list(texts)
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[i : i + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                chunks.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(chunks)


def _pick_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_predictor(path) -> BasePredictor:
    """``*.joblib`` file -> TF-IDF baseline; directory with config.json -> transformer."""
    p = Path(path)
    if p.is_dir() and (p / "config.json").exists():
        return TransformerPredictor(p)
    if p.is_file() and p.suffix == ".joblib":
        return TfidfPredictor(p)
    raise FileNotFoundError(
        f"No model found at {p}. Expected a .joblib file (baseline) or a directory with "
        "config.json (transformer). Train one first - see the README."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict the intent of a banking query")
    parser.add_argument("text")
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)
    pred = load_predictor(args.model).predict(args.text, args.top_k)
    print(f"{pred.intent}  ({pred.confidence:.1%})")
    for intent, conf in pred.top_k:
        print(f"  {conf:6.1%}  {intent}")


if __name__ == "__main__":
    main()
