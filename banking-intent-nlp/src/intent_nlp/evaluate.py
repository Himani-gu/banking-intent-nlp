"""Model-agnostic evaluation: accuracy/F1, top-k, typo robustness, latency, error analysis.

Every model (baseline or transformer) is evaluated through the same function, given a
``predict_proba(texts) -> ndarray[n, n_classes]`` whose columns follow ``labels`` order.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

ProbaFn = Callable[[Sequence[str]], np.ndarray]
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def perturb(text: str, rate: float, rng: np.random.Generator) -> str:
    """Inject character-level typos: each letter is hit with probability ``rate`` and
    then deleted, swapped with its neighbour, doubled, or replaced by a random letter."""
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = int(rng.integers(0, 4))
            if op == 0:  # delete
                pass
            elif op == 1 and i + 1 < len(chars) and chars[i + 1].isalpha():  # swap
                out.extend([chars[i + 1], c])
                i += 1
            elif op == 2:  # double
                out.extend([c, c])
            else:  # replace (also the fallback when a swap is impossible)
                out.append(_LETTERS[int(rng.integers(0, 26))])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def noisy_copy(texts: Sequence[str], rate: float, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    return [perturb(t, rate, rng) for t in texts]


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    top_k = np.argsort(-proba, axis=1)[:, :k]
    return float((top_k == y_true[:, None]).any(axis=1).mean())


def measure_latency(
    predict_proba: ProbaFn, texts: Sequence[str], n: int = 200, warmup: int = 5
) -> tuple:
    """Single-query latency (p50, p95) in milliseconds, i.e. what an API call would see."""
    texts = list(texts)[:n]
    for t in texts[:warmup]:
        predict_proba([t])
    times = []
    for t in texts:
        start = time.perf_counter()
        predict_proba([t])
        times.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 95))


def evaluate_model(
    predict_proba: ProbaFn,
    test_df: pd.DataFrame,
    labels: list,
    out_dir,
    name: str,
    meta: Optional[dict] = None,
    noise_rate: float = 0.05,
    seed: int = 0,
    latency_samples: int = 200,
) -> dict:
    """Evaluate on ``test_df`` and write metrics + error-analysis files to ``out_dir/name``."""
    run_dir = Path(out_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    label2id = {label: i for i, label in enumerate(labels)}
    texts = test_df["text"].tolist()
    y_true = test_df["label"].map(label2id).to_numpy()

    proba = predict_proba(texts)
    y_pred = proba.argmax(axis=1)
    noisy_pred = predict_proba(noisy_copy(texts, noise_rate, seed)).argmax(axis=1)
    p50, p95 = measure_latency(predict_proba, texts, n=latency_samples)

    accuracy = float(accuracy_score(y_true, y_pred))
    noisy_accuracy = float(accuracy_score(y_true, noisy_pred))
    metrics = {
        "name": name,
        **(meta or {}),
        "n_test": int(len(test_df)),
        "n_classes": len(labels),
        "accuracy": accuracy,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "top3_accuracy": top_k_accuracy(proba, y_true, 3),
        "top5_accuracy": top_k_accuracy(proba, y_true, 5),
        "noise_rate": noise_rate,
        "noisy_accuracy": noisy_accuracy,
        "noisy_accuracy_drop": accuracy - noisy_accuracy,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
    }

    # Per-class report (sorted worst-first so weak intents are obvious)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).T.drop(
        index=["accuracy", "macro avg", "weighted avg"], errors="ignore"
    )
    per_class.sort_values("f1-score").to_csv(run_dir / "per_class_report.csv")

    # Most confused (true -> predicted) pairs
    true_labels = np.array(labels)[y_true]
    pred_labels = np.array(labels)[y_pred]
    wrong = y_true != y_pred
    pairs = Counter(zip(true_labels[wrong], pred_labels[wrong])).most_common(15)
    pd.DataFrame(
        [{"true": t, "predicted": p, "count": c} for (t, p), c in pairs]
    ).to_csv(run_dir / "top_confusions.csv", index=False)

    # Misclassified examples, most confident mistakes first (often label noise / ambiguity)
    errors = pd.DataFrame(
        {
            "text": np.array(texts)[wrong],
            "true": true_labels[wrong],
            "predicted": pred_labels[wrong],
            "confidence": proba[wrong, y_pred[wrong]],
        }
    ).sort_values("confidence", ascending=False)
    errors.to_csv(run_dir / "errors.csv", index=False)

    with open(run_dir / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics
