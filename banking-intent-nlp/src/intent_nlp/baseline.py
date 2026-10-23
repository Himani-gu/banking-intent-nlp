"""Strong classical baseline: word + character TF-IDF features with logistic regression.

The character n-grams make this baseline reasonably robust to typos, so the comparison
against the fine-tuned transformer is a fair one rather than a straw man.

Usage:
    python -m intent_nlp.baseline                      # full training data
    python -m intent_nlp.baseline --fraction 0.1       # 10% of the training data
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import DEFAULT_CACHE_DIR, get_splits
from .evaluate import evaluate_model


def build_pipeline(c: float = 10.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
                ),
            ),
        ]
    )
    return Pipeline([("tfidf", features), ("clf", LogisticRegression(C=c, max_iter=2000))])


def fit_baseline(train_df, val_df=None, c_grid=(10.0,)):
    """Fit one pipeline per C and keep the best on validation accuracy.

    Returns (pipeline, best_c, val_accuracy). With a single C (or no validation set)
    no selection is performed and val_accuracy is None.
    """
    best = (None, None, -1.0)
    for c in c_grid:
        pipe = build_pipeline(c).fit(train_df["text"], train_df["label"])
        if val_df is None or len(c_grid) == 1:
            return pipe, c, None
        val_acc = float(np.mean(pipe.predict(val_df["text"]) == val_df["label"].to_numpy()))
        print(f"  C={c:<6g} val accuracy = {val_acc:.4f}")
        if val_acc > best[2]:
            best = (pipe, c, val_acc)
    return best


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train/evaluate the TF-IDF + LogReg baseline")
    p.add_argument("--dataset", default="banking77", choices=["banking77", "sample"])
    p.add_argument("--fraction", type=float, default=1.0, help="share of training data to use")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--c-grid", default="10,30,100", help="comma-separated C values to try")
    p.add_argument("--noise-rate", type=float, default=0.05, help="typo rate for robustness test")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    splits = get_splits(args.dataset, args.fraction, seed=args.seed, cache_dir=args.cache_dir)
    print(
        f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
        f"classes={len(splits.labels)}"
    )

    c_grid = tuple(float(c) for c in args.c_grid.split(","))
    start = time.perf_counter()
    pipe, best_c, val_acc = fit_baseline(splits.train, splits.val, c_grid)
    train_seconds = time.perf_counter() - start
    assert list(pipe.classes_) == splits.labels, "training data is missing some classes"

    name = f"tfidf_lr_frac{args.fraction:g}"
    model_dir = Path(args.models_dir) / name
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": splits.labels, "name": name}, model_dir / "model.joblib")

    meta = {
        "family": "tfidf_lr",
        "display_name": "TF-IDF (word+char) + LogReg",
        "fraction": args.fraction,
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "best_C": best_c,
        "val_accuracy": val_acc,
        "train_seconds": round(train_seconds, 1),
        "device": "cpu",
        "seed": args.seed,
    }
    metrics = evaluate_model(
        pipe.predict_proba,
        splits.test,
        splits.labels,
        args.results_dir,
        name,
        meta=meta,
        noise_rate=args.noise_rate,
    )
    print(
        f"[{name}] test accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f} "
        f"noisy accuracy={metrics['noisy_accuracy']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    main()
