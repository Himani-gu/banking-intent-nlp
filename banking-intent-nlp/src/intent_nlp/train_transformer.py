"""Fine-tune a pretrained transformer (default: DistilBERT) for intent classification.

A plain PyTorch loop (AdamW, linear warm-up/decay, gradient clipping, optional mixed
precision) is used instead of ``transformers.Trainer`` so that every step is explicit and
the script does not depend on Trainer API changes between library versions.

The best epoch is chosen on the *validation* split; the test split is evaluated once at the
end, through the same ``TransformerPredictor`` that the API serves.

Usage (GPU strongly recommended, e.g. a free Colab T4 - see notebooks/colab_train.ipynb):
    python -m intent_nlp.train_transformer --fp16
    python -m intent_nlp.train_transformer --fp16 --fraction 0.1      # data-efficiency run
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .data import DEFAULT_CACHE_DIR, get_splits
from .evaluate import evaluate_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune a transformer on Banking77")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--run-name", default=None, help="defaults to <model>_frac<fraction>")
    p.add_argument("--dataset", default="banking77", choices=["banking77", "sample"])
    p.add_argument("--fraction", type=float, default=1.0, help="share of training data to use")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--fp16", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-rate", type=float, default=0.05, help="typo rate for robustness test")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run(args) -> dict:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    from .predict import TransformerPredictor

    set_seed(args.seed)
    splits = get_splits(args.dataset, args.fraction, seed=args.seed, cache_dir=args.cache_dir)
    labels = splits.labels
    print(
        f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} "
        f"classes={len(labels)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=splits.id2label,
        label2id=splits.label2id,
    )
    device = pick_device()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model_name} params={n_params / 1e6:.1f}M device={device}")

    class IntentDataset(Dataset):
        def __init__(self, df):
            enc = tokenizer(df["text"].tolist(), truncation=True, max_length=args.max_length)
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = df["label"].map(splits.label2id).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i],
            }

    def collate(batch):
        labels_t = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        feats = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        padded = tokenizer.pad(feats, return_tensors="pt")
        padded["labels"] = labels_t
        return padded

    train_loader = DataLoader(
        IntentDataset(splits.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        IntentDataset(splits.val), batch_size=args.batch_size * 2, collate_fn=collate
    )

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def validate() -> float:
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                correct += (logits.argmax(-1) == batch["labels"]).sum().item()
                total += batch["labels"].numel()
        return correct / total

    run_name = args.run_name or f"{Path(args.model_name).name.replace('-', '_')}_frac{args.fraction:g}"
    best_dir = Path(args.models_dir) / run_name
    best_dir.mkdir(parents=True, exist_ok=True)

    history, best_val, best_epoch = [], -1.0, 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * batch["labels"].numel()
            seen += batch["labels"].numel()

        val_acc = validate()
        history.append(
            {"epoch": epoch, "train_loss": running_loss / seen, "val_accuracy": val_acc}
        )
        print(f"epoch {epoch}/{args.epochs}  train_loss={running_loss / seen:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
    train_seconds = time.perf_counter() - start

    # Free training memory, then evaluate the *saved best* checkpoint via the serving path.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictor = TransformerPredictor(best_dir, device=device.type, max_length=args.max_length)

    meta = {
        "family": "transformer",
        "display_name": Path(args.model_name).name,
        "fraction": args.fraction,
        "n_train": int(len(splits.train)),
        "n_val": int(len(splits.val)),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "fp16": use_amp,
        "n_params_millions": round(n_params / 1e6, 1),
        "train_seconds": round(train_seconds, 1),
        "device": device.type,
        "seed": args.seed,
    }
    metrics = evaluate_model(
        predictor.predict_proba,
        splits.test,
        splits.labels,
        args.results_dir,
        run_name,
        meta=meta,
        noise_rate=args.noise_rate,
    )
    with open(Path(args.results_dir) / run_name / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    print(
        f"[{run_name}] test accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f} "
        f"noisy accuracy={metrics['noisy_accuracy']:.4f}"
    )
    return metrics


def main(argv=None) -> dict:
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
