"""End-to-end smoke test of the transformer pipeline using a *tiny, randomly initialised*
BERT built locally, so it needs no internet and no pretrained weights. It checks that the
training loop, checkpointing, serving path and evaluation all run - not model quality.
Skipped automatically when torch/transformers are not installed."""
import json
import re

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from intent_nlp.data import load_sample
from intent_nlp.predict import load_predictor
from intent_nlp.train_transformer import main


def _make_tiny_model_dir(path):
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # [PAD] must have id 0
    words = sorted({w for t in load_sample()["text"] for w in re.findall(r"\w+|[^\w\s]", t.lower())})
    vocab = {tok: i for i, tok in enumerate(specials + words)}

    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    config = BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=64, pad_token_id=0,
    )
    BertModel(config).save_pretrained(path)
    fast.save_pretrained(path)


def test_tiny_transformer_trains_saves_and_serves(tmp_path):
    base = tmp_path / "tiny_base"
    _make_tiny_model_dir(base)

    metrics = main(
        [
            "--model-name", str(base),
            "--run-name", "tiny",
            "--dataset", "sample",
            "--epochs", "3",
            "--batch-size", "16",
            "--lr", "1e-3",
            "--max-length", "32",
            "--results-dir", str(tmp_path / "results"),
            "--models-dir", str(tmp_path / "models"),
        ]
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["family"] == "transformer"
    run_dir = tmp_path / "results" / "tiny"
    assert (run_dir / "metrics.json").exists()
    history = json.loads((run_dir / "history.json").read_text())
    assert len(history) == 3 and all("val_accuracy" in h for h in history)

    saved = tmp_path / "models" / "tiny"
    assert (saved / "config.json").exists()
    predictor = load_predictor(saved)
    assert len(predictor.labels) == 8
    pred = predictor.predict("my card has not arrived", top_k=3)
    assert pred.intent in predictor.labels and len(pred.top_k) == 3
