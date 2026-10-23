import numpy as np

from intent_nlp.evaluate import noisy_copy, perturb, top_k_accuracy


def test_perturb_zero_rate_is_identity():
    rng = np.random.default_rng(0)
    assert perturb("my card has not arrived", 0.0, rng) == "my card has not arrived"


def test_perturb_is_deterministic_for_a_seed_and_changes_text():
    texts = ["my card has not arrived yet", "what is the exchange rate today"]
    a, b = noisy_copy(texts, 0.3, seed=1), noisy_copy(texts, 0.3, seed=1)
    assert a == b
    assert a != texts


def test_perturb_keeps_spaces_and_punctuation_positions_sane():
    out = perturb("hello, world?", 1.0, np.random.default_rng(0))
    assert "," in out and "?" in out and " " in out


def test_top_k_accuracy():
    proba = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([1, 1, 0])
    assert top_k_accuracy(proba, y, 1) == 1 / 3
    assert top_k_accuracy(proba, y, 2) == 2 / 3
    assert top_k_accuracy(proba, y, 3) == 1.0
    assert top_k_accuracy(proba, y, 10) == 1.0  # k larger than n_classes is clipped
