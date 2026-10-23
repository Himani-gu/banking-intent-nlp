import json

from intent_nlp.predict import load_predictor


def test_baseline_learns_the_sample_task(baseline_artifacts):
    _, metrics = baseline_artifacts
    assert metrics["accuracy"] >= 0.8  # 8 well-separated intents; chance is 12.5%
    assert 0.0 <= metrics["noisy_accuracy"] <= 1.0
    assert metrics["top3_accuracy"] >= metrics["accuracy"]


def test_baseline_writes_expected_outputs(baseline_artifacts):
    model_path, metrics = baseline_artifacts
    run_dir = model_path.parents[2] / "results" / metrics["name"]
    for fname in ("metrics.json", "per_class_report.csv", "top_confusions.csv", "errors.csv"):
        assert (run_dir / fname).exists(), fname
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["accuracy"] == metrics["accuracy"]
    assert saved["family"] == "tfidf_lr"


def test_saved_model_round_trips_and_ranks_probabilities(baseline_artifacts):
    model_path, _ = baseline_artifacts
    predictor = load_predictor(model_path)
    pred = predictor.predict("my card never arrived in the post", top_k=3)
    assert pred.intent == "card_arrival"
    assert len(pred.top_k) == 3
    confidences = [c for _, c in pred.top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert abs(predictor.predict_proba(["hello"]).sum() - 1.0) < 1e-6
