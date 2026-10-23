import pytest

from intent_nlp import baseline


@pytest.fixture(scope="session")
def baseline_artifacts(tmp_path_factory):
    """Train the baseline once on the tiny sample dataset; return (model_path, metrics)."""
    root = tmp_path_factory.mktemp("artifacts")
    metrics = baseline.main(
        [
            "--dataset", "sample",
            "--c-grid", "10",
            "--results-dir", str(root / "results"),
            "--models-dir", str(root / "models"),
        ]
    )
    model_path = root / "models" / metrics["name"] / "model.joblib"
    return model_path, metrics
