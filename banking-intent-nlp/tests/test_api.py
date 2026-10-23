from fastapi.testclient import TestClient

from intent_nlp.api import create_app


def _client(model_path):
    return TestClient(create_app(str(model_path)))


def test_health_and_intents(baseline_artifacts):
    model_path, _ = baseline_artifacts
    with _client(model_path) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["num_intents"] == 8
        assert "card_arrival" in client.get("/intents").json()["intents"]


def test_predict_returns_ranked_intents(baseline_artifacts):
    model_path, _ = baseline_artifacts
    with _client(model_path) as client:
        resp = client.post("/predict", json={"text": "I lost my card yesterday", "top_k": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "lost_or_stolen_card"
        assert len(body["top_k"]) == 2
        assert body["top_k"][0]["intent"] == body["intent"]
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["latency_ms"] >= 0


def test_predict_rejects_bad_input(baseline_artifacts):
    model_path, _ = baseline_artifacts
    with _client(model_path) as client:
        assert client.post("/predict", json={"text": "   "}).status_code == 422
        assert client.post("/predict", json={"text": "hi", "top_k": 0}).status_code == 422
        assert client.post("/predict", json={}).status_code == 422
