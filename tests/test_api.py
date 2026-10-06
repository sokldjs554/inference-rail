from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        demo = client.get("/")
        assert demo.status_code == 200
        assert "InferenceRail" in demo.text
        build_body = client.get("/ops/build").json()
        assert build_body["app"] == "InferenceRail"
        assert build_body["version"] == "0.3.0"
        assert build_body["backend_mode"] == "mock"
        status_body = client.get("/ops/status").json()
        assert status_body["backend_mode"] == "mock"
        assert "primary_timeouts" in status_body
        response = client.post("/v1/predict", json={"text": "normal request"})
        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == "primary-model"
        assert body["fallback_used"] is False
        assert body["batch_size"] >= 1


def test_primary_failure_is_served_by_fallback():
    with TestClient(app) as client:
        response = client.post("/v1/predict", json={"text": "__FAIL_PRIMARY__"})
        assert response.status_code == 200
        body = response.json()
        assert body["backend"] == "fallback-model"
        assert body["fallback_used"] is True
