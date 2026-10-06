from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        demo = client.get("/")
        assert demo.status_code == 200
        assert "InferenceRail" in demo.text
        assert "SLO를 입력하면" in demo.text
        assert "SLO Compiler" in demo.text
        assert "Deployment Contract" in demo.text
        assert "}\\nfunction" not in demo.text
        assert "Safe Operating Envelope" in demo.text
        assert "CALLS/100 SUCCESS" in demo.text
        build_body = client.get("/ops/build").json()
        assert build_body["app"] == "InferenceRail"
        assert build_body["version"] == "0.6.0"
        assert build_body["backend_mode"] == "mock"
        status_body = client.get("/ops/status").json()
        assert status_body["backend_mode"] == "mock"
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


def test_request_flight_recorder_captures_serving_lifecycle():
    with TestClient(app) as client:
        response = client.post(
            "/v1/predict",
            headers={"x-request-id": "flight-demo-1"},
            json={"text": "flight recorder request"},
        )
        assert response.status_code == 200
        flight = client.get("/ops/flights/flight-demo-1")
        assert flight.status_code == 200
        body = flight.json()
        assert body["status"] == "ok"
        stages = [event["stage"] for event in body["events"]]
        for stage in (
            "received",
            "admitted",
            "queued",
            "batch_assigned",
            "backend_dispatched",
            "backend_result",
        ):
            assert stage in stages


def test_shadow_compare_never_changes_live_serving_decision():
    with TestClient(app) as client:
        response = client.post(
            "/v1/shadow-compare",
            json={"text": "synthetic shadow request"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "shadow_only"
        assert body["serving_decision"] == "stable-v17"
        assert body["stable"]["backend"] == "stable-v17"
        assert body["candidate"]["backend"] == "candidate-v18"


def test_policy_lab_compares_two_real_policy_paths():
    with TestClient(app) as client:
        response = client.post("/v1/policy-lab", json={"profile": "steady"})
        assert response.status_code == 200
        body = response.json()
        assert set(body["policies"]) == {
            "latency_guard",
            "throughput_guard",
            "availability_guard",
        }
        for result in body["policies"].values():
            assert "p95_ms" in result
            assert "throughput_rps" in result
            assert "status_counts" in result


def test_slo_governor_emits_decision_receipt():
    with TestClient(app) as client:
        response = client.post(
            "/v1/slo-decision",
            json={
                "profile": "flash_crowd",
                "target_p95_ms": 250,
                "min_success_rate": 0.995,
            },
        )
        assert response.status_code == 200
        body = response.json()
        receipt = body["decision_receipt"]
        assert receipt["selected_policy"] in {
            "latency_guard",
            "throughput_guard",
            "availability_guard",
        }
        assert len(receipt["workload_fingerprint"]) == 16
        assert receipt["objective"]["target_p95_ms"] == 250
        assert set(receipt["evidence"]) == {
            "latency_guard",
            "throughput_guard",
            "availability_guard",
        }
        for result in receipt["evidence"].values():
            assert "slo_pass" in result
            assert "backend_calls" in result
            assert "backend_calls_per_100_success" in result
            assert "throughput_rps" in result
            assert "p95_ms" in result
        selected = receipt["evidence"][receipt["selected_policy"]]
        assert selected["policy"] == receipt["selected_config"]
        envelope = receipt["safe_operating_envelope"]
        assert envelope["method"] == "measured_concurrency_sweep"
        assert [point["concurrency"] for point in envelope["points"]] == [
            4,
            8,
            16,
            24,
            32,
            40,
        ]
        contract = receipt["deployment_contract"]
        assert contract["runtime_config"] == receipt["selected_config"]
        assert contract["evidence_basis"].startswith("measured concurrency sweep")
