from __future__ import annotations

import hashlib
import json
from typing import Any


def choose_serving_policy(
    *,
    profile: str,
    traffic: dict[str, int],
    results: dict[str, dict[str, Any]],
    target_p95_ms: float,
    min_success_rate: float,
) -> dict[str, Any]:
    evaluated: dict[str, dict[str, Any]] = {}
    feasible: list[str] = []

    for name, result in results.items():
        reasons: list[str] = []
        if result["success_rate"] < min_success_rate:
            reasons.append(
                f"success {result['success_rate']:.3f} < {min_success_rate:.3f}"
            )
        if result["p95_ms"] > target_p95_ms:
            reasons.append(f"p95 {result['p95_ms']:.2f}ms > {target_p95_ms:.2f}ms")
        passed = not reasons
        if passed:
            feasible.append(name)
        evaluated[name] = {
            **result,
            "slo_pass": passed,
            "rejection_reasons": reasons,
        }

    if feasible:
        selected = min(
            feasible,
            key=lambda name: (
                evaluated[name]["backend_calls_per_100_success"],
                evaluated[name]["p95_ms"],
                -evaluated[name]["throughput_rps"],
            ),
        )
        why = (
            "SLO를 만족한 정책 중 backend 호출/100 성공 요청이 가장 낮고, "
            "동률이면 p95와 admitted throughput을 순서대로 비교했습니다."
        )
        status = "slo_met"
    else:
        selected = min(
            evaluated,
            key=lambda name: (
                -evaluated[name]["success_rate"],
                evaluated[name]["p95_ms"],
                evaluated[name]["backend_calls_per_100_success"],
            ),
        )
        why = (
            "모든 후보가 SLO를 동시에 만족하지 못해 success rate를 우선하고, "
            "그 다음 p95와 backend 호출 비용을 비교했습니다."
        )
        status = "best_effort"

    fingerprint_payload = json.dumps(
        {"profile": profile, "traffic": traffic},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()[:16]

    return {
        "status": status,
        "workload_fingerprint": fingerprint,
        "objective": {
            "target_p95_ms": target_p95_ms,
            "min_success_rate": min_success_rate,
            "cost_proxy": "backend calls per 100 successful requests",
        },
        "selected_policy": selected,
        "selected_config": evaluated[selected]["policy"],
        "why": why,
        "evidence": evaluated,
    }
