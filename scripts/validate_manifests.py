from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def validate() -> dict:
    checked: list[str] = []
    for relative in [
        "docker-compose.yml",
        "render.yaml",
        "prometheus/prometheus.yml",
        "prometheus/alerts.yml",
        "otel/collector.yaml",
        "grafana/provisioning/datasources/datasource.yml",
        "grafana/provisioning/dashboards/dashboard.yml",
        "k8s/configmap.yaml",
        "k8s/configmap-triton.yaml",
        "k8s/deployment.yaml",
        "k8s/service.yaml",
        "k8s/hpa.yaml",
        "k8s/pdb.yaml",
        "k8s/keda-scaledobject.yaml",
        "k8s/servicemonitor.yaml",
    ]:
        path = ROOT / relative
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        if not docs or all(doc is None for doc in docs):
            raise RuntimeError(f"{relative}: no YAML document")
        checked.append(relative)

    dashboard = json.loads(
        (ROOT / "grafana/dashboards/inference-rail.json").read_text(encoding="utf-8")
    )
    if len(dashboard.get("panels", [])) < 4:
        raise RuntimeError("Grafana dashboard must expose at least four reliability panels")

    render = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    services = render.get("services", [])
    if not services or services[0].get("healthCheckPath") != "/health/ready":
        raise RuntimeError("Render service must use /health/ready")

    deployment = yaml.safe_load((ROOT / "k8s/deployment.yaml").read_text(encoding="utf-8"))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        if probe not in container:
            raise RuntimeError(f"deployment missing {probe}")
    if container.get("securityContext", {}).get("allowPrivilegeEscalation") is not False:
        raise RuntimeError("container must disable privilege escalation")

    return {
        "passed": True,
        "checked_yaml": checked,
        "grafana_panels": len(dashboard["panels"]),
        "render_health_check": services[0]["healthCheckPath"],
    }


def main() -> None:
    print(json.dumps(validate(), indent=2))


if __name__ == "__main__":
    main()
