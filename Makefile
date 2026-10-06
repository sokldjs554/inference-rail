.PHONY: run test lint benchmark slo manifests verify docker
run:
	uvicorn app.main:app --reload

test:
	python -m pytest -q

lint:
	ruff check .

benchmark:
	PYTHONPATH=. python scripts/benchmark_compare.py --output benchmark-results.json

slo:
	python scripts/slo_gate.py benchmark-results.json

manifests:
	PYTHONPATH=. python scripts/validate_manifests.py

verify:
	PYTHONPATH=. python scripts/release_verify.py

docker:
	docker compose up --build
