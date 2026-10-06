from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--max-p95-ms", type=float, default=250)
    parser.add_argument("--min-success-rate", type=float, default=0.995)
    args = parser.parse_args()
    with open(args.result, encoding="utf-8") as fp:
        data = json.load(fp)
    target = data.get("batched", data)
    failures = []
    if target["p95_ms"] > args.max_p95_ms:
        failures.append(f"p95 {target['p95_ms']}ms > {args.max_p95_ms}ms")
    if target["success_rate"] < args.min_success_rate:
        failures.append(f"success_rate {target['success_rate']} < {args.min_success_rate}")
    if failures:
        print("SLO GATE FAILED:", "; ".join(failures))
        sys.exit(1)
    print("SLO GATE PASSED")


if __name__ == "__main__":
    main()
