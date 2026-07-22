"""Run deterministic CPU retrieval-quality and output-budget gates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpu_service.bench import run_quality_manifest  # noqa: E402
from gpu_service.quality_benchmark import (  # noqa: E402
    compare_baseline,
    make_baseline,
)


CASES = {
    "csharp": {
        "fixture": "benchmarks/fixtures/csharp",
        "manifest": "benchmarks/manifests/csharp.json",
        "modes": ["exact", "symbol"],
    },
    "typescript": {
        "fixture": "benchmarks/fixtures/typescript",
        "manifest": "benchmarks/manifests/typescript.json",
        "modes": ["exact"],
    },
    "python": {
        "fixture": "benchmarks/fixtures/python",
        "manifest": "benchmarks/manifests/python.json",
        "modes": ["exact"],
    },
    "mixed": {
        "fixture": "benchmarks/fixtures/mixed",
        "manifest": "benchmarks/manifests/mixed.json",
        "modes": ["exact"],
    },
}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_gate(
    *,
    baseline_dir: Path,
    output_dir: Path | None = None,
    update_baselines: bool = False,
    max_quality_drop: float = 0.0,
    max_token_increase_pct: float = 10.0,
    max_returned_tokens: int = 1024,
) -> dict:
    os.environ["GPU_SEARCH_DEVICE"] = "cpu"
    reports: dict[str, dict] = {}
    regressions: list[dict] = []

    for name, case in CASES.items():
        report = run_quality_manifest(
            str(ROOT / case["fixture"]),
            str(ROOT / case["manifest"]),
            modes=list(case["modes"]),
            iterations=1,
            top_k=10,
        )
        reports[name] = make_baseline(report)
        if output_dir is not None:
            _write_json(output_dir / f"{name}.json", report)

        baseline_path = baseline_dir / f"{name}.json"
        if update_baselines:
            _write_json(baseline_path, reports[name])
            continue
        if not baseline_path.is_file():
            regressions.append({
                "case": name,
                "kind": "missing_baseline",
                "baseline": str(baseline_path),
            })
            continue

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for field in ("repository", "language", "query_count", "top_k"):
            if baseline.get(field) != reports[name].get(field):
                regressions.append({
                    "case": name,
                    "kind": "baseline_contract",
                    "field": field,
                    "baseline": baseline.get(field),
                    "current": reports[name].get(field),
                })
        for regression in compare_baseline(
            reports[name],
            baseline,
            max_quality_drop=max_quality_drop,
            max_token_increase_pct=max_token_increase_pct,
            max_returned_tokens=max_returned_tokens,
        ):
            regressions.append({"case": name, **regression})

    return {
        "passed": not regressions,
        "device": "cpu",
        "policy": {
            "max_quality_drop": max_quality_drop,
            "max_token_increase_pct": max_token_increase_pct,
            "max_returned_tokens": max_returned_tokens,
            "latency_gate": None,
        },
        "cases": reports,
        "regressions": regressions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic CPU retrieval-quality regression gates"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "benchmarks" / "baselines",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--update-baselines", action="store_true")
    parser.add_argument("--max-quality-drop", type=float, default=0.0)
    parser.add_argument("--max-token-increase-pct", type=float, default=10.0)
    parser.add_argument("--max-returned-tokens", type=int, default=1024)
    args = parser.parse_args(argv)

    result = run_gate(
        baseline_dir=args.baseline_dir,
        output_dir=args.output_dir,
        update_baselines=args.update_baselines,
        max_quality_drop=args.max_quality_drop,
        max_token_increase_pct=args.max_token_increase_pct,
        max_returned_tokens=args.max_returned_tokens,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
