from __future__ import annotations

import json
from pathlib import Path

from gpu_service.quality_benchmark import compare_baseline, load_manifest
from scripts import quality_gate


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "benchmarks" / "baselines"
MANIFESTS = ROOT / "benchmarks" / "manifests"
CASES = ("csharp", "typescript", "python", "mixed")


def test_checked_in_baselines_are_portable_and_match_manifests() -> None:
    for name in CASES:
        baseline = json.loads(
            (BASELINES / f"{name}.json").read_text(encoding="utf-8")
        )
        manifest = load_manifest(MANIFESTS / f"{name}.json")

        assert baseline["repository"] == manifest.repository
        assert baseline["language"] == manifest.language
        assert baseline["query_count"] == len(manifest.queries)
        assert baseline["top_k"] == 10
        assert "runtime" not in baseline
        for mode in baseline["modes"].values():
            assert "latency_ms_p95" not in mode["aggregate"]
            assert mode["aggregate"]["returned_tokens_max"] <= 1024
        assert not compare_baseline(
            baseline,
            baseline,
            max_quality_drop=0.0,
            max_token_increase_pct=10.0,
            max_returned_tokens=1024,
        )


def test_gate_uses_all_language_baselines_without_latency(
    monkeypatch,
) -> None:
    def fake_run(
        directory: str,
        manifest_path: str,
        *,
        modes: list[str],
        iterations: int,
        top_k: int,
    ) -> dict:
        del directory, modes, iterations, top_k
        name = Path(manifest_path).stem
        baseline = json.loads(
            (BASELINES / f"{name}.json").read_text(encoding="utf-8")
        )
        return {**baseline, "runtime": {"index_build_ms": 123.0}}

    monkeypatch.setattr(quality_gate, "run_quality_manifest", fake_run)

    result = quality_gate.run_gate(baseline_dir=BASELINES)

    assert result["passed"] is True
    assert set(result["cases"]) == set(CASES)
    assert result["policy"]["latency_gate"] is None
    assert result["regressions"] == []
