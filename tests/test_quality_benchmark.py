from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from gpu_service.bench import run_quality_manifest
from gpu_service.quality_benchmark import (
    BenchmarkManifest,
    BenchmarkQuery,
    compare_baseline,
    load_manifest,
    run_quality_benchmark,
    score_response,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "benchmarks" / "manifests"
FIXTURES = ROOT / "benchmarks" / "fixtures"


def test_language_manifests_cover_required_repository_types() -> None:
    loaded = [
        load_manifest(MANIFESTS / name)
        for name in ("csharp.json", "typescript.json", "python.json", "mixed.json")
    ]

    assert {manifest.language for manifest in loaded} == {
        "csharp",
        "typescript",
        "python",
        "mixed",
    }
    assert all(manifest.queries for manifest in loaded)
    assert all(
        query.expected_files and query.expected_symbols
        for manifest in loaded
        for query in manifest.queries
    )


def test_score_response_calculates_rank_symbol_and_test_metrics(
    tmp_path: Path,
) -> None:
    query = BenchmarkQuery(
        id="jwt",
        query="where is JWT expiration checked?",
        expected_files=("src/Auth/JwtValidator.cs", "src/Auth/TokenService.cs"),
        expected_symbols=("Sample.JwtValidator.ValidateExpiration",),
        expected_tests=("tests/Auth/JwtValidatorTests.cs",),
    )
    response = {
        "results": [
            {
                "absoluteFile": str(tmp_path / "src/Auth/JwtValidator.cs"),
                "qualifiedName": "Sample.JwtValidator.ValidateExpiration",
            },
            {"absoluteFile": str(tmp_path / "src/Auth/TokenService.cs")},
        ],
        "primary_results": [
            {"absoluteFile": str(tmp_path / "src/Auth/JwtValidator.cs")},
            {"absoluteFile": str(tmp_path / "src/Auth/TokenService.cs")},
        ],
        "related_files": {
            "tests": [
                {
                    "absoluteFile": str(
                        tmp_path / "tests/Auth/JwtValidatorTests.cs"
                    )
                }
            ]
        },
    }

    metrics = score_response(query, response, repository_root=str(tmp_path))

    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["precision_at_5"] == 0.4
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["exact_symbol_recall"] == 1.0
    assert metrics["related_test_recall"] == 1.0


def test_quality_report_is_deterministic_except_measured_latency() -> None:
    manifest = BenchmarkManifest(
        repository="fixture",
        language="python",
        modes=("exact",),
        queries=(
            BenchmarkQuery(
                id="one",
                query="find target",
                expected_files=("src/target.py",),
                expected_symbols=("target",),
                expected_tests=("tests/test_target.py",),
            ),
        ),
    )

    def search(query: BenchmarkQuery, mode: str, top_k: int) -> dict:
        assert query.id == "one"
        assert mode == "exact"
        assert top_k == 10
        return {
            "results": [
                {"file": "src/target.py", "qualifiedName": "target"},
                {"file": "tests/test_target.py"},
            ],
            "primary_results": [{"file": "src/target.py"}],
            "related_files": {"tests": [{"file": "tests/test_target.py"}]},
            "warnings": [],
        }

    first = run_quality_benchmark(manifest, search)
    second = run_quality_benchmark(manifest, search)
    first_query = first["modes"]["exact"]["queries"][0]
    second_query = second["modes"]["exact"]["queries"][0]

    assert first_query["metrics"] == second_query["metrics"]
    assert first_query["returned_tokens"] == second_query["returned_tokens"]
    assert first["modes"]["exact"]["aggregate"]["recall_at_10"] == 1.0
    assert first["modes"]["exact"]["aggregate"]["exact_symbol_recall"] == 1.0
    assert first["modes"]["exact"]["aggregate"]["related_test_recall"] == 1.0


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({}, "repository is required"),
        (
            {
                "repository": "x",
                "language": "rust",
                "queries": [{"id": "q", "query": "q", "expected_files": ["a"]}],
            },
            "language must be",
        ),
        (
            {
                "repository": "x",
                "language": "python",
                "queries": [
                    {"id": "q", "query": "q", "expected_files": ["a"]},
                    {"id": "q", "query": "q2", "expected_files": ["b"]},
                ],
            },
            "query ids must be unique",
        ),
    ],
)
def test_manifest_validation_is_explicit(raw: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        BenchmarkManifest.from_dict(raw)


def test_baseline_comparison_requires_explicit_limits() -> None:
    baseline = {
        "modes": {
            "exact": {
                "aggregate": {
                    "recall_at_1": 1.0,
                    "recall_at_5": 1.0,
                    "recall_at_10": 1.0,
                    "precision_at_5": 0.2,
                    "mean_reciprocal_rank": 1.0,
                    "exact_symbol_recall": None,
                    "related_test_recall": None,
                    "latency_ms_p95": 10.0,
                    "returned_tokens_max": 100,
                }
            }
        }
    }
    current = copy.deepcopy(baseline)
    current["modes"]["exact"]["aggregate"].update({
        "recall_at_10": 0.8,
        "latency_ms_p95": 13.0,
        "returned_tokens_max": 140,
    })

    regressions = compare_baseline(
        current,
        baseline,
        max_quality_drop=0.1,
        max_latency_increase_pct=20,
        max_token_increase_pct=25,
    )

    assert {
        (item["metric"], item["kind"]) for item in regressions
    } == {
        ("recall_at_10", "quality_drop"),
        ("latency_ms_p95", "resource_increase"),
        ("returned_tokens_max", "resource_increase"),
    }


def test_cpu_exact_manifest_runs_end_to_end(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "python-fixture"
    shutil.copytree(FIXTURES / "python", repository)
    monkeypatch.setenv("GPU_SEARCH_DEVICE", "cpu")

    report = run_quality_manifest(
        str(repository),
        str(MANIFESTS / "python.json"),
        modes=["exact"],
        iterations=1,
        top_k=10,
    )

    aggregate = report["modes"]["exact"]["aggregate"]
    assert aggregate["recall_at_5"] == 1.0
    assert aggregate["mean_reciprocal_rank"] == 1.0
    assert report["runtime"]["index_build_ms"] >= 0
    assert report["runtime"]["cache_size_bytes"] > 0
    json.dumps(report)
