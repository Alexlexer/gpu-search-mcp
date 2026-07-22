"""Deterministic retrieval-quality benchmark contracts and metrics.

The evaluator is deliberately independent from the search implementation.  A
caller supplies a search function, which keeps metric tests fast and lets the
CLI compare exact, semantic, symbol, dependency-reranked, and external modes
without changing the manifest or scoring rules.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import yaml


SCHEMA_VERSION = 1
DEFAULT_MODES = ("exact", "symbol", "hybrid_dependencies")
QUALITY_METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "precision_at_5",
    "mean_reciprocal_rank",
    "exact_symbol_recall",
    "related_test_recall",
)


def _strings(value: object, field: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        values: list[object] = []
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise ValueError(f"{field} must be a list of strings")
    result = tuple(str(item).strip() for item in values if str(item).strip())
    if required and not result:
        raise ValueError(f"{field} must contain at least one value")
    return result


def _normal_path(value: str, root: str | None = None) -> str:
    value = str(value).strip()
    if root and os.path.isabs(value):
        try:
            value = os.path.relpath(value, root)
        except ValueError:
            pass
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/").casefold()


def _normal_symbol(value: str) -> str:
    return value.strip().casefold()


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    id: str
    query: str
    expected_files: tuple[str, ...]
    expected_symbols: tuple[str, ...] = ()
    expected_tests: tuple[str, ...] = ()
    exact_query: str | None = None
    symbol_query: str | None = None

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> "BenchmarkQuery":
        if not isinstance(raw, dict):
            raise ValueError(f"queries[{index}] must be an object")
        query_id = str(raw.get("id", "")).strip()
        query = str(raw.get("query", "")).strip()
        if not query_id:
            raise ValueError(f"queries[{index}].id is required")
        if not query:
            raise ValueError(f"queries[{index}].query is required")
        return cls(
            id=query_id,
            query=query,
            expected_files=_strings(
                raw.get("expected_files"),
                f"queries[{index}].expected_files",
                required=True,
            ),
            expected_symbols=_strings(
                raw.get("expected_symbols"), f"queries[{index}].expected_symbols"
            ),
            expected_tests=_strings(
                raw.get("expected_tests"), f"queries[{index}].expected_tests"
            ),
            exact_query=(str(raw["exact_query"]).strip() or None)
            if raw.get("exact_query") is not None else None,
            symbol_query=(str(raw["symbol_query"]).strip() or None)
            if raw.get("symbol_query") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    repository: str
    language: str
    queries: tuple[BenchmarkQuery, ...]
    modes: tuple[str, ...] = DEFAULT_MODES
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: dict) -> "BenchmarkManifest":
        if not isinstance(raw, dict):
            raise ValueError("benchmark manifest must be an object")
        schema_version = int(raw.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported benchmark schema_version {schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        repository = str(raw.get("repository", "")).strip()
        language = str(raw.get("language", "")).strip().casefold()
        if not repository:
            raise ValueError("repository is required")
        if language not in {"csharp", "typescript", "python", "mixed"}:
            raise ValueError("language must be csharp, typescript, python, or mixed")
        queries_raw = raw.get("queries")
        if not isinstance(queries_raw, list) or not queries_raw:
            raise ValueError("queries must contain at least one query")
        queries = tuple(
            BenchmarkQuery.from_dict(item, index)
            for index, item in enumerate(queries_raw)
        )
        ids = [query.id for query in queries]
        if len(ids) != len(set(ids)):
            raise ValueError("query ids must be unique")
        modes = _strings(raw.get("modes"), "modes") or DEFAULT_MODES
        return cls(
            repository=repository,
            language=language,
            queries=queries,
            modes=modes,
            schema_version=schema_version,
        )


def load_manifest(path: str | Path) -> BenchmarkManifest:
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.casefold() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)
    return BenchmarkManifest.from_dict(raw)


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _result_paths(response: dict, root: str | None) -> list[str]:
    results = response.get("primary_results")
    if not isinstance(results, list):
        results = response.get("results", [])
    return _dedupe(
        _normal_path(item.get("absoluteFile") or item.get("file") or "", root)
        for item in results if isinstance(item, dict)
    )


def _result_symbols(response: dict) -> list[str]:
    results = response.get("results", [])
    if not isinstance(results, list):
        return []
    return _dedupe(
        _normal_symbol(
            item.get("qualifiedName")
            or item.get("qualified_name")
            or item.get("symbol")
            or ""
        )
        for item in results if isinstance(item, dict)
    )


def _related_tests(response: dict, root: str | None) -> list[str]:
    related = response.get("related_files", {})
    tests = related.get("tests", []) if isinstance(related, dict) else []
    paths = [
        _normal_path(item.get("absoluteFile") or item.get("file") or "", root)
        for item in tests if isinstance(item, dict)
    ]
    all_results = response.get("results", [])
    if isinstance(all_results, list):
        paths.extend(
            _normal_path(item.get("absoluteFile") or item.get("file") or "", root)
            for item in all_results
            if isinstance(item, dict)
            and any(
                token in _normal_path(
                    item.get("absoluteFile") or item.get("file") or "", root
                ).split("/")
                for token in ("test", "tests", "spec", "specs")
            )
        )
    return _dedupe(paths)


def _recall(expected: set[str], ranked: list[str], k: int) -> float:
    if not expected:
        return 1.0
    return len(expected.intersection(ranked[:k])) / len(expected)


def _precision(expected: set[str], ranked: list[str], k: int) -> float:
    return len(expected.intersection(ranked[:k])) / k


def _mrr(expected: set[str], ranked: list[str]) -> float:
    for index, value in enumerate(ranked, start=1):
        if value in expected:
            return 1.0 / index
    return 0.0


def score_response(
    query: BenchmarkQuery,
    response: dict,
    *,
    repository_root: str | None = None,
) -> dict:
    ranked = _result_paths(response, repository_root)
    expected_files = {_normal_path(path) for path in query.expected_files}
    symbols = set(_result_symbols(response))
    expected_symbols = {_normal_symbol(value) for value in query.expected_symbols}
    tests = set(_related_tests(response, repository_root))
    expected_tests = {_normal_path(path) for path in query.expected_tests}
    return {
        "recall_at_1": round(_recall(expected_files, ranked, 1), 6),
        "recall_at_5": round(_recall(expected_files, ranked, 5), 6),
        "recall_at_10": round(_recall(expected_files, ranked, 10), 6),
        "precision_at_5": round(_precision(expected_files, ranked, 5), 6),
        "mean_reciprocal_rank": round(_mrr(expected_files, ranked), 6),
        "exact_symbol_recall": (
            round(len(symbols.intersection(expected_symbols)) / len(expected_symbols), 6)
            if expected_symbols else None
        ),
        "related_test_recall": (
            round(len(tests.intersection(expected_tests)) / len(expected_tests), 6)
            if expected_tests else None
        ),
        "ranked_files": ranked,
        "returned_symbols": sorted(symbols),
        "related_tests": sorted(tests),
    }


SearchFunction = Callable[[BenchmarkQuery, str, int], dict]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * fraction))]


def _mean_metric(results: list[dict], name: str) -> float | None:
    values = [result["metrics"][name] for result in results]
    available = [float(value) for value in values if value is not None]
    return round(statistics.fmean(available), 6) if available else None


def run_quality_benchmark(
    manifest: BenchmarkManifest,
    search: SearchFunction,
    *,
    modes: Iterable[str] | None = None,
    top_k: int = 10,
    iterations: int = 1,
    repository_root: str | None = None,
    runtime: dict | None = None,
) -> dict:
    if top_k < 10:
        raise ValueError("top_k must be at least 10 to calculate Recall@10")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    selected_modes = tuple(modes or manifest.modes)
    if not selected_modes:
        raise ValueError("at least one retrieval mode is required")

    mode_reports: dict[str, dict] = {}
    for mode in selected_modes:
        query_reports: list[dict] = []
        for query in manifest.queries:
            latencies: list[float] = []
            response: dict = {}
            for _ in range(iterations):
                started = time.perf_counter()
                response = search(query, mode, top_k)
                latencies.append((time.perf_counter() - started) * 1000)
            metrics = score_response(
                query, response, repository_root=repository_root
            )
            encoded = json.dumps(
                response, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            query_reports.append({
                "id": query.id,
                "query": query.query,
                "metrics": metrics,
                "latency_ms": {
                    "p50": round(statistics.median(latencies), 3),
                    "p95": round(_percentile(latencies, 0.95), 3),
                },
                "returned_tokens": math.ceil(len(encoded) / 4),
                "warnings": list(response.get("warnings", [])),
            })
        latency_values = [item["latency_ms"]["p50"] for item in query_reports]
        token_values = [item["returned_tokens"] for item in query_reports]
        mode_reports[mode] = {
            "aggregate": {
                **{
                    metric: _mean_metric(query_reports, metric)
                    for metric in QUALITY_METRICS
                },
                "latency_ms_p50": round(statistics.median(latency_values), 3),
                "latency_ms_p95": round(_percentile(latency_values, 0.95), 3),
                "returned_tokens_mean": round(statistics.fmean(token_values), 3),
                "returned_tokens_max": max(token_values),
            },
            "queries": query_reports,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": manifest.repository,
        "language": manifest.language,
        "query_count": len(manifest.queries),
        "top_k": top_k,
        "iterations": iterations,
        "runtime": runtime or {},
        "modes": mode_reports,
    }


def make_baseline(report: dict) -> dict:
    """Strip machine-dependent measurements from a quality report.

    Baselines retain retrieval metrics and serialized output size. Runtime,
    latency, device, and cache measurements stay in the full report so a future
    runner-specific policy can gate them without polluting portable baselines.
    """
    modes: dict[str, dict] = {}
    for mode, mode_report in sorted(report.get("modes", {}).items()):
        aggregate = mode_report.get("aggregate", {})
        queries = mode_report.get("queries", [])
        modes[mode] = {
            "aggregate": {
                **{
                    metric: aggregate.get(metric)
                    for metric in QUALITY_METRICS
                },
                "returned_tokens_mean": aggregate.get("returned_tokens_mean"),
                "returned_tokens_max": aggregate.get("returned_tokens_max"),
            },
            "queries": [
                {
                    "id": query.get("id"),
                    "metrics": query.get("metrics", {}),
                    "returned_tokens": query.get("returned_tokens"),
                }
                for query in queries
            ],
        }
    return {
        "schema_version": report.get("schema_version", SCHEMA_VERSION),
        "repository": report.get("repository"),
        "language": report.get("language"),
        "query_count": report.get("query_count"),
        "top_k": report.get("top_k"),
        "modes": modes,
    }


def compare_baseline(
    current: dict,
    baseline: dict,
    *,
    max_quality_drop: float = 0.0,
    max_latency_increase_pct: float | None = None,
    max_token_increase_pct: float | None = None,
    max_returned_tokens: int | None = None,
) -> list[dict]:
    """Return deterministic regression records; no thresholds are implicit."""
    regressions: list[dict] = []
    current_modes = current.get("modes", {})
    baseline_modes = baseline.get("modes", {})
    for mode in sorted(set(current_modes).intersection(baseline_modes)):
        now = current_modes[mode].get("aggregate", {})
        before = baseline_modes[mode].get("aggregate", {})
        for metric in QUALITY_METRICS:
            current_value = now.get(metric)
            baseline_value = before.get(metric)
            if current_value is None or baseline_value is None:
                continue
            drop = float(baseline_value) - float(current_value)
            if drop > max_quality_drop:
                regressions.append({
                    "mode": mode,
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "limit": max_quality_drop,
                    "kind": "quality_drop",
                })
        for metric, limit in (
            ("latency_ms_p95", max_latency_increase_pct),
            ("returned_tokens_max", max_token_increase_pct),
        ):
            if limit is None:
                continue
            current_value = float(now.get(metric, 0.0))
            baseline_value = float(before.get(metric, 0.0))
            if baseline_value <= 0:
                continue
            increase = ((current_value - baseline_value) / baseline_value) * 100
            if increase > limit:
                regressions.append({
                    "mode": mode,
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "limit_pct": limit,
                    "increase_pct": round(increase, 3),
                    "kind": "resource_increase",
                })
        if max_returned_tokens is not None:
            returned_tokens = int(now.get("returned_tokens_max", 0))
            if returned_tokens > max_returned_tokens:
                regressions.append({
                    "mode": mode,
                    "metric": "returned_tokens_max",
                    "current": returned_tokens,
                    "limit": max_returned_tokens,
                    "kind": "output_budget",
                })
    return regressions


def manifest_as_dict(manifest: BenchmarkManifest) -> dict:
    return asdict(manifest)


__all__ = [
    "BenchmarkManifest",
    "BenchmarkQuery",
    "QUALITY_METRICS",
    "SCHEMA_VERSION",
    "compare_baseline",
    "load_manifest",
    "make_baseline",
    "manifest_as_dict",
    "run_quality_benchmark",
    "score_response",
]
