"""Lightweight validation that the OpenAPI contract file is well-formed."""

import pathlib

import pytest
import yaml

_SPEC_PATH = pathlib.Path(__file__).parent.parent / "docs" / "openapi" / "gpu-search-mcp.openapi.yaml"

_EXPECTED_PATHS = [
    "/health",
    "/stats",
    "/diagnostics",
    "/search/code",
    "/search/semantic",
    "/search/hybrid",
    "/semantic/model/status",
    "/read/block",
    "/read/skeleton",
    "/dependency/impact",
]

_EXPECTED_SCHEMAS = [
    "ErrorResponse",
    "HealthResponse",
    "StatsResponse",
    "DiagnosticsResponse",
    "DiagnosticsDevice",
    "DiagnosticsIndexStatus",
    "DiagnosticsCacheStatus",
    "DiagnosticsCapabilities",
    "SemanticModelStatus",
    "SearchRequest",
    "SearchResult",
    "SearchRelatedFile",
    "SearchRelatedFiles",
    "SearchIndexStatus",
    "SearchResponse",
    "ReadBlockRequest",
    "ReadBlockResponse",
    "ReadSkeletonRequest",
    "ReadSkeletonResponse",
    "DependencyImpactRequest",
    "ImpactedFile",
    "DependencyImpactResponse",
]


@pytest.fixture(scope="module")
def spec():
    with open(_SPEC_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_spec_file_exists():
    assert _SPEC_PATH.exists(), f"OpenAPI spec not found at {_SPEC_PATH}"


def test_top_level_keys(spec):
    for key in ("openapi", "info", "paths", "components"):
        assert key in spec, f"Missing top-level key: {key}"


def test_openapi_version(spec):
    assert spec["openapi"].startswith("3.1"), "Expected OpenAPI 3.1.x"


def test_info_fields(spec):
    info = spec["info"]
    assert "title" in info
    assert "version" in info


def test_all_expected_paths_present(spec):
    missing = [p for p in _EXPECTED_PATHS if p not in spec["paths"]]
    assert not missing, f"Missing paths in spec: {missing}"


def test_all_expected_schemas_present(spec):
    schemas = spec.get("components", {}).get("schemas", {})
    missing = [s for s in _EXPECTED_SCHEMAS if s not in schemas]
    assert not missing, f"Missing schemas in components/schemas: {missing}"


def test_each_path_has_at_least_one_operation(spec):
    for path, item in spec["paths"].items():
        ops = {k for k in item if k in ("get", "post", "put", "patch", "delete", "head", "options")}
        assert ops, f"Path {path!r} has no HTTP operations"


def test_each_operation_has_operation_id(spec):
    for path, item in spec["paths"].items():
        for method, op in item.items():
            if method not in ("get", "post", "put", "patch", "delete"):
                continue
            assert "operationId" in op, f"Missing operationId on {method.upper()} {path}"


def test_error_response_schema(spec):
    schemas = spec["components"]["schemas"]
    err = schemas["ErrorResponse"]
    assert "properties" in err
    assert "error" in err["properties"]


def test_impacted_file_schema_includes_optional_reason(spec):
    impacted = spec["components"]["schemas"]["ImpactedFile"]
    props = impacted["properties"]
    assert "reason" in props
    assert props["reason"]["type"] == "string"
    assert "Human-readable heuristic explanation" in props["reason"]["description"]
    assert "reason" not in impacted.get("required", [])


def test_dependency_impact_example_includes_reason(spec):
    response = spec["paths"]["/dependency/impact"]["post"]["responses"]["200"]
    examples = response["content"]["application/json"]["examples"]
    impacted_files = examples["with_impact"]["value"]["impactedFiles"]
    assert impacted_files
    assert impacted_files[0]["reason"] == "references type UserService"


def test_server_url_is_localhost(spec):
    servers = spec.get("servers", [])
    assert servers, "No servers defined in spec"
    url = servers[0]["url"]
    assert "127.0.0.1" in url or "localhost" in url, f"Server URL should be localhost, got: {url}"


def test_stats_schema_includes_semantic_model(spec):
    props = spec["components"]["schemas"]["StatsResponse"]["properties"]
    assert "semanticModel" in props
    assert props["semanticModel"]["$ref"] == "#/components/schemas/SemanticModelStatus"


def test_semantic_model_status_endpoint_schema(spec):
    op = spec["paths"]["/semantic/model/status"]["get"]
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"] == "#/components/schemas/SemanticModelStatus"


def test_openapi_includes_diagnostics_path(spec):
    assert "/diagnostics" in spec["paths"]
    op = spec["paths"]["/diagnostics"]["get"]
    assert op["operationId"] == "getDiagnostics"
    schema = op["responses"]["200"]["content"]["application/json"]["schema"]
    assert schema["$ref"] == "#/components/schemas/DiagnosticsResponse"


def test_unified_search_request_contract(spec):
    request = spec["components"]["schemas"]["SearchRequest"]
    properties = request["properties"]

    assert properties["mode"]["enum"] == [
        "auto", "exact", "pattern", "semantic", "hybrid", "symbol", "path"
    ]
    assert properties["intent"]["enum"] == [
        "locate", "understand", "modify", "debug", "audit"
    ]
    assert {
        "contextMode",
        "context_mode",
        "topK",
        "top_k",
        "includeDependencies",
        "include_dependencies",
        "includeTests",
        "include_tests",
    } <= set(properties)


def test_unified_search_response_contract(spec):
    response = spec["components"]["schemas"]["SearchResponse"]
    required = set(response["required"])
    properties = response["properties"]

    assert {
        "result",
        "query",
        "mode",
        "contextMode",
        "results",
        "mode_used",
        "intent",
        "primary_results",
        "related_files",
        "warnings",
        "index_status",
    } <= required
    assert properties["related_files"]["$ref"].endswith("/SearchRelatedFiles")
    assert properties["index_status"]["$ref"].endswith("/SearchIndexStatus")


def test_search_related_files_schema_has_all_relation_buckets(spec):
    related = spec["components"]["schemas"]["SearchRelatedFiles"]

    assert set(related["required"]) == {
        "callers",
        "dependencies",
        "implementations",
        "tests",
        "configuration",
    }