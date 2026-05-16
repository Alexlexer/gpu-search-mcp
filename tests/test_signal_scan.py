"""Tests for POST /scan/signals and scan_repository_signals MCP tool."""

import sys
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server

_SPEC_PATH = REPO_ROOT / "docs" / "openapi" / "gpu-search-mcp.openapi.yaml"


def _post_http(path: str, payload: dict) -> tuple[int, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _FakeIndex:
    """Minimal in-memory index that searches file content by substring.

    Requires at least one file so the handler does not hit the early-return
    "no index" branch. Pass a dummy file when testing structure, not signal hits.
    """

    def __init__(self, base: str, file_contents: dict[str, str]):
        self._base = base
        self._files = file_contents  # {abs_path: content}

    def stats(self):
        # Report at least 1 so the endpoint skips the "not indexed" early return.
        return {"files": max(1, len(self._files)), "base_dir": self._base, "vram_mb": 0}

    def search(self, query, case_sensitive=False, max_files=50):
        results = []
        for fpath, content in self._files.items():
            matches = [
                {"line": i, "content": line}
                for i, line in enumerate(content.splitlines(), start=1)
                if query in line
            ]
            if matches:
                results.append({
                    "file": fpath,
                    "matches": matches,
                    "_total_files": len(results) + 1,
                })
        return results


class _EmptyIndex:
    def stats(self):
        return {"files": 0, "base_dir": None, "vram_mb": 0}

    def search(self, query, case_sensitive=False, max_files=50):
        return []


class _FakeSemantic:
    def stats(self):
        return {"chunks": 0, "base_dir": None, "vram_mb": 0}


# ---------------------------------------------------------------------------
# Basic response structure
# ---------------------------------------------------------------------------

def test_scan_signals_returns_structured_response(tmp_path, monkeypatch):
    src = tmp_path / "Startup.cs"
    src.write_text("using System.Web;\n", encoding="utf-8")

    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {str(src): src.read_text()}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    assert isinstance(body["result"], str)
    assert "categories" in body
    assert "summary" in body
    assert "signals" in body
    assert "limitations" in body
    assert "warnings" in body
    assert isinstance(body["signals"], list)
    assert isinstance(body["limitations"], list)
    assert isinstance(body["warnings"], list)


def test_scan_signals_summary_fields_present(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    s = body["summary"]
    assert "signalCount" in s
    assert "matchCount" in s
    assert "categories" in s
    assert isinstance(s["categories"], dict)


def test_scan_signals_no_index_returns_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _EmptyIndex())
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    assert body["warnings"]
    assert any("gpu_index" in w or "index" in w.lower() for w in body["warnings"])
    assert body["signals"] == []


# ---------------------------------------------------------------------------
# Default scan covers expected categories
# ---------------------------------------------------------------------------

def test_default_scan_includes_all_builtin_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    expected = {"legacy-dotnet", "config", "sql", "async-risk", "exception-risk", "di", "tests"}
    actual = set(body["categories"])
    assert expected <= actual, f"Missing categories: {expected - actual}"


def test_default_scan_signals_list_covers_all_builtin_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    returned_ids = {s["id"] for s in body["signals"]}
    builtin_ids = {s["id"] for s in mcp_server._BUILTIN_SIGNALS}
    assert builtin_ids == returned_ids


# ---------------------------------------------------------------------------
# Category filter
# ---------------------------------------------------------------------------

def test_category_filter_limits_returned_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["sql"]})
    assert status == 200
    assert body["categories"] == ["sql"]
    assert all(s["category"] == "sql" for s in body["signals"])


def test_category_filter_multiple(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["legacy-dotnet", "di"]})
    assert status == 200
    actual_cats = set(body["categories"])
    assert actual_cats == {"legacy-dotnet", "di"}


# ---------------------------------------------------------------------------
# topKPerSignal cap
# ---------------------------------------------------------------------------

def test_top_k_per_signal_is_capped_at_20(tmp_path, monkeypatch):
    # Build a file with 30 occurrences of SqlConnection
    content = "\n".join(f"var c{i} = new SqlConnection(cs);" for i in range(30))
    src = tmp_path / "Db.cs"
    src.write_text(content, encoding="utf-8")

    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {str(src): content}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    # Request more than the max
    status, body = _post_http("/scan/signals", {"categories": ["sql"], "topKPerSignal": 99})
    assert status == 200
    for sig in body["signals"]:
        assert len(sig["matches"]) <= 20, f"Signal '{sig['id']}' exceeded cap"


def test_top_k_per_signal_defaults_to_5(tmp_path, monkeypatch):
    content = "\n".join(f"new SqlConnection(cs{i});" for i in range(10))
    src = tmp_path / "Db.cs"
    src.write_text(content, encoding="utf-8")

    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {str(src): content}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["sql"]})
    assert status == 200
    sql_conn = next((s for s in body["signals"] if s["id"] == "sql-connection"), None)
    assert sql_conn is not None
    assert len(sql_conn["matches"]) <= 5


# ---------------------------------------------------------------------------
# Signal detection from fixture content
# ---------------------------------------------------------------------------

def test_web_config_signal_detected(tmp_path, monkeypatch):
    # The signal query "web.config" searches file *content* (the GPU index is content-based).
    # Use a .csproj that references the file — this is realistic and contains the query token.
    csproj = tmp_path / "App.csproj"
    csproj.write_text(
        '<Project Sdk="Microsoft.NET.Sdk">\n'
        '  <ItemGroup>\n'
        '    <Content Include="web.config" />\n'
        '  </ItemGroup>\n'
        '</Project>\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(csproj): csproj.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["legacy-dotnet"]})
    assert status == 200
    web_cfg = next((s for s in body["signals"] if s["id"] == "legacy-web-config"), None)
    assert web_cfg is not None
    assert web_cfg["matches"], "Expected web.config signal to have matches"
    assert web_cfg["confidence"] == "high"


def test_packages_config_signal_detected(tmp_path, monkeypatch):
    # Same principle: put "packages.config" in a file's content (e.g., a README or .csproj).
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Legacy App\n\nThis project uses packages.config for NuGet package management.\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(readme): readme.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["legacy-dotnet"]})
    assert status == 200
    pkgs = next((s for s in body["signals"] if s["id"] == "legacy-packages-config"), None)
    assert pkgs is not None
    assert pkgs["matches"]


def test_sql_connection_signal_detected(tmp_path, monkeypatch):
    src = tmp_path / "Repo.cs"
    src.write_text(
        "using System.Data.SqlClient;\n"
        "var conn = new SqlConnection(connectionString);\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["sql"]})
    assert status == 200
    sig = next((s for s in body["signals"] if s["id"] == "sql-connection"), None)
    assert sig is not None
    assert sig["matches"]
    assert sig["matches"][0]["engine"] == "pattern"


def test_catch_exception_signal_detected(tmp_path, monkeypatch):
    src = tmp_path / "Handler.cs"
    src.write_text(
        "try { DoWork(); }\ncatch (Exception ex) { Log(ex); }\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["exception-risk"]})
    assert status == 200
    sig = next((s for s in body["signals"] if s["id"] == "broad-catch-exception"), None)
    assert sig is not None
    assert sig["matches"]


# ---------------------------------------------------------------------------
# Snippet redaction
# ---------------------------------------------------------------------------

def test_signal_snippets_are_redacted(tmp_path, monkeypatch):
    src = tmp_path / "Config.cs"
    src.write_text(
        'var conn = new SqlConnection("password=mysecret123");\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["sql"]})
    assert status == 200
    sig = next((s for s in body["signals"] if s["id"] == "sql-connection"), None)
    assert sig is not None
    assert sig["matches"]
    snippet = sig["matches"][0]["snippet"]
    assert "mysecret123" not in snippet
    assert "[REDACTED]" in snippet


def test_include_snippets_false_omits_snippet_field(tmp_path, monkeypatch):
    src = tmp_path / "Db.cs"
    src.write_text("var c = new SqlConnection(cs);\n", encoding="utf-8")

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["sql"], "includeSnippets": False})
    assert status == 200
    sig = next((s for s in body["signals"] if s["id"] == "sql-connection"), None)
    assert sig is not None
    if sig["matches"]:
        assert "snippet" not in sig["matches"][0]


# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------

def test_signal_summary_counts_are_correct(tmp_path, monkeypatch):
    src = tmp_path / "Startup.cs"
    src.write_text(
        "services.AddSingleton<IFoo, Foo>();\n"
        "services.AddScoped<IBar, Bar>();\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["di"]})
    assert status == 200
    match_count = sum(len(s["matches"]) for s in body["signals"])
    assert body["summary"]["matchCount"] == match_count
    signal_count = sum(1 for s in body["signals"] if s["matches"])
    assert body["summary"]["signalCount"] == signal_count


# ---------------------------------------------------------------------------
# Each signal has required fields
# ---------------------------------------------------------------------------

def test_each_signal_has_required_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(tmp_path), {}))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {})
    assert status == 200
    required = {"id", "category", "label", "description", "confidence", "query", "matches"}
    for sig in body["signals"]:
        missing = required - sig.keys()
        assert not missing, f"Signal '{sig.get('id')}' missing: {missing}"
        assert sig["confidence"] in ("high", "medium", "low")
        assert isinstance(sig["matches"], list)


# ---------------------------------------------------------------------------
# Match records have required fields when present
# ---------------------------------------------------------------------------

def test_signal_matches_have_required_fields(tmp_path, monkeypatch):
    src = tmp_path / "App.cs"
    src.write_text("services.AddSingleton<IFoo, Foo>();\n", encoding="utf-8")

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())

    status, body = _post_http("/scan/signals", {"categories": ["di"]})
    assert status == 200
    for sig in body["signals"]:
        for m in sig["matches"]:
            for key in ("file", "absoluteFile", "lineStart", "lineEnd", "score", "reason", "engine"):
                assert key in m, f"Match missing key '{key}' in signal '{sig['id']}'"
            assert m["engine"] == "pattern"


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------

def test_scan_repository_signals_mcp_returns_string(tmp_path, monkeypatch):
    src = tmp_path / "Startup.cs"
    src.write_text("services.AddSingleton<IFoo, Foo>();\n", encoding="utf-8")

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    result = mcp_server.scan_repository_signals()
    assert isinstance(result, str)
    assert "signal" in result.lower()


def test_scan_repository_signals_mcp_no_index(monkeypatch):
    monkeypatch.setattr(mcp_server, "index", _EmptyIndex())
    result = mcp_server.scan_repository_signals()
    assert "gpu_index" in result or "index" in result.lower()


def test_scan_repository_signals_mcp_category_filter(tmp_path, monkeypatch):
    src = tmp_path / "Db.cs"
    src.write_text("var c = new SqlConnection(cs);\n", encoding="utf-8")

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(tmp_path), {str(src): src.read_text()})
    )
    result = mcp_server.scan_repository_signals(categories=["sql"])
    assert "sql" in result.lower() or "SqlConnection" in result or "sql-connection" in result


# ---------------------------------------------------------------------------
# OpenAPI contract includes /scan/signals
# ---------------------------------------------------------------------------

def test_openapi_includes_scan_signals_path():
    with open(_SPEC_PATH, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    assert "/scan/signals" in spec["paths"], "/scan/signals not found in OpenAPI spec paths"


def test_openapi_includes_signal_scan_schemas():
    with open(_SPEC_PATH, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    schemas = spec.get("components", {}).get("schemas", {})
    for name in ("SignalScanRequest", "SignalScanResponse", "RepositorySignal", "SignalScanSummary"):
        assert name in schemas, f"Schema '{name}' missing from OpenAPI spec"
