import sys
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


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


def test_http_path_guard_allows_indexed_root(tmp_path: Path):
    allowed = tmp_path / "src" / "app.py"
    allowed.parent.mkdir()
    allowed.write_text("print('ok')\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        assert mcp_server._require_under_root(str(allowed)) == str(allowed.resolve())
    finally:
        mcp_server._http_roots = old_roots


def test_http_path_guard_rejects_outside_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-secret.txt"
    outside.write_text("secret\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path / "project")]
        with pytest.raises(ValueError, match="outside indexed roots"):
            mcp_server._require_under_root(str(outside))
    finally:
        mcp_server._http_roots = old_roots


def test_http_structured_pattern_results(tmp_path: Path):
    src = tmp_path / "UserService.cs"
    src.write_text("public class UserService {}\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        mcp_server.index.index_directory(str(tmp_path))
        payload = mcp_server._http_search_structured("UserService", mode="pattern")
        assert payload["results"]
        first = payload["results"][0]
        assert first["file"] == "UserService.cs"
        assert first["lineStart"] == 1
        assert first["reason"] == "exact token match"
        assert first["engine"] == "pattern"
    finally:
        mcp_server._http_roots = old_roots


def test_read_block_endpoint_rejects_parent_traversal(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("TOKEN = 'secret'\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http(
            "/read/block",
            {"filepath": str(project / ".." / "secret.py"), "line": 1},
        )
        assert status == 400
        assert "outside indexed roots" in body["error"]
    finally:
        mcp_server._http_roots = old_roots


def test_read_block_endpoint_rejects_absolute_path_outside_root(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/read/block", {"filepath": str(outside), "line": 1})
        assert status == 400
        assert "outside indexed roots" in body["error"]
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_endpoint_rejects_outside_root_path(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(outside)})
        assert status == 400
        assert "outside indexed roots" in body["error"]
    finally:
        mcp_server._http_roots = old_roots


def test_search_semantic_endpoint_returns_structured_results(tmp_path: Path, monkeypatch):
    src = tmp_path / "UserService.cs"
    src.write_text("public class UserService {}\n", encoding="utf-8")

    class FakeIndex:
        def stats(self):
            return {"files": 0, "base_dir": None, "vram_mb": 0}

    class FakeSemantic:
        def stats(self):
            return {"chunks": 1, "base_dir": str(tmp_path), "vram_mb": 0}

        def search(self, query, top_k=5):
            return [{
                "file": str(src),
                "start_line": 1,
                "end_line": 1,
                "score": 0.87,
                "snippet": "public class UserService {}",
            }]

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())
    status, body = _post_http(
        "/search/semantic",
        {"query": "user service", "topK": 1, "contextMode": "compact"},
    )

    assert status == 200
    assert isinstance(body["result"], str)
    assert body["mode"] == "semantic"
    assert body["results"]
    first = body["results"][0]
    assert first["file"] == "UserService.cs"
    assert first["lineStart"] == 1
    assert first["lineEnd"] == 1
    assert first["score"] == 0.87
    assert first["reason"] == "semantic match"
    assert first["engine"] == "semantic"


def test_require_under_root_rejects_empty_filepath(tmp_path: Path):
    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        with pytest.raises(ValueError, match="Missing filepath"):
            mcp_server._require_under_root("")
    finally:
        mcp_server._http_roots = old_roots


def test_require_under_root_handles_normalized_paths(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    src_dir = project / "src"
    src_dir.mkdir()
    src = src_dir / "app.py"
    src.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        # Path with redundant .. that still resolves inside the root
        redundant = str(project / "src" / ".." / "src" / "app.py")
        result = mcp_server._require_under_root(redundant)
        assert result == str(src.resolve())
    finally:
        mcp_server._http_roots = old_roots


def test_read_skeleton_endpoint_rejects_outside_root(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/read/skeleton", {"filepath": str(outside), "matchLines": []})
        assert status == 400
        assert "outside indexed roots" in body["error"]
    finally:
        mcp_server._http_roots = old_roots


def test_search_code_endpoint_returns_result_string_and_results_array(tmp_path: Path, monkeypatch):
    src = tmp_path / "Auth.cs"
    src.write_text("public class AuthService {}\n", encoding="utf-8")

    class FakeIndex:
        def stats(self):
            return {"files": 1, "base_dir": str(tmp_path), "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return [{
                "file": str(src),
                "matches": [{"line": 1, "content": "public class AuthService {}"}],
                "_total_files": 1,
            }]

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())
    status, body = _post_http(
        "/search/code",
        {"query": "AuthService", "mode": "pattern", "contextMode": "compact"},
    )

    assert status == 200
    assert isinstance(body["result"], str)
    assert "AuthService" in body["result"]
    assert isinstance(body["results"], list)
    assert body["results"]
    first = body["results"][0]
    assert first["file"] == "Auth.cs"
    assert first["lineStart"] == 1
    assert first["reason"] == "exact token match"
    assert first["engine"] == "pattern"


# ---------------------------------------------------------------------------
# Part 2 — structured search response schema
# ---------------------------------------------------------------------------

def test_pattern_structured_returns_all_required_fields(tmp_path: Path):
    src = tmp_path / "Repo.cs"
    src.write_text("public class Repo {}\n", encoding="utf-8")
    results = [{"file": str(src), "matches": [{"line": 1, "content": "public class Repo {}"}], "_total_files": 1}]
    stats = {"base_dir": str(tmp_path)}
    out = mcp_server._pattern_structured(results, stats, context_mode="compact")
    assert out
    first = out[0]
    for key in ("file", "absoluteFile", "lineStart", "lineEnd", "score", "reason", "snippet", "engine"):
        assert key in first, f"missing key: {key}"
    assert first["engine"] == "pattern"
    assert first["lineStart"] == 1


def test_semantic_structured_returns_all_required_fields(tmp_path: Path):
    src = tmp_path / "Service.py"
    src.write_text("class Service: pass\n", encoding="utf-8")
    results = [{"file": str(src), "start_line": 1, "end_line": 1, "score": 0.9, "snippet": "class Service: pass"}]
    stats = {"base_dir": str(tmp_path)}
    out = mcp_server._semantic_structured(results, stats, context_mode="compact")
    assert out
    first = out[0]
    for key in ("file", "absoluteFile", "lineStart", "lineEnd", "score", "reason", "snippet", "engine"):
        assert key in first, f"missing key: {key}"
    assert first["engine"] == "semantic"
    assert first["score"] == 0.9


def test_http_search_structured_returns_required_fields(monkeypatch):
    class FakeIndex:
        def stats(self):
            return {"files": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return [{"file": "/project/Foo.cs", "matches": [{"line": 10, "content": "class Foo {}"}], "_total_files": 1}]

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())

    result = mcp_server._http_search_structured("Foo", mode="pattern", context_mode="compact")
    assert "query" in result
    assert "mode" in result
    assert "contextMode" in result
    assert "results" in result
    assert isinstance(result["results"], list)


def test_snippets_are_redacted():
    secret_line = "password=mysecretpass123"
    results = [{"file": "/project/config.py", "matches": [{"line": 1, "content": secret_line}], "_total_files": 1}]
    stats = {"base_dir": "/project"}
    out = mcp_server._pattern_structured(results, stats, context_mode="compact")
    assert out
    assert "mysecretpass123" not in out[0]["snippet"]
    assert "[REDACTED]" in out[0]["snippet"]


def test_compact_mode_keeps_snippets_short():
    long_snippet = "x" * 700
    results = [{"file": "/project/big.py", "start_line": 1, "end_line": 10, "score": 0.9, "snippet": long_snippet}]
    stats = {"base_dir": "/project"}
    out = mcp_server._semantic_structured(results, stats, context_mode="compact")
    assert out
    assert len(out[0]["snippet"]) <= 160


def test_hybrid_mode_does_not_duplicate_results(monkeypatch):
    shared_file = "/project/Shared.cs"

    class FakeIndex:
        def stats(self):
            return {"files": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return [{"file": shared_file, "matches": [{"line": 5, "content": "class Shared {}"}], "_total_files": 1}]

    class FakeSemantic:
        def stats(self):
            return {"chunks": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, top_k=5):
            return [{"file": shared_file, "start_line": 1, "end_line": 5, "score": 0.9, "snippet": "class Shared {}"}]

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())

    result = mcp_server._http_search_structured("Shared", mode="hybrid", context_mode="compact")
    rel_files = [r["file"] for r in result["results"]]
    assert rel_files.count("Shared.cs") == 1


def test_http_search_structured_semantic_mode(monkeypatch):
    class FakeIndex:
        def stats(self):
            return {"files": 0, "base_dir": None, "vram_mb": 0}

    class FakeSemantic:
        def stats(self):
            return {"chunks": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, top_k=5):
            return [{"file": "/project/A.py", "start_line": 1, "end_line": 5, "score": 0.8, "snippet": "def foo(): pass"}]

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())

    result = mcp_server._http_search_structured("foo function", mode="semantic", context_mode="compact")
    assert result["mode"] == "semantic"


def test_http_search_structured_empty_results_returns_valid_response(monkeypatch):
    class FakeIndex:
        def stats(self):
            return {"files": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return []

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())

    result = mcp_server._http_search_structured("nonexistent_xyz_function", mode="pattern")
    assert result["query"] == "nonexistent_xyz_function"
    assert "mode" in result
    assert "contextMode" in result
    assert result["results"] == []



@pytest.mark.parametrize(
    ("mode", "intent", "semantic_ready", "expected_mode", "warning_fragment"),
    [
        ("exact", "understand", False, "pattern", None),
        ("auto", "modify", True, "hybrid", None),
        ("symbol", "locate", False, "pattern", "Symbol index"),
        ("path", "locate", False, "pattern", "path search"),
        ("auto", "debug", False, "pattern", "Semantic index"),
    ],
)
def test_resolve_search_request_modes_and_intents(
    mode, intent, semantic_ready, expected_mode, warning_fragment
):
    effective, normalized_intent, warnings = mcp_server._resolve_search_request(
        "TokenValidator",
        mode=mode,
        intent=intent,
        semantic_ready=semantic_ready,
    )

    assert effective == expected_mode
    assert normalized_intent == intent
    if warning_fragment is None:
        assert warnings == []
    else:
        assert any(warning_fragment in warning for warning in warnings)


def test_resolve_search_request_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unsupported search mode"):
        mcp_server._resolve_search_request("query", mode="unknown")
    with pytest.raises(ValueError, match="Unsupported search intent"):
        mcp_server._resolve_search_request("query", intent="rewrite")


def test_unified_structured_search_adds_related_context(tmp_path: Path, monkeypatch):
    source = tmp_path / "TokenValidator.cs"
    dependency = tmp_path / "TokenPolicy.cs"
    caller = tmp_path / "AuthController.cs"
    test_file = tmp_path / "tests" / "TokenValidatorTests.cs"
    config = tmp_path / "appsettings.json"
    test_file.parent.mkdir()
    for path in (source, dependency, caller, test_file, config):
        path.write_text("// fixture\n", encoding="utf-8")

    class FakeIndex:
        def stats(self):
            return {"files": 3, "base_dir": str(tmp_path), "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return [
                {
                    "file": str(source),
                    "matches": [{"line": 1, "content": "class TokenValidator {}"}],
                    "_total_files": 3,
                },
                {
                    "file": str(test_file),
                    "matches": [{"line": 1, "content": "class TokenValidatorTests {}"}],
                    "_total_files": 3,
                },
                {
                    "file": str(config),
                    "matches": [{"line": 1, "content": "{\"TokenValidator\": {}}"}],
                    "_total_files": 3,
                },
            ]

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    class FakeDeps:
        def stats(self):
            return {"files": 4, "base_dir": str(tmp_path), "vram_mb": 0}

        def direct_imports(self, filepath):
            return [str(dependency)] if filepath == str(source) else []

        def impact(self, filepath):
            if filepath != str(source):
                return []
            return [{"file": str(caller), "hops": 1, "reason": "direct importer"}]

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())
    monkeypatch.setattr(mcp_server, "deps", FakeDeps())

    result = mcp_server._http_search_structured(
        "TokenValidator",
        mode="auto",
        intent="modify",
        include_dependencies=True,
        include_tests=True,
    )

    assert result["mode"] == "pattern"
    assert result["mode_used"] == "pattern"
    assert result["intent"] == "modify"
    assert result["results"]
    assert result["primary_results"]
    assert result["index_status"] == {
        "pattern_ready": True,
        "semantic_ready": False,
        "symbol_ready": False,
    }
    assert result["related_files"]["callers"][0]["file"] == "AuthController.cs"
    assert result["related_files"]["dependencies"][0]["file"] == "TokenPolicy.cs"
    assert result["related_files"]["tests"][0]["file"].replace("\\", "/").endswith(
        "tests/TokenValidatorTests.cs"
    )
    assert result["related_files"]["configuration"][0]["file"] == "appsettings.json"
    assert any("Semantic index" in warning for warning in result["warnings"])


def test_unified_structured_search_empty_result_has_complete_schema(monkeypatch):
    class FakeIndex:
        def stats(self):
            return {"files": 1, "base_dir": "/project", "vram_mb": 0}

        def search(self, query, case_sensitive=False, max_files=50):
            return []

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())

    result = mcp_server._http_search_structured("missing", mode="exact")

    assert result["results"] == []
    assert result["primary_results"] == []
    assert result["related_files"] == {
        "callers": [],
        "dependencies": [],
        "implementations": [],
        "tests": [],
        "configuration": [],
    }
    assert isinstance(result["warnings"], list)
    assert result["index_status"]["pattern_ready"] is True
# ---------------------------------------------------------------------------
# Part 3 — structured read endpoint responses
# ---------------------------------------------------------------------------

def test_read_block_endpoint_returns_structured_fields(tmp_path: Path):
    src = tmp_path / "Hello.cs"
    src.write_text("public class Hello {}\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        status, body = _post_http("/read/block", {"filepath": str(src), "line": 1})
        assert status == 200
        assert isinstance(body["result"], str)
        for key in ("file", "absoluteFile", "lineStart", "lineEnd", "content", "language"):
            assert key in body, f"missing key: {key}"
        assert body["language"] == "csharp"
        assert isinstance(body["lineStart"], int)
        assert isinstance(body["lineEnd"], int)
    finally:
        mcp_server._http_roots = old_roots


def test_read_skeleton_endpoint_returns_structured_fields(tmp_path: Path):
    src = tmp_path / "module.py"
    src.write_text("def foo():\n    pass\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        status, body = _post_http("/read/skeleton", {"filepath": str(src), "matchLines": [1]})
        assert status == 200
        assert isinstance(body["result"], str)
        for key in ("file", "absoluteFile", "content", "matchLines", "language"):
            assert key in body, f"missing key: {key}"
        assert body["language"] == "python"
        assert isinstance(body["matchLines"], list)
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_endpoint_returns_structured_fields(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "module.py"
    src.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(src)})
        assert status == 200
        assert isinstance(body["result"], str)
        for key in ("file", "absoluteFile", "impactedFiles"):
            assert key in body, f"missing key: {key}"
        assert isinstance(body["impactedFiles"], list)
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_endpoint_includes_reason_when_available(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "UserService.cs"
    target.write_text("namespace Demo; public class UserService {}\n", encoding="utf-8")
    impacted = project / "UserController.cs"
    impacted.write_text("using Demo; public class UserController { UserService S; }\n", encoding="utf-8")

    class FakeDeps:
        def stats(self):
            return {"files": 2, "edges": 1, "base_dir": str(project), "cache": "test"}

        def impact(self, fpath):
            return [{"file": str(impacted), "hops": 1, "reason": "references type UserService"}]

    monkeypatch.setattr(mcp_server, "deps", FakeDeps())
    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(target)})
        assert status == 200
        assert body["impactedFiles"]
        assert body["impactedFiles"][0]["file"] == "UserController.cs"
        assert body["impactedFiles"][0]["hops"] == 1
        assert body["impactedFiles"][0]["reason"] == "references type UserService"
        assert "references type UserService" in body["result"]
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_endpoint_remains_compatible_without_reason(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    target = project / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    impacted = project / "app.py"
    impacted.write_text("import module\n", encoding="utf-8")

    class FakeDeps:
        def stats(self):
            return {"files": 2, "edges": 1, "base_dir": str(project), "cache": "test"}

        def impact(self, fpath):
            return [{"file": str(impacted), "hops": 1}]

    monkeypatch.setattr(mcp_server, "deps", FakeDeps())
    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(target)})
        assert status == 200
        first = body["impactedFiles"][0]
        assert first["file"] == "app.py"
        assert first["absoluteFile"] == str(impacted)
        assert first["hops"] == 1
        assert "reason" not in first
    finally:
        mcp_server._http_roots = old_roots


# ---------------------------------------------------------------------------
# Part 4 — confidence metadata on /dependency/impact
# ---------------------------------------------------------------------------

def test_dependency_impact_returns_confidence_field(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "module.py"
    src.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(src)})
        assert status == 200
        assert "confidence" in body
        assert body["confidence"] in ("low", "medium", "high")
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_returns_analysis_mode_heuristic(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "module.py"
    src.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(src)})
        assert status == 200
        assert body.get("analysisMode") == "heuristic"
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_returns_limitations_list(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "module.py"
    src.write_text("x = 1\n", encoding="utf-8")

    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(src)})
        assert status == 200
        assert "limitations" in body
        assert isinstance(body["limitations"], list)
        assert len(body["limitations"]) > 0
        assert any("heuristic" in lim.lower() or "compiler" in lim.lower() for lim in body["limitations"])
    finally:
        mcp_server._http_roots = old_roots


def test_dependency_impact_graph_unavailable_returns_low_confidence(tmp_path: Path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    src = project / "module.py"
    src.write_text("x = 1\n", encoding="utf-8")

    class FakeDeps:
        def stats(self):
            return {"files": 0, "edges": 0, "base_dir": None, "cache": "n/a"}

        def impact(self, fpath):
            return []

    monkeypatch.setattr(mcp_server, "deps", FakeDeps())
    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(project)]
        status, body = _post_http("/dependency/impact", {"filepath": str(src)})
        assert status == 200
        assert body["confidence"] == "low"
        assert body["impactedFiles"] == []
        assert any("dep_index" in w or "not built" in w for w in body.get("warnings", []))
    finally:
        mcp_server._http_roots = old_roots


def test_stats_returns_capabilities_and_limitations(monkeypatch):
    class FakeIndex:
        def stats(self):
            return {"files": 10, "base_dir": "/project", "vram_mb": 0, "cache": "ok"}

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    class FakeDeps:
        def stats(self):
            return {"files": 5, "edges": 3, "base_dir": "/project", "cache": "ok"}

    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer
    import threading

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())
    monkeypatch.setattr(mcp_server, "deps", FakeDeps())

    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", "/stats")
        resp = conn.getresponse()
        import json as _json
        body = _json.loads(resp.read().decode("utf-8"))
        conn.close()
        assert resp.status == 200
        assert "capabilities" in body
        caps = body["capabilities"]
        assert caps["patternSearch"] is True
        assert caps["semanticSearch"] is False
        assert caps["dependencyImpact"] is True
        assert caps["httpStructuredResponses"] is True
        assert "limitations" in body
        assert isinstance(body["limitations"], list)
        assert len(body["limitations"]) > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
