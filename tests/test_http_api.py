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
