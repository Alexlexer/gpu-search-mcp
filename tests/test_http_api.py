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
