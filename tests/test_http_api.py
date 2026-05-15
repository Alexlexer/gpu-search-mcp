import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


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
