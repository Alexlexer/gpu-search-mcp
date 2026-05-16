"""Tests for semantic embedding model configuration and preflight."""
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server
from gpu_semantic_index import MODEL_ID, SemanticIndex
import semantic_model_manager as smm


def test_default_semantic_model_id(monkeypatch):
    monkeypatch.delenv(smm.SEMANTIC_MODEL_ENV, raising=False)

    assert smm.resolve_semantic_model_id(config_model=None) == "BAAI/bge-small-en-v1.5"
    assert MODEL_ID == "BAAI/bge-small-en-v1.5"


def test_env_var_overrides_semantic_model(monkeypatch):
    monkeypatch.setenv(smm.SEMANTIC_MODEL_ENV, "local/test-embed")

    assert smm.resolve_semantic_model_id() == "local/test-embed"


def test_cli_argument_overrides_env_and_config(monkeypatch):
    monkeypatch.setenv(smm.SEMANTIC_MODEL_ENV, "env/model")

    assert smm.resolve_semantic_model_id("cli/model", "config/model") == "cli/model"


def test_prepare_startup_cli_semantic_model_overrides_env(monkeypatch):
    monkeypatch.setenv(smm.SEMANTIC_MODEL_ENV, "env/model")
    args = mcp_server._parse_args([
        "--semantic-model",
        "cli/model",
        "--download-semantic-model",
    ])

    cli_targets, all_targets = mcp_server._prepare_startup(args)

    assert cli_targets == []
    assert all_targets == []
    assert mcp_server._SEMANTIC_MODEL_ID == "cli/model"
    assert smm.get_configured_semantic_model_id() == "cli/model"


def test_preflight_does_not_download_and_reports_missing(monkeypatch):
    calls = []

    def fake_snapshot_download(model_id, local_files_only=False):
        calls.append((model_id, local_files_only))
        raise RuntimeError("not cached")

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.append(("st", kwargs.get("local_files_only")))
            raise RuntimeError("not cached")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    status = smm.get_semantic_model_status("missing/model", device="cpu")

    assert status["available"] is False
    assert status["requiresDownload"] is True
    assert ("missing/model", True) in calls
    assert ("st", True) in calls
    assert "--download-semantic-model" in status["message"]


def test_preflight_available_local_model(monkeypatch, tmp_path: Path):
    def fake_snapshot_download(model_id, local_files_only=False):
        assert model_id == "cached/model"
        assert local_files_only is True
        return str(tmp_path / "hf" / "cached-model")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=fake_snapshot_download))

    status = smm.get_semantic_model_status("cached/model", device="cpu")

    assert status["available"] is True
    assert status["cached"] is True
    assert status["requiresDownload"] is False
    assert status["cachePath"].endswith("cached-model")


def test_download_flag_uses_sentence_transformer_without_local_files_only(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        cache_folder = "C:/hf/cache"

        def __init__(self, model_id, **kwargs):
            calls.append((model_id, kwargs))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    status = smm.download_semantic_model("download/model", device="cpu")

    assert status["available"] is True
    assert calls == [("download/model", {"device": "cpu"})]


def test_semantic_unavailable_message_mentions_download_command():
    message = SemanticIndex().semantic_unavailable_message()

    assert "--download-semantic-model" in message
    assert "gpu-search-mcp --semantic-model" in message


def test_stats_includes_semantic_model_metadata(monkeypatch, tmp_path: Path):
    class FakeIndex:
        def stats(self):
            return {"files": 0, "base_dir": str(tmp_path), "vram_mb": 0}

    class FakeSemantic:
        def stats(self):
            return {"chunks": 0, "base_dir": str(tmp_path), "vram_mb": 0}

    class FakeDeps:
        def stats(self):
            return {"files": 0, "edges": 0, "base_dir": str(tmp_path), "cache": "cold"}

    def fake_status(model_id=None, device=None):
        return {
            "modelId": model_id or "test/model",
            "provider": "sentence-transformers",
            "available": False,
            "cached": False,
            "requiresDownload": True,
            "device": device,
            "message": "not cached",
        }

    monkeypatch.setattr(mcp_server, "index", FakeIndex())
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic())
    monkeypatch.setattr(mcp_server, "deps", FakeDeps())
    monkeypatch.setattr(mcp_server, "get_semantic_model_status", fake_status)
    monkeypatch.setattr(mcp_server, "_SEMANTIC_MODEL_ID", "test/model")

    status, body = _get_http("/stats")

    assert status == 200
    assert body["semanticModel"]["modelId"] == "test/model"
    assert body["semanticModel"]["provider"] == "sentence-transformers"
    assert body["semanticModel"]["requiresDownload"] is True


def test_semantic_model_status_endpoint(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "get_semantic_model_status",
        lambda model_id=None, device=None: {
            "modelId": model_id or "endpoint/model",
            "provider": "sentence-transformers",
            "available": True,
            "cached": True,
            "requiresDownload": False,
            "message": "ok",
        },
    )
    monkeypatch.setattr(mcp_server, "_SEMANTIC_MODEL_ID", "endpoint/model")

    status, body = _get_http("/semantic/model/status")

    assert status == 200
    assert body["modelId"] == "endpoint/model"
    assert body["available"] is True


def _get_http(path: str) -> tuple[int, dict]:
    import json
    import threading
    from http.client import HTTPConnection
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
