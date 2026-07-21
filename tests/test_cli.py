import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


def test_parse_args_accepts_doctor_json_and_port():
    args = mcp_server._parse_args(["doctor", "--json", "--port", "9000"])

    assert args.command == "doctor"
    assert args.json_output is True
    assert args.port == 9000


def test_parse_args_preserves_legacy_server_mode():
    args = mcp_server._parse_args(["--directory", "repo", "--http"])

    assert args.command == "serve"
    assert args.directories == ["repo"]
    assert args.http is True


def test_version_flag_uses_server_version(capsys):
    with pytest.raises(SystemExit) as exc:
        mcp_server._parse_args(["--version"])

    assert exc.value.code == 0
    assert mcp_server.VERSION in capsys.readouterr().out


def test_run_doctor_json_is_machine_readable(monkeypatch, capsys):
    report = {
        "version": "0.2.0",
        "status": "degraded",
        "warnings": ["Semantic model is not available locally."],
    }
    monkeypatch.setattr(mcp_server, "doctor_snapshot", lambda port: report)

    code = mcp_server._run_doctor(SimpleNamespace(port=8765, json_output=True))
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output == report


def test_run_doctor_not_ready_returns_nonzero(monkeypatch, capsys):
    report = {
        "version": "0.2.0",
        "status": "not_ready",
        "system": {},
        "device": {},
        "indexes": {},
        "indexedRoots": [],
        "clientConfiguration": {},
        "http": {"url": "http://127.0.0.1:8765", "reachable": False},
        "warnings": ["Pattern index is not ready."],
    }
    monkeypatch.setattr(mcp_server, "doctor_snapshot", lambda port: report)

    code = mcp_server._run_doctor(SimpleNamespace(port=8765, json_output=False))
    output = capsys.readouterr().out

    assert code == 1
    assert "gpu-search-mcp doctor" in output
    assert "Pattern index is not ready." in output


def test_cli_main_dispatches_doctor_without_starting_server(monkeypatch):
    args = SimpleNamespace(command="doctor")
    monkeypatch.setattr(mcp_server, "_parse_args", lambda: args)
    monkeypatch.setattr(mcp_server, "_run_doctor", lambda received: 7)

    def fail_if_started(received):
        raise AssertionError("server must not start for doctor")

    monkeypatch.setattr(mcp_server, "_start_server", fail_if_started)

    assert mcp_server.cli_main() == 7


def test_client_configuration_status_does_not_read_contents(tmp_path, monkeypatch):
    (tmp_path / ".codex").mkdir()
    (tmp_path / ".codex" / "config.toml").write_text(
        "secret = 'must-not-be-read'", encoding="utf-8"
    )
    monkeypatch.setattr(mcp_server.Path, "home", lambda: tmp_path)

    status = mcp_server._client_configuration_status()

    assert status["codex"]["configured"] is True
    assert "secret" not in json.dumps(status)


def test_doctor_snapshot_reports_configured_roots_without_indexing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "diagnostics_snapshot",
        lambda: {"status": "not_ready", "warnings": []},
    )
    monkeypatch.setattr(mcp_server, "_load_config_dirs", lambda: [str(tmp_path)])
    monkeypatch.setattr(mcp_server, "_client_configuration_status", lambda: {})
    monkeypatch.setattr(
        mcp_server,
        "_probe_local_http",
        lambda port: {"url": f"http://127.0.0.1:{port}", "reachable": False},
    )

    report = mcp_server.doctor_snapshot(port=9000)

    assert report["configuredRoots"] == [
        {"path": str(tmp_path.resolve()), "exists": True}
    ]
    assert report["http"]["url"].endswith(":9000")

def test_parse_args_setup_options(tmp_path):
    args = mcp_server._parse_args([
        "setup",
        "--client", "claude",
        "--client", "codex",
        "--directory", str(tmp_path),
        "--no-model",
        "--dry-run",
        "--yes",
    ])

    assert args.command == "setup"
    assert args.clients == ["claude", "codex"]
    assert args.directories == [str(tmp_path)]
    assert args.no_model is True
    assert args.dry_run is True
    assert args.yes is True
