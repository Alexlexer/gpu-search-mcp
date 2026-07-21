import json
from argparse import Namespace

from gpu_service import setup_cli


def _args(**overrides):
    values = {
        "clients": ["claude"], "directories": None, "no_index": False,
        "no_model": True, "dry_run": False, "yes": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_dry_run_does_not_write_files(tmp_path, capsys):
    project = tmp_path / "project"
    project.mkdir()
    result = setup_cli.run_setup(
        _args(directories=[str(project)], dry_run=True),
        home=tmp_path, executable="python-test",
    )
    assert result == 0
    assert not (tmp_path / ".claude.json").exists()
    assert not (tmp_path / ".gpu-search-config.json").exists()
    assert "Dry run complete" in capsys.readouterr().out


def test_setup_preserves_claude_config_and_is_idempotent(tmp_path, capsys):
    project = tmp_path / "project"
    project.mkdir()
    claude_path = tmp_path / ".claude.json"
    claude_path.write_text(json.dumps({"theme": "dark"}), encoding="utf-8")
    args = _args(directories=[str(project)])
    assert setup_cli.run_setup(args, home=tmp_path, executable="python-test") == 0
    config = json.loads(claude_path.read_text(encoding="utf-8"))
    assert config["theme"] == "dark"
    assert config["mcpServers"]["gpu-search"] == {
        "command": "python-test",
        "args": ["-m", "gpu_service.mcp_server", "--directory", str(project.resolve())],
    }
    startup = json.loads((tmp_path / ".gpu-search-config.json").read_text(encoding="utf-8"))
    assert startup["directories"] == [str(project.resolve())]
    assert (tmp_path / ".claude.json.bak").exists()
    capsys.readouterr()
    assert setup_cli.run_setup(args, home=tmp_path, executable="python-test") == 0
    assert "Already configured" in capsys.readouterr().out
    assert not (tmp_path / ".claude.json.bak.1").exists()


def test_codex_setup_replaces_only_managed_table(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(
        'model = "example"\n\n[mcp_servers.gpu-search]\ncommand = "old"\nargs = []\n\n[notice]\nseen = true\n',
        encoding="utf-8",
    )
    args = _args(clients=["codex"], directories=[str(project)])
    assert setup_cli.run_setup(args, home=tmp_path, executable="python-test") == 0
    content = config_path.read_text(encoding="utf-8")
    assert 'model = "example"' in content
    assert '[mcp_servers."gpu-search"]' in content
    assert 'command = "python-test"' in content
    assert "[notice]\nseen = true" in content
    assert content.count("gpu-search") == 1


def test_no_index_omits_directory_and_startup_config(tmp_path):
    result = setup_cli.run_setup(_args(no_index=True), home=tmp_path, executable="python-test")
    assert result == 0
    config = json.loads((tmp_path / ".claude.json").read_text(encoding="utf-8"))
    assert config["mcpServers"]["gpu-search"]["args"] == ["-m", "gpu_service.mcp_server"]
    assert not (tmp_path / ".gpu-search-config.json").exists()


def test_invalid_existing_json_aborts_without_changes(tmp_path, capsys):
    config_path = tmp_path / ".claude.json"
    config_path.write_text("{invalid", encoding="utf-8")
    result = setup_cli.run_setup(_args(no_index=True), home=tmp_path, executable="python-test")
    assert result == 2
    assert config_path.read_text(encoding="utf-8") == "{invalid"
    assert "Refusing to overwrite invalid JSON" in capsys.readouterr().err


def test_missing_client_requires_explicit_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(setup_cli, "detect_clients", lambda home: [])
    result = setup_cli.run_setup(_args(clients=None, no_index=True), home=tmp_path)
    assert result == 2
    assert "Pass --client" in capsys.readouterr().err


def test_model_check_is_local_status_only(tmp_path, monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(
        setup_cli, "get_semantic_model_status",
        lambda: checked.append(True) or {"modelId": "local-model", "cached": False},
    )
    result = setup_cli.run_setup(
        _args(no_index=True, no_model=False, dry_run=True),
        home=tmp_path, executable="python-test",
    )
    assert result == 0
    assert checked == [True]
    assert "no download attempted" in capsys.readouterr().out
