from __future__ import annotations

from pathlib import Path
import sys

import pytest

from gpu_service.codex_eval_adapter import (
    CodexEventNormalizer,
    build_codex_command,
    run_adapter,
)


def _request(tmp_path: Path, mode: str = "baseline") -> dict:
    return {
        "mode": mode,
        "model": "codex-test-model",
        "workspace": str(tmp_path),
        "task": {
            "id": "task",
            "description": "Fix the storefront behavior and run appropriate tests.",
            "language": "csharp",
            "category": "bugfix",
        },
        "runner_config": {
            "codex_command": ["codex"],
            "codex_version": "codex-cli test",
        },
        "limits": {"timeout_seconds": 10, "max_tool_calls": 20},
    }


def _without_gpu_config(command: list[str]) -> list[str]:
    result = []
    index = 0
    while index < len(command):
        if (
            command[index] == "-c"
            and index + 1 < len(command)
            and command[index + 1].startswith("mcp_servers.gpu_search.")
        ):
            index += 2
            continue
        result.append(command[index])
        index += 1
    return result


def test_codex_command_isolates_gpu_search_as_only_mode_difference(
    tmp_path: Path,
) -> None:
    baseline = build_codex_command(_request(tmp_path, "baseline"))
    gpu = build_codex_command(_request(tmp_path, "gpu_search"))

    assert not any("mcp_servers" in item for item in baseline)
    assert any("mcp_servers.gpu_search.command" in item for item in gpu)
    assert any("mcp_servers.gpu_search.args" in item for item in gpu)
    assert _without_gpu_config(gpu) == baseline
    assert "--ignore-user-config" in baseline
    assert "--model" in baseline
    assert "codex-test-model" in baseline


def test_codex_prompt_rejects_evaluator_only_labels(tmp_path: Path) -> None:
    request = _request(tmp_path)
    request["task"]["relevant_files"] = ["secret-answer.cs"]

    with pytest.raises(ValueError, match="evaluator-only"):
        build_codex_command(request)


def test_codex_parser_normalizes_tools_usage_and_omits_raw_output(
    tmp_path: Path,
) -> None:
    normalizer = CodexEventNormalizer(tmp_path, "codex-cli 1")
    source = tmp_path / "src" / "Service.cs"
    source.parent.mkdir()
    source.write_text("class Service {}", encoding="utf-8")

    started = normalizer.normalize({
        "type": "item.started",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": "Get-Content src/Service.cs",
        },
    }, 10)
    completed = normalizer.normalize({
        "type": "item.completed",
        "item": {
            "id": "command-1",
            "type": "command_execution",
            "command": "Get-Content src/Service.cs",
            "aggregated_output": "class Service { private string secret; }",
        },
    }, 20)
    finished = normalizer.normalize({
        "type": "turn.completed",
        "usage": {
            "input_tokens": 1200,
            "cached_input_tokens": 800,
            "output_tokens": 200,
            "reasoning_output_tokens": 50,
        },
    }, 30)

    assert started[0]["category"] == "file_read"
    assert started[0]["file_paths"] == ["src/Service.cs"]
    assert completed[0]["type"] == "tool_result"
    assert completed[0]["result_size_bytes"] > 0
    assert "aggregated_output" not in completed[0]
    assert finished[0]["usage_semantics"] == "cumulative"
    assert finished[0]["token_usage"] == {
        "input_tokens": 1200,
        "output_tokens": 200,
        "cached_input_tokens": 800,
        "reasoning_tokens": 50,
    }
    assert finished[1]["data"]["completed"] is True
    assert finished[1]["data"]["provider"]["version"] == "codex-cli 1"


def test_codex_parser_classifies_gpu_search_and_sanitizes_arguments(
    tmp_path: Path,
) -> None:
    normalizer = CodexEventNormalizer(tmp_path, "test")
    source = tmp_path / "src" / "Services" / "OrderService.cs"
    source.parent.mkdir(parents=True)
    source.write_text("class OrderService {}", encoding="utf-8")

    events = normalizer.normalize({
        "type": "item.completed",
        "item": {
            "id": "mcp-1",
            "type": "mcp_tool_call",
            "server": "gpu_search",
            "tool": "search_code",
            "arguments": {
                "query": "OrderService",
                "top_k": 5,
                "api_key": "must-not-persist",
            },
            "result": "src/Services/OrderService.cs:10",
        },
    }, 15)

    assert [event["type"] for event in events] == [
        "tool_call", "tool_result"
    ]
    assert events[0]["category"] == "gpu_search"
    assert events[0]["arguments"] == {
        "query": "OrderService",
        "top_k": 5,
    }
    assert events[1]["file_paths"] == ["src/Services/OrderService.cs"]
    assert "result" not in events[1]


def _fake_codex_script(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_codex.py"
    script.write_text(body, encoding="utf-8")
    return script


def test_codex_adapter_records_malformed_events_and_failed_process(
    tmp_path: Path,
) -> None:
    script = _fake_codex_script(
        tmp_path,
        "import json\n"
        "print('not-json', flush=True)\n"
        "print(json.dumps({'type':'turn.failed'}), flush=True)\n"
        "raise SystemExit(7)\n",
    )
    request = _request(tmp_path)
    request["runner_config"] = {
        "codex_command": [sys.executable, str(script)],
        "codex_version": "fake",
    }
    emitted = []

    exit_code = run_adapter(request, emitted.append)

    assert exit_code == 1
    assert any(
        event.get("data", {}).get("warning", "").startswith(
            "malformed Codex event"
        )
        for event in emitted
    )
    assert emitted[-1]["type"] == "final"
    assert emitted[-1]["data"]["completed"] is False
    assert emitted[-1]["data"]["error"] == "codex_process_failed"
    assert emitted[-1]["data"]["exit_code"] == 7


def test_codex_adapter_enforces_timeout_without_paid_call(
    tmp_path: Path,
) -> None:
    script = _fake_codex_script(
        tmp_path,
        "import time\ntime.sleep(5)\n",
    )
    request = _request(tmp_path)
    request["runner_config"] = {
        "codex_command": [sys.executable, str(script)],
        "codex_version": "fake",
    }
    request["limits"]["timeout_seconds"] = 1
    emitted = []

    exit_code = run_adapter(request, emitted.append)

    assert exit_code == 1
    assert emitted[-1]["data"]["error"] == "codex_timeout"
    assert emitted[-1]["data"]["completed"] is False
