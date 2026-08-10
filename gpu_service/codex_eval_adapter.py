"""Opt-in Codex CLI adapter for the coding-agent evaluation harness."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gpu_service.redact import redact
else:
    from .redact import redact


_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z]:[\\/][^\s\"']+|"
    r"(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.?*-]+|"
    r"[A-Za-z0-9_.-]+\.(?:cs|csproj|sln|json|toml|ya?ml|md|py))"
)
_SAFE_ARGUMENTS = {
    "query", "mode", "intent", "context_mode", "top_k",
    "include_dependencies", "include_tests", "relationship", "kind",
}
_TOOL_ITEM_TYPES = {"command_execution", "mcp_tool_call", "file_change"}


def _json_line(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _safe_path(value: str, workspace: Path) -> str | None:
    candidate = value.strip("'\"()[]{}:,;")
    if not candidate or "*" in candidate or "?" in candidate:
        return None
    path = Path(candidate)
    try:
        if path.is_absolute():
            path = path.resolve().relative_to(workspace.resolve())
    except (OSError, ValueError):
        return None
    normalized = path.as_posix().removeprefix("./")
    if not normalized or normalized.startswith("../"):
        return None
    try:
        if not (workspace / normalized).is_file():
            return None
    except OSError:
        return None
    return normalized


def _paths_from_text(value: str, workspace: Path) -> list[str]:
    paths = {
        path
        for match in _PATH_PATTERN.finditer(value)
        if (path := _safe_path(match.group(1), workspace)) is not None
    }
    return sorted(paths)


def _paths_from_item(item: dict, workspace: Path) -> list[str]:
    paths: set[str] = set()

    def visit(value, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key).casefold())
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and key in {
            "path", "file", "file_path", "filename",
        }:
            if path := _safe_path(value, workspace):
                paths.add(path)

    visit(item)
    output = _item_output(item)
    paths.update(_paths_from_text(output, workspace))
    if item.get("type") == "command_execution":
        paths.update(_paths_from_text(str(item.get("command", "")), workspace))
    return sorted(paths)


def _item_output(item: dict) -> str:
    parts = []
    for key in ("aggregated_output", "output", "stdout", "stderr", "result"):
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            parts.append(json.dumps(value, sort_keys=True, default=str))
    return "\n".join(parts)


def _command_operations(command: str) -> list[str]:
    lowered = command.casefold()
    operations = []
    if re.search(
        r"\b(dotnet\s+(?:test|build|run)|pytest|npm\s+test)\b",
        lowered,
    ):
        operations.append("test")
    if re.search(
        r"\b(rg|grep|findstr|select-string)\b|\bgit\s+grep\b",
        lowered,
    ):
        operations.append("search")
    if re.search(
        r"\b(get-content|cat|type|head|tail|bat)\b|\bsed\s+-n\b",
        lowered,
    ):
        operations.append("file_read")
    if re.search(
        r"\b(apply_patch|set-content|add-content)\b",
        lowered,
    ):
        operations.append("edit")
    return operations or ["other"]


def _command_name(command: str) -> str:
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        parts = command.split()
    return Path(parts[0]).name.strip("'\"[]") if parts else "shell"


def _safe_mcp_arguments(item: dict) -> dict:
    arguments = item.get("arguments")
    if not isinstance(arguments, dict):
        return {}
    return {
        str(key): redact(str(value)) if isinstance(value, str) else value
        for key, value in arguments.items()
        if str(key) in _SAFE_ARGUMENTS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }


def _item_descriptor(item: dict, workspace: Path) -> tuple[str, str, dict, list[str]]:
    item_type = str(item.get("type", ""))
    paths = _paths_from_item(item, workspace)
    if item_type == "command_execution":
        command = str(item.get("command", ""))
        operations = _command_operations(command)
        return (
            "shell",
            operations[0],
            {
                "command_name": _command_name(command),
                "operations": operations,
            },
            paths,
        )
    if item_type == "file_change":
        return "file_change", "edit", {}, paths
    server = str(item.get("server", item.get("server_name", "")))
    tool = str(item.get("tool", item.get("tool_name", "mcp_tool")))
    combined = f"{server} {tool}".casefold().replace("-", "_")
    category = "gpu_search" if "gpu_search" in combined else "other"
    return tool, category, _safe_mcp_arguments(item), paths


class CodexEventNormalizer:
    """Convert Codex CLI JSONL events to the harness trajectory contract."""

    def __init__(self, workspace: str | Path, provider_version: str | None):
        self.workspace = Path(workspace)
        self.provider_version = provider_version
        self.started_items: set[str] = set()
        self.completed = False

    def normalize(self, raw: dict, elapsed_ms: float) -> list[dict]:
        if not isinstance(raw, dict):
            raise ValueError("Codex event must be an object")
        event_type = str(raw.get("type", ""))
        if event_type in {"turn.failed", "error"}:
            self.completed = False
            return [{
                "type": "final",
                "elapsed_ms": elapsed_ms,
                "data": {
                    "completed": False,
                    "error": event_type,
                    "provider": self._provider(),
                },
            }]
        if event_type == "turn.completed":
            events = []
            usage = raw.get("usage")
            if isinstance(usage, dict):
                token_usage = {
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "reasoning_tokens": usage.get(
                        "reasoning_tokens",
                        usage.get("reasoning_output_tokens"),
                    ),
                    "total_tokens": usage.get("total_tokens"),
                }
                token_usage = {
                    key: value for key, value in token_usage.items()
                    if isinstance(value, (int, float))
                }
                events.append({
                    "type": "usage",
                    "elapsed_ms": elapsed_ms,
                    "usage_semantics": "cumulative",
                    "token_usage": token_usage,
                })
            self.completed = True
            events.append({
                "type": "final",
                "elapsed_ms": elapsed_ms,
                "data": {
                    "completed": True,
                    "provider": self._provider(),
                },
            })
            return events
        if not event_type.startswith("item."):
            return []
        item = raw.get("item")
        if not isinstance(item, dict) or item.get("type") not in _TOOL_ITEM_TYPES:
            return []
        item_id = str(item.get("id", ""))
        tool, category, arguments, paths = _item_descriptor(
            item, self.workspace
        )
        if event_type == "item.started":
            if item_id:
                self.started_items.add(item_id)
            return [{
                "type": "tool_call",
                "elapsed_ms": elapsed_ms,
                "tool": tool,
                "category": category,
                "arguments": arguments,
                "file_paths": paths,
            }]
        if event_type != "item.completed":
            return []
        events = []
        if not item_id or item_id not in self.started_items:
            events.append({
                "type": "tool_call",
                "elapsed_ms": elapsed_ms,
                "tool": tool,
                "category": category,
                "arguments": arguments,
                "file_paths": paths,
            })
        output = _item_output(item)
        events.append({
            "type": "tool_result",
            "elapsed_ms": elapsed_ms,
            "tool": tool,
            "category": category,
            "arguments": arguments,
            "result_size_bytes": len(output.encode("utf-8")),
            "file_paths": paths,
        })
        return events

    def _provider(self) -> dict:
        return {
            "name": "codex-cli",
            "version": self.provider_version,
        }


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _prompt(request: dict) -> str:
    task = request.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("description"), str):
        raise ValueError("request.task.description is required")
    forbidden = {"relevant_files", "expected_changed_files", "validation"}
    leaked = forbidden & set(task)
    if leaked:
        raise ValueError(
            f"evaluator-only task fields were exposed: {', '.join(sorted(leaked))}"
        )
    return (
        "Complete the following coding task in the current repository. "
        "Inspect the repository to locate the relevant implementation, make "
        "the smallest correct change, and run appropriate repository tests. "
        "Apply the edits directly in the working tree; do not merely describe "
        "a proposed patch. Do not modify unrelated files. Use the repository "
        "tools available to you as appropriate. When a specialized repository "
        "search tool is available, prefer it over broad shell traversal.\n\n"
        "Task:\n" + task["description"]
    )


def build_codex_command(request: dict) -> list[str]:
    config = request.get("runner_config") or {}
    if not isinstance(config, dict):
        raise ValueError("runner_config must be an object")
    prefix = config.get("codex_command", ["codex"])
    if not isinstance(prefix, list) or not prefix or not all(
        isinstance(item, str) and item for item in prefix
    ):
        raise ValueError("codex_command must be a non-empty string list")
    sandbox = str(config.get("sandbox", "workspace-write"))
    if sandbox not in {"read-only", "workspace-write"}:
        raise ValueError("adapter only permits read-only or workspace-write sandbox")
    command = [
        *prefix,
        "exec",
        "--json",
        "--ephemeral",
        "--ignore-user-config",
        "--sandbox",
        sandbox,
        "--color",
        "never",
        "--model",
        str(request["model"]),
        "--cd",
        str(request["workspace"]),
    ]
    if request.get("mode") == "gpu_search":
        gpu_command = str(config.get("gpu_search_command", sys.executable))
        raw_args = config.get("gpu_search_args", [
            "-m",
            "gpu_service.mcp_server",
            "--directory",
            "{workspace}",
        ])
        if not isinstance(raw_args, list) or not all(
            isinstance(item, str) for item in raw_args
        ):
            raise ValueError("gpu_search_args must be a string list")
        gpu_args = [
            item.replace("{workspace}", str(request["workspace"]))
            for item in raw_args
        ]
        command.extend([
            "-c",
            f"mcp_servers.gpu_search.command={_toml_string(gpu_command)}",
            "-c",
            f"mcp_servers.gpu_search.args={json.dumps(gpu_args)}",
            "-c",
            "mcp_servers.gpu_search.startup_timeout_sec=120",
            "-c",
            "mcp_servers.gpu_search.required=true",
        ])
    elif request.get("mode") != "baseline":
        raise ValueError("mode must be baseline or gpu_search")
    extra_args = config.get("codex_extra_args", [])
    if not isinstance(extra_args, list) or not all(
        isinstance(item, str) for item in extra_args
    ):
        raise ValueError("codex_extra_args must be a string list")
    if any(
        item in {
            "--dangerously-bypass-approvals-and-sandbox",
            "--danger-full-access",
        }
        for item in extra_args
    ):
        raise ValueError("dangerous Codex arguments are not allowed by the adapter")
    command.extend(extra_args)
    command.append(_prompt(request))
    return command


def _codex_version(command_prefix: list[str], configured: str | None) -> str | None:
    if configured:
        return configured
    try:
        completed = subprocess.run(
            [*command_prefix, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return redact(completed.stdout.strip()) or None


def _stop_process(process: subprocess.Popen) -> None:
    """Terminate the adapter-owned process tree without touching unrelated agents."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            process.wait(timeout=3)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    elif process.pid:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
            return
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
                return
            except (OSError, subprocess.TimeoutExpired):
                pass
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)

def run_adapter(
    request: dict,
    emit: Callable[[dict], None],
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    command = build_codex_command(request)
    config = request.get("runner_config") or {}
    prefix = config.get("codex_command", ["codex"])
    version = _codex_version(prefix, config.get("codex_version"))
    normalizer = CodexEventNormalizer(request["workspace"], version)
    limits = request.get("limits") or {}
    timeout_seconds = int(limits.get("timeout_seconds", 1800))
    max_tool_calls = int(limits.get("max_tool_calls", 250))
    started = time.perf_counter()
    stderr_tail: deque[str] = deque(maxlen=100)
    messages: queue.Queue[tuple[str, str, float]] = queue.Queue()

    try:
        process = popen(
            command,
            cwd=request["workspace"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            ),
            start_new_session=os.name != "nt",
        )
    except OSError as exc:
        emit({
            "type": "final",
            "elapsed_ms": 0,
            "data": {
                "completed": False,
                "error": "codex_start_failed",
                "detail": redact(str(exc)),
                "provider": {"name": "codex-cli", "version": version},
            },
        })
        return 1

    def read_stdout() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            messages.put(("stdout", line, time.perf_counter()))
        messages.put(("eof", "", time.perf_counter()))

    def read_stderr() -> None:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_tail.append(redact(line.rstrip())[-1000:])

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=read_stderr, daemon=True).start()

    tool_calls = 0
    saw_final = False
    timed_out = False
    exceeded_tools = False
    while True:
        elapsed = time.perf_counter() - started
        if elapsed >= timeout_seconds:
            timed_out = True
            _stop_process(process)
            break
        try:
            kind, line, observed = messages.get(timeout=min(0.2, timeout_seconds - elapsed))
        except queue.Empty:
            continue
        if kind == "eof":
            break
        elapsed_ms = round((observed - started) * 1000, 3)
        try:
            raw = json.loads(line)
            normalized = normalizer.normalize(raw, elapsed_ms)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            emit({
                "type": "tool_result",
                "elapsed_ms": elapsed_ms,
                "category": "other",
                "data": {"warning": f"malformed Codex event: {redact(str(exc))}"},
            })
            continue
        for event in normalized:
            if event["type"] == "tool_call":
                tool_calls += 1
            if event["type"] == "final":
                saw_final = True
            emit(event)
        if tool_calls > max_tool_calls:
            exceeded_tools = True
            _stop_process(process)
            break

    if process.poll() is None:
        _stop_process(process)
    return_code = process.wait()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if timed_out or exceeded_tools or return_code != 0 or not saw_final:
        error = (
            "codex_timeout" if timed_out else
            "max_tool_calls_exceeded" if exceeded_tools else
            "codex_process_failed" if return_code != 0 else
            "codex_missing_final_event"
        )
        emit({
            "type": "final",
            "elapsed_ms": elapsed_ms,
            "data": {
                "completed": False,
                "error": error,
                "exit_code": return_code,
                "stderr_tail": list(stderr_tail)[-20:],
                "provider": {"name": "codex-cli", "version": version},
            },
        })
        return 1
    return 0


def main() -> int:
    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        return run_adapter(request, lambda event: print(_json_line(event), flush=True))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        print(_json_line({
            "type": "final",
            "elapsed_ms": 0,
            "data": {
                "completed": False,
                "error": "invalid_adapter_request",
                "detail": redact(str(exc)),
            },
        }), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
