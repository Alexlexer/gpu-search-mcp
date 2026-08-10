"""Opt-in coding-agent evaluation harness for gpu-search-mcp."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import random
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Callable, Iterable, Protocol
import uuid

from .redact import redact

SCHEMA_VERSION = 1
HARNESS_VERSION = "2"
MODES = ("baseline", "gpu_search")
EVALUATION_TYPES = ("instrumentation-smoke", "benchmark")
_TOOL_CATEGORIES = {"file_read", "search", "gpu_search", "edit", "test", "other"}
_SECRET_KEYS = {"authorization", "api_key", "apikey", "password", "secret", "token"}


def _strings(value, field_name: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    result = tuple(item.strip().replace("\\", "/") for item in value if item.strip())
    if required and not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize(value):
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if str(key).casefold() in _SECRET_KEYS else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact(str(value))


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    argv: tuple[str, ...]
    timeout_seconds: int = 600

    @classmethod
    def from_dict(cls, raw: dict, index: int) -> "ValidationCommand":
        if not isinstance(raw, dict):
            raise ValueError(f"validation[{index}] must be an object")
        argv = _strings(raw.get("argv"), f"validation[{index}].argv", required=True)
        timeout = int(raw.get("timeout_seconds", 600))
        if timeout < 1:
            raise ValueError(f"validation[{index}].timeout_seconds must be positive")
        return cls(argv=argv, timeout_seconds=timeout)


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    id: str
    repository: str
    base_commit: str
    description: str
    language: str
    category: str
    evaluation_type: str = "instrumentation-smoke"
    validation: tuple[ValidationCommand, ...] = ()
    relevant_files: tuple[str, ...] = ()
    expected_changed_files: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)
    definition_hash: str = ""
    evaluator_root: str = ""

    @classmethod
    def from_dict(cls, raw: dict, index: int, base_dir: Path) -> "EvaluationTask":
        if not isinstance(raw, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        values = {
            name: str(raw.get(name, "")).strip()
            for name in ("id", "repository", "base_commit", "description", "language", "category")
        }
        for name, value in values.items():
            if not value:
                raise ValueError(f"tasks[{index}].{name} is required")
        repository = values["repository"]
        if not repository.startswith(("http://", "https://", "ssh://", "git@")):
            candidate = Path(repository).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            repository = str(candidate.resolve())
        validation_raw = raw.get("validation", [])
        if not isinstance(validation_raw, list):
            raise ValueError(f"tasks[{index}].validation must be a list")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"tasks[{index}].metadata must be an object")
        evaluation_type = str(raw.get("evaluation_type", "")).strip().casefold()
        if not evaluation_type:
            evaluation_type = (
                "instrumentation-smoke"
                if values["category"].casefold() == "instrumentation-smoke"
                else "benchmark"
            )
        if evaluation_type not in EVALUATION_TYPES:
            raise ValueError(
                f"tasks[{index}].evaluation_type must be one of: "
                f"{', '.join(EVALUATION_TYPES)}"
            )
        validation = tuple(
            ValidationCommand.from_dict(item, pos)
            for pos, item in enumerate(validation_raw)
        )
        if evaluation_type == "benchmark":
            if not validation:
                raise ValueError(
                    f"tasks[{index}] benchmark tasks require deterministic validation"
                )
            if not re.fullmatch(r"[0-9a-fA-F]{40}", values["base_commit"]):
                raise ValueError(
                    f"tasks[{index}] benchmark base_commit must be a full 40-character SHA"
                )
        return cls(
            id=values["id"], repository=repository, base_commit=values["base_commit"],
            description=values["description"], language=values["language"].casefold(),
            category=values["category"].casefold(),
            evaluation_type=evaluation_type,
            validation=validation,
            relevant_files=_strings(raw.get("relevant_files"), f"tasks[{index}].relevant_files"),
            expected_changed_files=_strings(raw.get("expected_changed_files"), f"tasks[{index}].expected_changed_files"),
            metadata=_sanitize(metadata), definition_hash=_canonical_hash(raw),
            evaluator_root=str(base_dir.resolve()),
        )


@dataclass(frozen=True, slots=True)
class TaskManifest:
    suite: str
    tasks: tuple[EvaluationTask, ...]
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, raw: dict, base_dir: Path) -> "TaskManifest":
        if not isinstance(raw, dict):
            raise ValueError("task manifest must be an object")
        version = int(raw.get("schema_version", SCHEMA_VERSION))
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported task schema_version {version}; expected {SCHEMA_VERSION}")
        suite = str(raw.get("suite", "")).strip()
        if not suite:
            raise ValueError("suite is required")
        tasks_raw = raw.get("tasks")
        if not isinstance(tasks_raw, list) or not tasks_raw:
            raise ValueError("tasks must contain at least one task")
        tasks = tuple(EvaluationTask.from_dict(item, pos, base_dir) for pos, item in enumerate(tasks_raw))
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")
        return cls(suite=suite, tasks=tasks, schema_version=version)


def load_task_manifest(path: str | Path) -> TaskManifest:
    manifest_path = Path(path).resolve()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return TaskManifest.from_dict(raw, manifest_path.parent)


@dataclass(frozen=True, slots=True)
class RunLimits:
    timeout_seconds: int = 1800
    max_tool_calls: int = 250
    token_budget: int | None = None

    def __post_init__(self):
        if self.timeout_seconds < 1 or self.max_tool_calls < 1:
            raise ValueError("run limits must be positive")
        if self.token_budget is not None and self.token_budget < 1:
            raise ValueError("token_budget must be positive when provided")


@dataclass(frozen=True, slots=True)
class RunRequest:
    run_id: str
    mode: str
    task: EvaluationTask
    workspace: str
    resolved_commit: str
    model: str
    runner_name: str
    runner_config: dict
    limits: RunLimits

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "harness_version": HARNESS_VERSION,
            "run_id": self.run_id, "mode": self.mode,
            "gpu_search_enabled": self.mode == "gpu_search",
            "task": {"id": self.task.id, "description": self.task.description,
                     "language": self.task.language, "category": self.task.category},
            "workspace": self.workspace, "repository_commit": self.resolved_commit,
            "model": self.model, "runner": self.runner_name,
            "runner_config": _sanitize(self.runner_config), "limits": asdict(self.limits),
            "event_protocol": {"format": "jsonl", "types": ["tool_call", "tool_result", "milestone", "usage", "final"]},
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEvent:
    sequence: int
    type: str
    elapsed_ms: float
    tool: str | None = None
    category: str | None = None
    arguments: dict = field(default_factory=dict)
    result_size_bytes: int | None = None
    file_paths: tuple[str, ...] = ()
    token_usage: dict = field(default_factory=dict)
    usage_semantics: str | None = None
    milestone: str | None = None
    data: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict, sequence: int, default_elapsed_ms: float) -> "TrajectoryEvent":
        if not isinstance(raw, dict):
            raise ValueError("trajectory event must be an object")
        event_type = str(raw.get("type", "")).strip()
        if event_type not in {"tool_call", "tool_result", "milestone", "usage", "final"}:
            raise ValueError(f"unsupported trajectory event type: {event_type or '<empty>'}")
        category = raw.get("category")
        if category is not None:
            category = str(category).strip().casefold()
            if category not in _TOOL_CATEGORIES:
                category = "other"
        token_usage = raw.get("token_usage", {})
        arguments = raw.get("arguments", {})
        data = raw.get("data", {})
        if not all(isinstance(item, dict) for item in (token_usage, arguments, data)):
            raise ValueError("token_usage, arguments, and data must be objects")
        usage_semantics = raw.get("usage_semantics")
        if usage_semantics is not None:
            usage_semantics = str(usage_semantics).strip().casefold()
            if usage_semantics not in {"delta", "cumulative"}:
                usage_semantics = "invalid"
        result_size = raw.get("result_size_bytes")
        return cls(
            sequence=sequence, type=event_type,
            elapsed_ms=round(float(raw.get("elapsed_ms", default_elapsed_ms)), 3),
            tool=str(raw["tool"]) if raw.get("tool") is not None else None,
            category=category, arguments=_sanitize(arguments),
            result_size_bytes=None if result_size is None else max(0, int(result_size)),
            file_paths=_strings(raw.get("file_paths"), "file_paths"),
            token_usage=_sanitize(token_usage),
            usage_semantics=usage_semantics,
            milestone=str(raw["milestone"]) if raw.get("milestone") is not None else None,
            data=_sanitize(data),
        )

    def to_dict(self) -> dict:
        result = asdict(self)
        result["file_paths"] = list(self.file_paths)
        return result


class AgentRunner(Protocol):
    name: str
    def run(self, request: RunRequest) -> list[TrajectoryEvent]: ...

class CommandAgentRunner:
    """Run an external adapter using JSON stdin and normalized JSONL stdout."""

    def __init__(self, command: Iterable[str], name: str = "command") -> None:
        self.command = tuple(command)
        self.name = name
        if not self.command:
            raise ValueError("runner command must not be empty")

    def run(self, request: RunRequest) -> list[TrajectoryEvent]:
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                self.command, cwd=request.workspace,
                input=json.dumps(request.to_dict()) + "\n", text=True,
                capture_output=True,
                    timeout=request.limits.timeout_seconds + 15,
                    check=False,
            )
        except subprocess.TimeoutExpired:
            return [TrajectoryEvent(sequence=0, type="final",
                elapsed_ms=request.limits.timeout_seconds * 1000,
                data={"completed": False, "error": "runner_timeout"})]
        events: list[TrajectoryEvent] = []
        for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                events.append(TrajectoryEvent.from_dict(
                    raw, len(events), (time.perf_counter() - started) * 1000
                ))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                events.append(TrajectoryEvent(
                    sequence=len(events), type="tool_result",
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    category="other",
                    data={"warning": f"invalid runner JSONL line {line_number}: {exc}"},
                ))
        if completed.returncode != 0 or not events or events[-1].type != "final":
            events.append(TrajectoryEvent(
                sequence=len(events), type="final",
                elapsed_ms=(time.perf_counter() - started) * 1000,
                data={"completed": completed.returncode == 0,
                      "runner_exit_code": completed.returncode,
                      "runner_stderr": redact(completed.stderr[-4000:])},
            ))
        return events


def _git(args: list[str], cwd: str | Path, timeout: int = 120) -> str:
    completed = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                               text=True, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(redact(completed.stderr.strip() or completed.stdout.strip()))
    return completed.stdout.strip()


def _prepare_workspace(task: EvaluationTask, root: Path) -> tuple[Path, str]:
    workspace = root / task.id
    _git(["clone", "--quiet", "--no-hardlinks", task.repository, str(workspace)], root)
    resolved = _git(["rev-parse", f"{task.base_commit}^{{commit}}"], workspace)
    _git(["checkout", "--quiet", "--detach", resolved], workspace)
    return workspace, resolved


def _validation_results(task: EvaluationTask, workspace: Path) -> list[dict]:
    results = []
    for command in task.validation:
        started = time.perf_counter()
        argv = tuple(
            value
            .replace("{manifest_dir}", task.evaluator_root)
            .replace("{workspace}", str(workspace))
            for value in command.argv
        )
        try:
            completed = subprocess.run(argv, cwd=workspace, capture_output=True,
                text=True, timeout=command.timeout_seconds, check=False)
            results.append({
                "argv": _sanitize(list(argv)), "passed": completed.returncode == 0,
                "exit_code": completed.returncode,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "stdout_tail": redact(completed.stdout[-4000:]),
                "stderr_tail": redact(completed.stderr[-4000:]),
            })
        except subprocess.TimeoutExpired:
            results.append({"argv": _sanitize(list(argv)), "passed": False,
                "exit_code": None,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                "error": "validation_timeout"})
    return results


def _patch_metrics(workspace: Path, expected: tuple[str, ...]) -> dict:
    tracked = _git(["diff", "--name-only", "HEAD", "--"], workspace).splitlines()
    untracked = _git(
        ["ls-files", "--others", "--exclude-standard"], workspace
    ).splitlines()
    changed_files = tuple(sorted({
        path.replace("\\", "/") for path in [*tracked, *untracked] if path
    }))
    diff = _git(["diff", "--binary", "HEAD"], workspace)
    numstat = _git(["diff", "--numstat", "HEAD"], workspace)
    insertions = deletions = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            insertions += int(parts[0]) if parts[0].isdigit() else 0
            deletions += int(parts[1]) if parts[1].isdigit() else 0
    untracked_bytes = 0
    for relative_path in untracked:
        data = (workspace / relative_path).read_bytes()
        untracked_bytes += len(data)
        insertions += data.count(b"\n") + int(
            bool(data) and not data.endswith(b"\n")
        )
    expected_set = {item.casefold() for item in expected}
    changed_set = {item.casefold() for item in changed_files}
    overlap = len(expected_set & changed_set)
    return {
        "produced": bool(changed_files),
        "size_bytes": len(diff.encode("utf-8")) + untracked_bytes,
        "insertions": insertions, "deletions": deletions,
        "changed_files": list(changed_files),
        "expected_file_recall": round(overlap / len(expected_set), 6) if expected_set else None,
        "unexpected_changed_files": sorted(changed_set - expected_set) if expected_set else [],
    }


def _sum_tokens(events: list[TrajectoryEvent]) -> dict:
    keys = (
        "input_tokens", "output_tokens", "cached_input_tokens",
        "reasoning_tokens", "total_tokens",
    )
    usage_events = [event for event in events if event.type == "usage"]
    if not usage_events:
        return {
            **{key: None for key in keys},
            "usage_semantics": None,
            "usage_valid": True,
            "usage_error": None,
        }

    semantics = {event.usage_semantics for event in usage_events}
    if len(usage_events) == 1 and semantics == {None}:
        normalized_semantics = "single-event"
    elif semantics == {"delta"}:
        normalized_semantics = "delta"
    elif semantics == {"cumulative"}:
        normalized_semantics = "cumulative"
    else:
        return {
            **{key: None for key in keys},
            "usage_semantics": "invalid",
            "usage_valid": False,
            "usage_error": (
                "multiple usage events require one explicit, consistent "
                "delta or cumulative semantic"
            ),
        }

    result: dict[str, int | str | bool | None] = {}
    for key in keys:
        values = [
            int(event.token_usage[key])
            for event in usage_events
            if isinstance(event.token_usage.get(key), (int, float))
        ]
        if not values:
            result[key] = None
        elif normalized_semantics == "delta":
            result[key] = sum(values)
        else:
            if normalized_semantics == "cumulative" and any(
                later < earlier for earlier, later in zip(values, values[1:])
            ):
                return {
                    **{item: None for item in keys},
                    "usage_semantics": "invalid",
                    "usage_valid": False,
                    "usage_error": f"cumulative {key} decreased",
                }
            result[key] = values[-1]

    if result["total_tokens"] is None:
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        result["total_tokens"] = (
            input_tokens + output_tokens
            if isinstance(input_tokens, int) and isinstance(output_tokens, int)
            else None
        )
    result.update({
        "usage_semantics": normalized_semantics,
        "usage_valid": True,
        "usage_error": None,
    })
    return result


def _trajectory_metrics(task: EvaluationTask, events: list[TrajectoryEvent]) -> dict:
    tool_calls = [event for event in events if event.type == "tool_call"]

    def has_operation(event: TrajectoryEvent, operation: str) -> bool:
        operations = event.arguments.get("operations", [])
        return (
            event.category == operation
            or isinstance(operations, list) and operation in operations
        )

    reads = [
        event for event in tool_calls if has_operation(event, "file_read")
    ]
    file_reads = [path for event in reads for path in event.file_paths]
    unique_files = sorted(set(file_reads))
    relevant = {path.casefold() for path in task.relevant_files}
    irrelevant = [path for path in unique_files if relevant and path.casefold() not in relevant]

    def first_time(predicate: Callable[[TrajectoryEvent], bool]) -> float | None:
        match = next((event.elapsed_ms for event in events if predicate(event)), None)
        return None if match is None else round(match, 3)

    first_relevant = first_time(
        lambda event: bool(relevant & {path.casefold() for path in event.file_paths})
    )
    first_impl = first_time(
        lambda event: event.milestone == "likely_implementation" or
        (event.category == "edit" and bool(relevant & {path.casefold() for path in event.file_paths}))
    )
    first_patch = first_time(
        lambda event: event.milestone == "first_patch" or event.category == "edit"
    )
    duration = max((event.elapsed_ms for event in events), default=0.0)
    gpu_context_bytes = sum(event.result_size_bytes or 0 for event in events
        if event.type == "tool_result" and event.category == "gpu_search")
    repo_context_bytes = sum(event.result_size_bytes or 0 for event in events
        if event.type == "tool_result" and has_operation(event, "file_read"))
    return {
        "files_read": unique_files, "unique_files_read": len(unique_files),
        "total_file_reads": len(file_reads),
        "irrelevant_files_inspected": irrelevant if relevant else None,
        "irrelevant_file_count": len(irrelevant) if relevant else None,
        "search_operations": sum(
            has_operation(event, "search") for event in tool_calls
        ),
        "gpu_search_operations": sum(event.category == "gpu_search" for event in tool_calls),
        "total_tool_calls": len(tool_calls),
        "repository_context_tokens_estimate": math.ceil(repo_context_bytes / 4),
        "gpu_search_context_tokens_estimate": math.ceil(gpu_context_bytes / 4),
        "duration_ms": round(duration, 3),
        "time_to_first_relevant_file_ms": first_relevant,
        "time_to_first_likely_implementation_ms": first_impl,
        "time_to_first_patch_ms": first_patch,
        "time_to_final_patch_ms": round(duration, 3) if first_patch is not None else None,
    }


def _gpu_search_commit() -> str | None:
    try:
        return _git(["rev-parse", "HEAD"], Path(__file__).resolve().parents[1])
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None


def run_task(
    task: EvaluationTask,
    mode: str,
    runner: AgentRunner,
    *,
    model: str,
    output_dir: str | Path,
    limits: RunLimits | None = None,
    runner_config: dict | None = None,
    experiment_metadata: dict | None = None,
    keep_workspace: bool = False,
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of: {', '.join(MODES)}")
    limits = limits or RunLimits()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="gpu-search-eval-"))
    run_id = f"{task.id}-{mode}-{uuid.uuid4().hex[:12]}"
    started_at = _now()
    try:
        workspace, resolved = _prepare_workspace(task, work_root)
        request = RunRequest(run_id=run_id, mode=mode, task=task,
            workspace=str(workspace), resolved_commit=resolved, model=model,
            runner_name=runner.name, runner_config=runner_config or {}, limits=limits)
        events = runner.run(request)
        if sum(event.type == "tool_call" for event in events) > limits.max_tool_calls:
            events.append(TrajectoryEvent(sequence=len(events), type="final",
                elapsed_ms=max((event.elapsed_ms for event in events), default=0.0),
                data={"completed": False, "error": "max_tool_calls_exceeded"}))
        validation = _validation_results(task, workspace)
        patch = _patch_metrics(workspace, task.expected_changed_files)
        final = next((event for event in reversed(events) if event.type == "final"), None)
        completed = bool(final and final.data.get("completed", False))
        validation_passed = all(item["passed"] for item in validation) if validation else None
        eligible = task.evaluation_type == "benchmark" and bool(validation)
        metrics = _trajectory_metrics(task, events)
        tokens = _sum_tokens(events)
        result = {
            "schema_version": SCHEMA_VERSION, "harness_version": HARNESS_VERSION,
            "identity": {"task_id": task.id, "task_definition_hash": task.definition_hash,
                "run_id": run_id, "mode": mode, "repository": task.repository,
                "base_commit": task.base_commit, "resolved_commit": resolved,
                "model": model, "runner": runner.name,
                "runner_config": _sanitize(runner_config or {}),
                "experiment": _sanitize(experiment_metadata or {}),
                "timestamp": started_at,
                "gpu_search_commit": _gpu_search_commit()},
            "outcome": {"task_completed": completed,
                "evaluation_type": task.evaluation_type,
                "eligible_for_success_comparison": eligible,
                "validation_passed": validation_passed,
                "tests_passed": sum(item["passed"] for item in validation),
                "tests_failed": sum(not item["passed"] for item in validation),
                "patch_produced": patch["produced"],
                "execution_success": completed and validation_passed is not False,
                "success": (
                    validation_passed is True
                    if task.evaluation_type == "benchmark"
                    else completed and validation_passed is not False
                )},
            "patch": patch, "validation": validation,
            "quality": {
                "patch_correctness_score": (
                    1.0 if validation_passed is True else
                    0.0 if validation_passed is False else None
                ),
                "regression_status": (
                    "not_detected" if validation_passed is True else
                    "detected" if validation_passed is False else "not_available"
                ),
            },
            "exploration": {key: metrics[key] for key in (
                "files_read", "unique_files_read", "total_file_reads",
                "irrelevant_files_inspected", "irrelevant_file_count",
                "search_operations", "gpu_search_operations", "total_tool_calls")},
            "tokens": {**tokens,
                "repository_context_tokens_estimate": metrics["repository_context_tokens_estimate"],
                "gpu_search_context_tokens_estimate": metrics["gpu_search_context_tokens_estimate"]},
            "timing": {key: metrics[key] for key in (
                "duration_ms", "time_to_first_relevant_file_ms",
                "time_to_first_likely_implementation_ms", "time_to_first_patch_ms",
                "time_to_final_patch_ms")},
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "processor": platform.processor() or None,
                "provider": final.data.get("provider") if final else None,
            },
            "limits": asdict(limits), "trajectory_file": f"trajectories/{run_id}.jsonl",
        }
        trajectory_path = output / result["trajectory_file"]
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text("".join(
            json.dumps(event.to_dict(), sort_keys=True) + "\n" for event in events
        ), encoding="utf-8")
        return result
    finally:
        if not keep_workspace:
            shutil.rmtree(work_root, ignore_errors=True)


def append_run(path: str | Path, result: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")


def load_runs(paths: Iterable[str | Path]) -> list[dict]:
    runs = []
    for path in paths:
        for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
                raise ValueError(f"{path}:{line_number}: unsupported run schema")
            runs.append(raw)
    return runs


def _stats(values: Iterable[float | int | None]) -> dict | None:
    clean = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not clean:
        return None
    position = max(0, math.ceil(0.95 * len(clean)) - 1)
    return {"count": len(clean), "mean": round(statistics.fmean(clean), 3),
        "median": round(statistics.median(clean), 3),
        "p50": round(statistics.median(clean), 3), "p95": round(clean[position], 3),
        "min": round(min(clean), 3), "max": round(max(clean), 3)}


def _metric(run: dict, section: str, name: str):
    value = run.get(section, {}).get(name)
    return value if isinstance(value, (int, float)) else None


def aggregate_runs(runs: list[dict]) -> dict:
    ordered = sorted(runs, key=lambda item: (
        item["identity"]["task_id"], item["identity"]["mode"],
        item["identity"]["run_id"],
    ))
    numeric = {
        "input_tokens": ("tokens", "input_tokens"),
        "output_tokens": ("tokens", "output_tokens"),
        "total_tokens": ("tokens", "total_tokens"),
        "repository_context_tokens": (
            "tokens", "repository_context_tokens_estimate",
        ),
        "gpu_search_context_tokens": (
            "tokens", "gpu_search_context_tokens_estimate",
        ),
        "unique_files_read": ("exploration", "unique_files_read"),
        "total_file_reads": ("exploration", "total_file_reads"),
        "irrelevant_file_count": ("exploration", "irrelevant_file_count"),
        "search_operations": ("exploration", "search_operations"),
        "gpu_search_operations": ("exploration", "gpu_search_operations"),
        "total_tool_calls": ("exploration", "total_tool_calls"),
        "duration_ms": ("timing", "duration_ms"),
        "time_to_first_relevant_file_ms": (
            "timing", "time_to_first_relevant_file_ms",
        ),
        "time_to_first_likely_implementation_ms": (
            "timing", "time_to_first_likely_implementation_ms",
        ),
        "time_to_first_patch_ms": ("timing", "time_to_first_patch_ms"),
        "time_to_final_patch_ms": ("timing", "time_to_final_patch_ms"),
    }
    modes: dict[str, dict] = {}
    eligible_by_mode: dict[str, list[dict]] = {}
    for mode in MODES:
        selected = [
            run for run in ordered if run["identity"]["mode"] == mode
        ]
        eligible = [
            run for run in selected
            if run["outcome"].get("eligible_for_success_comparison", True)
        ]
        successful = [run for run in eligible if run["outcome"]["success"]]
        validation_known = [
            run for run in eligible
            if run["outcome"].get("validation_passed") is not None
        ]
        validation_passes = sum(
            run["outcome"].get("validation_passed") is True
            for run in validation_known
        )
        eligible_by_mode[mode] = eligible
        modes[mode] = {
            "runs": len(selected),
            "eligible_runs": len(eligible),
            "tasks": len({
                run["identity"]["task_id"] for run in selected
            }),
            "eligible_tasks": len({
                run["identity"]["task_id"] for run in eligible
            }),
            "successes": len(successful),
            "success_rate": (
                round(len(successful) / len(eligible), 6)
                if eligible else None
            ),
            "validation_passes": validation_passes,
            "validation_runs": len(validation_known),
            "validation_pass_rate": (
                round(validation_passes / len(validation_known), 6)
                if validation_known else None
            ),
            "metrics": {
                name: _stats(_metric(run, section, field) for run in eligible)
                for name, (section, field) in numeric.items()
            },
            "per_success": {
                "tokens_per_successful_task": _stats(
                    _metric(run, "tokens", "total_tokens")
                    for run in successful
                ),
                "input_tokens_per_successful_task": _stats(
                    _metric(run, "tokens", "input_tokens")
                    for run in successful
                ),
                "files_read_per_successful_task": _stats(
                    _metric(run, "exploration", "unique_files_read")
                    for run in successful
                ),
                "tool_calls_per_successful_task": _stats(
                    _metric(run, "exploration", "total_tool_calls")
                    for run in successful
                ),
            },
        }

    per_task = []
    regressions = []
    task_ids = sorted({
        run["identity"]["task_id"] for run in ordered
    })
    for task_id in task_ids:
        item = {"task_id": task_id, "modes": {}}
        for mode in MODES:
            selected = [
                run for run in eligible_by_mode[mode]
                if run["identity"]["task_id"] == task_id
            ]
            successes = sum(run["outcome"]["success"] for run in selected)
            validation_passes = sum(
                run["outcome"].get("validation_passed") is True
                for run in selected
            )
            item["modes"][mode] = {
                "runs": len(selected),
                "successes": successes,
                "success_rate": (
                    round(successes / len(selected), 6) if selected else None
                ),
                "validation_passes": validation_passes,
            }
        baseline_rate = item["modes"]["baseline"]["success_rate"]
        gpu_rate = item["modes"]["gpu_search"]["success_rate"]
        if (
            baseline_rate is not None and gpu_rate is not None
            and gpu_rate < baseline_rate
        ):
            regressions.append(task_id)
        per_task.append(item)

    baseline_runs = eligible_by_mode["baseline"]
    gpu_runs = eligible_by_mode["gpu_search"]

    def comparison_signature(run: dict) -> str:
        identity = run["identity"]
        return _canonical_hash({
            name: identity.get(name)
            for name in (
                "task_id", "task_definition_hash", "repository",
                "resolved_commit", "model", "runner", "runner_config",
            )
        })

    paired_configuration = (
        bool(baseline_runs)
        and sorted(map(comparison_signature, baseline_runs))
        == sorted(map(comparison_signature, gpu_runs))
    )

    def reduction(metric_name: str) -> float | None:
        baseline = modes["baseline"]["metrics"][metric_name]
        gpu = modes["gpu_search"]["metrics"][metric_name]
        if (
            not paired_configuration or not baseline or not gpu
            or baseline["count"] != len(baseline_runs)
            or gpu["count"] != len(gpu_runs)
            or baseline["mean"] == 0
        ):
            return None
        return round(
            100 * (baseline["mean"] - gpu["mean"]) / baseline["mean"], 3
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_count": len(ordered),
        "eligible_run_count": sum(
            len(items) for items in eligible_by_mode.values()
        ),
        "task_count": len(task_ids),
        "modes": modes,
        "comparison": {
            "success_delta": (
                modes["gpu_search"]["successes"]
                - modes["baseline"]["successes"]
            ),
            "input_token_reduction_pct": reduction("input_tokens"),
            "total_token_reduction_pct": reduction("total_tokens"),
            "repository_context_reduction_pct": reduction(
                "repository_context_tokens"
            ),
            "file_read_reduction_pct": reduction("unique_files_read"),
            "irrelevant_file_reduction_pct": reduction(
                "irrelevant_file_count"
            ),
            "tool_call_reduction_pct": reduction("total_tool_calls"),
            "search_call_reduction_pct": reduction("search_operations"),
            "duration_reduction_pct": reduction("duration_ms"),
            "regressed_tasks": regressions,
            "paired_configuration": paired_configuration,
            "comparability_note": (
                "Reductions are null unless both modes have matching task, "
                "commit, model, runner, and configuration multisets and every "
                "eligible run exposes the metric."
            ),
        },
        "per_task": per_task,
    }


def report_markdown(report: dict) -> str:
    baseline = report["modes"]["baseline"]
    gpu = report["modes"]["gpu_search"]

    def mean(mode: dict, key: str) -> str:
        stats = mode["metrics"][key]
        return "n/a" if not stats else str(stats["mean"])

    def rate(mode: dict) -> str:
        if mode["validation_pass_rate"] is None:
            return "n/a"
        percentage = round(mode["validation_pass_rate"] * 100, 1)
        return (
            f"{mode['validation_passes']}/{mode['validation_runs']} "
            f"({percentage}%)"
        )

    lines = [
        "# GPU Search agent evaluation",
        "",
        f"Tasks: {report['task_count']}  ",
        f"Runs: {report['run_count']} "
        f"({report['eligible_run_count']} correctness-eligible)",
        "",
        "| Metric | Baseline | GPU Search |",
        "|---|---:|---:|",
        (
            f"| Successful eligible runs | "
            f"{baseline['successes']}/{baseline['eligible_runs']} | "
            f"{gpu['successes']}/{gpu['eligible_runs']} |"
        ),
        f"| Validation pass rate | {rate(baseline)} | {rate(gpu)} |",
    ]
    rows = (
        ("Input tokens (provider mean)", "input_tokens"),
        ("Total tokens (provider mean)", "total_tokens"),
        ("Repository context estimate mean", "repository_context_tokens"),
        ("GPU Search context estimate mean", "gpu_search_context_tokens"),
        ("Unique files read mean", "unique_files_read"),
        ("Irrelevant files read mean", "irrelevant_file_count"),
        ("Tool calls mean", "total_tool_calls"),
        ("Search calls mean", "search_operations"),
        (
            "Time to first relevant file mean (ms)",
            "time_to_first_relevant_file_ms",
        ),
        (
            "Time to likely implementation mean (ms)",
            "time_to_first_likely_implementation_ms",
        ),
        ("Time to first patch mean (ms)", "time_to_first_patch_ms"),
        ("Total duration mean (ms)", "duration_ms"),
    )
    for label, key in rows:
        lines.append(
            f"| {label} | {mean(baseline, key)} | {mean(gpu, key)} |"
        )

    comparison = report["comparison"]
    lines.extend(["", "## Derived comparison", ""])
    for label, key in (
        ("Success delta", "success_delta"),
        ("Input-token reduction", "input_token_reduction_pct"),
        ("Total-token reduction", "total_token_reduction_pct"),
        ("Repository-context reduction", "repository_context_reduction_pct"),
        ("File-read reduction", "file_read_reduction_pct"),
        ("Irrelevant-file reduction", "irrelevant_file_reduction_pct"),
        ("Tool-call reduction", "tool_call_reduction_pct"),
        ("Search-call reduction", "search_call_reduction_pct"),
        ("Duration reduction", "duration_reduction_pct"),
    ):
        value = comparison[key]
        suffix = "%" if key.endswith("_pct") and value is not None else ""
        rendered = "not available" if value is None else f"{value}{suffix}"
        lines.append(f"- {label}: {rendered}")
    lines.append(
        f"- Paired configuration valid: "
        f"{str(comparison['paired_configuration']).lower()}"
    )
    lines.append(
        f"- Regressed tasks: "
        f"{', '.join(comparison['regressed_tasks']) or 'none'}"
    )
    lines.extend([
        "",
        "> Provider token counts come from the agent runtime. Repository and "
        "GPU Search context values are byte/4 estimates, not provider tokens.",
        "",
        "> This small, nondeterministic experiment is preliminary and does "
        "not justify broad product claims.",
        "",
        "## Per-task outcome",
        "",
        "| Task | Baseline | GPU Search |",
        "|---|---:|---:|",
    ])
    for task in report["per_task"]:
        left = task["modes"]["baseline"]
        right = task["modes"]["gpu_search"]
        lines.append(
            f"| {task['task_id']} | "
            f"{left['successes']}/{left['runs']} | "
            f"{right['successes']}/{right['runs']} |"
        )
    return "\n".join(lines) + "\n"


def _parse_runner_command(value: str) -> list[str]:
    import shlex
    return shlex.split(value, posix=os.name != "nt")


def _load_runner_config(path: str | None) -> dict:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runner config must contain a JSON object")
    return value


def _build_schedule(
    tasks: list[EvaluationTask],
    modes: list[str],
    runs: int,
    order: str,
    seed: int,
) -> list[tuple[EvaluationTask, str, int]]:
    schedule: list[tuple[EvaluationTask, str, int]] = []
    for repetition in range(runs):
        for task_index, task in enumerate(tasks):
            task_modes = list(modes)
            if order == "alternating" and (repetition + task_index) % 2:
                task_modes.reverse()
            for mode in task_modes:
                schedule.append((task, mode, repetition))
    if order == "random":
        random.Random(seed).shuffle(schedule)
    return schedule


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in coding-agent evaluation harness")
    subs = parser.add_subparsers(dest="command", required=True)
    run_parser = subs.add_parser("run", help="Run tasks through an external agent adapter")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--runner-command", required=True)
    run_parser.add_argument("--runner-name", default="command")
    run_parser.add_argument("--runner-config-file")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--mode", choices=MODES, action="append", required=True)
    run_parser.add_argument("--runs", type=int, default=1)
    run_parser.add_argument(
        "--order", choices=("alternating", "random"), default="alternating"
    )
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument(
        "--max-total-runs", type=int, default=100,
        help="Refuse schedules above this total number of runs (default: 100)",
    )
    run_parser.add_argument("--task", action="append", dest="tasks")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--timeout-seconds", type=int, default=1800)
    run_parser.add_argument("--max-tool-calls", type=int, default=250)
    run_parser.add_argument("--token-budget", type=int)
    run_parser.add_argument("--keep-workspace", action="store_true")
    report_parser = subs.add_parser("report", help="Aggregate run JSONL files")
    report_parser.add_argument("--input", action="append", required=True)
    report_parser.add_argument("--output-json", required=True)
    report_parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args(argv)

    if args.command == "report":
        report = aggregate_runs(load_runs(args.input))
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(args.output_markdown).write_text(report_markdown(report), encoding="utf-8")
        return 0
    if args.runs < 1 or args.max_total_runs < 1:
        parser.error("--runs and --max-total-runs must be positive")
    manifest = load_task_manifest(args.manifest)
    selected = [task for task in manifest.tasks if not args.tasks or task.id in args.tasks]
    if not selected:
        parser.error("no tasks selected")
    modes = list(dict.fromkeys(args.mode))
    scheduled_runs = len(selected) * len(modes) * args.runs
    if scheduled_runs > args.max_total_runs:
        parser.error(
            f"scheduled {scheduled_runs} runs, exceeding --max-total-runs "
            f"{args.max_total_runs}"
        )
    output_dir = Path(args.output_dir).resolve()
    runner = CommandAgentRunner(
        _parse_runner_command(args.runner_command),
        name=args.runner_name,
    )
    try:
        runner_config = _load_runner_config(args.runner_config_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"invalid runner config: {exc}")
    limits = RunLimits(
        args.timeout_seconds, args.max_tool_calls, args.token_budget
    )
    run_file = output_dir / "runs.jsonl"
    schedule = _build_schedule(
        selected, modes, args.runs, args.order, args.seed
    )
    for schedule_index, (task, mode, repetition) in enumerate(schedule):
        result = run_task(
            task,
            mode,
            runner,
            model=args.model,
            output_dir=output_dir,
            limits=limits,
            runner_config=runner_config,
            experiment_metadata={
                "order": args.order,
                "seed": args.seed,
                "schedule_index": schedule_index,
                "repetition": repetition,
            },
            keep_workspace=args.keep_workspace,
        )
        append_run(run_file, result)
        print(json.dumps({
            "task": task.id,
            "mode": mode,
            "success": result["outcome"]["success"],
            "run_id": result["identity"]["run_id"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
