from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gpu_service.agent_eval import (
    CommandAgentRunner,
    EvaluationTask,
    RunLimits,
    TaskManifest,
    TrajectoryEvent,
    ValidationCommand,
    aggregate_runs,
    append_run,
    load_runs,
    load_task_manifest,
    main,
    report_markdown,
    run_task,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "eval@example.test")
    _git(repo, "config", "user.name", "Eval Test")
    source = repo / "src"
    source.mkdir()
    (source / "Service.cs").write_text(
        "public class Service { public int Value() => 1; }\n", encoding="utf-8"
    )
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD")


def _task(repo: Path, commit: str, *, validation_passes: bool = True) -> EvaluationTask:
    expected = "2" if validation_passes else "999"
    return EvaluationTask(
        id="service-value",
        repository=str(repo),
        base_commit=commit,
        description="Change Service.Value to return 2",
        language="csharp",
        category="bugfix",
        validation=(ValidationCommand(
            argv=(sys.executable, "-c", (
                "from pathlib import Path; "
                f"assert '=> {expected}' in Path('src/Service.cs').read_text()"
            )),
            timeout_seconds=10,
        ),),
        relevant_files=("src/Service.cs",),
        expected_changed_files=("src/Service.cs",),
        definition_hash="task-hash",
    )


class FakeRunner:
    name = "fake"

    def __init__(self, *, complete: bool = True, include_tokens: bool = True):
        self.complete = complete
        self.include_tokens = include_tokens

    def run(self, request):
        path = Path(request.workspace) / "src" / "Service.cs"
        path.write_text(path.read_text(encoding="utf-8").replace("=> 1", "=> 2"), encoding="utf-8")
        usage = {"input_tokens": 100, "output_tokens": 20} if self.include_tokens else {}
        return [
            TrajectoryEvent(0, "tool_call", 10, tool="read", category="file_read",
                            file_paths=("README.md",)),
            TrajectoryEvent(1, "tool_result", 15, category="file_read",
                            result_size_bytes=80, file_paths=("README.md",)),
            TrajectoryEvent(2, "tool_call", 20, tool="search_code", category="gpu_search",
                            file_paths=("src/Service.cs",)),
            TrajectoryEvent(3, "tool_result", 25, category="gpu_search",
                            result_size_bytes=120, file_paths=("src/Service.cs",)),
            TrajectoryEvent(4, "tool_call", 30, tool="read", category="file_read",
                            file_paths=("src/Service.cs",)),
            TrajectoryEvent(5, "tool_result", 35, category="file_read",
                            result_size_bytes=160, file_paths=("src/Service.cs",)),
            TrajectoryEvent(6, "tool_call", 40, tool="edit", category="edit",
                            file_paths=("src/Service.cs",)),
            TrajectoryEvent(7, "usage", 45, token_usage=usage),
            TrajectoryEvent(8, "final", 50, data={"completed": self.complete}),
        ]


def test_task_manifest_parses_versioned_multi_repository_shape(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    manifest_path = tmp_path / "tasks.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "suite": "dotnet-smoke",
        "tasks": [{
            "id": "one",
            "repository": str(repo),
            "base_commit": commit,
            "description": "Fix the value",
            "language": "CSharp",
            "category": "BugFix",
            "relevant_files": ["src\\Service.cs"],
            "expected_changed_files": ["src/Service.cs"],
            "validation": [{"argv": ["dotnet", "test"], "timeout_seconds": 30}],
        }],
    }), encoding="utf-8")

    manifest = load_task_manifest(manifest_path)

    assert manifest.suite == "dotnet-smoke"
    assert manifest.tasks[0].language == "csharp"
    assert manifest.tasks[0].relevant_files == ("src/Service.cs",)
    assert manifest.tasks[0].definition_hash
    assert manifest.tasks[0].validation[0].argv == ("dotnet", "test")


def test_manifest_rejects_duplicate_tasks_and_unknown_schema(tmp_path: Path) -> None:
    task = {
        "id": "same", "repository": ".", "base_commit": "HEAD",
        "description": "x", "language": "csharp", "category": "bugfix",
    }
    with pytest.raises(ValueError, match="unique"):
        TaskManifest.from_dict(
            {"schema_version": 1, "suite": "x", "tasks": [task, task]}, tmp_path
        )
    with pytest.raises(ValueError, match="unsupported"):
        TaskManifest.from_dict(
            {"schema_version": 99, "suite": "x", "tasks": [task]}, tmp_path
        )


def test_trajectory_parsing_sanitizes_secrets_and_validates_types() -> None:
    event = TrajectoryEvent.from_dict({
        "type": "tool_call",
        "elapsed_ms": 12,
        "tool": "request",
        "category": "unknown-category",
        "arguments": {"token": "secret-value", "query": "safe"},
        "file_paths": ["src/a.cs"],
    }, 0, 0)

    assert event.category == "other"
    assert event.arguments["token"] == "[REDACTED]"
    assert event.file_paths == ("src/a.cs",)
    with pytest.raises(ValueError, match="unsupported"):
        TrajectoryEvent.from_dict({"type": "mystery"}, 0, 0)


def test_run_task_records_outcome_patch_trajectory_and_context_metrics(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    output = tmp_path / "output"

    result = run_task(
        _task(repo, commit), "gpu_search", FakeRunner(), model="test-model",
        output_dir=output, limits=RunLimits(timeout_seconds=30, max_tool_calls=20),
    )

    assert result["identity"]["resolved_commit"] == commit
    assert result["identity"]["mode"] == "gpu_search"
    assert result["outcome"]["success"] is True
    assert result["outcome"]["tests_passed"] == 1
    assert result["patch"]["changed_files"] == ["src/Service.cs"]
    assert result["patch"]["expected_file_recall"] == 1
    assert result["exploration"]["unique_files_read"] == 2
    assert result["exploration"]["irrelevant_files_inspected"] == ["README.md"]
    assert result["exploration"]["gpu_search_operations"] == 1
    assert result["tokens"]["input_tokens"] == 100
    assert result["tokens"]["gpu_search_context_tokens_estimate"] == 30
    assert result["timing"]["time_to_first_relevant_file_ms"] == 20
    trajectory = output / result["trajectory_file"]
    assert trajectory.exists()
    assert len(trajectory.read_text(encoding="utf-8").splitlines()) == 9


def test_run_task_preserves_missing_tokens_and_failed_validation(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    result = run_task(
        _task(repo, commit, validation_passes=False), "baseline",
        FakeRunner(include_tokens=False), model="test-model", output_dir=tmp_path / "out",
    )

    assert result["outcome"]["task_completed"] is True
    assert result["outcome"]["validation_passed"] is False
    assert result["outcome"]["success"] is False
    assert result["tokens"]["input_tokens"] is None
    assert result["tokens"]["total_tokens"] is None


def test_run_task_counts_untracked_files_in_patch_metrics(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)

    class NewFileRunner:
        name = "new-file"

        def run(self, request):
            (Path(request.workspace) / "src" / "NewService.cs").write_text(
                "public class NewService {}\n", encoding="utf-8"
            )
            return [TrajectoryEvent(0, "final", 1, data={"completed": True})]

    result = run_task(
        _task(repo, commit), "baseline", NewFileRunner(), model="test-model",
        output_dir=tmp_path / "out",
    )

    assert result["patch"]["changed_files"] == ["src/NewService.cs"]
    assert result["patch"]["size_bytes"] == len(("public class NewService {}" + os.linesep).encode())
    assert result["patch"]["insertions"] == 1


def test_cli_refuses_schedules_above_maximum_run_count(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "suite": "cap-test",
        "tasks": [{
            "id": "one", "repository": str(repo), "base_commit": commit,
            "description": "x", "language": "csharp", "category": "bugfix",
        }],
    }), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main([
            "run", "--manifest", str(manifest), "--runner-command", "unused",
            "--model", "test", "--mode", "baseline", "--mode", "gpu_search",
            "--runs", "2", "--max-total-runs", "3",
            "--output-dir", str(tmp_path / "out"),
        ])


def _run_record(task: str, mode: str, run_id: str, success: bool, *,
                input_tokens: int | None, files: int, calls: int) -> dict:
    return {
        "schema_version": 1,
        "identity": {"task_id": task, "mode": mode, "run_id": run_id},
        "outcome": {"success": success, "validation_passed": success},
        "tokens": {"input_tokens": input_tokens, "output_tokens": 10,
                   "total_tokens": None if input_tokens is None else input_tokens + 10,
                   "repository_context_tokens_estimate": 20,
                   "gpu_search_context_tokens_estimate": 5 if mode == "gpu_search" else 0},
        "exploration": {"unique_files_read": files, "total_file_reads": files,
                        "irrelevant_file_count": 1, "total_tool_calls": calls},
        "timing": {"duration_ms": 100, "time_to_first_relevant_file_ms": 20,
                   "time_to_first_likely_implementation_ms": 30},
    }


def test_aggregation_handles_repeated_runs_missing_metrics_and_regressions() -> None:
    runs = [
        _run_record("one", "baseline", "b1", True, input_tokens=100, files=10, calls=20),
        _run_record("one", "baseline", "b2", True, input_tokens=120, files=12, calls=22),
        _run_record("one", "gpu_search", "g1", False, input_tokens=80, files=5, calls=12),
        _run_record("one", "gpu_search", "g2", True, input_tokens=None, files=7, calls=14),
    ]

    report = aggregate_runs(runs)

    assert report["run_count"] == 4
    assert report["modes"]["baseline"]["metrics"]["input_tokens"]["mean"] == 110
    assert report["modes"]["gpu_search"]["metrics"]["input_tokens"]["count"] == 1
    assert report["comparison"]["success_delta"] == -1
    assert report["comparison"]["input_token_reduction_pct"] is None
    assert report["comparison"]["paired_configuration"] is True
    assert report["comparison"]["regressed_tasks"] == ["one"]
    assert report["comparison"]["file_read_reduction_pct"] == 45.455
    assert "Regressed tasks: one" in report_markdown(report)


def test_aggregation_does_not_compare_unpaired_tasks() -> None:
    baseline = _run_record(
        "one", "baseline", "b1", True, input_tokens=100, files=10, calls=20
    )
    gpu = _run_record(
        "two", "gpu_search", "g1", True, input_tokens=50, files=5, calls=10
    )

    comparison = aggregate_runs([baseline, gpu])["comparison"]

    assert comparison["paired_configuration"] is False
    assert comparison["input_token_reduction_pct"] is None
    assert comparison["file_read_reduction_pct"] is None


def test_result_jsonl_round_trip_and_aggregation_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    first = _run_record("b", "gpu_search", "2", True, input_tokens=50, files=2, calls=3)
    second = _run_record("a", "baseline", "1", True, input_tokens=70, files=4, calls=5)
    append_run(path, first)
    append_run(path, second)

    loaded = load_runs([path])
    assert loaded == [first, second]
    assert aggregate_runs(loaded) == aggregate_runs(list(reversed(loaded)))
    json.dumps(aggregate_runs(loaded), sort_keys=True)


def test_command_runner_consumes_json_and_emits_jsonl(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text(
        "import json,sys\n"
        "request=json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'milestone','elapsed_ms':1,'milestone':'likely_implementation'}))\n"
        "print(json.dumps({'type':'final','elapsed_ms':2,'data':{'completed':request['mode']=='baseline'}}))\n",
        encoding="utf-8",
    )
    task = _task(repo, commit)
    request_workspace = tmp_path / "workspace"
    subprocess.run(["git", "clone", "--quiet", str(repo), str(request_workspace)], check=True)
    from gpu_service.agent_eval import RunRequest
    request = RunRequest("run", "baseline", task, str(request_workspace), commit,
                         "model", "command", {}, RunLimits(timeout_seconds=10))

    assert "relevant_files" not in request.to_dict()["task"]
    events = CommandAgentRunner([sys.executable, str(script)]).run(request)

    assert [event.type for event in events] == ["milestone", "final"]
    assert events[-1].data["completed"] is True


def test_command_runner_nonzero_exit_overrides_success(tmp_path: Path) -> None:
    repo, commit = _repo(tmp_path)
    script = tmp_path / "runner.py"
    script.write_text(
        "import json,sys\n"
        "json.loads(sys.stdin.readline())\n"
        "print(json.dumps({'type':'final','data':{'completed':True}}))\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    task = _task(repo, commit)
    request_workspace = tmp_path / "workspace"
    subprocess.run(["git", "clone", "--quiet", str(repo), str(request_workspace)], check=True)
    from gpu_service.agent_eval import RunRequest
    request = RunRequest("run", "baseline", task, str(request_workspace), commit,
                         "model", "command", {}, RunLimits(timeout_seconds=10))

    events = CommandAgentRunner([sys.executable, str(script)]).run(request)

    assert events[-1].type == "final"
    assert events[-1].data["completed"] is False
    assert events[-1].data["runner_exit_code"] == 7
