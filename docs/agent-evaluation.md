# Coding-agent evaluation harness

The agent evaluation harness measures coding-agent effectiveness and context efficiency, not merely GPU Search latency. It compares otherwise-equivalent runs in two modes:

- `baseline`: the runner exposes its normal repository tools but not GPU Search.
- `gpu_search`: the same runner, model, instructions, task, limits, and base commit may additionally use GPU Search.

The harness is opt-in. Unit tests use fake adapters and temporary local Git repositories; normal CI never invokes a paid agent or a network API.

## Task manifests

Task suites use JSON schema version 1:

```json
{
  "schema_version": 1,
  "suite": "dotnet-tasks-v1",
  "tasks": [
    {
      "id": "task-001",
      "repository": "D:/repos/sample-app",
      "base_commit": "0123456789abcdef",
      "description": "Fix expired tokens being accepted.",
      "language": "csharp",
      "category": "bugfix",
      "relevant_files": ["src/Auth/JwtValidator.cs"],
      "expected_changed_files": ["src/Auth/JwtValidator.cs"],
      "validation": [
        {"argv": ["dotnet", "test", "tests/Auth.Tests"], "timeout_seconds": 600}
      ]
    }
  ]
}
```

Repository paths may be local paths or Git URLs. Relative paths resolve from the manifest. Each run clones the repository into an isolated temporary workspace, resolves `base_commit` to an exact commit, and records both values. Validation commands are argument arrays and are executed without a shell. Manifests are trusted inputs because validation commands execute local programs.

`relevant_files` enables time-to-relevant-file and irrelevant-file metrics. These values are evaluator labels, not hints that should be passed to the agent. `expected_changed_files` enables patch-file recall; omit it when the valid patch shape is intentionally open-ended.

## Runner adapter protocol

Agent integration is behind `AgentRunner`. The included command adapter sends one JSON request on stdin and accepts normalized JSONL events on stdout. The request includes the task, workspace, exact commit, mode, model, limits, and `gpu_search_enabled`.

Supported event types are `tool_call`, `tool_result`, `milestone`, `usage`, and `final`. Example:

```jsonl
{"type":"tool_call","elapsed_ms":25,"tool":"read","category":"file_read","file_paths":["src/Auth/JwtValidator.cs"],"arguments":{"path":"src/Auth/JwtValidator.cs"}}
{"type":"tool_result","elapsed_ms":30,"category":"file_read","file_paths":["src/Auth/JwtValidator.cs"],"result_size_bytes":2048}
{"type":"usage","elapsed_ms":500,"token_usage":{"input_tokens":1200,"output_tokens":220,"cached_input_tokens":400}}
{"type":"final","elapsed_ms":900,"data":{"completed":true}}
```

Tool categories are `file_read`, `search`, `gpu_search`, `edit`, `test`, and `other`. Optional milestones include `likely_implementation` and `first_patch`. Adapters should report provider token usage exactly when available and omit unavailable fields. The harness represents omitted token metrics as `null`; it does not fabricate them.

Arguments and result metadata are sanitized before persistence. Secret-like keys are redacted. Adapters should emit metadata and sizes, not raw source or model responses, unless a controlled evaluation explicitly requires them.

## Running an evaluation

Install the package or run the module from a checkout:

```powershell
gpu-search-agent-eval run `
  --manifest benchmarks/agent_eval/tasks.example.json `
  --runner-command "python D:\path\to\adapter.py" `
  --runner-name codex-adapter `
  --model MODEL_AND_CONFIGURATION `
  --mode baseline `
  --mode gpu_search `
  --runs 3 `
  --max-total-runs 100 `
  --timeout-seconds 1800 `
  --max-tool-calls 250 `
  --output-dir D:\eval-results\run-001
```

The checked-in `example_runner.py` is a deterministic protocol smoke adapter, not a real agent and not evidence of product value.

`--max-total-runs` refuses unexpectedly large task x mode x repetition schedules. The timeout is enforced by the command adapter. `max_tool_calls` is sent to adapters and a run that exceeds it is marked incomplete; adapters should also enforce it online so they can stop an expensive run immediately.

Aggregate one or more run files:

```powershell
gpu-search-agent-eval report `
  --input D:\eval-results\run-001\runs.jsonl `
  --output-json D:\eval-results\run-001\report.json `
  --output-markdown D:\eval-results\run-001\report.md
```

## Outputs and metrics

Each line in `runs.jsonl` is a versioned run record with:

- task/run/mode/repository/commit/model/runner/GPU Search identity
- completion, validation, test, patch, and changed-file outcomes
- files read, repeated reads, searches, GPU Search calls, total tools, and derivable irrelevant files
- provider token usage when available
- approximate normal-file and GPU Search context tokens based on reported result bytes
- duration and times to relevant file, likely implementation, first patch, and final patch
- limits, validation details, and a reference to a sanitized trajectory JSONL file

The aggregate report includes counts, success rates, per-task regressions, mean/median/p50/p95/min/max metrics, and per-success token/file/tool metrics. Reduction percentages remain `null` unless both modes have matching task, commit, model, runner, and configuration multisets and every run exposes that metric.

Approximate context tokens use `ceil(bytes / 4)` and are labeled estimates. They do not replace provider token accounting.

## Interpretation

Use the same task commit, model/configuration, instructions, runner, and limits in both modes. Repeat runs when model behavior is nondeterministic. Show failures and regressions. A small fixture suite validates instrumentation but cannot establish statistically significant product claims.

The initial target is approximately 20 realistic C#/.NET tasks spanning bugs, business logic, endpoints, interfaces, dependency impact, DI, configuration, tests, regressions, and multi-file changes. Building and reviewing that task corpus is separate from the harness-foundation PR.
