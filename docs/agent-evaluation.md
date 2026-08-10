# Coding-agent evaluation harness

The harness measures coding-agent correctness and context efficiency, not merely GPU Search latency. It compares otherwise-equivalent runs in two modes:

- `baseline`: Codex receives normal repository capabilities and no GPU Search MCP server.
- `gpu_search`: the same Codex model, prompt, workspace commit, sandbox, limits, and runner configuration additionally receive one GPU Search MCP server.

The harness is opt-in. Normal tests use fake processes and temporary local Git repositories; CI never invokes Codex or a paid API.

## Correctness eligibility

Task manifests distinguish:

- `instrumentation-smoke`: validates protocol wiring only and is excluded from correctness comparisons.
- `benchmark`: requires at least one deterministic validation command and a full 40-character Git commit SHA.

For benchmark tasks, `success` requires validation to pass. Agent telemetry such as `final.completed=true`, patch production, or a confident final message is not correctness proof. A programmatically constructed benchmark task without validation is recorded as ineligible rather than successful.

Evaluator-only fields—validation commands, `relevant_files`, and `expected_changed_files`—are never included in the request sent to the agent.

## Task manifest

Schema version 1 remains additive. A real task resembles:

```json
{
  "id": "task-001",
  "repository": "../..",
  "base_commit": "0123456789abcdef0123456789abcdef01234567",
  "description": "Fix expired promotions without changing rounding.",
  "language": "csharp",
  "category": "bugfix",
  "evaluation_type": "benchmark",
  "relevant_files": ["src/Pricing/DiscountCalculator.cs"],
  "expected_changed_files": ["src/Pricing/DiscountCalculator.cs"],
  "validation": [
    {
      "argv": [
        "python",
        "{manifest_dir}/validate_task.py",
        "task-001"
      ],
      "timeout_seconds": 180
    }
  ]
}
```

Relative repository paths resolve from the manifest. Each run clones the repository into an isolated workspace and checks out the exact commit. `{manifest_dir}` and `{workspace}` placeholders are expanded only when evaluator validation runs, after the agent has exited.

Manifests are trusted inputs because validation commands execute local programs.

## Usage-event semantics

Every provider with multiple usage events must declare whether events are deltas or cumulative snapshots:

```jsonl
{"type":"usage","usage_semantics":"delta","token_usage":{"input_tokens":5000}}
{"type":"usage","usage_semantics":"cumulative","token_usage":{"input_tokens":20000}}
```

Rules:

- `delta`: reported values are summed.
- `cumulative`: the latest non-decreasing value for each metric is authoritative.
- one event without semantics remains backward-compatible because summing and taking the snapshot are equivalent.
- multiple events with missing, mixed, invalid, or decreasing cumulative semantics are marked invalid; provider-token fields remain `null`.
- missing provider metrics remain `null`.

Provider counts can include input, output, cached input, reasoning, and total tokens. Cached and reasoning tokens are not added to input/output when deriving a missing total.

`repository_context_tokens_estimate` and `gpu_search_context_tokens_estimate` remain `ceil(result_bytes / 4)` estimates. They are not provider token counts.

## Codex adapter

The opt-in adapter uses Codex non-interactive JSONL output. It follows the documented `codex exec --json` event stream and uses `--sandbox workspace-write`; see [official Codex non-interactive-mode documentation](https://learn.chatgpt.com/docs/non-interactive-mode).

Isolation rules:

1. Both modes use `--ignore-user-config`, the same prompt, model, sandbox, limits, and extra arguments.
2. Baseline receives no MCP configuration.
3. GPU mode adds only `mcp_servers.gpu_search`, pointed at the isolated workspace.
4. Dangerous full-access/bypass arguments are rejected.
5. Raw model messages, command output, source text, environment variables, and credentials are not persisted. The adapter stores bounded metadata, result byte sizes, paths, safe search arguments, usage, and sanitized error tails.

The adapter normalizes Codex command executions, file changes, MCP calls, usage, process failures, malformed events, and timeouts into the harness trajectory protocol.

### Prerequisites

```powershell
codex --version
codex login status
python -m pip install -e ".[test]"
```

The initial suite requires a .NET 8-or-newer SDK. It has no external test-package dependency.

### Run the 30-run experiment

Use an exact model identifier/configuration and record it with the results:

```powershell
gpu-search-agent-eval run `
  --manifest benchmarks/agent_eval/tasks.dotnet-real.json `
  --runner-command "gpu-search-codex-eval-adapter" `
  --runner-name codex-cli `
  --runner-config-file benchmarks/agent_eval/codex-runner-config.example.json `
  --model YOUR_EXACT_CODEX_MODEL `
  --mode baseline `
  --mode gpu_search `
  --runs 3 `
  --order alternating `
  --seed 20260810 `
  --timeout-seconds 1800 `
  --max-tool-calls 250 `
  --max-total-runs 30 `
  --output-dir agent-eval-results/storefront-codex-v1
```

From a development checkout without refreshed console scripts, use the adapter's absolute path:

```powershell
$adapter = (Resolve-Path gpu_service/codex_eval_adapter.py).Path
python -m gpu_service.agent_eval run `
  --manifest benchmarks/agent_eval/tasks.dotnet-real.json `
  --runner-command "python $adapter" `
  --runner-name codex-cli `
  --runner-config-file benchmarks/agent_eval/codex-runner-config.example.json `
  --model YOUR_EXACT_CODEX_MODEL `
  --mode baseline --mode gpu_search --runs 3 `
  --order alternating --seed 20260810 `
  --max-total-runs 30 `
  --output-dir agent-eval-results/storefront-codex-v1
```

`alternating` reverses mode order across task/repetition pairs so one mode does not always receive the same warm-cache position. `random` is also available with a recorded seed.

### Aggregate

```powershell
gpu-search-agent-eval report `
  --input agent-eval-results/storefront-codex-v1/runs.jsonl `
  --output-json agent-eval-results/storefront-codex-v1/report.json `
  --output-markdown agent-eval-results/storefront-codex-v1/report.md
```

## Initial real suite

`benchmarks/agent_eval/tasks.dotnet-real.json` contains five independent tasks against immutable fixture commit `84e6c6d499f3e6152c8e439ef1f2089a8476f1e7`:

1. expired-promotion bug fix
2. repository interface + implementation + service change
3. DI and options configuration
4. endpoint/business-logic correction
5. multi-file cancellation regression

The fixture is purpose-built but production-shaped. Tasks were selected across different boundaries and were not designed around one GPU Search query. Hidden deterministic validation is outside the agent's checked-out base commit.

## Outputs and interpretation

Each run stores identity/configuration, exact commit, environment/provider metadata, correctness eligibility, validation, patch metrics, sanitized JSONL trajectory, provider usage when available, context-size estimates, repository exploration, and timing.

Reports include mean, median, p50, p95, min, and max statistics; per-task regressions are always shown. Percentage reductions remain `null` unless task/commit/model/runner/configuration multisets are paired and every eligible run exposes the metric.

This five-task experiment is preliminary. It proves the experiment path; it cannot establish broad claims about token savings, speed, or task success.

## First local pilot status

A Windows pilot on 2026-08-11 confirmed Codex authentication and JSONL parsing, but the nested Codex process could not write its isolated workspace because the local Codex sandbox helper was unavailable. Both pilot modes therefore produced no patch. They are not valid A/B results and are excluded from product claims.

Run the documented 30-run command in an environment where a direct Codex workspace-write smoke task can actually modify a disposable checkout. See [`benchmarks/agent-eval-codex-pilot-2026-08-11.md`](benchmarks/agent-eval-codex-pilot-2026-08-11.md) for the blocker record.
