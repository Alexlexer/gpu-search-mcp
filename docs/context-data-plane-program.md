# Context data plane engineering program

Last updated: 2026-08-11

## Product outcome

GPU Search retrieves, combines, ranks, and compresses the minimum high-confidence repository evidence a coding agent needs to solve a task. The primary metrics are task success, patch correctness, tests, context tokens, unnecessary files, tool calls, time to implementation, and physical repository I/O. GPU acceleration is optional; CPU correctness and local/private operation are mandatory.

## Program status

| Phase | Outcome | Status | Exit gate |
|---|---|---|---|
| A | Agent evaluation harness and current baseline | Instrumentation complete; real baseline pending | Real Codex adapter and five-task suite are ready; collect a valid 30-run A/B result in a writable nested-agent environment |
| B | `prepare_context` v1 | Planned | One high-level request reuses current evidence/planner and returns structured task context |
| C | Context ranking and token budgeting | Planned | Evaluation-driven ranking, deduplication, provenance, budget allocation, and omission reporting |
| D | 1/10/30/100 GiB corpus benchmark | Planned | Quantified physical reads, initialization, RAM/VRAM, and stage timing at 100 GiB |
| E | Smarter candidate selection | Planned | All/first/rarest/intersection comparison with exact equivalence |
| F | Adaptive CPU/GPU verification | Planned | Measured crossover policy with identical correctness |
| G | `StructureProvider` boundary | Planned | Generic provider contract supports different structural backends |
| H | Existing C# behind provider | Planned | Existing C# behavior passes unchanged behind the contract |
| I | Second structurally different backend | Planned | Context pipeline consumes another backend without core redesign |

After Phase I, measured agent trajectories and large-corpus profiles determine priorities. Do not continue a feature list without evidence.

## Invariants

- Python remains the authoritative runtime; do not reintroduce Rust.
- Exact verification is authoritative and CPU-compatible.
- MCP and HTTP remain backward-compatible unless a documented migration is required.
- GPU Search remains local/private by default.
- Structural-provider absence never disables generic retrieval.
- Do not optimize candidate selection, CPU/GPU policy, GDS, or language breadth without measurements.
- Every phase is a focused PR with tests, validation, documentation, and reported limitations.

## PR log

| Date | Phase | Branch/PR | Result | Next action |
|---|---|---|---|---|
| 2026-08-10 | A | `eval/agent-baseline-harness` | Reusable harness, command-runner protocol, deterministic fixture, reports, and tests complete | Build and review a realistic C# task corpus, then collect the current baseline before Phase B |
| 2026-08-11 | A | eval/agent-baseline-harness / PR #96 | Added explicit usage semantics, validation-gated success, a real Codex adapter, five .NET tasks, hidden validators, deterministic ordering, and stronger reports | Run the 30-run A/B suite where nested Codex has workspace-write access; local pilot was invalid because the sandbox was read-only |
