# Codex A/B pilot status — 2026-08-11

## Status

**No valid A/B result was produced.** Do not use the pilot as evidence that GPU Search helps or hurts coding-agent performance.

## Environment checked

- Host: Windows
- Codex CLI: 0.147.0
- Authentication: available through ChatGPT login
- Requested model/config: gpt-5.6-sol, high reasoning effort
- Requested sandbox: workspace-write
- Suite: five-task Storefront .NET suite
- Intended experiment: 5 tasks × 2 modes × 3 repetitions

## What happened

One baseline pilot and one GPU Search pilot reached the real Codex CLI and emitted machine-readable events, but neither produced a patch. A direct diagnostic run reported that the workspace was read-only. The local Codex Windows sandbox setup helper was unavailable, so the nested agent could not apply edits even though workspace-write was requested.

An elevated-sandbox diagnostic was also attempted and hung without producing a patch or final result. It was terminated. These attempts tested environment wiring only.

The pilots are not comparable product measurements:

- task edits were impossible;
- validation necessarily failed;
- the full randomized/alternating 30-run schedule was not executed;
- no token, file-read, tool-call, or timing reduction should be calculated from them.

## Reproduction gate

Before starting the paid/long-running experiment, verify in a disposable checkout that a direct Codex exec command with JSON output, ephemeral mode, ignored user config, workspace-write sandbox, and the chosen model can create a requested file.

Do not run the A/B suite until that file is actually created. Then run the exact 30-run command documented in docs/agent-evaluation.md.

Preserve failures and regressions; do not rerun only failed conditions unless the rerun policy is declared for both modes.
