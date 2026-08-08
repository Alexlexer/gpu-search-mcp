# gpu-search-mcp Python-Only Roadmap

> Source: .agents/plan.txt
> Status: Active
> Direction: Python-only. No Rust code, bindings, sidecars, Cargo files, or migration work.

## Product outcome

Build a production-ready, local-first code-intelligence engine for AI coding agents. One request should return a compact, explainable bundle of the relevant implementation, symbols, callers, dependencies, configuration, tests, risks, and recommended inspection order.

The product is more than GPU grep or vector search. GPU acceleration is optional; correct and useful CPU-only behavior is mandatory.

## Principles

- **Local-first:** no account, API key, cloud database, or telemetry for normal operation.
- **Agent-first:** compact, structured, explainable, change-oriented responses.
- **Progressive capability:** exact search works without embeddings; optional features degrade explicitly.
- **Stable contracts:** MCP and HTTP remain backward-compatible while versioned surfaces are added.
- **Python-only:** never add a Rust core, bindings, sidecars, Cargo workspace, or Rust migration tasks.
- **Secure by default:** canonical paths, local binding, redaction, resource limits, and safe diagnostics.
- **Evidence-based:** performance and retrieval claims require reproducible benchmarks.
- **Small slices:** every change includes affected files, tests, validation, and known limitations.

## Baseline to preserve

- PyTorch exact byte-pattern search.
- CUDA, Apple Silicon MPS, and CPU execution.
- Sentence-transformer semantic retrieval and hybrid search.
- Persistent pattern, semantic, line-offset, and dependency caches.
- Filesystem watching and incremental reindexing.
- MCP stdio and local HTTP transports.
- Dependency impact analysis.
- AST block expansion and file skeletons.
- C#/.NET-aware heuristics.
- Secret redaction and indexed-root validation.
- LegacyLens signal scanning.
- Console entry point, installer, smoke tests, and structured HTTP results.

Any replacement must match current behavior before an old field, route, command, or tool is deprecated.

## Delivery map

| Milestone | Outcome | Status | Exit gate |
|---|---|---:|---|
| 1. Usable product | Unified search, setup, diagnostics, packaging, onboarding | In progress | Fresh CPU install configures a client and completes a diagnostic search |
| 2. C# intelligence | Language-neutral symbol graph and high-quality C# lookup | Completed | C# fixtures pass symbol, caller, DI, endpoint, implementation, and test queries |
| 3. Change planning | Token-budgeted plans with risks and inspection order | Completed | Change requests return implementation, impact, config, tests, omissions, and risks |
| 4. Quality/reliability | Benchmarks, regression gates, reliable caches | In progress | CI detects quality, latency, budget, CPU, and cache regressions |
| 5. Languages/distribution | TypeScript/Python symbols, bundles, multi-root | Planned | Language fixtures and packaged-install smoke gates pass |
| 6. Security/API | Versioned API, limits, authentication, injection warnings | Planned | Security and transport end-to-end matrices pass |

# Milestone 1 — Usable product

## Current implementation map

- MCP tools and unified text search: **gpu_service/mcp_tools.py**
- Shared indexes, routing, formatting, structured search: **gpu_service/mcp_server.py**
- HTTP transport and root filtering: **gpu_service/http_server.py**
- Existing installer/configuration logic: **install.py**
- Package metadata and console scripts: **pyproject.toml**
- HTTP schema: **docs/openapi/gpu-search-mcp.openapi.yaml**
- Smoke coverage: **scripts/smoke_test.py**
- Contract coverage: **tests/test_http_api.py**
- Root, cache, security, device, and language coverage: **tests/**

## Unified search_code contract

Request:

~~~json
{
  "query": "where is JWT expiration validated?",
  "mode": "auto",
  "intent": "understand",
  "top_k": 8,
  "context_mode": "compact",
  "include_dependencies": true,
  "include_tests": true
}
~~~

Modes:

- **auto:** select the best ready engines.
- **exact:** exact byte-pattern search; alias of existing pattern mode.
- **pattern:** backward-compatible exact-search name.
- **semantic:** embedding retrieval when ready.
- **hybrid:** merge exact and semantic results without duplicate files.
- **symbol:** explicit exact fallback until Milestone 2 is ready.
- **path:** explicit exact fallback until dedicated path retrieval exists.

Intents:

- **locate:** prioritize direct matches.
- **understand:** add structural context.
- **modify:** prefer hybrid and expand dependencies/tests.
- **debug:** prioritize callers, tests, recent changes, errors, and configuration.
- **audit:** prioritize breadth, provenance, warnings, and signals.

Response:

~~~json
{
  "query": "...",
  "mode_used": "hybrid",
  "intent": "modify",
  "primary_results": [],
  "related_files": {
    "callers": [],
    "dependencies": [],
    "implementations": [],
    "tests": [],
    "configuration": []
  },
  "warnings": [],
  "index_status": {
    "pattern_ready": true,
    "semantic_ready": true,
    "symbol_ready": false
  }
}
~~~

Compatibility rules:

- Preserve the MCP text response until a versioned structured MCP result is introduced.
- Preserve HTTP fields result, mode, contextMode, and results.
- Add new snake_case fields without removing legacy fields.
- Accept existing pattern mode and camelCase HTTP input.
- Warn when semantic, symbol, dependency, or test capabilities are unavailable.
- Never silently truncate structural results.

Implementation checklist:

- [x] Audit current routing and structured HTTP behavior.
- [x] Add shared mode and intent normalization.
- [x] Add exact alias plus explicit symbol/path fallbacks.
- [x] Add mode_used, intent, primary_results, related_files, warnings, and index_status.
- [x] Accept dependency/test expansion over HTTP and MCP.
- [x] Reuse the dependency graph for callers and direct dependencies.
- [x] Add conservative test/configuration path classification.
- [x] Add routing and backward-compatibility tests.
- [x] Update OpenAPI and user documentation after the contract stabilizes.

Acceptance criteria:

- Existing callers and tests pass.
- Identifiers use exact search.
- Prose uses semantic when ready and exact fallback otherwise.
- Modify intent uses hybrid when semantic is ready and expands related context.
- Hybrid results contain no duplicate files.
- Readiness and fallback warnings are deterministic.
- Empty results return the complete schema.
- CPU-only operation remains fully functional.

## One-command setup

Command: **gpu-search-mcp setup**

Required options:

- --client codex, claude, cursor, or windsurf
- --no-index
- --no-model
- --dry-run
- --yes

Workflow:

1. Validate OS and Python.
2. Resolve CUDA, MPS, or CPU.
3. Detect supported clients.
4. Preview changes.
5. Back up configuration before writes.
6. Validate the local model cache.
7. Ask before model downloads.
8. Index the current repository unless disabled.
9. Run an exact diagnostic query.
10. Print success, warnings, changed files, and next steps.

Acceptance criteria:

- Dry-run changes nothing.
- Setup is idempotent.
- Existing configuration is preserved.
- Model downloads are explicit.
- Setup works outside the checkout.
- Failures provide safe remediation.

## Diagnostics

Commands:

- **gpu-search-mcp doctor**
- **gpu-search-mcp doctor --json**

Report safe metadata only: app/API/cache versions, OS/Python, selected backend, device availability, model cache state, indexed-root readiness, MCP configuration presence, local HTTP health, warnings, and remediation.

Doctor must not modify files, download models, rebuild indexes, expose secrets, or print source.

## Packaging and onboarding

- Split extras into semantic, ast, cuda, test, and all.
- Keep base CPU/exact installation usable.
- Support pipx, uv tool, and uvx.
- Add gpu-search-mcp --version.
- Verify execution outside the checkout.
- Document Codex and Claude first; Cursor and Windsurf follow.
- Add wheel build and isolated-install smoke tests.

## Milestone 1 test gate

- Query classification and routing.
- New and legacy response fields.
- Exact alias and capability fallbacks.
- Setup dry-run, backups, idempotency, and client detection.
- Doctor text and JSON output.
- Wheel build and isolated installation.
- MCP initialize, tools/list, and tools/call.
- HTTP schema, errors, root restrictions, and concurrency.
- CPU smoke with no model download.

# Milestone 2 — C# symbol intelligence

Status: completed on 2026-07-22.

- Stable Python Symbol/SymbolEdge graph with deterministic identifiers.
- Dependency-free C# fallback for declarations, relationships, ASP.NET endpoints, DI, and tests.
- Symbol, reference, implementation, caller, callee, test, and impact MCP operations.
- Golden ASP.NET exit-gate coverage with confidence and provenance.

Create stable Symbol and SymbolEdge models.

Symbol kinds: namespace, module, class, struct, interface, enum, record, method, function, constructor, property, field, constant, endpoint, and test.

Edge kinds: imports, calls, inherits, implements, instantiates, references, overrides, configured_by, and tested_by.

Every edge records confidence, provenance, source/target identifiers, source location, parser, and parser version.

C# work:

- Extract namespaces, types, methods, constructors, properties, fields, and signatures.
- Extract inheritance, interfaces, using directives, and probable calls.
- Detect ASP.NET controllers/endpoints and dependency-injection registrations.
- Detect tests and probable target symbols.
- Preserve regex/heuristic fallback without AST extras.

Advanced operations: find_symbol, find_references, find_implementations, find_callers, find_callees, find_tests, and explain_impact.

Exit gate: a golden ASP.NET fixture answers implementation, instantiation, controller caller, DI registration, test coverage, and interface-impact queries with confidence and provenance.

# Milestone 3 — Agent change planning

Status: completed on 2026-07-22.

- Deterministic `plan_change(request, top_k, max_context_tokens)` MCP operation.
- Ordered context bundles spanning implementation, graph context, tests, configuration, and Git state.
- Explicit token estimates, truncation, omitted-item metadata, risks, unknowns, likely changes, and inspection order.
- Exact symbol matches remain ahead of Git-boosted candidates.

Add plan_change with request, top_k, and max_context_tokens.

Bundle order:

1. Primary implementation.
2. Parent class/module context.
3. Direct callers.
4. Direct dependencies.
5. Implementations/overrides.
6. Related configuration/documentation.
7. Tests and missing coverage.
8. Relevant Git changes.
9. Match reasons/confidence.
10. Risks, unknowns, and inspection order.

Use deterministic token budgets and explicit omitted-item metadata. Git state may influence ranking, but must never outrank an exact symbol match.

# Milestone 4 — Retrieval quality and reliability

Status: in progress from 2026-07-22.

Completed benchmark-foundation slice:

- Versioned JSON/YAML manifest contract with checked-in C#, TypeScript, Python, and mixed fixtures.
- Deterministic Recall@1/5/10, Precision@5, MRR, exact-symbol recall, and related-test recall.
- Comparable ripgrep, exact, symbol, semantic, hybrid, hybrid-plus-symbol, and dependency-expanded modes.
- Index/search latency, throughput, incremental update, cache, VRAM, returned-token, and available peak-RAM reporting.
- Explicit baseline comparison thresholds; no unapproved threshold is implicit.
- Reviewed CPU fixture baselines plus CI gates for zero quality drop, bounded token growth, and a hard compact-output ceiling.

Completed cache-reliability slice:

- SHA-256 source-content identities cover repository paths, cache schemas, app versions, and parser/model/chunking/configuration components.
- Pattern, dependency, and semantic artifacts now commit with their metadata under a repository lock.
- Same-directory temporary files, fsync, atomic promotion, rollback backups, active-transaction detection, stale-lock recovery, and interrupted-commit recovery are covered by failure-injection tests.
- Cache schema versions were advanced so older non-content-addressed artifacts rebuild once instead of being trusted.

Next reliability slice: exercise repository reconciliation across rename/branch/worktree/event-storm scenarios and establish runner-specific latency gates.

Benchmark manifests cover C#, TypeScript, Python, and mixed repositories with expected files and symbols.

Metrics:

- Recall at 1, 5, and 10; Precision at 5; mean reciprocal rank.
- Exact-symbol and related-test recall.
- Cold start, indexing throughput, incremental/search latency.
- Peak RAM/VRAM, cache size, and returned tokens.

Compare ripgrep, exact CPU/device, semantic, hybrid, hybrid plus symbols, and hybrid plus dependency reranking. Define baselines before thresholds.

Cache work:

- Content-address source, parser, model, chunking, schema, configuration, and app metadata.
- Invalidate only affected artifacts.
- Use temporary writes, atomic rename, repository locks, stale-lock recovery, transaction detection, validation, and rollback.
- Reconcile create, modify, delete, rename, branch switch, rebase, worktree, and watcher storms.

# Milestone 5 — Multi-language and distribution

- TypeScript symbols: imports, exports, classes, interfaces, types, functions, methods, calls, React components, tests.
- Python symbols: modules, imports, classes, functions, decorators, inheritance, calls, pytest, unittest.
- Standalone Python bundles where practical.
- Multi-repository workspaces with root isolation.
- Language-specific quality gates before advertising support.

# Milestone 6 — Security and public API

Versioned routes:

- /v1/search/code
- /v1/search/symbol
- /v1/change/plan
- /v1/index/root
- /v1/index/status
- /v1/diagnostics

Every response includes api_version and server_version.

Security work:

- Canonicalize every path under indexed roots.
- Reject traversal, outside-root paths, symlink escapes, Windows case/UNC bypasses, and encoded traversal.
- Bound file/repository size, file count, query length, results, context, concurrency, and semantic batches.
- Redact credentials, tokens, passwords, connection strings, private keys, database URLs, and JWTs.
- Support custom redaction rules.
- Flag instruction-like content as possible prompt injection while retaining provenance.
- Bind HTTP to 127.0.0.1 by default.
- Require explicit external binding, authentication, restricted CORS, request limits, and rate limiting.
- Hide internal traces by default.

# CI and documentation target

CI:

- Windows, Linux, macOS.
- Python 3.10–3.13 where supported.
- Unit, integration, MCP, HTTP, security, migration, package, and CPU smoke tests.
- Ruff, type checking, build, and isolated install.
- Retrieval quality, latency, and output-budget gates after baselines are approved.

Documentation target:

- README.md
- docs/getting-started.md
- docs/installation.md
- docs/clients/claude.md
- docs/clients/codex.md
- docs/clients/cursor.md
- docs/configuration.md
- docs/search.md
- docs/symbol-intelligence.md
- docs/change-planning.md
- docs/http-api.md
- docs/security.md
- docs/benchmarks.md
- docs/troubleshooting.md
- docs/architecture.md
- docs/cache-format.md
- CHANGELOG.md
- ROADMAP.md
- CONTRIBUTING.md

# Definition of done for the next release

A new user can run pipx install gpu-search-mcp and gpu-search-mcp setup, then ask Codex or Claude to change JWT expiration behavior and receive:

- Primary implementation and relevant symbols.
- Callers and dependencies.
- Configuration and documentation.
- Tests and missing coverage.
- Match reasons and confidence.
- Risks and unknowns.
- Readiness, omissions, and inspection order.

The flow works locally on CPU. CUDA, MPS, embeddings, and AST grammars are optional accelerators or quality improvements.

# Execution policy

For every slice:

1. State affected files and behavior.
2. Check dependency impact before editing.
3. Add or update tests.
4. Run focused and relevant broader suites.
5. Run Ruff and packaging checks when applicable.
6. Record limitations.
7. Keep commits small and reviewable.
8. Never report success without validation.
9. Never introduce Rust artifacts.

# Progress log

- **2026-07-20:** Added the first backward-compatible unified search contract slice: intent-aware routing, exact/symbol/path normalization, structured primary and related results, dependency/test expansion, readiness metadata, warnings, root filtering, and focused regression coverage.
- **2026-07-21:** Published the unified request/response contract in OpenAPI and README, added schema regression tests, and implemented the read-only doctor command with JSON output, configured-root/client detection, loopback health probing, and a version flag.
- **2026-07-21:** Added the packaged setup command for Claude and Codex with explicit or detected client selection, dry-run, confirmation, atomic writes, pre-write backups, preserved unrelated configuration, local-only model checks, startup-root registration, and idempotence tests.
- **2026-07-22:** Started Milestone 4 with versioned multi-language quality manifests, deterministic retrieval metrics, live mode comparisons, runtime/resource measurements, and opt-in baseline regression gates.
- **2026-07-22:** Added portable CPU retrieval baselines and CI gates for quality regressions, relative token growth, and absolute output budgets.
- **2026-07-23:** Added content-addressed pattern/dependency/semantic cache identities and crash-safe multi-artifact transactions with repository locks, stale recovery, rollback, and failure-injection coverage.

# Immediate queue

1. **Now:** split packaging extras and add isolated-install smoke coverage.
2. **Next:** add Codex and Claude onboarding documents plus CPU-only end-to-end setup coverage.
3. **Then:** extend setup with device reporting, Cursor/Windsurf adapters, and a post-setup exact diagnostic query.
4. **Milestone gate:** validate a fresh installation from package install through diagnostic search.
