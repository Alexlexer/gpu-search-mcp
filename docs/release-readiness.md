# Release Readiness Checklist — gpu-search-mcp

Pre-release validation pass for a local alpha demo or stable release cut.

---

## 1. Tests

```bash
python -m pytest tests/ -v --timeout=60 --ignore=tests/test_thread_safety.py -k "not with_semantic"
```

All tests must pass with zero failures or errors. Thread-safety and semantic tests are excluded from the standard pass — run them separately on appropriate hardware if needed.

---

## 2. Lint

```bash
ruff check gpu_service/ tests/
```

Must exit 0 with no lint errors.

---

## 3. Import check

```bash
python -c "from gpu_service.mcp_server import cli_main; print('ok')"
```

Must print `ok` with no import errors.

---

## 4. Smoke test

```bash
python scripts/smoke_test.py
```

Must complete with no failures. This exercises pattern search, HTTP endpoints, and basic MCP tool routing without downloading the semantic model.

---

## 5. HTTP endpoint checks

Start the server against a test directory:

```bash
gpu-search-mcp --directory . --http --port 8765 --device cpu
```

Then verify each endpoint:

```bash
curl http://127.0.0.1:8765/health
curl http://127.0.0.1:8765/stats
curl http://127.0.0.1:8765/diagnostics
```

```bash
curl -X POST http://127.0.0.1:8765/scan/signals \
  -H "Content-Type: application/json" \
  -d '{"categories": ["legacy-dotnet"], "topKPerSignal": 3}'
```

Verify:

- [ ] `/health` returns `{ "ok": true }`.
- [ ] `/stats` returns index sizes, device, and cache metadata.
- [ ] `/diagnostics` returns device, roots, pattern/semantic/dependency readiness, and capabilities without triggering downloads or reindexing.
- [ ] `/scan/signals` returns categorized findings without error.

---

## 6. Semantic model status

```bash
curl http://127.0.0.1:8765/stats
```

Confirm `semanticModel` field is present in the response and reflects the correct preflight status (`available` or `unavailable`).

If unavailable, confirm pattern search still works and `/diagnostics` reports the status clearly without crashing.

---

## 7. Verify no secret leakage

Review the `/stats` and `/diagnostics` responses and confirm:

- [ ] No API keys, bearer tokens, or passwords appear.
- [ ] No connection strings or private key material.
- [ ] No raw environment variable values.
- [ ] No absolute paths outside indexed roots.

`/diagnostics` is explicitly designed to exclude raw config, tokens, secrets, and environment values.

---

## 8. Cache directory

Confirm `.gpu-search-cache/` behaviour:

- [ ] Deleting `.gpu-search-cache/` is safe — source files are never modified.
- [ ] Server restarts cleanly after cache deletion and rebuilds from source.
- [ ] `cache-meta.json` is present after a successful startup and contains schema version and fingerprint fields.

Test cache rebuild:

```bash
gpu-search-mcp --directory . --http --rebuild-cache --device cpu
```

---

## 9. Verify no source file modification

Confirm that running the server, smoke tests, or cache rebuild operations do not modify any file in the indexed source directory:

```bash
# Before test
git -C /path/to/indexed/repo status

# Run server / smoke tests

# After test
git -C /path/to/indexed/repo status
# must show: nothing to commit, working tree clean
```

---

## 10. README and docs links

Review `README.md` and confirm:

- [ ] Demo workflow references LegacyLens `docs/demo-alpha.md`.
- [ ] Release readiness link points to `docs/release-readiness.md`.
- [ ] `--download-semantic-model` command is documented and correct.
- [ ] `/diagnostics` endpoint is documented.
- [ ] Sentence-transformers vs Ollama distinction is clearly stated.

---

## Known limitations

Document before release:

- **Dependency graph** is regex/heuristic, not compiler-accurate. Dynamic imports, conditional imports, and generated code are not modelled. Use `dep_impact` results as advisory starting points.
- **Secret redaction** is best-effort pattern matching, not a DLP scanner. Do not rely on it as the only protection.
- **Tree-sitter coverage** targets Python, TypeScript/JavaScript, and C#. Other file types fall back to line-window snippets.
- **C# dependency analysis** is best-effort; Roslyn integration (compiler-accurate) is future work.
- **Semantic model** is not downloaded on first startup — requires an explicit `--download-semantic-model` pass.
- **HTTP binds to `127.0.0.1` by default** — local-first. Binding to `0.0.0.0` requires an explicit `--host 0.0.0.0`.

---

## Related

- [README.md](../README.md) — installation, usage, and endpoint reference
- [docs/signal-scan.md](signal-scan.md) — `/scan/signals` endpoint documentation
- [LegacyLens release-readiness](https://github.com/Alexlexer/LegacyLens/blob/main/docs/release-readiness.md)
- [LegacyLens demo-alpha](https://github.com/Alexlexer/LegacyLens/blob/main/docs/demo-alpha.md)
