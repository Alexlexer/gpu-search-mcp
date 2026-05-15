# Release Checklist — v0.1.0

Run every step in order before tagging. All required steps must pass.

---

## 1. Run tests

```bash
pytest tests/ -v --timeout=60 \
  --ignore=tests/test_thread_safety.py \
  -k "not with_semantic"
```

Expected: all collected tests pass. Thread-safety tests are excluded — run them locally on a
well-resourced machine if needed (`pytest tests/test_thread_safety.py -v`).

---

## 2. Run linter

```bash
ruff check gpu_service/ tests/
```

Expected: `All checks passed!`. No errors permitted before release.

---

## 3. Run smoke test

```bash
python scripts/smoke_test.py
```

Expected: `[PASS]` for pattern index, dependency index, pattern search, dep impact, read block,
and semantic formatter. The semantic model download step (`--with-semantic`) is optional in CI
but worth running locally if the model is already cached.

---

## 4. Run install test

```bash
pip install -e ".[test,ast]"
python -c "from gpu_service.mcp_server import cli_main; print('ok')"
```

Expected: `ok`. Verifies the package installs cleanly and the entry point imports without error.

---

## 5. Run optional benchmark (local only)

```bash
gpu-search-bench --directory <path-to-large-repo> --output results.json
```

Compare against the reference numbers in README.md. Not required to pass; use as a regression
check if touching indexing or search logic.

---

## 6. Verify version consistency

All three sources must agree:

```bash
grep "^version" pyproject.toml
grep "^VERSION" gpu_service/mcp_server.py
head -1 README.md
```

Expected output (adjust to current release):

```
version = "0.1.0"
VERSION = "0.1.0"
# gpu-search-mcp `v0.1.0`
```

Also confirm `CHANGELOG.md` heading and `docs/releases/` file name match.

---

## 7. Verify README examples

- HTTP endpoint table in README matches the actual routes in `gpu_service/mcp_server.py`.
- Example JSON responses match the actual response shapes returned by the server.
- Install command (`pip install -e ".[test,ast]"`) is current.
- Link to `docs/releases/v0.1.0.md` resolves.

---

## 8. Tag and release

Only run after all steps above pass:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then create the GitHub release manually:

1. Go to the repository → **Releases** → **Draft a new release**.
2. Choose the `v0.1.0` tag.
3. Set the title to `v0.1.0`.
4. Paste the `## 0.1.0` section from `CHANGELOG.md` as the release body.
5. Attach no build artifacts (no PyPI publish for this release).
6. Publish.

No automated PyPI publish is configured. Distribution is manual for v0.1.0.
