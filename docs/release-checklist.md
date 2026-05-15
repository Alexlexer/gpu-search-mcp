# Release Checklist

Steps to follow before creating a GitHub release.

## 1. Run tests

```bash
pytest tests/ -v
```

All tests must pass. Thread-safety tests may be skipped in resource-constrained environments:

```bash
pytest tests/ -v --ignore=tests/test_thread_safety.py
```

## 2. Run linter

```bash
ruff check gpu_service/ tests/
```

No errors permitted.

## 3. Run smoke test

```bash
python scripts/smoke_test.py
```

Verifies that the server starts, indexes, and searches without crashing.
Run with `--with-semantic` locally if the model is already cached.

## 4. Run install test

```bash
pip install -e ".[test,ast]"
python -c "from gpu_service.mcp_server import cli_main; print('ok')"
```

## 5. Run optional benchmark

```bash
gpu-search-bench --directory /path/to/large/repo --output results.json
```

Compare against the reference numbers in README.md.

## 6. Verify README

- HTTP endpoint table matches the actual routes in `mcp_server.py`.
- Example JSON responses match the actual response shapes.
- Version number in the title matches `pyproject.toml` and `VERSION` in `mcp_server.py`.

## 7. Verify version consistency

Check that the version string is the same in all three places:

```bash
grep -n "version" pyproject.toml
grep -n "^VERSION" gpu_service/mcp_server.py
head -1 README.md
```

## 8. Create GitHub release

1. Tag the commit: `git tag v0.1.0`
2. Push the tag: `git push origin v0.1.0`
3. Create a GitHub release from the tag with the CHANGELOG entry as the body.

No automated PyPI publish is configured. Distribution is manual for now.
