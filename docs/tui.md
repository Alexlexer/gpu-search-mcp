# TUI dashboard

`gpu-search-tui` is a read-only live view over the local API. It answers three
questions without a profiler: what is loaded, what is it costing, and is anything
actually using it.

```bash
gpu-search-tui --directory /path/to/repo     # start an embedded server and watch it
gpu-search-tui --url http://127.0.0.1:8765   # attach to `gpu-search-mcp --http`
gpu-search-tui --once                        # one snapshot, then exit
gpu-search-tui --json                        # the same snapshot as JSON
```

The live view needs the optional `tui` extra (`pip install "gpu-search-mcp[tui]"`,
which installs `rich`). `--once` and `--json` use only the standard library.

CTRL+C stops the view. In `--directory` mode the embedded server is shut down with it.

## Panels

| Panel | Data |
|---|---|
| Overview | indexed files, chunks, corpus size, ready features |
| Indexes | cache status (`loaded`/`rebuilt`), candidate selector, trigram index size, semantic/dependency/symbol counts, index root |
| Last exact query | latency, candidate chunks vs total, bytes read, storage/kernel/host-to-gpu split, pipeline mode |
| Resources | RSS and peak RSS, VRAM, host staging/device bytes, pool leases, buffer size and budget, watcher queue |
| Usage (HTTP API) | requests, errors, searches by engine, p50/p95/max latency, top endpoints, latency sparkline, uptime |

Everything comes from `GET /stats` and `GET /runtime`, so the dashboard cannot
diverge from the API contract. `/stats` and `/runtime` are excluded from the usage
counters (the dashboard polls them), which keeps "requests" equal to real client
traffic. MCP tool calls over stdio are not HTTP requests; use the *Last exact query*
panel for those, because every entry point shares the same index instance.

## Notes

- A corpus smaller than the 2 MiB chunk size is a single chunk, so *candidates* shows
  `1/1 (100%)` and the *corpus read* percentage is 100% by construction.
- On Windows consoles that cannot encode the sparkline blocks or the arrow, the
  dashboard falls back to ASCII decorations instead of failing.
- `--json` output includes the full `/stats` and `/runtime` payloads plus the resolved
  URL and fetch timestamp.
