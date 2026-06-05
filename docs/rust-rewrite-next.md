# Rust rewrite — next concrete steps

This file lists the immediate, concrete tasks to finish Rust core parity with the
existing Python runtime. Treat these as short, testable milestones so we can
remove or reduce Python usage safely.

1) Validate Rust core builds and tests
   - Run: `cargo test --manifest-path crates/gpu-search-core/Cargo.toml`
   - Fix any failing unit tests in crates/gpu-search-core (file_discovery, pattern,
	 line_index, cache modules already present). Aim for `cargo test` green.

2) Add small integration parity tests
   - Create an integration test that: discovers files in a tiny temp repo,
	 runs `search_files` for a token, and asserts the Rust result shape.
   - Place integration tests under `crates/gpu-search-core/tests/`.

3) Implement missing deterministic subsystems (if any)
   - File discovery: ensure `discover_files` respects INDEXED_EXTS, SKIP_DIRS,
	 allow_env_files, and max file size. (file_discovery.rs exists)
   - Pattern search: ensure behavior matches Python compact/normal/full modes
	 (pattern.rs implemented; review snippet/line mapping).
   - Line offsets: ensure LineIndex behavior matches Python newline handling.
   - Cache meta: ensure write/read shape compatibility with `.gpu-search-cache`.

4) Add a minimal Rust HTTP shim for pattern search
   - Add a small HTTP handler in `crates/gpu-search-http` that exposes
	 `POST /search/code` and routes to the Rust `pattern::search_files` path.
   - Keep DTO shapes compatible with current Python HTTP responses.

5) Define and implement the Python semantic sidecar contract
   - Sidecar endpoints: `/health`, `/status`, `/embed`, `/index`, `/merge`, `/shutdown`.
   - Sidecar connection info should be written to `.gpu-search-cache/sidecar.json`.
   - Rust should treat semantic as "sidecar available" when the file exists and is reachable.

6) Add parity integration tests comparing Python vs Rust outputs
   - A simple test runner (Python or Rust) that starts Python MCP (or its test harness)
	 and the Rust HTTP shim, runs a set of search/read/deps requests against both,
	 and asserts behavioral parity on compact result shapes.

7) Add CI jobs and benchmarks
   - Add a GitHub Actions job to run `cargo test` for the Rust crates.
   - Add a benchmark script (`crates/gpu-search-core/examples/pattern_benchmark.rs`) and
	 a CI benchmark action (optional, run nightly or on demand).

8) Measure and gate Python removal
   - Define parity gates: unit tests green, integration parity tests pass, and
	 benchmarks meet expected thresholds vs Python for pattern search.
   - When gates pass, remove non-essential Python modules and keep only the
	 semantic sidecar (or migrate semantics to a Rust sidecar later).

Quick commands and notes
 - Run all Rust tests: `cargo test --workspace`
 - Run a single crate tests: `cargo test --manifest-path crates/gpu-search-core/Cargo.toml`
 - Build http crate locally: `cargo build --manifest-path crates/gpu-search-http/Cargo.toml`
 - Run Python smoke tests (before removing Python): `python scripts/smoke_test.py`

Where to start now
 - If you want me to implement one concrete item next, tell me which: run Rust tests,
   add an integration parity test, scaffold the HTTP shim, or implement the sidecar JSON
   contract in code. I can apply the change in the repository.
