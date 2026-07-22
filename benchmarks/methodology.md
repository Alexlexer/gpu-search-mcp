# Benchmark Methodology

The benchmark suite separates raw exact-search speed from retrieval quality,
indexing cost, and agent-context size. Results are machine-specific evidence,
not universal performance claims.

## Pattern latency benchmark

Run the existing direct-call benchmark against any checkout:

```bash
gpu-search-bench \
  --directory /path/to/repository \
  --queries benchmarks/queries.json \
  --iterations 20 \
  --device cpu \
  --output pattern-results.json
```

This records repository size, pattern-index build time, device information,
p50/p95/p99 direct Python latency, and a warm ripgrep timing when `rg` is
installed. It does not force the operating system to evict disk caches.

## Retrieval-quality manifests

Version 1 manifests are JSON or safe YAML documents:

```yaml
schema_version: 1
repository: sample-dotnet-app
language: csharp
modes: [exact, symbol, hybrid_dependencies]
queries:
  - id: jwt-validation
    query: Where is JWT expiration validated?
    exact_query: ValidateExpiration
    symbol_query: ValidateExpiration
    expected_files:
      - src/Auth/JwtValidator.cs
      - src/Auth/TokenService.cs
    expected_symbols:
      - Sample.Auth.JwtValidator.ValidateExpiration
    expected_tests:
      - tests/Auth/JwtValidatorTests.cs
```

Required fields and unique query IDs are validated before indexing. Paths use
repository-relative forward slashes and are compared case-insensitively so the
same manifest can run on Windows, Linux, and macOS.

Four checked-in fixtures establish the format for C#, TypeScript, Python, and
mixed repositories:

- `benchmarks/manifests/csharp.json`
- `benchmarks/manifests/typescript.json`
- `benchmarks/manifests/python.json`
- `benchmarks/manifests/mixed.json`

Run a CPU-only quality benchmark:

```bash
gpu-search-bench \
  --directory benchmarks/fixtures/csharp \
  --manifest benchmarks/manifests/csharp.json \
  --modes exact,symbol,hybrid_dependencies \
  --iterations 3 \
  --device cpu \
  --output csharp-quality.json
```

Semantic embeddings are never downloaded implicitly. Add `--build-semantic`
only after the configured model is available locally or after an explicit
model download.

## Retrieval modes

- `ripgrep`: fixed-string file baseline, if `rg` is installed.
- `exact`: device-selected `GpuFileIndex` search using `exact_query`.
- `symbol`: symbol graph lookup using `symbol_query`.
- `semantic`: semantic retrieval using the natural-language query.
- `hybrid`: exact plus semantic retrieval.
- `hybrid_symbols`: symbol results first, then unique hybrid results.
- `hybrid_dependencies`: hybrid retrieval with dependency and test expansion.

Run exact mode once with `--device cpu` and once with the relevant device to
compare CPU and accelerator behavior. Do not combine those reports as if they
were collected in one process.

## Quality metrics

Metrics are calculated independently for every query and then macro-averaged:

- Recall@1, Recall@5, and Recall@10.
- Precision@5, with a fixed denominator of five.
- Mean reciprocal rank of the first expected file.
- Exact-symbol recall when `expected_symbols` is present.
- Related-test recall when `expected_tests` is present.

The report also records p50/p95 search latency, mean/max returned token
estimates, index build time, indexing throughput, incremental update latency,
cache size, VRAM reported by the pattern index, and peak RSS where the platform
standard library exposes it. On Windows, peak RSS is reported as unavailable
rather than approximated.

## Baselines and regression gates

First record and review a baseline without thresholds:

```bash
gpu-search-bench ... --write-baseline baseline.json --output current.json
```

The baseline contains deterministic quality metrics and returned-token counts
only. Machine, device, cache, and latency fields remain in the full current
report and are deliberately excluded from the portable baseline.

A later run may opt into explicit gates:

```bash
gpu-search-bench ... \
  --baseline baseline.json \
  --max-quality-drop 0.02 \
  --max-latency-increase-pct 20 \
  --max-token-increase-pct 10 \
  --max-returned-tokens 1024 \
  --output current.json
```

No threshold is implicit. A baseline comparison without threshold flags is
informational and cannot fail the command. With thresholds, any reported
regression returns exit code 1.

The checked-in CPU policy runs all four fixtures with zero permitted quality
drop, at most 10% returned-token growth, and a hard 1,024-token ceiling:

~~~bash
python scripts/quality_gate.py
~~~

Use python scripts/quality_gate.py --update-baselines only after reviewing an
intentional retrieval change. Latency remains informational until
runner-specific baselines are approved.

## Reproducibility checklist

Record the commit, manifest, repository revision, operating system, Python and
PyTorch versions, selected device, semantic model identifier, warm-up policy,
iteration count, and whether ripgrep was available. Compare reports only when
those inputs are compatible.
