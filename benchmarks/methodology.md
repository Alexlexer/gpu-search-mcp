# Benchmark Methodology

## How to reproduce

```bash
# 1. Install gpu-search-mcp with a GPU-capable PyTorch build
python install.py

# 2. Choose a large repository to benchmark against
git clone https://github.com/microsoft/vscode /tmp/vscode

# 3. Run the benchmark
.venv/bin/python benchmarks/run_benchmark.py --repo /tmp/vscode --runs 10
```

## What is measured

### gpu-search
- **Setup:** `index_directory()` is called once. All file bytes are loaded into VRAM.
- **Measurement:** `search(query)` is called `--runs` times. Includes the GPU kernel dispatch, synchronization, and CPU decode of matching lines.
- **Timing:** `time.perf_counter()` wall-clock time around each `search()` call.

### ripgrep warm
- **Setup:** The OS file cache is assumed warm (all files already in RAM) from prior reads.
- **Measurement:** `rg --count-matches -r "" <query> <repo>` is called `--runs` times.
- **Timing:** Same wall-clock approach.

### ripgrep cold (optional)
- Attempts `echo 3 > /proc/sys/vm/drop_caches` before each run (requires root on Linux).
- Not feasible in most CI environments; document manually if you run it.

## Metrics reported

| Metric | Description |
|--------|-------------|
| median | 50th percentile of run times |
| p95 | 95th percentile of run times |

Median is the best single-number summary for latency. P95 shows tail latency.

## Known biases

1. **GPU warmup:** The first 1–2 runs may be slower due to CUDA kernel compilation. These are included in the statistics to represent realistic cold-start behavior within a session.
2. **OS scheduler noise:** On laptops or shared machines, variance can be high. Run on a quiet machine for stable results.
3. **ripgrep parallelism:** `rg` uses multiple threads. The comparison is not perfectly apples-to-apples; it reflects real-world conditions where both tools compete for CPU.
4. **Corpus size:** gpu-search pre-loads all files into VRAM at startup. For very large repos (> VRAM capacity), this will fail or spill to system RAM.

## Example results (2026-05, RTX 4060)

These numbers were collected on the VS Code repo (12,259 files, 285 MB) on Windows 11 with an RTX 4060 (8 GB VRAM).

> **These are example results, not guaranteed performance.** Your results will vary by hardware, OS, repo size, and system load.

| Query | Matches | gpu-search median | rg warm median |
|-------|---------|-------------------|----------------|
| `ICodeEditor` | 428 files | **10ms** | ~110ms |
| `createTextModel` | 95 files | **8ms** | ~135ms |
| `disposeOnReturn` | 3 files | **7ms** | ~200ms |
| `handleError` | 14 files | **8ms** | ~120ms |
| `addEventListener` | 109 files | **10ms** | ~115ms |

Move these numbers into your own README only after re-running the benchmark yourself on comparable hardware.
