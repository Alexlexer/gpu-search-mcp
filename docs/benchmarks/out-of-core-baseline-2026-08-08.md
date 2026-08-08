# Out-of-core benchmark baseline — 2026-08-08

This baseline validates the file-backed packed-corpus path on Windows with an NVIDIA CUDA device. Each query had one warm-up followed by five timed runs. Configuration: 2 MiB chunks, two reusable buffers, `FileStorageBackend`, and the default all-chunks candidate selector.

Two corpora were measured:

- **gpu-search-mcp:** 829,546 searchable corpus bytes, one chunk. The existing packed cache loaded in 417.0 ms.
- **Synthetic source corpus:** 64 Python-like files totaling 64 MiB, 32 chunks. A clean packed build completed in 1,027.7 ms.

The synthetic corpus repeats source-like lines containing every benchmark term. It is intended to exercise transport and verification throughput, not retrieval selectivity.

## gpu-search-mcp results

| Query | p50 latency | p95 latency | Candidate % | Physical bytes | Read ratio | Storage p50 | H2D p50 | GPU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `class` | 4.324 ms | 6.832 ms | 100% | 832,555 | 1.003627 | 0.388 ms | 0.226 ms | 1.937 ms |
| `function` | 4.295 ms | 6.766 ms | 100% | 831,263 | 1.002070 | 0.304 ms | 0.233 ms | 2.082 ms |
| `authentication` | 3.497 ms | 3.971 ms | 100% | 830,679 | 1.001366 | 0.255 ms | 0.219 ms | 1.231 ms |
| `TODO` | 2.451 ms | 2.638 ms | 100% | 829,631 | 1.000102 | 0.193 ms | 0.218 ms | 1.265 ms |

## 64 MiB results

| Query | p50 latency | p95 latency | Candidate % | Physical bytes | Read ratio | Storage p50 | H2D p50 | GPU p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `class` | 841.041 ms | 1,093.329 ms | 100% | 67,111,548 | 1.000097 | 23.919 ms | 19.900 ms | 101.731 ms |
| `function` | 858.963 ms | 1,044.524 ms | 100% | 67,111,641 | 1.000099 | 22.217 ms | 60.912 ms | 151.314 ms |
| `authentication` | 760.119 ms | 991.754 ms | 100% | 67,111,827 | 1.000101 | 19.460 ms | 60.519 ms | 164.332 ms |
| `TODO` | 954.698 ms | 1,024.618 ms | 100% | 67,111,517 | 1.000097 | 23.161 ms | 62.616 ms | 160.085 ms |

The 64 MiB corpus used 4.0 MiB of reusable buffer VRAM (20.0 MiB reserved by PyTorch), demonstrating that corpus size is no longer tied to VRAM allocation. Physical-read ratios slightly above 1.0 include query overlap and result-line reads.

## Legacy all-in-VRAM comparison

The pre-refactor implementation from commit `0914a67` was run against the same 64 MiB corpus, CUDA device, five timed runs, and `max_files=10`. This uses the actual former whole-corpus raw/lower GPU allocations and result mapper.

| Query | Legacy p50 | Out-of-core p50 | Ratio |
|---|---:|---:|---:|
| `class` | 575.554 ms | 841.041 ms | 1.46x |
| `function` | 615.424 ms | 858.963 ms | 1.40x |
| `authentication` | 595.222 ms | 760.119 ms | 1.28x |
| `TODO` | 691.895 ms | 954.698 ms | 1.38x |

The legacy clean build took 2,713.1 ms and allocated 127.99 MiB of corpus VRAM. The out-of-core build took 1,027.7 ms and used 4.0 MiB of reusable-buffer VRAM. The out-of-core path therefore traded 28–46% dense-query latency for a 32x reduction in measured VRAM and a 2.6x faster clean build on this workload. The next profiling step should focus on per-chunk synchronization and Python hit mapping; candidate pruning will reduce both work and physical reads for selective queries.

## Initial interpretation

- The default selector reads every chunk, so physical-read ratio remains approximately 1.0. Candidate filtering is the main route to storage reduction.
- Storage time is a small part of large-corpus latency on this NVMe-backed run.
- The gap between storage/H2D/GPU time and end-to-end latency is dominated by dense match collection and Python result mapping/line formatting in this intentionally match-heavy corpus.
- The fixed 4 MiB buffer allocation confirms the intended out-of-core memory behavior for a 64 MiB searchable corpus.

Run a new baseline with:

```powershell
gpu-search-bench --directory D:\path\to\repo --output result.json --iterations 5 --chunk-mib 2 --buffers 2 --storage file
```
