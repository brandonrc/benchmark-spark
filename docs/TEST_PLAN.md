# Test Plan v2 — DGX Spark Container Memory Characterization

**Status:** Design spec for reproducible re-run on a clean DGX Spark build.
**Supersedes:** Phase 1 (N=1 "Iteration 5") and the informal Phase 2 comprehensive runner.
**Authored in response to:** External peer review of `dgx-spark-paper-1` draft (April 2026).

---

## 1. Research Question

> **On NVIDIA DGX Spark (GB10, 128 GB unified LPDDR5X), does Docker containerization cause a real reduction in usable KV-cache capacity, or is the observed reduction an artifact of TensorRT-LLM's free-memory probe interacting with host-resident state under cgroups v2?**

Phase 1 observed a 50.6% KV-cache reduction and a 35.9% peak-memory increase inside a container versus chroot. Peer review flagged that this pattern is consistent with a **framework probe artifact**: TensorRT-LLM (like vLLM) sizes the paged KV cache from a run-time free-memory query, so anything that reduces observed free memory at probe time (host page cache, cgroup accounting, shm reservations) mechanically shrinks the cache — without any real resource being "consumed" by containerization.

The Phase 2 plan below is designed so its **result can falsify either hypothesis**.

## 2. Primary Hypothesis and Falsification Rule

- **H0 (null, "probe artifact"):** If we override TensorRT-LLM's dynamic sizing and pin an absolute KV-cache block count, the container-vs-native gap in KV-cache bytes **collapses to within measurement noise (< 2 GiB mean difference)**.
- **H1 (alternative, "real overhead"):** If we pin the KV-cache block count and the container still fails to allocate, OOMs, or lands with materially less available block memory, then containerization is genuinely consuming memory the native run can use.

The decision is made by **experiment E4** below. H0 versus H1 must be declared **before** results are unblinded.

## 3. Secondary Questions

1. Which Docker configuration knob closes the gap? (`--shm-size`, `--ipc=host`, `--cgroup-parent=/`, `--memory=unlimited`, `--memory-swap=-1`, `--privileged`?)
2. Does compute throughput actually match between environments? (Phase 1 claim — needs an equivalence test, not a null-result handwave.)
3. Does the effect scale with model size? (7B vs 72B.)
4. Where in the process lifecycle does free memory diverge between the two environments? (Probe-time snapshot instrumentation.)

## 4. Independent Variables (what we vary)

| Variable | Levels |
|---|---|
| **Environment** | `native` (chroot from extracted rootfs), `container` (Docker + NCT) |
| **Docker config** | `E0-baseline`, `E1-shm32g`, `E2-memunlim`, `E3-cgroup-root`, `E4-pinned-kv`, `E5-privileged`, `E6-all` |
| **Model** | `DeepSeek-R1-Distill-Qwen-7B` (bf16), `Qwen2.5-72B-Instruct` (AWQ-4bit) |
| **KV-cache sizing mode** | `dynamic` (default, `free_gpu_memory_fraction=0.90`), `pinned` (absolute block count via `--max_num_tokens` / engine build override) |

## 5. Controlled Variables (what we pin)

All must be captured in `environment.json` by `scripts/capture_environment.sh` **before every benchmark batch**. A batch is aborted if any of these change mid-run.

- Kernel version, cgroup version, hugepage config, transparent_hugepage setting, NUMA topology
- NVIDIA driver version, CUDA runtime version, `nvidia-container-toolkit` version and CDI config
- Container image **digest** (not tag), extracted rootfs **sha256**
- Model weights: HuggingFace commit SHA, local path, file count, total bytes
- TensorRT-LLM version (git SHA inside the container), TensorRT version, cuDNN version
- `trtllm-bench` CLI args verbatim
- `ulimit -a`, `sysctl -a | grep -E 'vm\\.|kernel\\.shm'`
- GPU persistence mode, GPU clocks (locked via `nvidia-smi --lock-gpu-clocks`)
- Ambient temperature proxy: pre-run GPU temperature must be ≤ 45 °C

## 6. Dependent Variables (what we measure)

All captured per run into a single `run_<id>.json`.

**Memory — primary:**
- KV-cache bytes allocated (parsed from TensorRT-LLM log: "Memory used for [paged] KV cache")
- KV-cache blocks allocated
- Peak process RSS (`smaps_rollup`)
- Peak `nvidia-smi memory.used` for the inference PID
- cgroup `memory.current` / `memory.peak` at engine-build-end and at benchmark-end
- System `/proc/meminfo` MemAvailable at: baseline, pre-import, post-import, pre-engine-build, post-engine-build, pre-bench, post-bench (7 snapshots per run)

**Performance — secondary (for the equivalence test):**
- End-to-end throughput (tokens/sec, from `trtllm-bench`)
- Request-level latency P50 / P95 / P99
- Time-to-first-token (TTFT), inter-token latency (ITL) if available
- GPU utilization (avg / P95) from `nvidia-smi dmon` at 1 s cadence
- GPU SM clock and power (for thermal-throttle detection)

**Diagnostic:**
- Log of the exact `nvidia-smi --query-gpu=memory.free` value at the moment TensorRT-LLM probes
- Stdout of `docker stats --no-stream` at benchmark mid-point (container runs only)

## 7. Experiment Matrix

N = **10 replications per cell**, minimum. This is the smallest N at which paired bootstrap CIs on the headline KV-cache delta can have a half-width below 2 GiB under Phase 1's observed variance.

| ID | Environment | Config | KV sizing | Purpose |
|---|---|---|---|---|
| **E0N** | native | chroot, no extra flags | dynamic | Baseline control |
| **E0C** | container | default: `--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864` | dynamic | Reproduce Phase 1 finding |
| **E1** | container | E0C + `--shm-size=32g` | dynamic | Tests shm hypothesis |
| **E2** | container | E0C + `--memory=128g --memory-swap=-1` | dynamic | Tests cgroup memory.max hypothesis |
| **E3** | container | E0C + `--cgroup-parent=/` | dynamic | Tests cgroup hierarchy hypothesis |
| **E4N** | native | chroot | **pinned** (`max_num_tokens` override) | Falsifier control |
| **E4C** | container | E0C | **pinned** (same block count as E4N) | **Primary falsifying experiment** |
| **E5** | container | `--privileged` (all caps, all cgroups relaxed) | dynamic | Upper bound: if this closes gap, cgroups are the cause |
| **E6** | container | `E1 + E2 + E3` combined | dynamic | Best-case container tuning |

**Total runs: 9 cells × 2 models × 10 reps = 180 runs.**
At ~10 min/run (including cooldown), total wall time ≈ **30 hours**. Break into 3 overnight batches.

## 8. Run Order Discipline

Run order matters because of thermal drift, cache warming, and host memory fragmentation. Phase 1 alternated but did not randomize.

- Before the batch starts, `orchestrate.py` generates a uniformly random permutation of all `(cell × rep)` tuples and writes `run_plan.json` **once**. This file is immutable and version-controlled with the results.
- A fixed PRNG seed (captured in `environment.json`) makes the permutation reproducible.
- Every run logs its ordinal position in the permutation. Order is a covariate in analysis.
- Warm-up: the **first 2 runs of every batch are discarded** as warm-up and marked `warmup=true` in metadata. They still execute so the page cache / thermal state reaches steady state.
- Cooldown: ≥ 5 min AND GPU temp ≤ 45 °C AND a forced `sync && echo 3 > /proc/sys/vm/drop_caches` between runs. The drop-caches resets page-cache state so runs are independent.

## 9. Workloads

Phase 1 used 1000 requests × 128 output tokens × 512 input tokens with a 5-prompt rotation. This under-stresses the KV cache and is unrealistically repetitive.

**W1 — Long-context capacity stressor** (primary workload for the KV-cache claim):
- Input length: uniform 2048-4096 tokens
- Output length: 512 tokens
- 500 requests, 32 concurrent
- Prompt source: ShareGPT-v3 truncated+padded to the length window (or if unavailable, deterministic Lorem-ipsum padding around a single seed prompt for full reproducibility)

**W2 — Short-request throughput** (for the throughput equivalence test):
- Input length: uniform 128-256 tokens
- Output length: 128 tokens
- 2000 requests, 64 concurrent

**W3 — Mixed** (sanity check only, not part of primary analysis):
- Length distribution from the MLPerf Inference LLM reference (2025 v5.1)

Only **W1** data feeds the primary KV-cache hypothesis test. **W2** data feeds the throughput equivalence test. W3 is qualitative.

## 10. Statistical Analysis Plan (pre-registered)

Analysis code: `analysis/statistics.py`. All tests and thresholds below are fixed before any result is inspected.

**Primary test — H0 vs H1 falsifier (E4C vs E4N):**
- Paired difference in KV-cache bytes (container − native).
- Paired bootstrap 95% CI with 10,000 resamples.
- **Decision rule:** reject H0 (probe artifact) if the bootstrap 95% CI lower bound exceeds **+2 GiB**. Otherwise the finding is declared consistent with a probe artifact.

**Secondary — baseline replication (E0C vs E0N):**
- Same paired bootstrap CI.
- Report Cohen's d on the paired differences.
- Report Welch's t p-value as a companion, not a primary statistic.

**Throughput equivalence test (W2 data, E0C vs E0N):**
- **Two one-sided tests (TOST)** with pre-registered equivalence margin **±5%** of the native mean.
- Report tokens/sec means, 90% CI on the difference, TOST verdict.
- "Negligible" is reported **only** if TOST passes. Otherwise report "undetermined at N=10" or "container is slower/faster by X%."

**Power disclosure:**
- Report minimum detectable effect size (MDES) at α=0.05, power=0.80 for each N achieved.
- N=10 with the Phase 1 observed SD of ~1.4 GiB on KV-cache means MDES ≈ ±1.8 GiB. Any smaller difference cannot be distinguished from zero.

**Multiple comparisons:**
- The 7 pairwise container configs (E1-E6 vs E0N) are reported with Holm-Bonferroni corrected p-values.
- The primary H0/H1 decision is **not** corrected — it is pre-registered and singular.

**Exclusion rules:**
- A run is excluded only if: (a) it crashed, (b) GPU temp exceeded 55 °C at probe time, or (c) the environment hash mismatches the batch's `environment.json`. All exclusions logged in `exclusions.json` with reason; exclusions are never applied post-hoc based on values.

## 11. Outputs

Every run produces exactly one file: `results/<batch_id>/<run_id>.json` with this schema (validated by `analysis/schema.py`):

```json
{
  "run_id": "e0c_deepseek7b_rep03",
  "batch_id": "2026-04-12_clean_spark",
  "cell": "E0C",
  "model": "DeepSeek-R1-Distill-Qwen-7B",
  "rep": 3,
  "warmup": false,
  "plan_ordinal": 47,
  "started_at": "2026-04-12T03:14:22.881Z",
  "ended_at":   "2026-04-12T03:24:05.112Z",
  "environment_hash": "sha256:...",
  "config": { ... Docker / chroot flags verbatim ... },
  "lifecycle_snapshots": [
    {"phase": "baseline",         "meminfo": {...}, "nvidia_smi": {...}, "cgroup": {...}},
    {"phase": "post_import",      ...},
    {"phase": "pre_engine_build", ...},
    {"phase": "post_engine_build",...},
    {"phase": "pre_bench",        ...},
    {"phase": "post_bench",       ...}
  ],
  "metrics": {
    "kv_cache_bytes": 47360000000,
    "kv_cache_blocks": 1445,
    "peak_rss_bytes":  75000000000,
    "peak_nvsmi_used_bytes": 95500000000,
    "throughput_tokens_per_sec": 119.83,
    "latency_ms":   {"p50": ..., "p95": ..., "p99": ...},
    "gpu_util_avg_pct": 39.7,
    "sm_clock_mhz_avg": 1950
  },
  "raw_logs": { "trtllm_bench_stdout_path": "...", "nvsmi_dmon_csv_path": "..." }
}
```

`batch_id` is a timestamp + short description. Every file under a batch directory is immutable.

## 12. Reproducibility Contract

A reviewer with a clean DGX Spark must be able to execute:

```bash
git clone https://github.com/brandonrc/benchmark-spark
cd benchmark-spark
git checkout <sha-in-paper>
./scripts/bootstrap_clean_spark.sh       # installs pinned Docker / NCT, downloads pinned image, extracts rootfs
./scripts/capture_environment.sh         # writes environment.json, aborts on mismatch with configs/versions.lock
./scripts/download_models.sh             # fetches pinned HF revisions into /data/models
./scripts/orchestrate.py --plan configs/experiment.yaml --out results/$(date +%F)_reproduce
./analysis/statistics.py results/<batch>
```

...and get a `results_table.md` whose numbers match the paper within the reported 95% CIs. If they don't, the paper is wrong, not the reviewer.

## 13. What is explicitly out of scope

- vLLM (the SPARK image ships TensorRT-LLM; cross-framework is left for a future paper).
- systemd-nspawn (mentioned in Phase 1 future work; defer).
- Multi-GPU or multi-node (GB10 is single-GPU).
- Fine-tuning workloads, RAG pipelines, streaming inference.
- Any comparison to discrete-GPU systems (H100, A100) — the claim is specific to GB10.

## 14. Deliverables

- `results/<batch_id>/` — all raw JSON runs + raw logs
- `results/<batch_id>/environment.json` — immutable env capture
- `results/<batch_id>/run_plan.json` — the pre-randomized order
- `results/<batch_id>/statistical_analysis/results_table.md` — paper-ready table
- `results/<batch_id>/statistical_analysis/figures/` — CI plots, not bar charts
- `results/<batch_id>/exclusions.json` — any excluded runs with reason
- `docs/VERIFICATION_CHECKLIST.md` — what a reviewer should eyeball before trusting the result
