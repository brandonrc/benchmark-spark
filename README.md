# benchmark-spark — DGX Spark Container Memory Characterization

> **Status.** Phase 1 (Nov 2025) and Phase 2 (Apr 2026) produced a preliminary
> finding that Docker containerization reduces usable KV-cache capacity by
> ~50% on the NVIDIA GB10 Superchip. External peer review flagged that this
> result is **not distinguishable from a TensorRT-LLM free-memory probe
> artifact** under the methodology as published. This repository now targets
> a pre-registered **v2** replication and falsification study.
>
> **Read [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) before running anything.**

---

## The research question

On DGX Spark (GB10, 128 GB unified LPDDR5X):

> Does Docker containerization cause a **real reduction** in usable KV-cache
> capacity, or is the observed reduction an **artifact of TensorRT-LLM's
> free-memory probe** interacting with host-resident state under cgroups v2?

Phase 1 observed −50.6% KV cache and +35.9% peak memory in containers versus
chroot. The v2 test plan is designed so that the result can **falsify either
hypothesis**.

## How v2 answers it

v2 adds a single critical control — cells **E4N** and **E4C** — in which the
TensorRT-LLM KV cache is sized by an **absolute pinned block count** rather
than by its dynamic free-memory probe. If pinning closes the gap, the Phase 1
finding is a probe artifact. If pinning preserves the gap, the finding is a
real container overhead worth characterizing. The decision rule and threshold
(2 GiB on the lower bound of the paired bootstrap 95% CI) are pre-registered
in the test plan so no one can tune them after the fact.

## What's in the repo

```
benchmark-spark/
├── docs/
│   └── TEST_PLAN.md              ← v2 experimental design (READ FIRST)
├── REPRODUCING.md                ← step-by-step reproduction guide for a clean SPARK
├── configs/
│   ├── experiment.yaml           ← v2 experiment definition (cells, workloads, models)
│   └── versions.lock             ← pinned software stack (kernel, driver, Docker, NCT, image digest)
├── scripts/
│   ├── capture_environment.sh    ← writes environment.json, verifies against versions.lock
│   ├── snapshot_memory.py        ← lifecycle memory probe (meminfo, smaps, cgroup, nvsmi, docker stats)
│   ├── orchestrate.py            ← v2 orchestrator: random run plan, cooldown, dispatch
│   ├── extract_container_rootfs.sh  ← chroot extract (shared with Phase 1)
│   └── …(legacy Phase 1/2 scripts preserved unchanged for reference)
├── benchmarks/
│   ├── run_benchmark.py          ← v2 instrumented runner (emits one result JSON per run)
│   └── trtllm_benchmark.py       ← legacy Phase 1/2 runner (kept for archival reproduction)
├── analysis/
│   ├── statistics.py             ← v2 pre-registered analysis (bootstrap CIs, TOST, Holm, Cohen's d)
│   └── phase2_statistical_analysis.py   ← legacy Phase 2 analysis (kept for comparison)
├── results/
│   ├── phase1/                   ← original Phase 1 outputs (Nov 2025)
│   └── 20XX-XX-XX_clean_spark/   ← v2 batches produced by orchestrate.py
└── README.md                     ← this file
```

## Running v2

See [`REPRODUCING.md`](REPRODUCING.md) for the full clean-SPARK runbook. In brief:

```bash
# 1. Install pinned stack per configs/versions.lock
# 2. Pull the image, extract the rootfs, download the models
./scripts/extract_container_rootfs.sh
./scripts/capture_environment.sh results/$(date +%F)/environment.json --verify configs/versions.lock

# 3. Dry-run to see what will execute
./scripts/orchestrate.py --experiment configs/experiment.yaml --out /tmp/dry --dry-run --skip-env-capture

# 4. Run the batch (~37 h for full, ~6.5 h for the minimal 40-run subset)
./scripts/orchestrate.py --experiment configs/experiment.yaml --out results/$(date +%F)_clean_spark

# 5. Analyze
./analysis/statistics.py results/$(date +%F)_clean_spark
```

## What v2 fixes relative to Phase 1 / Phase 2

| Phase 1/2 issue | v2 fix |
|---|---|
| Single "Iteration 5" treated as cross-model validation | N=10 per (cell × model), pre-shuffled random order, warmup runs marked and excluded from analysis |
| No falsifying experiment for the KV-cache finding | **E4N/E4C cells with pinned KV blocks** — the result directly decides H0 vs H1 |
| `gpu_memory_utilization` / free-memory probe never mentioned | Exposed as a first-class knob; pinned-block mode bypasses it entirely |
| No process-level memory accounting | Lifecycle snapshots at 6 points: baseline → post-dataset → pre-build → post-build → pre-bench → post-bench. Each snapshot includes `/proc/meminfo`, `/proc/<pid>/smaps_rollup`, cgroup `memory.current` / `memory.peak`, `nvidia-smi memory.free`, per-PID GPU memory, and `docker stats` |
| Version drift: `[FILL IN]` placeholders in methodology | `configs/versions.lock` + `capture_environment.sh --verify` refuse to run against a drifted stack |
| No inferential statistics in results | Paired bootstrap 95% CIs, Cohen's d (paired), Welch's t, Holm-Bonferroni over the sweep, **TOST equivalence test** for the "negligible throughput" claim with a pre-registered ±5% margin |
| "50.6% reduction" reported with false precision (SD 1.2% was wrong) | MDES reported for achieved N; third-sig-fig numbers always accompanied by 95% CIs |
| Framework contradiction (methodology said TRT-LLM, acknowledgments said vLLM) | v2 is **TensorRT-LLM only**. vLLM cross-check is out of scope |
| Workload never stressed the KV cache it claimed to measure | W1 (500 × 2048-4096 input × 512 output) is explicitly a capacity stressor |
| Container overhead attributed to "namespaces + COW layers" without measurement | Lifecycle snapshots let the analysis attribute each GiB to a specific phase and memory region |
| Run order interleaved but not logged or randomized | `run_plan.json` is an immutable pre-shuffled permutation with a fixed seed, written before the batch starts, with ordinal positions recorded per run |

## What remains out of scope

- vLLM (the SPARK image ships TensorRT-LLM; cross-framework is a future paper)
- systemd-nspawn or rootless container runtimes
- Multi-GPU or multi-node
- Fine-tuning workloads
- Comparison to discrete-GPU systems (H100, A100)

## Legacy Phase 1 / Phase 2

The original Phase 1 README is preserved below the horizontal rule (not in
this file — see `PHASE2_README.md` and `PHASE2_SUMMARY.md`). Phase 1's
interactive results site at `docs/index.html` also remains. Nothing from
Phase 1 has been deleted; v2 is additive so an auditor can compare the old
and new artifacts side by side.

## Citing this repo

```bibtex
@misc{benchmark-spark-v2,
  title  = {benchmark-spark v2: Pre-registered replication and falsification
            study of Docker containerization KV-cache effects on NVIDIA DGX Spark},
  author = {Geraci, Brandon},
  year   = {2026},
  url    = {https://github.com/brandonrc/benchmark-spark},
  note   = {v2 test plan in docs/TEST_PLAN.md; analysis is pre-registered}
}
```

## License

MIT — see LICENSE.
