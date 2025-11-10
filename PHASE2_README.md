# Phase 2: IEEE-Quality Docker Benchmarking on DGX Spark

> **Status:** Ready for execution ✅
> **Phase 1:** [Complete - Tagged as v1.0-phase1-complete](https://github.com/brandonrc/benchmark-spark/releases/tag/v1.0-phase1-complete)

---

## Quick Links

- **📖 Full Methodology:** [docs/PHASE2_METHODOLOGY.md](docs/PHASE2_METHODOLOGY.md)
- **🚀 Quick Start Guide:** [docs/PHASE2_QUICKSTART.md](docs/PHASE2_QUICKSTART.md)
- **⚙️ Model Configuration:** [configs/phase2_models.json](configs/phase2_models.json)
- **📊 Phase 1 Results:** [https://brandonrc.github.io/benchmark-spark/phase1/](https://brandonrc.github.io/benchmark-spark/phase1/)

---

## What Changed from Phase 1?

### Experimental Rigor

| Aspect | Phase 1 | Phase 2 | Improvement |
|--------|---------|---------|-------------|
| **Sample Size** | N=10 | **N=30** | +200% (adequate statistical power) |
| **Requests/Run** | 100 | **1000** | +900% (NVIDIA best practice) |
| **Models** | 3 | **10** | +233% (size/architecture diversity) |
| **Statistical Tests** | None | **t-tests, CIs, power** | IEEE publication standard |
| **Docker Config** | Custom | **NVIDIA official** | Reproducible, validated |

### Configuration Source

**Phase 1:** Custom configuration (`--shm-size=60g`) without documented rationale

**Phase 2:** Based on official [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)

**Baseline Docker command:**
```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  <command>
```

**Rationale:** Every flag documented in [PHASE2_METHODOLOGY.md](docs/PHASE2_METHODOLOGY.md#configuration-sources-and-validation)

---

## Research Questions

### Primary Question (Phase 1 carryover)

**Does Docker containerization introduce significant overhead on DGX Spark (Grace Hopper unified memory architecture)?**

**Phase 1 Answer:** Yes - 20-30 GB memory overhead, 1.6-2.7x less KV cache, but identical throughput

### New Questions for Phase 2

1. **Is the overhead consistent across model sizes?** (1.5B to 72B parameters)
2. **Do different architectures show different overhead patterns?** (Decoder-only vs MoE vs Efficient)
3. **Can Docker configuration optimizations reduce overhead?** (shm-size, CPU pinning, etc.)
4. **Is N=30 statistically sufficient?** (Power analysis validation)
5. **Are findings IEEE-publishable quality?** (Proper inferential statistics, effect sizes)

---

## 10-Model Test Suite

### Tier 1: Essential (IEEE Paper Core)

| Model | Size | Type | Why Include |
|-------|------|------|-------------|
| **Llama-3-8B** | 8B | Decoder | Meta's industry standard - most cited |
| **Llama-3-70B** | 70B (AWQ-4bit) | Decoder | Large model under memory pressure |
| **Mistral-7B-v0.3** | 7B | Decoder | Popular open-source baseline |
| **DeepSeek-R1-7B** | 7B | Reasoning | Phase 1 continuity, reasoning capability |
| **Qwen2.5-72B** | 72B | Decoder | Phase 1 continuity, long context (131K) |

### Tier 2: Important (Architectural Diversity)

| Model | Size | Type | Why Include |
|-------|------|------|-------------|
| **Mixtral-8x7B** | 45B (8x7B MoE) | MoE | Different memory access patterns |
| **Phi-3-mini** | 3.8B | Efficient | Microsoft's optimized architecture |
| **Gemma-2B** | 2B | Decoder | Google's efficient small model |

### Tier 3: Optional (Coverage)

| Model | Size | Type | Why Include |
|-------|------|------|-------------|
| **Qwen2-1.5B** | 1.5B | Decoder | Smallest model - minimal overhead test |
| **GPT-OSS-120B** | 120B (MoE) | MoE (MXFP4) | Phase 1 continuity (replace if unclear) |

**Total:** 10 models × 30 runs × 3 environments = **900 benchmark runs**

---

## Testing Phases

### Phase 2A: Baseline (Native vs Docker)

**Goal:** Establish performance gap with IEEE-quality rigor

**Configuration:**
- 10 models
- 30 runs per model per environment
- Native (chroot) vs Docker baseline (NVIDIA config)
- 1000 requests per run

**Estimated time:** ~80 hours

**Command:**
```bash
export TEST_PHASE="baseline"
export MODELS_TO_TEST="all"
export RUNS_PER_MODEL=30

./scripts/run_phase2_comprehensive.sh
```

### Phase 2B: Optimization (Docker Tuning)

**Goal:** Systematically reduce Docker overhead

**Optimizations to test:**
1. `--shm-size=8g` / `--shm-size=32g` / `--shm-size=60g`
2. `--network=host`
3. `--cpuset-cpus=0-9` (pin to performance cores)
4. `--privileged` (⚠️ security risk)
5. Combined "best" configuration

**Strategy:**
- Test individually on 3 representative models (small, medium, large)
- 10 runs per configuration for exploration
- Identify best 2-3 optimizations
- Combine and test "best" config on full suite (30 runs)

**Estimated time:** ~20 hours

**Command:**
```bash
export OPTIMIZATION_MODE="all"  # Test all configs
export MODELS_TO_TEST="Llama-3-8B,Mistral-7B-v0.3,Llama-3-70B"
export RUNS_PER_MODEL=10

./scripts/run_container_benchmark_optimized.sh trtllm
```

### Phase 2C: Statistical Analysis

**Goal:** Generate IEEE-quality statistical analysis

**Outputs:**
- Descriptive statistics (mean, std, CV, quartiles)
- Independent t-tests (native vs container)
- 95% confidence intervals
- Cohen's d effect sizes
- Statistical power calculation
- Bonferroni-corrected p-values

**Command:**
```bash
python analysis/phase2_statistical_analysis.py \
  --results-dir results/phase2
```

---

## Key Improvements for IEEE Publication

### 1. Documented Configuration Source

**Phase 1 Gap:** Custom `--shm-size=60g` without justification

**Phase 2 Fix:** All Docker flags sourced from NVIDIA official playbooks with rationale documented

### 2. Statistical Rigor

**Phase 1 Gap:** N=10, no inferential tests, no confidence intervals

**Phase 2 Fix:**
- N=30 (adequate statistical power)
- Independent t-tests with assumptions checked
- 95% confidence intervals
- Effect sizes (Cohen's d)
- Power analysis validation

### 3. Model Selection Justification

**Phase 1 Gap:** 3 models, unclear selection criteria

**Phase 2 Fix:**
- 10 models with explicit selection criteria
- Industry standards included (Llama, Mistral)
- Architecture diversity (Decoder, MoE, Efficient)
- Size range (1.5B to 72B)

### 4. Reproducibility

**Phase 1 Gap:** Some manual steps, incomplete documentation

**Phase 2 Fix:**
- Fully automated scripts (no manual intervention)
- All configurations in version control
- Complete audit trail (logs, metadata, monitoring)
- Open-source repository with MIT license

### 5. Optimization Path

**Phase 1 Gap:** No optimization attempted

**Phase 2 Contribution:**
- Systematic testing of 7 Docker optimization strategies
- Practical deployment recommendations
- "Best practices" for DGX Spark containerization

---

## Expected Contributions

### For Researchers

1. **First systematic benchmarking of DGX Spark (Grace Hopper GB10)**
   - No prior published work on this hardware
   - Novel unified memory architecture findings

2. **Containerization overhead on unified memory systems**
   - Prior work focuses on discrete GPUs (PCIe/NVLink)
   - Grace Hopper has no CPU-GPU transfer bottleneck
   - Different overhead sources and patterns

3. **Statistical rigor and reproducibility**
   - N=30, proper power analysis
   - All data and code open-sourced
   - Fully reproducible methodology

### For Practitioners

1. **Docker deployment best practices for DGX Spark**
   - Validated configuration from NVIDIA playbooks
   - Optimization strategies tested and ranked
   - Performance vs security trade-offs documented

2. **Model-specific deployment recommendations**
   - Which models fit in 119GB unified memory?
   - When is quantization necessary?
   - Batch size and concurrency tuning

3. **Cost-benefit analysis**
   - Docker overhead quantified (memory, throughput, latency)
   - When is native/chroot worth the complexity?
   - Production deployment decision framework

---

## Target Publication Venues

### Tier 1 Conferences (Target)

1. **ACM HPDC** - *ACM International Symposium on High-Performance Parallel and Distributed Computing*
   - Fit: Excellent (performance evaluation, GPU systems, containerization)
   - Deadline: ~January (for June conference)

2. **IEEE CCGrid** - *IEEE Conference on Cluster, Cloud and Grid Computing*
   - Fit: Excellent (cloud computing, container performance)
   - Deadline: ~November (for May conference)

3. **ACM ASPLOS** - *Architectural Support for Programming Languages and Operating Systems*
   - Fit: Good (systems architecture, hardware-software interaction)
   - Highly competitive (stretch goal)

### High-Impact Journals (Alternative)

4. **Future Generation Computer Systems** (Impact Factor: ~7.0)
   - Fit: Excellent (systems research, performance evaluation)
   - Advantage: No strict deadline, can extend and polish

5. **IEEE TPDS** - *Transactions on Parallel and Distributed Systems* (IF: ~5.3)
   - Fit: Good (parallel systems, performance)
   - Advantage: Prestigious, rigorous peer review

### Backup Plans

- ArXiv preprint (immediate visibility while under review)
- NVIDIA GTC paper/poster (high industry visibility)
- IEEE CloudCom, CLUSTER workshops

---

## Timeline

| Week | Phase | Activity | Deliverable |
|------|-------|----------|-------------|
| 1-3 | **Data Collection (Baseline)** | Run 10 models × 60 runs (native + Docker) | 600 benchmark results |
| 4 | **Optimization Exploration** | Test 7 configs on 3 models | Best optimization identified |
| 5-6 | **Optimized Testing** | Run 10 models × 30 runs (optimized) | 300 optimization results |
| 7 | **Statistical Analysis** | t-tests, CIs, power analysis, figures | Analysis package |
| 8-11 | **Paper Writing** | First draft → revisions → polish | Manuscript draft |
| 12-14 | **Submission** | Venue selection, formatting, submit | Submitted paper + ArXiv |

**Total:** 14 weeks (~3.5 months)

---

## Getting Started

### Prerequisites

- ✅ NVIDIA DGX Spark (Grace Hopper GB10)
- ✅ TensorRT-LLM container image
- ✅ Models downloaded to `/data/models/huggingface/`
- ✅ Docker with nvidia-container-toolkit
- ✅ Python 3.12 with scipy, pandas, numpy

### Quick Test (Before Full Run)

```bash
# Test one model with N=3 to verify everything works
export RUNS_PER_MODEL=3
export MODELS_TO_TEST="Llama-3-8B"

./scripts/run_phase2_comprehensive.sh
```

**Expected time:** ~1.5 hours

### Full Phase 2 Execution

```bash
# Run complete Phase 2 baseline
./scripts/run_phase2_comprehensive.sh
```

**Expected time:** ~80 hours (run overnight/weekend)

### Monitor Progress

```bash
# Count completed runs
ls results/phase2/native/*_metadata.json | wc -l
ls results/phase2/container_baseline/*_metadata.json | wc -l

# Watch latest log
tail -f results/phase2/logs/*_iter*.log
```

---

## Repository Structure (Phase 2 Additions)

```
benchmark-spark/
├── PHASE2_README.md                          # This file
├── docs/
│   ├── PHASE2_METHODOLOGY.md                 # Full methodology (IEEE quality)
│   ├── PHASE2_QUICKSTART.md                  # Quick start guide
│   ├── phase1/                               # Phase 1 results (preserved)
│   │   └── index.html
│   └── index.html                            # Landing page (both phases)
├── configs/
│   └── phase2_models.json                    # 10-model configuration
├── scripts/
│   ├── run_phase2_comprehensive.sh           # Main orchestrator (NEW)
│   ├── run_container_benchmark_optimized.sh  # Optimization testing (NEW)
│   ├── run_container_benchmark.sh            # Baseline (UPDATED)
│   └── run_native_benchmark.sh               # Native (unchanged)
├── benchmarks/
│   └── trtllm_benchmark.py                   # Updated: 1000 requests
├── analysis/
│   └── phase2_statistical_analysis.py        # Statistical tests (NEW)
└── results/
    ├── phase2/                               # Phase 2 results (NEW)
    │   ├── native/
    │   ├── container_baseline/
    │   ├── container_optimized/
    │   ├── logs/
    │   └── statistical_analysis/
    └── comprehensive/                        # Phase 1 (preserved)
```

---

## Next Steps

1. **Review methodology:** Read [docs/PHASE2_METHODOLOGY.md](docs/PHASE2_METHODOLOGY.md)

2. **Quick test:** Run one model (N=3) to verify setup

3. **Full baseline run:** Execute Phase 2A (10 models × 30 runs)

4. **Optimization:** Test Docker configurations (Phase 2B)

5. **Analysis:** Run statistical analysis scripts

6. **Paper writing:** Draft IEEE paper using methodology + results

7. **Submission:** Submit to HPDC/CCGrid + post ArXiv preprint

---

## Support

- **Phase 1 Results:** [https://brandonrc.github.io/benchmark-spark/phase1/](https://brandonrc.github.io/benchmark-spark/phase1/)
- **Full Methodology:** [docs/PHASE2_METHODOLOGY.md](docs/PHASE2_METHODOLOGY.md)
- **Quick Start:** [docs/PHASE2_QUICKSTART.md](docs/PHASE2_QUICKSTART.md)
- **Model Config:** [configs/phase2_models.json](configs/phase2_models.json)

---

**Ready to start Phase 2?**

```bash
./scripts/run_phase2_comprehensive.sh
```

🚀 Let's build an IEEE-quality paper!
