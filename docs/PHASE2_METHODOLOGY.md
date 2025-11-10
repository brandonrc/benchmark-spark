# Phase 2 Methodology: IEEE-Quality Benchmarking

## Overview

Phase 2 extends the initial Phase 1 investigation with a rigorous, publication-ready experimental design suitable for IEEE conference or journal submission.

### Key Improvements from Phase 1

| Aspect | Phase 1 | Phase 2 | Rationale |
|--------|---------|---------|-----------|
| **Sample Size** | N=10 runs/model | N=30 runs/model | Increased statistical power (80%+) |
| **Requests/Run** | 100 | 1000 | NVIDIA recommended minimum |
| **Models Tested** | 3 | 10 | Size diversity (1.5B-72B), architecture variety |
| **Docker Config** | Custom (`--shm-size=60g`) | NVIDIA official baseline | Reproducible, validated configuration |
| **Statistical Analysis** | Descriptive only | Full inferential (t-tests, CIs, power) | IEEE publication standards |
| **Optimization** | None | Systematic Docker tuning | Practical deployment guidance |

## Research Design

### Single-Variable Experimental Design

**Controlled Variable:** Containerization approach (Native/Chroot vs Docker)

**Constants:**
- Hardware: NVIDIA DGX Spark (Grace Hopper GB10) only
- OS: DGX OS / Ubuntu 24.04.3 LTS
- Software: TensorRT-LLM (`nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`)
- Prompts: Fixed dataset (1000 requests)
- Thermal conditions: 5-minute cooldown, GPU < 45°C before each run
- Alternating execution order: Mitigates systematic thermal bias

**Why This Design:**
- Any performance difference attributable solely to containerization
- Eliminates hardware, software, and workload confounds
- Highly reproducible with same DGX Spark hardware
- Clean causal inference

### Novel Contributions

1. **First DGX Spark (GB10) Benchmarking Study**
   - No prior published work on this hardware (released November 2024)
   - Grace Blackwell architecture fundamentally different from prior GPUs

2. **Unified Memory Architecture Focus**
   - Traditional studies examine discrete GPU systems (PCIe/NVLink bottlenecks)
   - GB10 has unified LPDDR5X memory (no CPU-GPU transfer overhead)
   - Research question: Does containerization behave differently in UMA systems?

3. **Real-World LLM Inference Workloads**
   - Not synthetic CUDA microbenchmarks
   - Production-representative models (Llama, Mistral, Qwen, etc.)
   - Actual TensorRT-LLM deployment framework

4. **Systematic Docker Optimization Path**
   - Most papers compare only default configurations
   - We test 7+ optimization strategies
   - Actionable deployment recommendations for practitioners

## Configuration Sources and Validation

### Docker Configuration: NVIDIA Official Playbooks

**Primary Reference:**
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- Specifically: `nvidia/nvfp4-quantization/README.md`

**Baseline Docker Configuration:**
```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  <image> <command>
```

**Rationale for Each Flag:**
- `--rm`: Auto-cleanup (no side effects between runs)
- `--gpus all`: Full GPU passthrough to container
- `--ipc=host`: Share host IPC namespace for performance
  - Avoids Docker's default IPC isolation overhead
  - Required for high-performance GPU workloads
- `--ulimit memlock=-1`: Unlimited locked memory
  - CUDA driver requires pinning GPU memory
  - Default limit (65KB) insufficient for LLM inference
- `--ulimit stack=67108864`: 64MB stack size
  - Deep neural networks require large stack
  - Default (8MB) can cause stack overflow

**What We Did NOT Use (Intentionally):**
- `--shm-size=<size>`: Not in NVIDIA baseline (tested separately in optimization phase)
- `--memory=<limit>`: No artificial constraints on container RAM
- `--privileged`: Security risk, not needed for GPU access
- `--network=host`: Not required for single-node inference

### TensorRT-LLM Configuration: NVIDIA Documentation

**Primary Reference:**
- [TensorRT-LLM Official Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Performance Benchmarking Guide](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html)

**Key Parameters:**
- **num_requests**: 1000 (NVIDIA recommends 1000-3000 for throughput)
- **target_output_len**: 128 tokens (common in chat applications)
- **max_batch_size**: Model-dependent (8 for 7B models, 2 for 70B models)
- **benchmark_type**: `throughput` (primary metric for capacity planning)

## Model Selection

### Criteria

1. **Size Diversity:** 1.5B to 72B parameters (4 size tiers)
2. **Architecture Variety:** Standard Transformer, MoE, Efficient architectures
3. **Practical Relevance:** Models deployed in production environments
4. **Memory Constraints:** Must fit in 119GB unified memory (with quantization if needed)
5. **Open Availability:** No gated/restricted models (for reproducibility)
6. **Academic Credibility:** Widely cited, industry-standard models

### Selected 10-Model Suite

| Model | Size | Type | Rationale |
|-------|------|------|-----------|
| **Qwen2-1.5B** | 1.5B | Decoder | Smallest model - minimal overhead baseline |
| **Gemma-2B** | 2B | Decoder | Google's efficient small model |
| **Phi-3-mini** | 3.8B | Efficient | Microsoft's optimized architecture |
| **Mistral-7B-v0.3** | 7B | Decoder | Popular open-source standard |
| **DeepSeek-R1-7B** | 7B | Reasoning | Cutting-edge reasoning model (Phase 1 continuity) |
| **Llama-3-8B** | 8B | Decoder | Meta's industry standard - most cited |
| **Mixtral-8x7B** | 45B (8x7B) | MoE | Different memory access pattern |
| **Llama-3-70B** | 70B | Decoder (AWQ-4bit) | Large model under memory pressure |
| **Qwen2.5-72B** | 72B | Decoder | Phase 1 continuity - long context |
| **Note** | - | - | GPT-OSS-120B replaced if provenance unclear |

**Architecture Breakdown:**
- 7 standard decoder-only transformers
- 1 MoE architecture (Mixtral)
- 1 efficient architecture (Phi-3)
- 1 reasoning-enhanced (DeepSeek-R1)

**Size Distribution:**
- Tier 1 (1-4B): 3 models
- Tier 2 (7-8B): 3 models
- Tier 3 (40-50B): 1 model
- Tier 4 (70-72B): 2 models

## Statistical Methodology

### Sample Size: N=30 Runs

**Justification:**
- **Statistical power**: 80%+ power to detect medium effects (d=0.5)
- **Convention**: N≥30 is standard for t-tests in experimental research
- **Practical**: 30 runs balances rigor with compute budget

**Per Model Testing:**
- 30 runs native (chroot)
- 30 runs Docker baseline
- 30 runs Docker optimized (best config)
- **Total per model:** 90 runs

**Full Suite:**
- 10 models × 90 runs = **900 total benchmark runs**
- Estimated time: 120-150 hours (~5-6 days continuous)

### Thermal Control Strategy

**Problem:** GPU performance degrades with temperature

**Solution:** Controlled cooldown + alternating execution

**Protocol:**
1. 5-minute cooldown between all runs
2. Monitor GPU temperature continuously
3. Wait until GPU < 45°C before starting run
4. Alternate execution order:
   ```
   Run 1: Native → Cooldown → Docker Baseline → Cooldown
   Run 2: Docker Baseline → Cooldown → Native → Cooldown
   Run 3: Native → Cooldown → Docker Baseline → Cooldown
   ...
   ```

**Benefits:**
- Thermal effects average out across conditions
- No systematic bias toward either environment
- Detectable if thermal throttling occurs

### Statistical Analysis Plan

#### 1. Descriptive Statistics
For each metric (throughput, latency, memory):
- Mean, median, standard deviation
- Minimum, maximum, quartiles (Q1, Q3)
- Coefficient of variation (CV)
- Percentiles (P50, P95, P99) for latency distributions

#### 2. Inferential Statistics
**Primary test:** Independent samples t-test
- **Null hypothesis (H₀):** μ_native = μ_container
- **Alternative hypothesis (H₁):** μ_native ≠ μ_container
- **Significance level:** α = 0.05 (two-tailed)

**Assumptions:**
- Check normality: Shapiro-Wilk test (if p > 0.05, assume normal)
- If violated: Use Welch's t-test (unequal variances) or Mann-Whitney U (non-parametric)

**Effect Size:**
- Compute Cohen's d for all comparisons
- Interpretation: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)

#### 3. Confidence Intervals
- 95% CI for mean differences
- Bootstrap CIs (10,000 resamples) if normality violated

#### 4. Power Analysis
- **Post-hoc:** Actual power given observed effect sizes
- **Validates:** Was N=30 sufficient?
- **Target:** Power ≥ 0.80 for all primary comparisons

#### 5. Multiple Comparisons Correction
**Issue:** Testing 10 models × 3 metrics = 30 comparisons
**Solution:** Bonferroni correction (α' = 0.05 / 30 = 0.0017)
**Alternative:** Benjamini-Hochberg FDR control (less conservative)

## Metrics Collection

### Primary Metrics (Main Paper)

1. **Token Throughput** (tokens/sec)
   - Most important for production capacity planning
   - Higher is better

2. **Request Throughput** (requests/sec)
   - System-level metric
   - Accounts for batching efficiency

3. **Average Latency** (milliseconds)
   - End-to-end request processing time
   - Lower is better

4. **Memory Usage** (GB)
   - Peak unified memory allocation
   - Critical for Grace Hopper UMA systems

5. **KV Cache Allocation** (GB)
   - Memory available for context caching
   - Directly impacts maximum concurrent users

### Secondary Metrics (Appendix)

6. **Time-to-First-Token** (TTFT, milliseconds)
   - User-perceived responsiveness
   - Separate latency benchmark run

7. **Inter-Token Latency** (milliseconds)
   - Generation smoothness
   - P50, P95, P99 percentiles

8. **GPU Utilization** (%)
   - How efficiently compute is used
   - Identifies bottlenecks

9. **Memory Bandwidth** (GB/s)
   - Unique to unified memory architecture
   - Shows if memory I/O is saturated

10. **Power Consumption** (watts)
    - Performance per watt analysis
    - Thermal headroom

## Docker Optimization Experiments

### Baseline Configuration
NVIDIA official recommendation (no optimizations):
```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  <image> <command>
```

### Optimization Dimensions

Test each optimization individually first, then combine best performers:

1. **Shared Memory Allocation**
   - Test: `--shm-size=8g`, `--shm-size=32g`, `--shm-size=60g`
   - Hypothesis: Larger SHM reduces memory pressure in UMA

2. **Network Mode**
   - Test: `--network=host`
   - Hypothesis: Eliminates bridge network overhead (minor expected)

3. **CPU Core Pinning**
   - Test: `--cpuset-cpus=0-9` (pin to performance cores: Cortex-X925)
   - Hypothesis: Reduces scheduler overhead, improves consistency

4. **Privileged Mode**
   - Test: `--privileged`
   - Hypothesis: Removes security overhead (⚠️ not recommended for production)

5. **Memory Limit Removal**
   - Test: `--memory=unlimited` (if possible)
   - Hypothesis: Prevents cgroup memory accounting issues

6. **Cgroup Parent Override**
   - Test: `--cgroup-parent=/`
   - Hypothesis: Bypass Docker cgroup memory accounting

7. **IPC Namespace (Verification)**
   - Confirm: `--ipc=host` vs default
   - Baseline already uses `--ipc=host`, verify it's necessary

### Experimental Plan

**Phase 2A: Baseline Testing** (Done first)
- 10 models × 30 runs × 2 environments (native + Docker baseline)
- Establishes performance gap to be optimized

**Phase 2B: Individual Optimization Testing** (On subset)
- Select 3-5 representative models (small, medium, large)
- Test each optimization individually (10 runs per config)
- Identify most impactful optimizations

**Phase 2C: Combined Optimization** (On subset)
- Combine top 2-3 optimizations
- Test "best combined" configuration (30 runs)

**Phase 2D: Full Suite Validation** (Final)
- Run full 10-model suite with best optimization (30 runs each)
- Compare: Baseline → Optimized reduction in overhead

### Expected Outcomes

**Hypothesis 1:** `--shm-size=32g` significantly reduces overhead
- **Basis:** Phase 1 used 60GB, but NVIDIA baseline omits this
- **Prediction:** 30-50% overhead reduction

**Hypothesis 2:** CPU pinning improves consistency
- **Basis:** Reduces scheduler variability
- **Prediction:** Lower standard deviation, no mean improvement

**Hypothesis 3:** Combined optimizations approach native performance
- **Basis:** Overhead is from multiple small sources
- **Prediction:** 60-80% overhead reduction vs baseline

## Data Collection and Quality Control

### Automated Execution
- All benchmarks run via shell scripts (no manual intervention)
- Eliminates human error and timing inconsistencies

### Audit Trail
- Every run produces:
  - Timestamped log file
  - JSON metadata file
  - GPU monitoring CSV
  - System memory monitoring CSV

### Quality Checks
- Verify GPU temperature < 45°C before each run
- Monitor for thermal throttling during runs
- Check for outliers (> 3σ from mean) - investigate and re-run if needed
- Validate all 30 runs completed successfully

### Data Integrity
- Results stored in git repository
- Version-controlled configuration files
- Reproducible analysis scripts (Jupyter notebooks + Python)

## Publication Strategy

### Target Venues (Priority Order)

1. **IEEE/ACM HPDC** (Tier 1 Conference)
   - *ACM International Symposium on High-Performance Parallel and Distributed Computing*
   - Fit: Excellent (performance evaluation, containerization, GPU systems)
   - Deadline: Typically January (for June conference)

2. **IEEE CCGrid** (Tier 1 Conference)
   - *IEEE International Conference on Cluster, Cloud and Grid Computing*
   - Fit: Excellent (cloud computing, container performance)
   - Deadline: Typically November (for May conference)

3. **ACM ASPLOS** (Top-Tier Conference)
   - *International Conference on Architectural Support for Programming Languages and Operating Systems*
   - Fit: Good (systems architecture, OS interaction with hardware)
   - Deadline: Typically April and August (for spring conference)
   - Note: Highly competitive, stretch goal

4. **Future Generation Computer Systems** (Journal)
   - Impact Factor: ~7.0
   - Fit: Excellent (systems research, performance evaluation)
   - Advantage: No strict deadline, can polish and extend

5. **IEEE Transactions on Parallel and Distributed Systems** (Journal)
   - Impact Factor: ~5.3
   - Fit: Good (parallel systems, performance)
   - Advantage: Prestigious venue, rigorous review

### Backup Plans

- **ArXiv Preprint:** Immediate visibility, citable while under review
- **NVIDIA GTC Paper/Poster:** High industry visibility
- **IEEE CloudCom, CLUSTER:** Solid conferences if top-tier rejected

### Success Metrics

- **Minimum:** Workshop or poster acceptance at major conference
- **Target:** Full paper at Tier 1 conference (HPDC, CCGrid)
- **Stretch:** Top-tier conference (ASPLOS) or high-IF journal (FGCS)

## Timeline

### Phase 2A: Baseline Data Collection (Weeks 1-3)
- Setup and validation: 3 days
- Tier 1 models (5 models × 60 runs): 7 days
- Tier 2+3 models (5 models × 60 runs): 7 days
- Buffer for issues: 4 days

### Phase 2B: Optimization Exploration (Week 4)
- Individual optimizations (3 models × 7 configs × 10 runs): 5 days
- Analysis and best-config selection: 2 days

### Phase 2C: Optimized Full Suite (Weeks 5-6)
- 10 models × 30 runs optimized: 10 days
- Data validation: 2 days

### Phase 2D: Statistical Analysis (Week 7)
- Descriptive statistics: 1 day
- Inferential tests: 1 day
- Power analysis: 1 day
- Figure generation: 2 days
- Table formatting: 1 day
- Sanity checks: 1 day

### Phase 2E: Paper Writing (Weeks 8-12)
- First draft: 2 weeks
- Internal review: 1 week
- Revisions: 1 week
- Final polish: 1 week

### Phase 2F: Submission (Weeks 13-14)
- Venue selection: 3 days
- Formatting: 3 days
- Final proofreading: 2 days
- Submission: 1 day
- ArXiv posting: 1 day

**Total Timeline:** 14 weeks (~3.5 months)

**Critical Path:** Data collection (5 weeks)

## Risk Mitigation

### Risk: Hardware Failure
- **Impact:** Data loss, delays
- **Mitigation:** Daily backups to external storage, git push results regularly

### Risk: Software Updates Breaking Reproducibility
- **Impact:** Inconsistent results
- **Mitigation:** Pin all software versions, use containers, document exact versions

### Risk: Insufficient Statistical Power
- **Impact:** Null findings, paper rejection
- **Mitigation:** N=30 provides 80%+ power for d=0.5 effects

### Risk: Reviewer Skepticism on Model Selection
- **Impact:** Paper rejection
- **Mitigation:** Include Llama (most cited), justify each model choice, provide diversity

### Risk: Docker Overhead Not Generalizable
- **Impact:** "Only applies to Grace Hopper" criticism
- **Mitigation:** Emphasize UMA-specific findings, discuss discrete GPU implications, suggest future work

### Risk: Optimization Phase Finds No Improvement
- **Impact:** Weakened contribution
- **Mitigation:** Still valuable negative result, focus on understanding WHY optimizations don't help

## Open Science Principles

### Repository Contents
- All benchmark scripts (shell + Python)
- Configuration files (models, Docker flags)
- Raw results (JSON metadata, logs, monitoring CSVs)
- Analysis code (Jupyter notebooks, Python scripts)
- Figures and tables (publication-ready)
- Complete documentation (methodology, setup, reproduction)

### Licensing
- **Code:** MIT License (permissive, widely used)
- **Data:** CC BY 4.0 (attribution required, can reuse)
- **Paper:** Traditional copyright to publisher (preprint on ArXiv)

### Reproducibility Checklist
✅ Exact hardware specifications documented
✅ Software versions pinned and recorded
✅ Configuration files version-controlled
✅ Scripts automated (no manual steps)
✅ Random seeds fixed (where applicable)
✅ Raw data available
✅ Analysis code provided
✅ Statistical methods detailed

### Benefits
- **Reproducibility:** Others can replicate on same hardware
- **Transparency:** Clear methodology, no hidden steps
- **Impact:** Data reuse increases citations
- **Trust:** Open process builds credibility

## References

### NVIDIA Documentation
1. [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) - Official Docker configurations
2. [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) - Framework API and usage
3. [TensorRT-LLM Performance Benchmarking Guide](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) - Best practices
4. [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) - GPU container runtime

### Academic References
5. Sani, A. F., et al. (2025). "Benchmarking GPU Passthrough Performance on Docker for AI Cloud System." *Brilliance: Research of Artificial Intelligence*, 5(2), 211-219.
6. Shetty, J., et al. (2017). "Container-based virtualization for GPU-enabled HPC applications." *Future Generation Computer Systems*.
7. Openja, M., et al. (2022). "Studying the practices of deploying machine learning projects on Docker." *ESEM 2022*.

### Model Documentation
8. Meta AI. (2024). "Llama 3 Model Card." [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
9. Mistral AI. (2024). "Mistral 7B." [https://mistral.ai/news/announcing-mistral-7b/](https://mistral.ai/news/announcing-mistral-7b/)
10. Alibaba Cloud. (2024). "Qwen2.5 Technical Report." [https://qwenlm.github.io/](https://qwenlm.github.io/)

---

**Last Updated:** 2025-01-XX
**Status:** Methodology Approved - Ready for Execution
**Version:** 2.0 (Phase 2)
