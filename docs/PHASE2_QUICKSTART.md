# Phase 2 Quick Start Guide

## TL;DR

```bash
# Run full Phase 2 baseline testing (Tier 1 models, N=30)
cd /home/khan/benchmark-spark
./scripts/run_phase2_comprehensive.sh
```

**Estimated time:** ~60-80 hours for full 10-model suite

---

## Configuration Summary

### Changes from Phase 1

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Requests/run | 100 | **1000** |
| Runs/model | 10 | **30** |
| Models | 3 | **10** |
| Docker config | Custom (`--shm-size=60g`) | **NVIDIA baseline** |
| Statistical tests | None | **t-tests, CIs, power analysis** |

### Docker Configuration

**Baseline (NVIDIA official):**
```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  <image> <command>
```

**Source:** [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nvfp4-quantization/README.md)

---

## Quick Test (Single Model, N=3)

Before running the full 80-hour suite, test with one model:

```bash
# Test Llama-3-8B with N=3 runs
export RUNS_PER_MODEL=3
export MODELS_TO_TEST="Llama-3-8B"

./scripts/run_phase2_comprehensive.sh
```

**Expected time:** ~1.5 hours

---

## Running Phase 2

### Option 1: Full Suite (Recommended for Paper)

```bash
# All 10 models, N=30 each, native + container baseline
export TEST_PHASE="baseline"
export MODELS_TO_TEST="all"
export RUNS_PER_MODEL=30

./scripts/run_phase2_comprehensive.sh
```

**Outputs:**
- `results/phase2/native/*_metadata.json` (300 files)
- `results/phase2/container_baseline/*_metadata.json` (300 files)
- `results/phase2/logs/*.log` (600 log files)

### Option 2: Tier 1 Models Only (Faster, Still Publishable)

```bash
# 5 essential models: Llama-3-8B, Llama-3-70B, Mistral-7B, DeepSeek-R1-7B, Qwen2.5-72B
export MODELS_TO_TEST="tier1"
export RUNS_PER_MODEL=30

./scripts/run_phase2_comprehensive.sh
```

**Estimated time:** ~40 hours

### Option 3: Custom Model Selection

```bash
# Test specific models
export MODELS_TO_TEST="Llama-3-8B,Mistral-7B-v0.3,Qwen2.5-72B-Instruct"
export RUNS_PER_MODEL=30

./scripts/run_phase2_comprehensive.sh
```

---

## Docker Optimization Testing

After baseline testing, run optimization experiments:

### Option 1: Test All Optimizations (Research Phase)

```bash
# Test all 7 optimization configs on 3 models
export OPTIMIZATION_MODE="all"
export MODELS_TO_TEST="tier1"
export RUNS_PER_MODEL=10  # Fewer runs for exploration

./scripts/run_container_benchmark_optimized.sh trtllm
```

### Option 2: Test Best Configuration (After Identifying Winner)

```bash
# Run "best" optimized config on full suite
export OPTIMIZATION_MODE="best"
export MODELS_TO_TEST="all"
export RUNS_PER_MODEL=30

./scripts/run_container_benchmark_optimized.sh trtllm
```

**Available optimization modes:**
- `baseline` - NVIDIA official (no optimization)
- `shm-8g` - Add 8GB shared memory
- `shm-32g` - Add 32GB shared memory
- `shm-60g` - Add 60GB shared memory (Phase 1 config)
- `host-network` - Use host network mode
- `cpu-pinning` - Pin to performance cores
- `privileged` - Privileged mode (⚠️ security risk)
- `best` - Best combination (update after experiments)
- `all` - Test all modes sequentially

---

## Statistical Analysis

After data collection, run analysis:

```bash
# Generate statistical analysis
python analysis/phase2_statistical_analysis.py \
  --results-dir results/phase2

# Output:
# - results/phase2/statistical_analysis/detailed_analysis.json
# - results/phase2/statistical_analysis/summary_table.csv
```

**Analysis includes:**
- Descriptive statistics (mean, std, CV)
- Independent t-tests (native vs container)
- 95% confidence intervals
- Cohen's d effect sizes
- Statistical power calculation
- Normality checks (Shapiro-Wilk)

---

## Model Configuration

Models are defined in `configs/phase2_models.json`:

```json
{
  "tier1_essential": [
    "Llama-3-8B",           // Meta's industry standard
    "Llama-3-70B",          // Large model (AWQ-4bit)
    "Mistral-7B-v0.3",      // Open-source baseline
    "DeepSeek-R1-7B",       // Reasoning model
    "Qwen2.5-72B-Instruct"  // Long context
  ],
  "tier2_important": [
    "Mixtral-8x7B",         // MoE architecture
    "Phi-3-mini",           // Efficient design
    "Gemma-2B"              // Small Google model
  ],
  "tier3_optional": [
    "Qwen2-1.5B",           // Smallest model
    "GPT-OSS-120B"          // Phase 1 continuity (if available)
  ]
}
```

### Adding Custom Models

Edit `configs/phase2_models.json` and update `scripts/run_phase2_comprehensive.sh` function `get_huggingface_id()`.

---

## Monitoring Progress

### Check Run Status

```bash
# Count completed runs
ls results/phase2/native/*_metadata.json | wc -l
ls results/phase2/container_baseline/*_metadata.json | wc -l

# Expected: 30 per model per environment
# For 10 models: 300 files each (600 total)
```

### View Latest Log

```bash
# Native runs
tail -f results/phase2/logs/*_native_*.log

# Container runs
tail -f results/phase2/logs/*_container_*.log
```

### GPU Monitoring

```bash
# Real-time GPU stats
watch -n 1 nvidia-smi

# Temperature check
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```

---

## Troubleshooting

### Issue: "Model not found"

**Solution:** Ensure models are downloaded to `/data/models/huggingface/`:

```bash
# Check model directory
ls -lh /data/models/huggingface/meta-llama/Meta-Llama-3-8B
ls -lh /data/models/huggingface/Qwen/Qwen2.5-72B-Instruct
```

### Issue: "Out of memory"

**Solution:** Large models may need quantization:

- Llama-3-70B: Use AWQ-4bit (`casperhansen/llama-3-70b-instruct-awq`)
- Qwen2.5-72B: May need quantization for container runs
- Mixtral-8x7B: May need AWQ-4bit (`TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ`)

### Issue: "Docker daemon not running"

**Solution:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### Issue: "nvidia-smi not found in container"

**Solution:** Check nvidia-container-toolkit installation:

```bash
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "Thermal throttling detected"

**Solution:** Increase cooldown time:

```bash
export COOLDOWN_MINUTES=10  # Increase from default 5 minutes
export MAX_GPU_TEMP=40       # Lower threshold from default 45°C
```

---

## Directory Structure

```
benchmark-spark/
├── scripts/
│   ├── run_phase2_comprehensive.sh       # Main orchestrator
│   ├── run_container_benchmark.sh        # Baseline Docker
│   ├── run_container_benchmark_optimized.sh  # Optimized Docker
│   └── run_native_benchmark.sh           # Native/chroot
├── configs/
│   └── phase2_models.json                # Model definitions
├── benchmarks/
│   └── trtllm_benchmark.py               # TensorRT-LLM wrapper
├── analysis/
│   └── phase2_statistical_analysis.py    # Statistical tests
├── results/
│   └── phase2/
│       ├── native/                       # Native benchmark results
│       ├── container_baseline/           # Baseline Docker results
│       ├── container_optimized/          # Optimized Docker results
│       ├── logs/                         # Execution logs
│       └── statistical_analysis/         # Analysis outputs
└── docs/
    ├── PHASE2_METHODOLOGY.md             # Full methodology
    └── PHASE2_QUICKSTART.md              # This file
```

---

## Expected Results

### Metrics per Run

Each benchmark produces:
- **Throughput:** Tokens/sec, Requests/sec
- **Latency:** Average ms, P50, P95, P99
- **Memory:** Peak GB, KV cache GB
- **GPU:** Utilization %, Temperature °C, Power W

### Statistical Outputs

- **Descriptive stats:** Mean, std, CV, min, max, quartiles
- **Inferential tests:** t-statistic, p-value, significance (α=0.05)
- **Effect sizes:** Cohen's d (small/medium/large)
- **Confidence intervals:** 95% CI for mean differences
- **Power analysis:** Actual statistical power achieved

### IEEE Paper Deliverables

1. **Main Results Table:** Native vs Container comparison (all models)
2. **Statistical Significance Table:** t-tests, p-values, effect sizes
3. **Optimization Impact Table:** Baseline → Optimized improvements
4. **Figures:**
   - Latency comparison (bar chart with error bars)
   - Memory overhead scaling (line plot)
   - Throughput under different configs (grouped bars)
   - Docker optimization waterfall (cumulative improvement)

---

## Next Steps After Data Collection

1. **Run statistical analysis:**
   ```bash
   python analysis/phase2_statistical_analysis.py
   ```

2. **Generate figures:**
   ```bash
   # TODO: Create phase2_generate_figures.py
   python analysis/phase2_generate_figures.py
   ```

3. **Write paper:**
   - Use `docs/PHASE2_METHODOLOGY.md` as methods section
   - Results from `statistical_analysis/summary_table.csv`
   - Figures from analysis output

4. **Submit to IEEE/ACM conference:**
   - Target: HPDC, CCGrid, or ASPLOS
   - Upload to ArXiv for preprint visibility

---

## Time Estimates

### Data Collection

| Configuration | Models | Runs/Model | Estimated Time |
|---------------|--------|------------|----------------|
| Quick test | 1 | 3 | ~1.5 hours |
| Tier 1 only | 5 | 30 | ~40 hours |
| Full suite (baseline) | 10 | 30 | ~80 hours |
| Optimization (subset) | 3 | 70 | ~20 hours |
| **Total Phase 2** | 10 | 90 | **~100 hours** |

**Assumptions:**
- ~10 min per benchmark run
- 5 min cooldown between runs
- ~15 min average per run (including cooldown)

### Analysis and Writing

- Statistical analysis: 1 day
- Figure generation: 2 days
- Paper writing (first draft): 2 weeks
- Revisions: 1 week
- **Total:** 3-4 weeks

### Overall Timeline

- Data collection: 5 weeks (continuous running)
- Analysis: 1 week
- Writing: 4 weeks
- **Total: 10 weeks** (~2.5 months)

---

## Support and Questions

**Phase 1 Results (Reference):**
- Results: `results/comprehensive/`
- Analysis: `ANALYSIS.md`
- GitHub Pages: `https://brandonrc.github.io/benchmark-spark/phase1/`

**Documentation:**
- Full methodology: `docs/PHASE2_METHODOLOGY.md`
- Model configs: `configs/phase2_models.json`
- Scripts: `scripts/run_phase2_*.sh`

**Logs and Debugging:**
- Execution logs: `results/phase2/logs/`
- GPU monitoring: `results/phase2/*/enhanced_metrics_*.csv`

---

**Ready to start? Run:**

```bash
./scripts/run_phase2_comprehensive.sh
```

Good luck! 🚀
