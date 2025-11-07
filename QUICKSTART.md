# Quick Start Guide

## 🚀 5-Minute Start

### Step 1: Container Benchmark (No Setup Required)

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/benchmark-spark.git
cd benchmark-spark

# Run container benchmarks
./scripts/run_container_benchmark.sh all
```

**That's it!** Results in `results/container/`

---

## 🔬 Full Comparison (Container vs Native)

### Step 2: Setup Native Environment (~30-60 min, one time only)

```bash
./scripts/setup_native.sh
```

Coffee break ☕ while it compiles...

### Step 3: Run Native Benchmarks

```bash
source ~/tensorrt-llm/activate.sh
./scripts/run_native_benchmark.sh all
```

### Step 4: Compare Results

```bash
pip install pandas matplotlib seaborn scipy
python analysis/compare_results.py
```

### Step 5: View Results

```bash
# Read the report
cat results/comparison/comparison_report.md

# View plots
open results/comparison/*.png  # macOS
xdg-open results/comparison/*.png  # Linux
```

---

## 📊 What Gets Measured

1. **Matrix Multiplication** (sanity check)
   - 10,000 x 10,000 matrix
   - Validates basic GPU operations

2. **LLM Inference** (real workload)
   - Various batch sizes (1, 4, 16, 32)
   - Different sequence lengths
   - Measures throughput (tokens/second)
   - Measures latency (milliseconds)

3. **GPU Utilization**
   - Continuous monitoring via nvidia-smi
   - Temperature, power, memory usage
   - Clock speeds

---

## 🎯 Interpreting Results

### Container Overhead < 10%
✅ **Good news:** Container is not the problem
- Look elsewhere: drivers, thermal throttling, power limits
- Performance issue likely in hardware/config

### Container Overhead 20-40%
⚠️ **Expected:** Normal container overhead
- Consider optimizations
- Balance convenience vs performance

### Container Overhead > 50%
🔴 **Problem:** Container configuration issue
- Dev container may be broken
- Try production container
- Report to NVIDIA

---

## 🛠️ Troubleshooting

### Docker not working?
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If fails, restart docker
sudo systemctl restart docker
```

### CUDA not found?
```bash
# Check installation
nvcc --version
nvidia-smi

# Add to ~/.bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Native build fails?
```bash
# Install dependencies
sudo apt-get install build-essential cmake ninja-build

# Check logs
tail -100 ~/tensorrt-llm/build.log
```

---

## 📁 Results Structure

```
results/
├── container/              # Container benchmark results
│   ├── matmul_results.csv
│   ├── llm_benchmark_*.csv
│   └── gpu_metrics_*.csv
├── native/                 # Native benchmark results
│   ├── matmul_results.csv
│   ├── llm_benchmark_*.csv
│   └── gpu_metrics_*.csv
└── comparison/             # Analysis outputs
    ├── comparison_report.md
    ├── matmul_comparison.png
    └── llm_comparison.png
```

---

## 🎓 Understanding the Output

### Execution Time
- **Lower is better**
- Native usually faster
- Measures end-to-end latency

### GPU Utilization
- **Higher can be better** (but not always)
- Shows how much GPU is being used
- 100% doesn't guarantee fastest execution

### Throughput (tokens/second)
- **Higher is better**
- Most important metric for LLM inference
- Shows real-world performance

### Overhead Percentage
- `(Container_time - Native_time) / Native_time × 100`
- Positive = container slower
- Negative = container faster (rare)

---

## 🔍 Next Steps After Benchmarking

1. **Review GPU Metrics**
   ```bash
   # Check for throttling
   grep temperature results/*/gpu_metrics_*.csv
   ```

2. **Test Different Container**
   ```bash
   # Try production container instead of dev
   export CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt-llm:latest"
   ./scripts/run_container_benchmark.sh
   ```

3. **Adjust Test Parameters**
   ```bash
   # Edit config
   nano benchmarks/config.yaml
   ```

4. **Share Results**
   - Generate GitHub Pages
   - Report to NVIDIA if issues found
   - Share with community

---

## 📞 Getting Help

- **Issues:** Open issue on GitHub
- **Docs:** Check `docs/` directory
- **Setup:** See `docs/setup.md`
- **Methodology:** See `docs/methodology.md`

---

## ⏱️ Time Investment

| Task | Time |
|------|------|
| Container benchmark | 5-10 min |
| Native setup (first time) | 30-60 min |
| Native benchmark | 10-20 min |
| Analysis | 2-5 min |
| **Total (first run)** | **~1 hour** |
| **Subsequent runs** | **~15 min** |

---

## 🎉 Success Criteria

You've succeeded when you have:
- [ ] Container benchmark results
- [ ] Native benchmark results
- [ ] Comparison report showing overhead %
- [ ] GPU utilization metrics
- [ ] Identified if container is the bottleneck

**Now you can:**
- Make informed decisions about containerization
- Report issues to NVIDIA with data
- Optimize your deployment strategy
- Share reproducible results
