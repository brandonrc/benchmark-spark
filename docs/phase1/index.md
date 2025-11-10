---
layout: default
title: DGX Spark Performance Benchmarking
---

# DGX Spark Performance Benchmarking
## Container vs Native GPU Performance Analysis

---

## 🎯 Project Overview

This project provides **reproducible benchmarks** to measure GPU performance differences between **containerized** and **native** execution environments on NVIDIA DGX Spark systems.

### Why This Matters

NVIDIA DGX Spark systems running TensorRT-LLM in the development container (`spark-single-gpu-dev`) are experiencing **~50% performance degradation** from expected throughput. This benchmark suite helps:

✅ Quantify container overhead
✅ Isolate performance bottlenecks
✅ Provide reproducible test cases
✅ Generate shareable results for NVIDIA

---

## 📊 Quick Results

> Results will be published here after benchmarking

### Expected Performance Patterns

Based on [prior research](reference_paper.html), we expect:

| Environment | Execution Speed | GPU Utilization | Trade-off |
|-------------|----------------|-----------------|-----------|
| **Native** | ⚡ Faster | Lower (45-60%) | Raw performance |
| **Container** | 🐌 Slower | Higher (80-95%) | Consistency & portability |

**Key Question:** Is container overhead the cause of 50% performance gap?

---

## 🚀 Getting Started

### Prerequisites

- NVIDIA DGX Spark (or similar GPU system)
- Docker with NVIDIA Container Toolkit
- CUDA 12.x and compatible drivers
- Python 3.10+

### Quick Start (5 minutes)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/benchmark-spark.git
cd benchmark-spark

# Run container benchmark
./scripts/run_container_benchmark.sh all

# Results saved to results/container/
```

[View Full Setup Guide →](setup.html)

---

## 📖 Documentation

### Core Documentation

- **[Quick Start Guide](../QUICKSTART.html)** - Get running in 5 minutes
- **[Setup Guide](setup.html)** - Detailed installation instructions
- **[Methodology](methodology.html)** - Benchmarking approach and rationale
- **[Reference Paper](reference_paper.html)** - Academic background

### Benchmarks

1. **Matrix Multiplication** (Sanity Check)
   - 10,000 x 10,000 matrix
   - Validates basic GPU operations
   - Comparable to prior research

2. **LLM Inference** (Production Workload)
   - TensorRT-LLM with various configurations
   - Batch sizes: 1, 4, 16, 32
   - Sequence lengths: 128, 512, 2048 tokens
   - Measures: throughput, latency, GPU utilization

### Analysis Tools

- **Automated comparison** - Statistical analysis with t-tests
- **Visualizations** - Box plots, line charts, overhead graphs
- **Markdown reports** - Shareable results

---

## 🎯 Key Features

### ✅ Reproducible
- Automated scripts
- Fixed random seeds
- Version-controlled configurations

### ✅ Comprehensive
- Multiple workload types
- Statistical significance testing
- GPU utilization monitoring

### ✅ Practical
- Real LLM inference workloads
- Production-relevant configurations
- Easy to share results

### ✅ Open Source
- MIT Licensed
- Community contributions welcome
- Documented methodology

---

## 📈 Benchmark Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. Container Benchmark (5-10 min)                         │
│     ./scripts/run_container_benchmark.sh                    │
│     ├─ Matrix multiplication                                │
│     └─ LLM inference tests                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Native Setup (30-60 min, one-time)                     │
│     ./scripts/setup_native.sh                               │
│     ├─ Install TensorRT-LLM                                 │
│     └─ Build from source                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Native Benchmark (5-10 min)                            │
│     ./scripts/run_native_benchmark.sh                       │
│     ├─ Same workloads                                       │
│     └─ Direct GPU access                                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. Analysis & Comparison (2-5 min)                        │
│     python analysis/compare_results.py                      │
│     ├─ Statistical tests                                    │
│     ├─ Visualizations                                       │
│     └─ Overhead calculation                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Understanding Results

### Overhead Interpretation

| Overhead | Interpretation | Action |
|----------|---------------|---------|
| **< 10%** | ✅ Minimal | Container not the issue; check elsewhere |
| **10-20%** | ⚠️ Moderate | Expected range; acceptable for most use cases |
| **20-40%** | ⚠️ Significant | Optimization opportunities exist |
| **> 40%** | 🔴 Critical | Configuration issue or dev container broken |

### What to Check Next

1. **GPU Utilization** - Is GPU fully loaded?
2. **Thermal Throttling** - Check temperatures
3. **Power Limits** - Verify power draw
4. **Driver Versions** - Match container and host
5. **Container Image** - Try production vs dev

---

## 📦 Repository Structure

```
benchmark-spark/
├── README.md                      # Project overview
├── QUICKSTART.md                  # 5-minute start guide
├── requirements.txt               # Python dependencies
├── benchmarks/                    # Benchmark scripts
│   ├── simple_matmul.py          # Matrix multiplication
│   ├── llm_inference.py          # LLM benchmark
│   └── config.yaml               # Test configurations
├── scripts/                       # Runner scripts
│   ├── setup_native.sh           # Native installation
│   ├── run_container_benchmark.sh
│   └── run_native_benchmark.sh
├── analysis/                      # Analysis tools
│   └── compare_results.py        # Statistical comparison
└── docs/                          # Documentation
    ├── setup.md
    ├── methodology.md
    └── reference_paper.md
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- Additional benchmark workloads
- Optimization suggestions
- Bug fixes
- Documentation improvements
- Results from different hardware

[Open an Issue](https://github.com/YOUR_USERNAME/benchmark-spark/issues) | [Submit PR](https://github.com/YOUR_USERNAME/benchmark-spark/pulls)

---

## 📚 Background Research

This project builds on academic research:

> **"Benchmarking GPU Passthrough Performance on Docker for AI Cloud System"**
> Sani et al., 2025
> Found 67% overhead on consumer GPU (RTX 3060)

[Read Full Paper Summary →](reference_paper.html)

**Key Differences:**
- Enterprise GPU (DGX) vs Consumer (RTX 3060)
- Production workload (LLM) vs Synthetic (matmul)
- Hardware-optimized container vs Generic

---

## 📄 License

MIT License - See [LICENSE](../LICENSE) for details

---

## 📞 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/YOUR_USERNAME/benchmark-spark/issues)
- **Discussions:** [GitHub Discussions](https://github.com/YOUR_USERNAME/benchmark-spark/discussions)
- **Documentation:** [docs/](https://github.com/YOUR_USERNAME/benchmark-spark/tree/main/docs)

---

## 🎉 Quick Links

- [Quick Start Guide](../QUICKSTART.html)
- [Setup Instructions](setup.html)
- [Methodology Details](methodology.html)
- [Reference Paper Summary](reference_paper.html)
- [GitHub Repository](https://github.com/YOUR_USERNAME/benchmark-spark)

---

<div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f0f0f0;">
<p><strong>Ready to benchmark your DGX Spark?</strong></p>
<p><a href="https://github.com/YOUR_USERNAME/benchmark-spark" style="background-color: #76B900; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold;">Get Started Now →</a></p>
</div>
