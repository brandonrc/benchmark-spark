# DGX Spark TensorRT-LLM Benchmarking
## Phase 1: Container vs Native Execution Analysis

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Results-blue)](https://USER.github.io/benchmark-spark/)
[![Analysis](https://img.shields.io/badge/Analysis-Complete-green)](ANALYSIS.md)

> **Phase 1 Status**: ✅ Complete (November 2025)
> **Phase 2**: Docker cgroup investigation → [See separate repository](#phase-2-next-steps)

---

## 🎯 Overview

This repository contains comprehensive benchmarking of large language model (LLM) inference on NVIDIA DGX Spark with Grace Hopper architecture, comparing **Docker containerized** vs **native (chroot)** execution.

### Key Findings

| Metric | Native | Container | Winner |
|--------|--------|-----------|--------|
| **Memory Overhead** | Baseline | +20-31 GB | 🏆 Native |
| **KV Cache** | 43-45 GB | 17-27 GB | 🏆 Native (1.6-2.7x more) |
| **Throughput** | 119-120 tok/s | 119-120 tok/s | 🤝 Identical |

**Bottom Line**: Native execution provides **20-30 GB memory savings** and **1.6-2.7x more KV cache** with no performance penalty.

📊 **[View Interactive Results](https://USER.github.io/benchmark-spark/)** | 📄 **[Read Full Analysis](ANALYSIS.md)**

---

## 🧪 Test Methodology

### Models Tested (60 total runs)

- **DeepSeek-R1-Distill-Qwen-7B** (7B parameters) - 10 native + 10 container
- **GPT-OSS-120B** (120B parameters, MXFP4) - 10 native + 10 container
- **Qwen2.5-72B-Instruct** (72B parameters) - 10 native + 10 container

### Benchmark Configuration

```bash
# Framework: TensorRT-LLM (trtllm-bench CLI)
# Workload: 50 requests × 128 output tokens
# Iterations: 10 per model per environment
# Cooldown: 5 minutes + GPU temp monitoring (<45°C)
```

### Environments

**Container (Docker)**:
```bash
docker run --rm --gpus all --ipc=host --shm-size=60g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
    python benchmarks/trtllm_benchmark.py ...
```

**Native (Chroot)**:
```bash
sudo chroot /path/to/container-rootfs \
    /usr/bin/env -i \
    LD_LIBRARY_PATH=... CUDA_HOME=/usr/local/cuda-12.9 \
    python benchmarks/trtllm_benchmark.py ...
```

---

## 📊 Results Summary

### Memory Usage

| Model | Native Peak | Container Peak | Overhead |
|-------|-------------|----------------|----------|
| DeepSeek-7B | 70.47 GiB | 101.30 GiB | **+30.83 GiB (44%)** |
| GPT-OSS-120B | 71.72 GiB | 93.43 GiB | **+21.71 GiB (30%)** |
| Qwen2.5-72B | 70.03 GiB | 90.02 GiB | **+19.99 GiB (29%)** |

### KV Cache Allocation

| Model | Native KV | Container KV | Difference |
|-------|-----------|--------------|------------|
| DeepSeek-7B | 44.31 GiB | 16.57 GiB | **-27.75 GiB (2.7x less in container)** |
| GPT-OSS-120B | 43.19 GiB | 23.65 GiB | **-19.54 GiB (1.8x less)** |
| Qwen2.5-72B | 44.71 GiB | 26.72 GiB | **-17.99 GiB (1.7x less)** |

### Performance (Throughput)

| Model | Native | Container | Difference |
|-------|--------|-----------|------------|
| DeepSeek-7B | 119.79 tok/s | 119.40 tok/s | **0.3%** |
| GPT-OSS-120B | 120.26 tok/s | 120.41 tok/s | **-0.1%** |
| Qwen2.5-72B | 119.33 tok/s | 119.51 tok/s | **-0.2%** |

**Identical performance** - differences are within measurement error (σ < 0.6 tokens/sec).

---

## 🔬 Why This Happens: Grace Hopper Unified Memory

### The Problem

Grace Hopper systems have **unified memory** (119.64 GB shared between CPU and GPU). Docker's cgroups were designed for discrete GPU systems and don't understand this architecture:

```
Traditional GPU System          Grace Hopper System
┌──────────┐  ┌──────────┐     ┌──────────────────────────┐
│ CPU RAM  │  │ GPU VRAM │     │  UNIFIED MEMORY (119 GB) │
│  (DDR)   │  │  (HBM)   │     │  CPU ←→ GPU (coherent)   │
└──────────┘  └──────────┘     └──────────────────────────┘
Docker manages CPU only        Docker sees ALL memory
GPU is outside cgroups         GPU allocations double-counted
```

**Result**: Docker's cgroup memory accounting **double-counts** GPU allocations as "container RAM", creating 20-30 GB overhead.

**Native execution avoids this** by running in the same memory namespace as the host.

---

## 🚀 Quick Start

### Prerequisites

- NVIDIA DGX Spark (Grace Hopper GB10)
- TensorRT-LLM container or rootfs
- CUDA 12.9+
- Python 3.12

### Running Benchmarks

```bash
# Clone repository
git clone https://github.com/USER/benchmark-spark
cd benchmark-spark

# Run single native benchmark
./scripts/run_native_benchmark.sh trtllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --model-path /data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Run single container benchmark
./scripts/run_container_benchmark.sh trtllm \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --model-path /data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Run comprehensive suite (60 runs, ~15-20 hours)
./scripts/run_comprehensive_benchmark_sudo.sh
```

### View Results

```bash
# Quick summary
./scripts/quick_summary.sh

# Open GitHub Pages locally
open docs/index.html

# Read full analysis
cat ANALYSIS.md
```

---

## 📂 Repository Structure

```
benchmark-spark/
├── ANALYSIS.md                      # Comprehensive analysis report
├── README.md                        # This file (Phase 1)
├── docs/
│   └── index.html                   # GitHub Pages site with charts
├── scripts/
│   ├── run_comprehensive_benchmark.sh      # Main orchestrator
│   ├── run_native_benchmark.sh            # Native execution
│   ├── run_container_benchmark.sh         # Container execution
│   └── quick_summary.sh                   # Progress monitoring
├── benchmarks/
│   └── trtllm_benchmark.py                # TensorRT-LLM wrapper
├── results/
│   └── comprehensive/
│       ├── *_metadata.json               # 60 run metadata files
│       └── *.log                         # 60 full benchmark logs
└── analysis/
    └── all_benchmark_results.json        # Processed results
```

---

## 💡 Recommendations

### For DGX Spark (Grace Hopper) Production

**Use Native/Chroot** when:
- ✅ Running large models (>10B parameters)
- ✅ Memory efficiency is critical
- ✅ Need maximum KV cache for throughput
- ✅ Operating in trusted environment

**Use Docker** when:
- ✅ Model is small (<10B) and overhead is acceptable
- ✅ Strong isolation is mandatory
- ✅ Deployment convenience is priority
- ✅ Multi-tenancy with strict boundaries

### For Other Hardware

⚠️ **This finding is specific to Grace Hopper unified memory**. Traditional discrete GPU systems (H100, A100) should not exhibit this overhead pattern because GPU VRAM is outside Docker's cgroups.

---

## 🔬 Phase 2: Next Steps

Phase 1 identified the problem. Phase 2 will investigate solutions:

1. **Test cgroup-level workarounds** (`--memory=unlimited`, `--cgroup-parent=/`)
2. **Evaluate systemd-nspawn** as lighter alternative to Docker
3. **Update nvidia-container-toolkit** and test for Grace Hopper optimizations
4. **Benchmark on discrete GPU** (H100) to confirm this is Grace-specific
5. **Investigate cgroup v2** unified hierarchy improvements

**Phase 2 Repository**: Coming soon - will be separate repo to keep this one focused and clean.

---

## 🛠️ Test Environment

### Hardware
- **System**: NVIDIA DGX Spark (ProMax GB10)
- **GPU**: NVIDIA GB10 (Grace Hopper, Compute Capability 12.1)
- **Memory**: 119.64 GB unified (shared CPU+GPU)
- **CPU**: ARM Cortex-X925 (10 cores) + Cortex-A725 (10 cores)

### Software
- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 6.11.0-1016-nvidia
- **TensorRT-LLM**: `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`
- **CUDA**: 12.9
- **Docker**: 27.x with nvidia-container-runtime
- **Python**: 3.12

See [VERSIONS.md](VERSIONS.md) for complete dependency list.

---

## 📖 Documentation

- **[Full Analysis Report](ANALYSIS.md)** - Detailed findings and methodology
- **[Interactive Results](https://USER.github.io/benchmark-spark/)** - Charts and visualizations
- **[Versions](VERSIONS.md)** - Complete software dependency list

---

## 📝 Citation

If you use this benchmark in your research or production decisions, please cite:

```bibtex
@misc{dgx-spark-benchmark-2025,
  title={DGX Spark TensorRT-LLM Benchmarking: Container vs Native Execution},
  author={Your Name},
  year={2025},
  url={https://github.com/USER/benchmark-spark}
}
```

---

## 🤝 Contributing

This is Phase 1 - analysis is complete. For Phase 2 (cgroup investigation), contributions will be welcome in the separate repository.

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- NVIDIA for DGX Spark hardware and TensorRT-LLM framework
- HuggingFace for model hosting
- The open-source LLM community

---

**Status**: Phase 1 Complete ✅ | **Next**: Phase 2 Docker Cgroup Investigation →
