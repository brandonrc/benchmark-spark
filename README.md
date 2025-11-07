# DGX Spark Performance Benchmarking: Container vs Native

## Overview

This repository contains benchmarking tools to measure and compare GPU performance between containerized and native execution environments on NVIDIA DGX Spark systems. The primary focus is quantifying the performance overhead of Docker containers when running TensorRT-LLM workloads.

## Motivation

DGX Spark systems are experiencing significantly lower performance (~50% of expected) when using NVIDIA's development container `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`. This benchmark suite aims to:

1. Quantify the performance difference between containerized and native execution
2. Identify whether container overhead is the primary cause of performance degradation
3. Provide reproducible benchmarks for performance validation
4. Generate data for community sharing and NVIDIA feedback

## Background Research

Based on ["Benchmarking GPU Passthrough Performance on Docker for AI Cloud System"](docs/reference_paper.md), prior research showed:
- **Native execution**: Faster execution time (1.52s avg) with lower GPU utilization (45.6%)
- **Docker containers**: Slower execution time (2.55s avg) but higher GPU utilization (86.2%)
- **Trade-off**: Container overhead vs. consistent environment and resource allocation

However, that study used consumer-grade GPUs (RTX 3060) with simple matrix multiplication. Our focus is enterprise DGX hardware with production LLM workloads.

## Test Methodology

### Environments

#### Container Environment
- **Image**: `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` (dev branch)
- **Runtime**: NVIDIA Container Toolkit
- **GPU Access**: `--gpus all` flag

#### Native Environment
- **Installation**: TensorRT-LLM built from source
- **CUDA**: Direct CUDA toolkit access
- **Dependencies**: System-level Python environment

### Workloads

We focus on realistic LLM inference scenarios using TensorRT-LLM:

1. **Standard Inference Benchmark**
   - Model: Llama-2-7B (or equivalent)
   - Batch sizes: 1, 4, 16, 32
   - Sequence lengths: 128, 512, 2048 tokens
   - Input/output split: 512 in / 128 out

2. **Throughput Testing**
   - Sustained inference over 100 iterations
   - Warm-up: 10 iterations
   - Measurements: tokens/second

### Metrics

**Primary Metrics:**
1. **Throughput**: Tokens per second (input + output)
2. **Latency**: Time per inference (P50, P95, P99)
3. **GPU Utilization**: Percentage from nvidia-smi

**Secondary Metrics:**
4. GPU Memory Usage
5. GPU Temperature
6. Power Consumption

## Repository Structure

```
benchmark-spark/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── scripts/
│   ├── setup_native.sh               # Native TensorRT-LLM installation
│   ├── run_container_benchmark.sh    # Container test runner
│   ├── run_native_benchmark.sh       # Native test runner
│   └── monitor_gpu.sh                # GPU monitoring script
├── benchmarks/
│   ├── llm_inference.py              # Main LLM benchmark
│   ├── simple_matmul.py              # Simple matrix multiply (sanity check)
│   └── config.yaml                   # Test configurations
├── analysis/
│   ├── compare_results.py            # Statistical analysis
│   ├── visualize.py                  # Generate charts
│   └── report_generator.py           # Markdown report generation
├── results/
│   ├── container/                    # Container test results
│   ├── native/                       # Native test results
│   └── comparison/                   # Comparative analysis outputs
└── docs/
    ├── setup.md                      # Detailed setup instructions
    ├── methodology.md                # Full methodology documentation
    ├── results.md                    # Published results
    └── reference_paper.md            # Summary of reference paper
```

## Quick Start

### Prerequisites

- NVIDIA DGX Spark system
- CUDA 12.x and compatible drivers
- Docker with NVIDIA Container Toolkit
- Python 3.10+

### Running Container Benchmark

```bash
# Pull the container
docker pull nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev

# Run benchmark
./scripts/run_container_benchmark.sh
```

### Running Native Benchmark

```bash
# Install TensorRT-LLM (first time only)
./scripts/setup_native.sh

# Run benchmark
./scripts/run_native_benchmark.sh
```

### Comparing Results

```bash
# Generate comparison report
python analysis/compare_results.py

# View results
cat results/comparison/report.md
```

## Results

Results will be published to [GitHub Pages](https://github.com/YOUR_USERNAME/benchmark-spark/pages) as they become available.

### Expected Outcomes

- **If container overhead < 10%**: Performance issue likely elsewhere (drivers, configuration, hardware)
- **If container overhead 20-40%**: Container configuration or dev branch optimization issue
- **If container overhead > 40%**: Dev container may be fundamentally broken

## Contributing

This is primarily a diagnostic tool, but contributions are welcome:
- Additional benchmark workloads
- Optimization suggestions
- Bug fixes
- Documentation improvements

## License

MIT License - See LICENSE file

## References

1. Sani, A. F., et al. (2025). "Benchmarking GPU Passthrough Performance on Docker for AI Cloud System." *Brilliance: Research of Artificial Intelligence*, 5(2).
2. NVIDIA TensorRT-LLM Documentation
3. NVIDIA Container Toolkit Documentation

## Contact

For questions or issues with DGX Spark performance, please open an issue in this repository.
