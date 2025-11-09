# Software Versions and Dependencies

This document lists all software versions used in the Phase 1 benchmarking study.

## Hardware

- **System**: NVIDIA DGX Spark (ProMax GB10-ED89)
- **GPU**: NVIDIA GB10 (Grace Hopper)
  - Compute Capability: 12.1
  - Architecture: Integrated (unified memory)
  - Total Memory: 119.64 GB (shared CPU+GPU)
- **CPU**: ARM Cortex (aarch64)
  - Cortex-X925: 10 cores
  - Cortex-A725: 10 cores
  - Total: 20 cores, 1 NUMA node
- **System Memory**: 119.6 GiB total

## Operating System

- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 6.11.0-1016-nvidia
- **Architecture**: aarch64 (ARM64)

## CUDA and GPU Drivers

- **NVIDIA Driver**: 580.95.05
- **CUDA Runtime**: 12.9
  - Path: `/usr/local/cuda-12.9`
- **cuDNN**: 9.10.2.21
- **TensorRT**: 10.11.0.33
- **NCCL**: 2.24.4
- **NVSHMEM**: 3.2.5

## Container Runtime

- **Docker**: 27.x
- **Container Runtime**: nvidia-container-runtime
- **Default Runtime**: runc
- **containerd**: 753481ec61c7c8955a23d6ff7bc8e4daed455734
- **runc**: v1.2.5-0-g59923ef

## TensorRT-LLM Environment

- **Container Image**: `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev`
- **TensorRT-LLM**: Version from spark-single-gpu-dev branch
- **Python**: 3.12
- **PyTorch**: 2.x (from container)
- **Transformers**: Latest from container

## Python Dependencies (from container)

```
tensorrt-llm (version from container)
torch>=2.0
transformers
datasets
pynvml
numpy
```

## Benchmark Configuration

- **Framework**: TensorRT-LLM (`trtllm-bench` CLI)
- **Benchmark Type**: Throughput
- **Requests per Run**: 50
- **Output Tokens**: 128 per request
- **Max Batch Size**: Varies by model
- **KV Cache Fraction**: 0.9 (90% of available memory)

## Docker Run Configuration

```bash
docker run --rm \
    --gpus all \
    --ipc=host \
    --shm-size=60g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v /home/khan/benchmark-spark:/workspace \
    -v /data/models/huggingface:/models:ro \
    nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

## Native (Chroot) Configuration

- **Chroot Root**: `/home/khan/container-rootfs`
- **Mounted Filesystems**:
  - `/proc` → host proc
  - `/sys` → host sys (rbind)
  - `/dev` → host dev (rbind)
  - `/home` → host /home
  - `/tmp` → host /tmp
  - `/data` → host /data
- **LD_LIBRARY_PATH**: Container libraries prioritized
- **CUDA_HOME**: `/usr/local/cuda-12.9`
- **MPI**: HPC-X OpenMPI from `/opt/hpcx/ompi`

## Models Tested

### DeepSeek-R1-Distill-Qwen-7B
- **Source**: HuggingFace (`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`)
- **Parameters**: 7 billion
- **Precision**: bfloat16
- **Path**: `/data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

### GPT-OSS-120B
- **Source**: HuggingFace (`openai/gpt-oss-120b`)
- **Parameters**: 120 billion (5.1B active with MoE)
- **Precision**: MXFP4 quantization
- **Path**: `/data/models/huggingface/openai/gpt-oss-120b`

### Qwen2.5-72B-Instruct
- **Source**: HuggingFace (`Qwen/Qwen2.5-72B-Instruct`)
- **Parameters**: 72 billion
- **Precision**: bfloat16
- **Path**: `/data/models/huggingface/Qwen/Qwen2.5-72B-Instruct`

## Monitoring Tools

- **GPU Monitoring**: nvidia-smi
- **System Monitoring**: Custom script (`monitor_memory.sh`)
- **Metrics Collected**:
  - Peak memory usage (torch + non-torch)
  - KV cache allocation
  - GPU temperature
  - Request throughput
  - Token throughput
  - Latency (average, P50, P99)

## Test Date

- **Start**: November 8, 2025, 19:00 UTC
- **End**: November 9, 2025, 09:15 UTC
- **Duration**: ~14 hours for 60 runs
- **Cooldown**: 5 minutes + temperature check (<45°C) between runs

## Version Collection Commands

To reproduce the version information:

```bash
# System info
lsb_release -a
uname -a
lscpu

# GPU info
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

# CUDA info
nvcc --version
cat /usr/local/cuda/version.txt

# Docker info
docker --version
docker info

# Python and packages (in container)
python3 --version
pip list | grep -E "torch|tensor|transform"
```

## Reproducibility Notes

1. **Container determinism**: The `spark-single-gpu-dev` tag may change. For exact reproducibility, use the specific image digest.

2. **Model weights**: Model checkpoints may be updated by providers. Download and freeze specific versions for reproducibility.

3. **Temperature variation**: GPU temperature affects performance slightly. We used 5-minute cooldown + temperature monitoring to minimize thermal throttling.

4. **System load**: Ensure no other GPU workloads are running during benchmarking.

---

Last updated: November 9, 2025
