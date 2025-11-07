# Benchmarking Methodology

## Objective

Quantify the performance overhead introduced by Docker containerization when running TensorRT-LLM workloads on NVIDIA DGX Spark systems, and determine if containerization is responsible for observed ~50% performance degradation.

## Hypothesis

The development container `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` introduces significant overhead that accounts for a substantial portion of the performance gap between observed and expected throughput.

## Experimental Design

### Independent Variable
- **Execution Environment**: Container vs. Native

### Dependent Variables
1. **Throughput** (tokens/second) - Primary metric
2. **Latency** (milliseconds) - Per-inference timing
3. **GPU Utilization** (percentage) - Hardware usage efficiency

### Controlled Variables
- GPU hardware (same GPU for all tests)
- Model architecture and size
- Input/output token lengths
- Batch sizes
- Software versions (CUDA, drivers, TensorRT-LLM)
- System load (isolated testing environment)
- GPU clock speeds (locked to consistent state)
- Power mode (maximum performance)

## Test Environments

### Container Environment

**Container Image:**
```
nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
```

**Docker Configuration:**
```bash
docker run --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd)/results:/results \
  [container_image] \
  python /benchmarks/llm_inference.py
```

**Rationale:**
- `--gpus all`: Full GPU passthrough
- `--ipc=host`: Shared memory for performance
- `--ulimit memlock=-1`: Unlimited locked memory for CUDA
- `--ulimit stack=67108864`: Large stack for deep networks

### Native Environment

**Installation:**
- TensorRT-LLM compiled from source
- System-level CUDA toolkit (12.x)
- Python 3.10 virtual environment

**Environment Variables:**
```bash
export CUDA_VISIBLE_DEVICES=0
export TRT_ENGINE_CACHE=/tmp/trt_cache
```

## Workload Specifications

### LLM Inference Benchmark

**Model Selection:**
- Primary: Llama-2-7B or similar 7B parameter model
- Fallback: GPT-J-6B if Llama unavailable
- Engine: TensorRT-LLM optimized engine

**Test Configurations:**

| Test ID | Batch Size | Input Length | Output Length | Total Tokens |
|---------|------------|--------------|---------------|--------------|
| T1      | 1          | 128          | 32            | 160          |
| T2      | 1          | 512          | 128           | 640          |
| T3      | 1          | 2048         | 512           | 2560         |
| T4      | 4          | 512          | 128           | 2560         |
| T5      | 16         | 512          | 128           | 10240        |
| T6      | 32         | 512          | 128           | 20480        |

**Iterations:**
- Warm-up runs: 10 (excluded from measurements)
- Measurement runs: 100
- Cool-down between tests: 30 seconds

**Rationale:**
- Warm-up eliminates GPU clock ramp-up effects
- 100 iterations provide statistical significance
- Cool-down prevents thermal throttling

### Simple Matrix Multiplication (Sanity Check)

**Purpose:** Validate that basic CUDA operations work correctly

**Configuration:**
- Matrix size: 10,000 x 10,000 (matching reference paper)
- Precision: FP32
- Iterations: 10
- Framework: TensorFlow 2.x

## Measurement Procedures

### Throughput Measurement

```python
import time

# Warm-up phase
for _ in range(10):
    model.generate(inputs)

# Measurement phase
start_time = time.perf_counter()
total_tokens = 0

for _ in range(100):
    outputs = model.generate(inputs)
    total_tokens += count_tokens(outputs)

end_time = time.perf_counter()
throughput = total_tokens / (end_time - start_time)
```

### GPU Utilization Monitoring

Run concurrently during benchmark:

```bash
nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,\
utilization.memory,memory.total,memory.used,memory.free,\
temperature.gpu,power.draw,clocks.sm,clocks.mem \
--format=csv -l 1 > gpu_metrics.csv
```

**Sampling rate:** 1 second (0.1 second for short runs)

### Latency Measurement

```python
latencies = []
for _ in range(100):
    start = time.perf_counter()
    output = model.generate(input)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # Convert to ms

p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
p99 = np.percentile(latencies, 99)
```

## Statistical Analysis

### Descriptive Statistics

For each metric (throughput, latency, GPU utilization):
- Mean (μ)
- Standard deviation (σ)
- Min/Max
- Percentiles (P50, P95, P99)

### Comparative Analysis

**Overhead Calculation:**
```
Overhead % = ((T_container - T_native) / T_native) × 100
```

Where T = execution time (inverse of throughput)

**Statistical Significance:**
- Independent t-test (α = 0.05)
- Null hypothesis: No difference between container and native
- Alternative hypothesis: Significant performance difference exists

**Confidence Intervals:**
- 95% confidence intervals for all mean measurements
- Bootstrap resampling if sample size is small

### Visualization

1. **Box plots**: Distribution comparison across environments
2. **Line charts**: Throughput vs. batch size
3. **Heatmaps**: GPU utilization over time
4. **Bar charts**: Mean comparison with error bars

## Quality Assurance

### Pre-Test Validation

1. **GPU Health Check:**
   ```bash
   nvidia-smi -q | grep -A 5 "GPU Current Temp"
   nvidia-smi -q | grep -A 5 "Performance State"
   ```
   - Verify GPU temperature < 80°C
   - Confirm Performance State = P0 (max performance)

2. **System Load Check:**
   ```bash
   top -bn1 | grep "Cpu(s)"
   ```
   - Ensure < 10% CPU usage before tests

3. **Memory Availability:**
   ```bash
   nvidia-smi --query-gpu=memory.free --format=csv,noheader
   ```
   - Confirm > 90% GPU memory available

### During Test Monitoring

- Monitor for thermal throttling
- Check for system warnings/errors
- Verify consistent GPU clock speeds
- Watch for memory leaks

### Post-Test Validation

1. **Data Integrity:**
   - Check for missing measurements
   - Verify CSV formatting
   - Confirm iteration counts

2. **Outlier Detection:**
   - Flag measurements > 3 standard deviations from mean
   - Investigate outlier causes
   - Re-run if > 5% outliers detected

## Reproducibility

### Environment Documentation

All tests will document:
- CUDA version
- Driver version
- TensorRT-LLM version/commit
- Container image digest
- Python package versions (`pip freeze`)
- GPU model and firmware
- System specs (CPU, RAM, OS)

### Random Seeds

All workloads use fixed random seeds:
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### Automation

All tests are scripted to eliminate manual intervention and ensure consistency.

## Limitations

1. **Single-GPU Focus:** Multi-GPU scenarios not covered in initial testing
2. **Model Constraints:** Limited to models that fit in single GPU memory
3. **Workload Specificity:** Results specific to LLM inference; may not generalize to training
4. **Hardware Specificity:** Results specific to DGX Spark architecture

## Expected Timeline

- **Setup & Validation:** 1 day
- **Container Benchmarking:** 2-4 hours
- **Native Setup & Benchmarking:** 1-2 days (includes TensorRT-LLM build)
- **Analysis & Reporting:** 4-8 hours

**Total:** 3-4 days for complete benchmark suite

## Success Criteria

1. **Data Quality:** < 5% coefficient of variation in repeated measurements
2. **Statistical Power:** Sufficient sample size to detect 10% performance difference
3. **Reproducibility:** Results within 5% when re-run on same hardware
4. **Completeness:** All planned test configurations executed successfully

## References

1. Sani, A. F., et al. (2025). "Benchmarking GPU Passthrough Performance on Docker for AI Cloud System."
2. NVIDIA TensorRT-LLM Performance Best Practices
3. Docker Performance Tuning Guide
