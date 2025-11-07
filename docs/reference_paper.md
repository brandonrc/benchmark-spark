# Reference Paper Summary

## Benchmarking GPU Passthrough Performance on Docker for AI Cloud System

**Authors:** Ahmad Faisal Sani, et al.
**Published:** Brilliance: Research of Artificial Intelligence, Vol. 5, No. 2, 2025
**DOI:** https://doi.org/10.47709/brilliance.v5i2.6794

## Key Findings

### Methodology
- **Hardware:** NVIDIA RTX 3060 (12GB VRAM), Intel Core i5-12400, 32GB RAM
- **OS:** Ubuntu 22.04 LTS
- **Workload:** 10,000 x 10,000 matrix multiplication using TensorFlow
- **Iterations:** 10 runs per environment
- **Environments:** Native vs. Docker (nvcr.io/nvidia/tensorflow:24.05-tf2-py3)

### Results Summary

| Metric | Native | Docker |
|--------|--------|--------|
| **Avg Execution Time** | 1.52s | 2.55s |
| **Avg GPU Utilization** | 45.6% | 86.2% |
| **Performance Overhead** | Baseline | +67.8% slower |

### Key Insights

1. **Native Execution**
   - ✅ Faster execution (1.52s average)
   - ❌ Lower GPU utilization (45.6%)
   - Direct hardware access
   - No containerization overhead

2. **Docker Execution**
   - ❌ Slower execution (2.55s average)
   - ✅ Higher GPU utilization (86.2%)
   - Better resource allocation
   - Containerization layer overhead

3. **Trade-offs**
   - **Performance:** Native wins for raw speed
   - **Efficiency:** Docker achieves better GPU saturation
   - **Consistency:** Docker provides reproducible environments
   - **Overhead:** ~67% increase in execution time for Docker

### Interpretation

The study reveals a fundamental trade-off:

- **Docker's higher GPU utilization** doesn't translate to faster execution due to:
  - Container runtime overhead
  - Memory management inefficiencies
  - Inter-process communication costs
  - Virtualization layer latency

- **Native's lower GPU utilization** still achieves faster execution because:
  - Direct memory access
  - No virtualization overhead
  - Optimized kernel launches
  - Better memory locality

### Limitations of Study

1. **Consumer-grade GPU:** RTX 3060 vs. enterprise GPUs (DGX systems)
2. **Simple workload:** Matrix multiplication doesn't represent complex AI workloads
3. **Small sample size:** Only 10 iterations
4. **Single framework:** TensorFlow only
5. **Generic container:** Not optimized for specific hardware

## Relevance to DGX Spark Testing

### Expected Differences

**DGX Spark advantages:**
- Enterprise-grade GPUs (A100/H100)
- NVLink interconnect
- ECC memory
- Fabric Manager for multi-GPU
- Optimized drivers

**Container improvements:**
- Hardware-specific containers (`spark-single-gpu-dev`)
- Better GPU passthrough on enterprise hardware
- Potentially lower overhead

### Our Enhanced Approach

1. **Real-world workloads:** LLM inference vs. simple matrix multiplication
2. **More iterations:** 100+ runs for statistical significance
3. **Multiple configurations:** Various batch sizes and sequence lengths
4. **Comprehensive metrics:** Throughput, latency, percentiles
5. **Statistical analysis:** T-tests, confidence intervals
6. **Hardware-specific:** Testing actual production container

### Expected Outcomes

**Hypothesis 1:** Container overhead on DGX Spark < 20%
- Enterprise hardware reduces virtualization costs
- Optimized containers minimize overhead
- Better PCIe/NVLink bandwidth

**Hypothesis 2:** Container overhead explains < 50% of performance gap
- If we see ~50% degradation and container adds only 10-20%
- Remaining 30-40% is elsewhere (drivers, config, thermal, etc.)

**Hypothesis 3:** LLM workloads show different patterns than matmul
- More memory-bound
- Different kernel characteristics
- Tensor Core utilization

## Comparison Matrix

| Aspect | Reference Paper | Our DGX Spark Study |
|--------|----------------|---------------------|
| **Hardware** | RTX 3060 (Consumer) | DGX Spark (Enterprise) |
| **Workload** | Matrix Multiply | LLM Inference |
| **Framework** | TensorFlow | TensorRT-LLM |
| **Container** | Generic TF | Spark-optimized |
| **Iterations** | 10 | 100+ |
| **Statistical** | Basic mean/std | T-tests, CI, percentiles |
| **Scope** | Proof of concept | Production diagnostic |

## Actionable Insights

Based on the reference paper, we should investigate:

1. **If overhead > 60%** like the paper:
   - Container is likely the problem
   - Configuration issue or dev branch broken

2. **If overhead 20-40%:**
   - Expected range for containers
   - Optimization opportunities exist

3. **If overhead < 10%:**
   - Performance issue is elsewhere
   - Check drivers, thermal, power limits

## Citations

```bibtex
@article{sani2025benchmarking,
  title={Benchmarking GPU Passthrough Performance on Docker for AI Cloud System},
  author={Sani, Ahmad Faisal and Khoirunisa, Rifa and Riatma, Darmawan Lahru and Rachman, Yusuf Fadlila and Masbahah},
  journal={Brilliance: Research of Artificial Intelligence},
  volume={5},
  number={2},
  pages={713--718},
  year={2025},
  doi={10.47709/brilliance.v5i2.6794}
}
```

## Additional References

1. Shetty, J., et al. (2017). "An empirical performance evaluation of docker container, openstack virtual machine and bare metal server."
2. Openja, M., et al. (2022). "Studying the Practices of Deploying Machine Learning Projects on Docker."
3. NVIDIA TensorRT-LLM Documentation
4. Docker GPU Passthrough Best Practices
