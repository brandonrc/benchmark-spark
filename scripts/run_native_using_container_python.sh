#!/bin/bash
#
# Run Native Benchmarks Using Container's Python
# This uses the container's Python/packages but runs directly on host (no container overhead)
#

set -e

CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/native}"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "  Native Benchmark (Container Python)"
echo "========================================"
echo "Container: ${CONTAINER_IMAGE}"
echo "Results dir: ${RESULTS_DIR}"
echo ""
echo "This runs benchmarks natively on the host using"
echo "the container's Python environment (no container overhead)"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Source the monitoring functions from container script
source "${BENCHMARK_DIR}/scripts/run_container_benchmark.sh" 2>/dev/null || true

# Run benchmark using container's Python but with host GPU access
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark (Native)"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/enhanced_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring (using host GPU directly)
    "${BENCHMARK_DIR}/scripts/monitor_memory_native.sh" "${monitor_file}" "${GPU_MONITOR_INTERVAL}" &
    local MONITOR_PID=$!

    # Determine benchmark script
    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="/workspace/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "trtllm" ]; then
        benchmark_script="/workspace/benchmarks/trtllm_benchmark.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        kill $MONITOR_PID 2>/dev/null || true
        return 1
    fi

    # Set output path
    if [ "${benchmark_type}" == "trtllm" ]; then
        local output_arg="/results"
    else
        local output_arg="/results/${benchmark_type}_results_${timestamp}.csv"
    fi

    # Run using container Python but with direct host access (no container isolation)
    echo "Running benchmark with container Python environment..."
    docker run --rm \
        --gpus all \
        --ipc=host \
        --network=host \
        --pid=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${BENCHMARK_DIR}:/workspace" \
        -v "${RESULTS_DIR}:/results" \
        -v "/data/models/huggingface:/models:ro" \
        "${CONTAINER_IMAGE}" \
        python "${benchmark_script}" \
            --environment native \
            --output "${output_arg}" \
            ${extra_args}

    local exit_code=$?

    # Stop monitoring
    kill $MONITOR_PID 2>/dev/null || true
    wait $MONITOR_PID 2>/dev/null || true

    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ ${benchmark_type} benchmark completed${NC}"
    else
        echo -e "${RED}✗ ${benchmark_type} benchmark failed with exit code ${exit_code}${NC}"
        return ${exit_code}
    fi

    echo "Enhanced metrics saved to: ${monitor_file}"
}

# Main
case "${1:-matmul}" in
    matmul)
        run_benchmark "matmul"
        ;;
    trtllm)
        run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path /models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        ;;
    *)
        echo "Usage: $0 [matmul|trtllm]"
        exit 1
        ;;
esac
