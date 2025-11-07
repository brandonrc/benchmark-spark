#!/bin/bash
#
# Container Benchmark Runner
# Runs benchmarks inside NVIDIA's TensorRT-LLM container for DGX Spark
#

set -e  # Exit on error

# Configuration
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/container}"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Container Benchmark Runner"
echo "========================================"
echo "Container: ${CONTAINER_IMAGE}"
echo "Results dir: ${RESULTS_DIR}"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check if nvidia-docker runtime is available
if ! docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA Docker runtime not working. Check nvidia-container-toolkit installation.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and NVIDIA runtime OK${NC}"

# Pull container if not already present
echo ""
echo "Checking container image..."
if ! docker image inspect "${CONTAINER_IMAGE}" &> /dev/null; then
    echo "Pulling container image (this may take a while)..."
    docker pull "${CONTAINER_IMAGE}"
else
    echo -e "${GREEN}✓ Container image already present${NC}"
fi

# Start GPU monitoring in background
MONITOR_PID=""
function start_gpu_monitor() {
    local output_file="$1"
    echo "Starting GPU monitor (output: ${output_file})..."

    nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,\
memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem \
        --format=csv -l "${GPU_MONITOR_INTERVAL}" > "${output_file}" &

    MONITOR_PID=$!
    echo -e "${GREEN}✓ GPU monitor started (PID: ${MONITOR_PID})${NC}"
}

function stop_gpu_monitor() {
    if [ -n "${MONITOR_PID}" ]; then
        echo "Stopping GPU monitor..."
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
        echo -e "${GREEN}✓ GPU monitor stopped${NC}"
    fi
}

# Ensure monitor is stopped on exit
trap stop_gpu_monitor EXIT

# Function to run benchmark in container
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/gpu_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Run benchmark in container
    echo "Starting container benchmark..."

    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="/workspace/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "llm" ]; then
        benchmark_script="/workspace/benchmarks/llm_inference.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        return 1
    fi

    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${BENCHMARK_DIR}:/workspace" \
        -v "${RESULTS_DIR}:/results" \
        "${CONTAINER_IMAGE}" \
        python "${benchmark_script}" \
            --environment container \
            --output /results \
            ${extra_args}

    local exit_code=$?

    # Stop monitoring
    stop_gpu_monitor

    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ ${benchmark_type} benchmark completed successfully${NC}"
    else
        echo -e "${RED}✗ ${benchmark_type} benchmark failed with exit code ${exit_code}${NC}"
        return ${exit_code}
    fi

    echo "GPU metrics saved to: ${monitor_file}"
}

# Main execution
main() {
    local benchmark_type="${1:-all}"

    case "${benchmark_type}" in
        matmul)
            run_benchmark "matmul"
            ;;
        llm)
            # Check if config file exists
            local config_arg=""
            if [ -f "${BENCHMARK_DIR}/benchmarks/config.yaml" ]; then
                config_arg="--config /workspace/benchmarks/config.yaml"
            fi
            run_benchmark "llm" "${config_arg}"
            ;;
        all)
            echo "Running all benchmarks..."
            run_benchmark "matmul"
            sleep 10  # Brief pause between benchmarks
            run_benchmark "llm"
            ;;
        *)
            echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
            echo "Usage: $0 [matmul|llm|all]"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================"
    echo "  All Container Benchmarks Complete"
    echo "========================================"
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Run native benchmarks: ./scripts/run_native_benchmark.sh"
    echo "  2. Compare results: python analysis/compare_results.py"
}

main "$@"
