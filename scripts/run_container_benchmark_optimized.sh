#!/bin/bash
#
# Container Benchmark Runner - OPTIMIZED CONFIGURATION
# Tests various Docker optimization flags to minimize containerization overhead
# Phase 2: Docker Optimization Experiments
#

set -e  # Exit on error

# Configuration
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/container_optimized}"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-/data/models/huggingface}"

# Optimization configuration selection
OPTIMIZATION_MODE="${OPTIMIZATION_MODE:-best}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Container Benchmark Runner (OPTIMIZED)"
echo "========================================"
echo "Container: ${CONTAINER_IMAGE}"
echo "Results dir: ${RESULTS_DIR}"
echo "Optimization mode: ${OPTIMIZATION_MODE}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check if nvidia-docker runtime is available
if ! docker image inspect "${CONTAINER_IMAGE}" &> /dev/null; then
    echo "Pulling container image..."
    docker pull "${CONTAINER_IMAGE}"
fi

if ! docker run --rm --gpus all "${CONTAINER_IMAGE}" nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA Docker runtime not working.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and NVIDIA runtime OK${NC}"

# Start enhanced monitoring in background
MONITOR_PID=""
GPU_MONITOR_PID=""
function start_gpu_monitor() {
    local output_file="$1"

    # Capture baseline metrics
    local baseline_file="${output_file%.csv}_baseline.txt"
    echo "Capturing baseline memory metrics..."
    echo "=== Baseline Memory Metrics ===" > "${baseline_file}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "${baseline_file}"
    echo "" >> "${baseline_file}"

    # System memory baseline
    echo "--- System Memory (from /proc/meminfo) ---" >> "${baseline_file}"
    grep -E "MemTotal|MemFree|MemAvailable|Cached|Buffers|Shmem" /proc/meminfo >> "${baseline_file}"
    echo "" >> "${baseline_file}"

    # GPU baseline
    echo "--- GPU Metrics (from nvidia-smi) ---" >> "${baseline_file}"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv >> "${baseline_file}" 2>&1
    echo "" >> "${baseline_file}"

    echo -e "${GREEN}✓ Baseline metrics captured: ${baseline_file}${NC}"

    # Start continuous monitoring
    echo "Starting enhanced GPU and memory monitor..."

    "${BENCHMARK_DIR}/scripts/monitor_memory.sh" "${output_file}" "${GPU_MONITOR_INTERVAL}" &
    MONITOR_PID=$!

    local gpu_basic_file="${output_file%.csv}_gpu_basic.csv"
    nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,\
memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem \
        --format=csv -l "${GPU_MONITOR_INTERVAL}" > "${gpu_basic_file}" &
    GPU_MONITOR_PID=$!

    echo -e "${GREEN}✓ Monitors started (PIDs: ${MONITOR_PID}, ${GPU_MONITOR_PID})${NC}"
}

function stop_gpu_monitor() {
    if [ -n "${MONITOR_PID}" ]; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
    fi

    if [ -n "${GPU_MONITOR_PID}" ]; then
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        GPU_MONITOR_PID=""
    fi
}

trap stop_gpu_monitor EXIT

# Function to get Docker flags based on optimization mode
function get_docker_flags() {
    local mode="$1"
    local flags=""

    case "${mode}" in
        baseline)
            # NVIDIA official recommendation (no optimization)
            flags="--gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        shm-8g)
            # Add 8GB shared memory
            flags="--gpus all --ipc=host --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        shm-32g)
            # Add 32GB shared memory
            flags="--gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        shm-60g)
            # Add 60GB shared memory (Phase 1 configuration)
            flags="--gpus all --ipc=host --shm-size=60g --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        host-network)
            # Use host network mode
            flags="--gpus all --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        cpu-pinning)
            # Pin to performance cores (Cortex-X925: cores 0-9)
            flags="--gpus all --ipc=host --cpuset-cpus=0-9 --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        privileged)
            # Privileged mode (WARNING: reduces security)
            flags="--gpus all --ipc=host --privileged --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        best)
            # Best combination based on initial testing
            # TODO: Update after experimentation shows which optimizations work best
            flags="--gpus all --ipc=host --shm-size=32g --cpuset-cpus=0-9 --ulimit memlock=-1 --ulimit stack=67108864"
            ;;
        *)
            echo -e "${RED}ERROR: Unknown optimization mode: ${mode}${NC}"
            exit 1
            ;;
    esac

    echo "${flags}"
}

# Function to run benchmark with specific optimization
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"
    local opt_mode="${3:-${OPTIMIZATION_MODE}}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark"
    echo "  Optimization: ${opt_mode}"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/enhanced_metrics_${benchmark_type}_${opt_mode}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Get Docker flags for this optimization mode
    local docker_flags=$(get_docker_flags "${opt_mode}")

    echo "Docker flags: ${docker_flags}"

    # Run benchmark in container
    local benchmark_script=""
    if [ "${benchmark_type}" == "trtllm" ]; then
        benchmark_script="/workspace/benchmarks/trtllm_benchmark.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        return 1
    fi

    local output_arg="/results"

    # Build docker command dynamically
    docker run --rm ${docker_flags} \
        -v "${BENCHMARK_DIR}:/workspace" \
        -v "${RESULTS_DIR}:/results" \
        -v "${LOCAL_MODELS_DIR}:/models:ro" \
        "${CONTAINER_IMAGE}" \
        python "${benchmark_script}" \
            --environment "container_opt_${opt_mode}" \
            --output "${output_arg}" \
            ${extra_args}

    local exit_code=$?

    # Stop monitoring
    stop_gpu_monitor

    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ ${benchmark_type} benchmark (${opt_mode}) completed successfully${NC}"
    else
        echo -e "${RED}✗ ${benchmark_type} benchmark (${opt_mode}) failed with exit code ${exit_code}${NC}"
        return ${exit_code}
    fi

    echo "Baseline metrics: ${monitor_file%.csv}_baseline.txt"
    echo "Enhanced metrics: ${monitor_file}"
    echo ""
}

# Main execution
main() {
    local benchmark_type="${1:-trtllm}"
    local model_args="${2:-}"

    if [ "${OPTIMIZATION_MODE}" == "all" ]; then
        # Test all optimization modes
        echo "Testing all optimization configurations..."
        for mode in baseline shm-8g shm-32g shm-60g host-network cpu-pinning privileged best; do
            echo ""
            echo "========================================"
            echo "  Testing optimization: ${mode}"
            echo "========================================"
            run_benchmark "${benchmark_type}" "${model_args}" "${mode}"
            sleep 10  # Cooldown between tests
        done
    else
        # Test single optimization mode
        run_benchmark "${benchmark_type}" "${model_args}" "${OPTIMIZATION_MODE}"
    fi

    echo ""
    echo "========================================"
    echo "  Optimization Benchmarks Complete"
    echo "========================================"
    echo "Results saved to: ${RESULTS_DIR}"
}

main "$@"
