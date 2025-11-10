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
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-/data/models/huggingface}"

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
# First ensure the container image is available
if ! docker image inspect "${CONTAINER_IMAGE}" &> /dev/null; then
    echo "Pulling container image for runtime check..."
    docker pull "${CONTAINER_IMAGE}"
fi

if ! docker run --rm --gpus all "${CONTAINER_IMAGE}" nvidia-smi &> /dev/null; then
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

# Start enhanced monitoring in background (GPU + system memory)
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

    # Process memory baseline (top processes)
    echo "--- Top 10 Memory Consumers ---" >> "${baseline_file}"
    ps aux --sort=-%mem | head -n 11 >> "${baseline_file}"
    echo "" >> "${baseline_file}"

    echo -e "${GREEN}✓ Baseline metrics captured: ${baseline_file}${NC}"

    # Start continuous monitoring
    echo "Starting enhanced GPU and memory monitor (output: ${output_file})..."

    # Use the enhanced monitor script
    "${BENCHMARK_DIR}/scripts/monitor_memory.sh" "${output_file}" "${GPU_MONITOR_INTERVAL}" &
    MONITOR_PID=$!

    # Also keep the original GPU metrics for compatibility
    local gpu_basic_file="${output_file%.csv}_gpu_basic.csv"
    nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,\
memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem \
        --format=csv -l "${GPU_MONITOR_INTERVAL}" > "${gpu_basic_file}" &
    GPU_MONITOR_PID=$!

    echo -e "${GREEN}✓ Enhanced monitor started (PID: ${MONITOR_PID})${NC}"
    echo -e "${GREEN}✓ GPU basic monitor started (PID: ${GPU_MONITOR_PID})${NC}"
}

function stop_gpu_monitor() {
    if [ -n "${MONITOR_PID}" ]; then
        echo "Stopping enhanced monitor..."
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
        MONITOR_PID=""
        echo -e "${GREEN}✓ Enhanced monitor stopped${NC}"
    fi

    if [ -n "${GPU_MONITOR_PID}" ]; then
        echo "Stopping GPU basic monitor..."
        kill "${GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${GPU_MONITOR_PID}" 2>/dev/null || true
        GPU_MONITOR_PID=""
        echo -e "${GREEN}✓ GPU basic monitor stopped${NC}"
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
    local monitor_file="${RESULTS_DIR}/enhanced_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Run benchmark in container
    echo "Starting container benchmark..."

    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="/workspace/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "llm" ]; then
        benchmark_script="/workspace/benchmarks/llm_inference.py"
    elif [ "${benchmark_type}" == "trtllm" ]; then
        benchmark_script="/workspace/benchmarks/trtllm_benchmark.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        return 1
    fi

    local timestamp=$(date +%Y%m%d_%H%M%S)

    # Set output path based on benchmark type
    if [ "${benchmark_type}" == "trtllm" ]; then
        # trtllm benchmark creates its own files in the output directory
        local output_arg="/results"
    else
        # Other benchmarks need a specific file path
        local output_arg="/results/${benchmark_type}_results_${timestamp}.csv"
    fi

    # Docker configuration based on NVIDIA DGX Spark Playbooks
    # Reference: https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nvfp4-quantization/README.md
    # Phase 2: Using NVIDIA-recommended baseline configuration
    docker run --rm \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v "${BENCHMARK_DIR}:/workspace" \
        -v "${RESULTS_DIR}:/results" \
        -v "${LOCAL_MODELS_DIR}:/models:ro" \
        "${CONTAINER_IMAGE}" \
        python "${benchmark_script}" \
            --environment container \
            --output "${output_arg}" \
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

    echo "Baseline metrics saved to: ${monitor_file%.csv}_baseline.txt"
    echo "Enhanced metrics saved to: ${monitor_file}"
    echo "GPU basic metrics saved to: ${monitor_file%.csv}_gpu_basic.csv"
    echo ""
    echo "Note: System memory usage includes baseline + benchmark container allocation"
    echo "      Check baseline file to estimate benchmark-specific memory usage"
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
        trtllm)
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path /models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        all)
            echo "Running all benchmarks..."
            run_benchmark "matmul"
            sleep 10  # Brief pause between benchmarks
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path /models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            sleep 10  # Brief pause between benchmarks
            if [ -f "${BENCHMARK_DIR}/benchmarks/llm_inference.py" ]; then
                local config_arg=""
                if [ -f "${BENCHMARK_DIR}/benchmarks/config.yaml" ]; then
                    config_arg="--config /workspace/benchmarks/config.yaml"
                fi
                run_benchmark "llm" "${config_arg}"
            fi
            ;;
        *)
            echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
            echo "Usage: $0 [matmul|trtllm|llm|all]"
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
