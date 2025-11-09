#!/bin/bash
#
# Native Benchmark Runner (Bare Metal)
# Runs benchmarks using extracted container binaries on bare metal (no Docker)
#

set -e  # Exit on error

# Configuration
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/native}"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
CONTAINER_ROOTFS="${CONTAINER_ROOTFS:-$HOME/container-rootfs}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-/data/models/huggingface}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Native Benchmark Runner (Bare Metal)"
echo "========================================"
echo "Results dir: ${RESULTS_DIR}"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo "Container rootfs: ${CONTAINER_ROOTFS}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check if container rootfs exists
if [ ! -d "${CONTAINER_ROOTFS}" ]; then
    echo -e "${RED}ERROR: Container rootfs not found: ${CONTAINER_ROOTFS}${NC}"
    echo "Please run: ./scripts/extract_container_rootfs.sh"
    exit 1
fi

if [ ! -f "${CONTAINER_ROOTFS}/run_in_rootfs.sh" ]; then
    echo -e "${RED}ERROR: Rootfs runner script not found${NC}"
    echo "Please re-run: ./scripts/extract_container_rootfs.sh"
    exit 1
fi

echo -e "${GREEN}✓ Container rootfs found${NC}"

# Verify GPU access
echo ""
echo "Verifying GPU access..."
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not accessible${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GPU access OK${NC}"

# Start enhanced monitoring in background (GPU + system memory)
MONITOR_PID=""
GPU_MONITOR_PID=""

function start_gpu_monitor() {
    local output_file="$1"

    # Capture baseline metrics
    local baseline_file="${output_file%.csv}_baseline.txt"
    echo ""
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

    # Use the enhanced monitor script (native version)
    "${BENCHMARK_DIR}/scripts/monitor_memory_native.sh" "${output_file}" "${GPU_MONITOR_INTERVAL}" &
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

# Function to run benchmark on bare metal
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark (Bare Metal)"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/enhanced_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Determine benchmark script
    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="/workspace/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "trtllm" ]; then
        benchmark_script="/workspace/benchmarks/trtllm_benchmark.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        return 1
    fi

    # Create workspace mount point in rootfs
    sudo mkdir -p "${CONTAINER_ROOTFS}/workspace"
    sudo mkdir -p "${CONTAINER_ROOTFS}/results"
    sudo mkdir -p "${CONTAINER_ROOTFS}/models"

    # Bind mount benchmark directory and results
    sudo mount --bind "${BENCHMARK_DIR}" "${CONTAINER_ROOTFS}/workspace"
    sudo mount --bind "${RESULTS_DIR}" "${CONTAINER_ROOTFS}/results"
    sudo mount --bind "${LOCAL_MODELS_DIR}" "${CONTAINER_ROOTFS}/models"

    # Set output path based on benchmark type
    if [ "${benchmark_type}" == "trtllm" ]; then
        local output_arg="/results"
    else
        local output_arg="/results/${benchmark_type}_results_${timestamp}.csv"
    fi

    # Run benchmark using rootfs Python
    echo "Running benchmark on bare metal (no Docker)..."
    echo ""

    # Build full command
    local full_cmd="python3 ${benchmark_script} --environment native --output ${output_arg} ${extra_args}"

    # Run in chroot
    "${CONTAINER_ROOTFS}/run_in_rootfs.sh" bash -c "${full_cmd}"

    local exit_code=$?

    # Unmount
    sudo umount "${CONTAINER_ROOTFS}/workspace" 2>/dev/null || true
    sudo umount "${CONTAINER_ROOTFS}/results" 2>/dev/null || true
    sudo umount "${CONTAINER_ROOTFS}/models" 2>/dev/null || true

    # Stop monitoring
    stop_gpu_monitor

    if [ ${exit_code} -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ ${benchmark_type} benchmark completed successfully${NC}"
    else
        echo ""
        echo -e "${RED}✗ ${benchmark_type} benchmark failed with exit code ${exit_code}${NC}"
        return ${exit_code}
    fi

    echo "Baseline metrics saved to: ${monitor_file%.csv}_baseline.txt"
    echo "Enhanced metrics saved to: ${monitor_file}"
    echo "GPU basic metrics saved to: ${monitor_file%.csv}_gpu_basic.csv"
    echo ""
    echo "Note: This ran on BARE METAL using container binaries (no Docker overhead)"
}

# Main execution
main() {
    local benchmark_type="${1:-matmul}"

    case "${benchmark_type}" in
        matmul)
            run_benchmark "matmul"
            ;;
        trtllm)
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path /models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        all)
            echo "Running all benchmarks..."
            run_benchmark "matmul"
            echo ""
            echo "Waiting 10 seconds before next benchmark..."
            sleep 10
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path /models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        *)
            echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
            echo "Usage: $0 [matmul|trtllm|all]"
            exit 1
            ;;
    esac

    echo ""
    echo "========================================"
    echo "  All Bare Metal Benchmarks Complete"
    echo "========================================"
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Compare with container results:"
    echo "  Container: results/container/"
    echo "  Bare metal: results/native/"
}

main "$@"
