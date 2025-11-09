#!/bin/bash
#
# Native Benchmark Runner
# Runs benchmarks directly on the host (no container)
#

set -e  # Exit on error

# Configuration
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results/native}"
BENCHMARK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-1}"
NATIVE_ENV="${NATIVE_ENV:-$HOME/container-rootfs/activate.sh}"
LOCAL_MODELS_DIR="${LOCAL_MODELS_DIR:-/data/models/huggingface}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "  Native Benchmark Runner"
echo "========================================"
echo "Results dir: ${RESULTS_DIR}"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo "Native env: ${NATIVE_ENV}"
echo "Models dir: ${LOCAL_MODELS_DIR}"
echo ""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Check if native environment activation script exists
if [ ! -f "${NATIVE_ENV}" ]; then
    echo -e "${YELLOW}WARNING: Native environment activation script not found: ${NATIVE_ENV}${NC}"
    echo "Attempting to use system Python environment..."
    echo ""

    # Check if required packages are installed
    if ! python3 -c "import torch; import tensorflow" 2>/dev/null; then
        echo -e "${RED}ERROR: Required packages (torch, tensorflow) not found${NC}"
        echo "Please run: ./scripts/setup_native.sh"
        exit 1
    fi
else
    echo "Activating native environment..."
    source "${NATIVE_ENV}"
    echo -e "${GREEN}✓ Native environment activated${NC}"
fi

# Verify GPU access
echo ""
echo "Verifying GPU access..."
if ! /home/khan/container-rootfs/run_python_simple.sh -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch cannot access CUDA${NC}"
    /home/khan/container-rootfs/run_python_simple.sh -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    exit 1
fi

echo -e "${GREEN}✓ GPU access OK${NC}"
/home/khan/container-rootfs/run_python_simple.sh -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Start enhanced monitoring in background (GPU + system memory)
# For native, we don't track container memory, but track native process instead
MONITOR_PID=""
GPU_MONITOR_PID=""
NATIVE_PROCESS_PID=""

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

    # Use the enhanced monitor script (modified for native - no container tracking)
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

# Function to run benchmark natively
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark (Native)"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/enhanced_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Determine benchmark script
    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="${BENCHMARK_DIR}/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "llm" ]; then
        benchmark_script="${BENCHMARK_DIR}/benchmarks/llm_inference.py"
    elif [ "${benchmark_type}" == "trtllm" ]; then
        benchmark_script="${BENCHMARK_DIR}/benchmarks/trtllm_benchmark.py"
    else
        echo -e "${RED}ERROR: Unknown benchmark type: ${benchmark_type}${NC}"
        return 1
    fi

    # Check if benchmark script exists
    if [ ! -f "${benchmark_script}" ]; then
        echo -e "${RED}ERROR: Benchmark script not found: ${benchmark_script}${NC}"
        return 1
    fi

    # Run benchmark
    echo "Running benchmark script: ${benchmark_script}"
    echo ""

    # Set output path based on benchmark type
    if [ "${benchmark_type}" == "trtllm" ]; then
        # trtllm benchmark creates its own files in the output directory
        local output_arg="${RESULTS_DIR}"
    else
        # Other benchmarks need a specific file path
        local output_arg="${RESULTS_DIR}/${benchmark_type}_results_${timestamp}.csv"
    fi

    /home/khan/container-rootfs/run_in_rootfs.sh python3 "${benchmark_script}" \
        --environment native \
        --output "${output_arg}" \
        ${extra_args}

    local exit_code=$?

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
    echo "Note: System memory usage includes baseline + benchmark process allocation"
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
                config_arg="--config ${BENCHMARK_DIR}/benchmarks/config.yaml"
            fi
            run_benchmark "llm" "${config_arg}"
            ;;
        trtllm)
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path ${LOCAL_MODELS_DIR}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        all)
            echo "Running all benchmarks..."
            run_benchmark "matmul"
            sleep 10  # Brief pause between benchmarks
            run_benchmark "trtllm" "--benchmark-type throughput --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --model-path ${LOCAL_MODELS_DIR}/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            sleep 10  # Brief pause between benchmarks
            if [ -f "${BENCHMARK_DIR}/benchmarks/llm_inference.py" ]; then
                local config_arg=""
                if [ -f "${BENCHMARK_DIR}/benchmarks/config.yaml" ]; then
                    config_arg="--config ${BENCHMARK_DIR}/benchmarks/config.yaml"
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
    echo "  All Native Benchmarks Complete"
    echo "========================================"
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Next steps:"
    echo "  python analysis/compare_results.py"
}

main "$@"
