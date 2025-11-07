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
NATIVE_ENV="${NATIVE_ENV:-$HOME/tensorrt-llm/activate.sh}"

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
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch cannot access CUDA${NC}"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    exit 1
fi

echo -e "${GREEN}✓ GPU access OK${NC}"
python3 -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Start GPU monitoring in background
MONITOR_PID=""
function start_gpu_monitor() {
    local output_file="$1"
    echo ""
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

# Function to run benchmark natively
function run_benchmark() {
    local benchmark_type="$1"
    local extra_args="${2:-}"

    echo ""
    echo "========================================"
    echo "  Running ${benchmark_type} Benchmark (Native)"
    echo "========================================"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local monitor_file="${RESULTS_DIR}/gpu_metrics_${benchmark_type}_${timestamp}.csv"

    # Start monitoring
    start_gpu_monitor "${monitor_file}"

    # Determine benchmark script
    local benchmark_script=""
    if [ "${benchmark_type}" == "matmul" ]; then
        benchmark_script="${BENCHMARK_DIR}/benchmarks/simple_matmul.py"
    elif [ "${benchmark_type}" == "llm" ]; then
        benchmark_script="${BENCHMARK_DIR}/benchmarks/llm_inference.py"
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

    python3 "${benchmark_script}" \
        --environment native \
        --output "${RESULTS_DIR}" \
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
                config_arg="--config ${BENCHMARK_DIR}/benchmarks/config.yaml"
            fi
            run_benchmark "llm" "${config_arg}"
            ;;
        all)
            echo "Running all benchmarks..."
            run_benchmark "matmul"
            echo ""
            echo "Waiting 10 seconds before next benchmark..."
            sleep 10
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
    echo "  All Native Benchmarks Complete"
    echo "========================================"
    echo "Results saved to: ${RESULTS_DIR}"
    echo ""
    echo "Next steps:"
    echo "  python analysis/compare_results.py"
}

main "$@"
