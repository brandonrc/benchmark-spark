#!/bin/bash
#
# Comprehensive Benchmark Runner
# Runs multiple iterations across different models and environments
#

set -e
set -x  # Enable debug tracing

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_DIR="${BENCHMARK_DIR}/results/comprehensive"
NUM_ITERATIONS=10
COOLDOWN_MINUTES=5
MAX_TEMP_THRESHOLD=45  # Wait until GPU is below this temp before next run

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Models to test
declare -A MODELS
MODELS=(
    ["deepseek"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|/data/models/huggingface/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    ["qwen72b"]="Qwen/Qwen2.5-72B-Instruct|/data/models/huggingface/Qwen/Qwen2.5-72B-Instruct"
    ["gpt120b"]="openai/gpt-oss-120b|/data/models/huggingface/openai/gpt-oss-120b"
)

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Function to get GPU temperature
get_gpu_temp() {
    nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0"
}

# Function to wait for GPU to cool down
wait_for_cooldown() {
    local current_temp=$(get_gpu_temp)

    if [ "$current_temp" -gt "$MAX_TEMP_THRESHOLD" ]; then
        echo -e "${YELLOW}GPU temperature: ${current_temp}°C (threshold: ${MAX_TEMP_THRESHOLD}°C)${NC}"
        echo "Waiting for cooldown..."

        while [ "$current_temp" -gt "$MAX_TEMP_THRESHOLD" ]; do
            sleep 30
            current_temp=$(get_gpu_temp)
            echo -e "${YELLOW}  Current temp: ${current_temp}°C${NC}"
        done
    fi

    echo -e "${GREEN}✓ GPU cooled down to ${current_temp}°C${NC}"

    # Additional fixed cooldown period
    echo "Additional ${COOLDOWN_MINUTES} minute cooldown..."
    sleep $((COOLDOWN_MINUTES * 60))
}

# Function to run single benchmark
run_single_benchmark() {
    local environment=$1  # "native" or "container"
    local model_name=$2
    local model_id=$3
    local model_path=$4
    local iteration=$5

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local result_prefix="${RESULTS_DIR}/${environment}_${model_name}_iter${iteration}_${timestamp}"

    echo ""
    echo "=========================================="
    echo "  Benchmark Run"
    echo "=========================================="
    echo "Environment:  ${environment}"
    echo "Model:        ${model_name}"
    echo "Model ID:     ${model_id}"
    echo "Iteration:    ${iteration}/${NUM_ITERATIONS}"
    echo "Temperature:  $(get_gpu_temp)°C"
    echo "Timestamp:    ${timestamp}"
    echo "=========================================="

    # Record start temperature
    local start_temp=$(get_gpu_temp)

    # Run the appropriate benchmark
    if [ "${environment}" == "native" ]; then
        "${SCRIPT_DIR}/run_native_benchmark.sh" trtllm \
            --model "${model_id}" \
            --model-path "${model_path}" \
            > "${result_prefix}.log" 2>&1
    else
        "${SCRIPT_DIR}/run_container_benchmark.sh" trtllm \
            --model "${model_id}" \
            --model-path "${model_path}" \
            > "${result_prefix}.log" 2>&1
    fi

    local exit_code=$?

    # Record end temperature
    local end_temp=$(get_gpu_temp)

    # Save metadata
    cat > "${result_prefix}_metadata.json" << EOF
{
  "environment": "${environment}",
  "model_name": "${model_name}",
  "model_id": "${model_id}",
  "model_path": "${model_path}",
  "iteration": ${iteration},
  "timestamp": "${timestamp}",
  "start_temp_celsius": ${start_temp},
  "end_temp_celsius": ${end_temp},
  "exit_code": ${exit_code}
}
EOF

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
    else
        echo -e "${RED}✗ Benchmark failed with exit code ${exit_code}${NC}"
        echo "  Check log: ${result_prefix}.log"
    fi

    return $exit_code
}

# Main execution
main() {
    echo ""
    echo "=========================================="
    echo "  COMPREHENSIVE BENCHMARK SUITE"
    echo "=========================================="
    echo "Models:      ${!MODELS[@]}"
    echo "Iterations:  ${NUM_ITERATIONS} per environment per model"
    echo "Total runs:  $((${#MODELS[@]} * 2 * NUM_ITERATIONS))"
    echo "Cooldown:    ${COOLDOWN_MINUTES} minutes between runs"
    echo "Temp limit:  ${MAX_TEMP_THRESHOLD}°C"
    echo "Results:     ${RESULTS_DIR}"
    echo "=========================================="
    echo ""

    # Check if we can use sudo without password for native benchmarks
    echo "Checking sudo access for native benchmarks..."
    if ! sudo -n true 2>/dev/null; then
        echo -e "${YELLOW}WARNING: sudo requires password. You may be prompted during native benchmarks.${NC}"
        echo "To avoid interruptions, either:"
        echo "  1. Run this script as: sudo $0"
        echo "  2. Configure passwordless sudo for mount/chroot operations"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
    else
        echo -e "${GREEN}✓ Sudo access OK${NC}"
    fi

    echo ""
    read -p "Start comprehensive benchmark? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    local total_runs=0
    local successful_runs=0
    local failed_runs=0

    # Loop through each model
    for model_key in "${!MODELS[@]}"; do
        IFS='|' read -r model_id model_path <<< "${MODELS[$model_key]}"

        echo ""
        echo "=========================================="
        echo "  Testing Model: ${model_key}"
        echo "  ID: ${model_id}"
        echo "=========================================="

        # Run native benchmarks
        echo -e "${BLUE}Running ${NUM_ITERATIONS} native iterations...${NC}"
        for iter in $(seq 1 $NUM_ITERATIONS); do
            echo ""
            echo "=========================================="
            echo "DEBUG: Starting iteration $iter of $NUM_ITERATIONS"
            echo "DEBUG: Model: ${model_key}"
            echo "DEBUG: Environment: native"
            echo "=========================================="

            wait_for_cooldown

            echo "DEBUG: About to run benchmark..."
            if run_single_benchmark "native" "${model_key}" "${model_id}" "${model_path}" "$iter"; then
                successful_runs=$((successful_runs + 1))
                echo "DEBUG: Benchmark succeeded. successful_runs=$successful_runs"
            else
                failed_runs=$((failed_runs + 1))
                echo "DEBUG: Benchmark failed. failed_runs=$failed_runs"
            fi
            total_runs=$((total_runs + 1))
            echo "DEBUG: Completed iteration $iter. total_runs=$total_runs"
        done

        echo ""
        echo "=========================================="
        echo "DEBUG: Finished all native iterations for ${model_key}"
        echo "=========================================="

        # Run container benchmarks
        echo -e "${BLUE}Running ${NUM_ITERATIONS} container iterations...${NC}"
        for iter in $(seq 1 $NUM_ITERATIONS); do
            echo ""
            echo "=========================================="
            echo "DEBUG: Starting iteration $iter of $NUM_ITERATIONS"
            echo "DEBUG: Model: ${model_key}"
            echo "DEBUG: Environment: container"
            echo "=========================================="

            wait_for_cooldown

            echo "DEBUG: About to run benchmark..."
            if run_single_benchmark "container" "${model_key}" "${model_id}" "${model_path}" "$iter"; then
                successful_runs=$((successful_runs + 1))
                echo "DEBUG: Benchmark succeeded. successful_runs=$successful_runs"
            else
                failed_runs=$((failed_runs + 1))
                echo "DEBUG: Benchmark failed. failed_runs=$failed_runs"
            fi
            total_runs=$((total_runs + 1))
            echo "DEBUG: Completed iteration $iter. total_runs=$total_runs"
        done

        echo ""
        echo "=========================================="
        echo "DEBUG: Finished all container iterations for ${model_key}"
        echo "=========================================="
    done

    echo ""
    echo "=========================================="
    echo "DEBUG: Finished all models!"
    echo "=========================================="

    # Summary
    echo ""
    echo "=========================================="
    echo "  BENCHMARK SUITE COMPLETE"
    echo "=========================================="
    echo "Total runs:      ${total_runs}"
    echo "Successful:      ${successful_runs}"
    echo "Failed:          ${failed_runs}"
    echo "Results saved:   ${RESULTS_DIR}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results: python ${BENCHMARK_DIR}/analysis/analyze_comprehensive_results.py"
    echo "  2. View summary: ls -lh ${RESULTS_DIR}"
}

main "$@"
