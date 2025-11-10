#!/bin/bash
#
# Phase 2 Comprehensive Benchmark Suite
# IEEE Paper Quality: N=30 runs per model, 1000 requests/run, 10 models
#
# Configuration based on NVIDIA DGX Spark Playbooks:
# https://github.com/NVIDIA/dgx-spark-playbooks
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_BASE="${RESULTS_BASE:-${BENCHMARK_DIR}/results/phase2}"
CONFIG_FILE="${CONFIG_FILE:-${BENCHMARK_DIR}/configs/phase2_models.json}"

# Testing parameters
RUNS_PER_MODEL="${RUNS_PER_MODEL:-30}"  # N=30 for statistical rigor
COOLDOWN_MINUTES="${COOLDOWN_MINUTES:-5}"  # 5 minute cooldown between runs
MAX_GPU_TEMP="${MAX_GPU_TEMP:-45}"  # Maximum GPU temperature before starting run

# Model selection (can be overridden)
MODELS_TO_TEST="${MODELS_TO_TEST:-tier1}"  # Options: tier1, tier2, all, or comma-separated list

# Phase selection
TEST_PHASE="${TEST_PHASE:-baseline}"  # Options: baseline, optimization, both

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================================================"
echo "                   PHASE 2 COMPREHENSIVE BENCHMARK SUITE"
echo "========================================================================"
echo "Configuration:"
echo "  - Runs per model: ${RUNS_PER_MODEL}"
echo "  - Cooldown: ${COOLDOWN_MINUTES} minutes"
echo "  - Max GPU temp: ${MAX_GPU_TEMP}°C"
echo "  - Models to test: ${MODELS_TO_TEST}"
echo "  - Test phase: ${TEST_PHASE}"
echo "  - Results: ${RESULTS_BASE}"
echo "========================================================================"
echo ""

# Create results directories
mkdir -p "${RESULTS_BASE}/native"
mkdir -p "${RESULTS_BASE}/container_baseline"
mkdir -p "${RESULTS_BASE}/container_optimized"
mkdir -p "${RESULTS_BASE}/logs"

# Function to check GPU temperature
wait_for_gpu_cooldown() {
    echo -e "${YELLOW}Checking GPU temperature...${NC}"
    local current_temp
    while true; do
        current_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | head -n 1)
        if [ "${current_temp}" -le "${MAX_GPU_TEMP}" ]; then
            echo -e "${GREEN}✓ GPU temperature: ${current_temp}°C (ready)${NC}"
            break
        else
            echo -e "${YELLOW}GPU temperature: ${current_temp}°C (waiting for ≤${MAX_GPU_TEMP}°C)${NC}"
            sleep 30
        fi
    done
}

# Function to wait for cooldown period
cooldown() {
    local reason="${1:-benchmark}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Cooldown period: ${COOLDOWN_MINUTES} minutes${NC}"
    echo -e "${BLUE}Reason: ${reason}${NC}"
    echo -e "${BLUE}========================================${NC}"

    local seconds=$((COOLDOWN_MINUTES * 60))
    local end_time=$(($(date +%s) + seconds))

    while [ $(date +%s) -lt ${end_time} ]; do
        local remaining=$((end_time - $(date +%s)))
        local mins=$((remaining / 60))
        local secs=$((remaining % 60))
        printf "\rRemaining: %02d:%02d " ${mins} ${secs}
        sleep 1
    done
    echo ""

    # Final temperature check
    wait_for_gpu_cooldown
}

# Function to get model list
get_model_list() {
    local selection="$1"

    case "${selection}" in
        tier1)
            echo "Llama-3-8B Llama-3-70B Mistral-7B-v0.3 DeepSeek-R1-Distill-Qwen-7B Qwen2.5-72B-Instruct"
            ;;
        tier2)
            echo "Mixtral-8x7B Phi-3-mini Gemma-2B"
            ;;
        tier3)
            echo "Qwen2-1.5B GPT-OSS-120B"
            ;;
        all)
            echo "Qwen2-1.5B Gemma-2B Phi-3-mini Mistral-7B-v0.3 DeepSeek-R1-Distill-Qwen-7B Llama-3-8B Mixtral-8x7B Llama-3-70B Qwen2.5-72B-Instruct GPT-OSS-120B"
            ;;
        *)
            # Treat as comma-separated list
            echo "${selection}" | tr ',' ' '
            ;;
    esac
}

# Function to get HuggingFace model ID
get_huggingface_id() {
    local model_name="$1"

    case "${model_name}" in
        "Qwen2-1.5B")
            echo "Qwen/Qwen2-1.5B"
            ;;
        "Gemma-2B")
            echo "google/gemma-2b"
            ;;
        "Phi-3-mini")
            echo "microsoft/Phi-3-mini-4k-instruct"
            ;;
        "Mistral-7B-v0.3")
            echo "mistralai/Mistral-7B-v0.3"
            ;;
        "DeepSeek-R1-Distill-Qwen-7B")
            echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
            ;;
        "Llama-3-8B")
            echo "meta-llama/Meta-Llama-3-8B"
            ;;
        "Mixtral-8x7B")
            echo "mistralai/Mixtral-8x7B-v0.1"
            ;;
        "Llama-3-70B")
            echo "casperhansen/llama-3-70b-instruct-awq"  # AWQ quantized
            ;;
        "Qwen2.5-72B-Instruct")
            echo "Qwen/Qwen2.5-72B-Instruct"
            ;;
        "GPT-OSS-120B")
            echo "openai/gpt-oss-120b"
            ;;
        *)
            echo "${model_name}"  # Use as-is
            ;;
    esac
}

# Function to run single benchmark
run_single_benchmark() {
    local model_name="$1"
    local environment="$2"  # native, container_baseline, container_optimized
    local iteration="$3"

    local huggingface_id=$(get_huggingface_id "${model_name}")
    local model_path="/data/models/huggingface/${huggingface_id}"

    echo ""
    echo "========================================================================"
    echo "Model: ${model_name} (${huggingface_id})"
    echo "Environment: ${environment}"
    echo "Iteration: ${iteration}/${RUNS_PER_MODEL}"
    echo "========================================================================"

    # Wait for cooldown and temperature
    if [ ${iteration} -gt 1 ]; then
        cooldown "After iteration $((iteration-1))"
    fi

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local results_dir="${RESULTS_BASE}/${environment}"
    local log_file="${RESULTS_BASE}/logs/${model_name}_${environment}_iter${iteration}_${timestamp}.log"

    # Export environment variables for the benchmark scripts
    export RESULTS_DIR="${results_dir}"
    export LOCAL_MODELS_DIR="/data/models/huggingface"

    # Run the appropriate benchmark script
    case "${environment}" in
        native)
            echo "Running native benchmark..."
            "${SCRIPT_DIR}/run_native_benchmark.sh" trtllm \
                --model "${huggingface_id}" \
                --model-path "${model_path}" \
                --benchmark-type throughput \
                > "${log_file}" 2>&1
            ;;
        container_baseline)
            echo "Running container baseline benchmark..."
            "${SCRIPT_DIR}/run_container_benchmark.sh" trtllm \
                --model "${huggingface_id}" \
                --model-path "${model_path}" \
                --benchmark-type throughput \
                > "${log_file}" 2>&1
            ;;
        container_optimized)
            echo "Running optimized container benchmark..."
            export OPTIMIZATION_MODE="best"
            "${SCRIPT_DIR}/run_container_benchmark_optimized.sh" trtllm \
                "--model ${huggingface_id} --model-path ${model_path} --benchmark-type throughput" \
                > "${log_file}" 2>&1
            ;;
        *)
            echo -e "${RED}ERROR: Unknown environment: ${environment}${NC}"
            return 1
            ;;
    esac

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
        echo "Log: ${log_file}"
    else
        echo -e "${RED}✗ Benchmark failed with exit code ${exit_code}${NC}"
        echo "Check log: ${log_file}"
        return ${exit_code}
    fi
}

# Function to run complete test suite for one model
run_model_suite() {
    local model_name="$1"

    echo ""
    echo "###################################################################"
    echo "#"
    echo "#  Testing Model: ${model_name}"
    echo "#  Total iterations: ${RUNS_PER_MODEL} per environment"
    echo "#"
    echo "###################################################################"

    # Determine which environments to test
    local environments=()
    if [ "${TEST_PHASE}" == "baseline" ] || [ "${TEST_PHASE}" == "both" ]; then
        environments+=("native" "container_baseline")
    fi
    if [ "${TEST_PHASE}" == "optimization" ] || [ "${TEST_PHASE}" == "both" ]; then
        environments+=("container_optimized")
    fi

    # Run N iterations for each environment
    for env in "${environments[@]}"; do
        echo ""
        echo "================================================================"
        echo "Environment: ${env}"
        echo "================================================================"

        for i in $(seq 1 ${RUNS_PER_MODEL}); do
            run_single_benchmark "${model_name}" "${env}" "${i}"
        done
    done

    echo ""
    echo -e "${GREEN}✓ Model ${model_name} completed all ${RUNS_PER_MODEL} iterations${NC}"
}

# Main execution
main() {
    local start_time=$(date +%s)

    # Get list of models to test
    local models=$(get_model_list "${MODELS_TO_TEST}")
    local model_array=(${models})
    local total_models=${#model_array[@]}

    echo "Models to test (${total_models}):"
    for model in ${models}; do
        echo "  - ${model}"
    done
    echo ""

    # Estimate total time
    local envs_to_test=2  # native + container_baseline by default
    if [ "${TEST_PHASE}" == "both" ]; then
        envs_to_test=3  # native + container_baseline + container_optimized
    elif [ "${TEST_PHASE}" == "optimization" ]; then
        envs_to_test=1  # only container_optimized
    fi

    local total_runs=$((total_models * RUNS_PER_MODEL * envs_to_test))
    local estimated_hours=$((total_runs * 15 / 60))  # ~15 min per run (including cooldown)

    echo "Estimated parameters:"
    echo "  - Total runs: ${total_runs}"
    echo "  - Estimated time: ~${estimated_hours} hours"
    echo ""

    read -p "Press Enter to start, or Ctrl+C to cancel..."

    # Run tests for each model
    local completed=0
    for model in ${models}; do
        run_model_suite "${model}"
        completed=$((completed + 1))
        echo ""
        echo "Progress: ${completed}/${total_models} models completed"
        echo ""
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    echo ""
    echo "###################################################################"
    echo "#"
    echo "#  PHASE 2 COMPREHENSIVE BENCHMARK SUITE COMPLETE"
    echo "#"
    echo "#  Models tested: ${total_models}"
    echo "#  Total runs: ${total_runs}"
    echo "#  Duration: ${hours}h ${minutes}m"
    echo "#"
    echo "#  Results: ${RESULTS_BASE}"
    echo "#"
    echo "###################################################################"

    echo ""
    echo "Next steps:"
    echo "  1. Run statistical analysis: python analysis/phase2_statistical_analysis.py"
    echo "  2. Generate figures: python analysis/phase2_generate_figures.py"
    echo "  3. Review results: ${RESULTS_BASE}"
}

main "$@"
