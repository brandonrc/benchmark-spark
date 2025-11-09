#!/bin/bash

RESULTS_DIR="/home/khan/benchmark-spark/results/comprehensive"

echo "=========================================="
echo "  COMPREHENSIVE BENCHMARK PROGRESS"
echo "=========================================="
echo ""

# Count completed runs
total_metadata=$(ls "$RESULTS_DIR"/*.json 2>/dev/null | wc -l)
echo "Completed runs: $total_metadata / 60"
echo ""

# Count by model and environment
echo "Breakdown by model and environment:"
echo "-----------------------------------"
for model in gpt120b deepseek qwen72b; do
    native_count=$(ls "$RESULTS_DIR"/native_${model}_*.json 2>/dev/null | wc -l)
    container_count=$(ls "$RESULTS_DIR"/container_${model}_*.json 2>/dev/null | wc -l)
    echo "$model:"
    echo "  Native:    $native_count / 10"
    echo "  Container: $container_count / 10"
done

echo ""
echo "=========================================="
echo "  PRELIMINARY PERFORMANCE METRICS"
echo "=========================================="

# Analyze DeepSeek results if available
if [ -f "$RESULTS_DIR"/native_deepseek_iter1_*.log ]; then
    echo ""
    echo "DeepSeek-7B Sample Results:"
    echo "-----------------------------------"

    # Native
    native_log=$(ls "$RESULTS_DIR"/native_deepseek_iter1_*.log 2>/dev/null | head -1)
    if [ -n "$native_log" ]; then
        echo "Native (iteration 1):"
        grep "Request Throughput" "$native_log" | tail -1 | sed 's/^/  /'
        grep "Total Output Throughput" "$native_log" | tail -1 | sed 's/^/  /'
        grep "Average Request Latency" "$native_log" | tail -1 | sed 's/^/  /'
    fi

    # Container
    container_log=$(ls "$RESULTS_DIR"/container_deepseek_iter1_*.log 2>/dev/null | head -1)
    if [ -n "$container_log" ]; then
        echo "Container (iteration 1):"
        grep "Request Throughput" "$container_log" | tail -1 | sed 's/^/  /'
        grep "Total Output Throughput" "$container_log" | tail -1 | sed 's/^/  /'
        grep "Average Request Latency" "$container_log" | tail -1 | sed 's/^/  /'
    fi
fi

# Show temperature stats
echo ""
echo "=========================================="
echo "  TEMPERATURE STATISTICS"
echo "=========================================="
echo ""

for env in native container; do
    echo "$env runs:"
    jq -r '.end_temp_celsius' "$RESULTS_DIR"/${env}_*_metadata.json 2>/dev/null | \
        awk '{sum+=$1; count++; if(min==""){min=max=$1}; if($1>max){max=$1}; if($1<min){min=$1}} END {if(count>0) printf "  Min: %d°C  Max: %d°C  Avg: %.1f°C  Count: %d\n", min, max, sum/count, count}'
done

echo ""
echo "=========================================="
