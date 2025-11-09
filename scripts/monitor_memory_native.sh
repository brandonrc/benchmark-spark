#!/bin/bash
#
# Enhanced Memory Monitor for Native Benchmarks
# Tracks GPU memory (via nvidia-smi) and system memory usage
# Tracks Python process memory instead of container memory
#

OUTPUT_FILE="$1"
INTERVAL="${2:-1}"

if [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <output_file> [interval_seconds]" >&2
    exit 1
fi

# CSV header
echo "timestamp,gpu_util_%,gpu_mem_util_%,gpu_mem_used_mb,gpu_mem_free_mb,gpu_temp_c,gpu_power_w,gpu_sm_clock_mhz,sys_mem_total_mb,sys_mem_used_mb,sys_mem_free_mb,sys_mem_available_mb,sys_mem_percent,python_mem_mb,python_mem_percent" > "$OUTPUT_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%3N')

    # Try to get GPU stats from nvidia-smi dmon (device monitoring)
    # This sometimes works better for memory on new GPUs
    GPU_STATS=$(nvidia-smi dmon -c 1 -s pucmt 2>/dev/null | tail -n 1 | awk '{print $2","$3","$4","$5","$6","$7}')

    # If dmon fails, try regular nvidia-smi query
    if [ -z "$GPU_STATS" ] || [ "$GPU_STATS" == ",,,,," ]; then
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_MEM_UTIL=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1)
        GPU_CLOCK=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -n1)

        # Replace [N/A] or empty with -1
        GPU_UTIL=${GPU_UTIL:-"-1"}
        GPU_MEM_UTIL=${GPU_MEM_UTIL:-"-1"}
        GPU_MEM_USED=${GPU_MEM_USED:-"-1"}
        GPU_MEM_FREE=${GPU_MEM_FREE:-"-1"}
        GPU_TEMP=${GPU_TEMP:-"-1"}
        GPU_POWER=${GPU_POWER:-"-1"}
        GPU_CLOCK=${GPU_CLOCK:-"-1"}

        # Clean up [N/A] strings
        GPU_UTIL=$(echo "$GPU_UTIL" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_MEM_UTIL=$(echo "$GPU_MEM_UTIL" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_MEM_USED=$(echo "$GPU_MEM_USED" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_MEM_FREE=$(echo "$GPU_MEM_FREE" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_TEMP=$(echo "$GPU_TEMP" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_POWER=$(echo "$GPU_POWER" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')
        GPU_CLOCK=$(echo "$GPU_CLOCK" | sed 's/\[N\/A\]/-1/g' | sed 's/ //g')

        GPU_STATS="${GPU_UTIL},${GPU_MEM_UTIL},${GPU_MEM_USED},${GPU_MEM_FREE},${GPU_TEMP},${GPU_POWER},${GPU_CLOCK}"
    fi

    # Get system memory from /proc/meminfo (in KB, convert to MB)
    MEM_TOTAL=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')
    MEM_FREE=$(grep MemFree /proc/meminfo | awk '{print int($2/1024)}')
    MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print int($2/1024)}')
    MEM_USED=$((MEM_TOTAL - MEM_AVAILABLE))
    MEM_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($MEM_USED/$MEM_TOTAL)*100}")

    # Get Python process memory usage (find python3 processes related to benchmarks)
    # Use ps to find python3 processes and sum their RSS (in KB, convert to MB)
    PYTHON_MEM_KB=$(ps aux | grep "python3.*benchmark" | grep -v grep | awk '{sum+=$6} END {print sum}')
    PYTHON_MEM_MB=$(awk "BEGIN {printf \"%.1f\", ${PYTHON_MEM_KB:-0}/1024}")

    # Calculate Python memory percentage
    if [ "$PYTHON_MEM_KB" != "" ] && [ "$PYTHON_MEM_KB" != "0" ]; then
        PYTHON_MEM_PCT=$(awk "BEGIN {printf \"%.2f\", (${PYTHON_MEM_KB}/${MEM_TOTAL}/1024)*100}")
    else
        PYTHON_MEM_PCT="-1"
    fi

    # Write combined stats
    echo "${TIMESTAMP},${GPU_STATS},${MEM_TOTAL},${MEM_USED},${MEM_FREE},${MEM_AVAILABLE},${MEM_PERCENT},${PYTHON_MEM_MB},${PYTHON_MEM_PCT}" >> "$OUTPUT_FILE"

    sleep "$INTERVAL"
done
