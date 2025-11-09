#!/bin/bash
#
# Sudo wrapper for comprehensive benchmark
# This ensures sudo credentials are cached before starting the long-running benchmark
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prompt for sudo password once and keep it cached
echo "This script requires sudo access for native benchmarks."
echo "Please enter your password to cache sudo credentials:"
sudo -v

# Keep sudo alive in background
while true; do
    sudo -n true
    sleep 50
    kill -0 "$$" 2>/dev/null || exit
done &
SUDO_KEEPER_PID=$!

# Trap to kill the sudo keeper on exit
trap "kill $SUDO_KEEPER_PID 2>/dev/null || true" EXIT

# Now run the actual benchmark script
exec "${SCRIPT_DIR}/run_comprehensive_benchmark.sh" "$@"
