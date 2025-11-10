#!/usr/bin/env python3
"""
TensorRT-LLM Benchmark
Uses the built-in trtllm-bench tool to run realistic LLM inference benchmarks.
"""

import sys
import subprocess
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path


def run_trtllm_benchmark(model, workspace, output_dir, benchmark_type="throughput", max_tokens=128, max_batch_size=8, model_path=None):
    """
    Run TensorRT-LLM benchmark using the trtllm-bench CLI.

    Args:
        model: HuggingFace model name
        workspace: Workspace directory for intermediate files
        output_dir: Output directory for results
        benchmark_type: "throughput" or "latency"
        max_tokens: Maximum number of output tokens
        max_batch_size: Maximum batch size for engine
        model_path: Optional path to local model checkpoint

    Returns:
        Exit code from benchmark
    """
    print("=" * 60)
    print(f"TensorRT-LLM {benchmark_type.upper()} Benchmark")
    print("=" * 60)
    print(f"Model: {model}")
    if model_path:
        print(f"Model path: {model_path}")
    print(f"Workspace: {workspace}")
    print(f"Output: {output_dir}")
    print(f"Max tokens: {max_tokens}")
    print(f"Max batch size: {max_batch_size}")
    print()

    # Create workspace directory
    Path(workspace).mkdir(parents=True, exist_ok=True)

    # Ensure output_dir is a directory (not a file path)
    output_dir_path = Path(output_dir)
    if output_dir_path.suffix:  # Has a file extension like .csv
        output_dir_path = output_dir_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Use the directory for output
    output_dir = str(output_dir_path)

    # Calculate max_seq_len (input + output tokens)
    target_input_len = 512  # Reasonable default for benchmarking
    max_seq_len = target_input_len + max_tokens

    # First build the engine
    print("Step 1: Building TensorRT engine...")
    print("-" * 60)

    # Use IFB Scheduler Limits approach (don't mix with tuning heuristics)
    build_cmd = [
        "trtllm-bench",
        "-m", model,
        "--workspace", workspace
    ]

    # Add model path if provided
    if model_path:
        build_cmd.extend(["--model_path", model_path])

    build_cmd.extend([
        "build",
        "--max_batch_size", str(max_batch_size),
        "--max_num_tokens", str(max_batch_size * target_input_len),
        "--max_seq_len", str(max_seq_len)
    ])

    print(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"ERROR: Engine build failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    print()
    print(f"✓ Engine build completed successfully")
    print()

    # Run the benchmark
    print(f"Step 2: Running {benchmark_type} benchmark...")
    print("-" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"trtllm_{benchmark_type}_{timestamp}.json"

    # Generate a simple dataset for the benchmark
    # Each line should be a JSON object with a "text" field
    # NVIDIA recommends 1000-3000 requests for throughput benchmarking
    # Using 1000 as per Phase 2 IEEE paper requirements
    num_requests = 1000
    dataset_lines = []
    sample_prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing to a 10 year old.",
        "What are the health benefits of drinking green tea?",
        "Describe the process of photosynthesis in plants.",
        "How does machine learning work in simple terms?"
    ]

    for i in range(num_requests):
        prompt = sample_prompts[i % len(sample_prompts)]
        dataset_lines.append(json.dumps({
            "task_id": 0,
            "prompt": prompt,
            "output_tokens": max_tokens
        }))

    # Write dataset to a temporary file
    dataset_file = Path(workspace) / f"benchmark_dataset_{timestamp}.jsonl"
    with open(dataset_file, 'w') as f:
        f.write("\n".join(dataset_lines))

    print(f"Generated dataset with {num_requests} requests: {dataset_file}")

    # Check if engine already exists from build step
    engine_dir = Path(workspace) / model.replace("/", "/") / "tp_1_pp_1"

    bench_cmd = [
        "trtllm-bench",
        "-m", model,
        "--workspace", workspace,
        benchmark_type,
        "--engine_dir", str(engine_dir),
        "--max_batch_size", str(max_batch_size),
        "--max_num_tokens", str(max_batch_size * 512),  # Use same as build
        "--dataset", str(dataset_file),
        "--target_output_len", str(max_tokens),
        "--num_requests", str(num_requests)
    ]

    print(f"Running: {' '.join(bench_cmd)}")

    # Run and capture output
    result = subprocess.run(bench_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR: Benchmark failed with exit code {result.returncode}", file=sys.stderr)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr, file=sys.stderr)
        return result.returncode

    # Print output
    print(result.stdout)

    # Save raw output
    output_file = Path(output_dir) / f"trtllm_{benchmark_type}_{timestamp}_output.txt"
    with open(output_file, 'w') as f:
        f.write(result.stdout)

    print()
    print(f"✓ Benchmark completed successfully")
    print(f"Results saved to: {output_file}")

    # Try to parse and save metrics in a more structured format
    try:
        parse_and_save_metrics(result.stdout, output_dir, benchmark_type, timestamp)
    except Exception as e:
        print(f"Warning: Could not parse metrics: {e}")

    return 0


def parse_and_save_metrics(output, output_dir, benchmark_type, timestamp):
    """
    Parse benchmark output and save metrics to CSV.
    This is a simple parser - adjust based on actual output format.
    """
    metrics = {}

    # Simple parsing - look for common metrics
    lines = output.split('\n')
    for line in lines:
        line = line.strip()
        # Look for key metrics (adjust based on actual output)
        if 'throughput' in line.lower() or 'tokens/sec' in line.lower():
            metrics['throughput_info'] = line
        elif 'latency' in line.lower() or 'time' in line.lower():
            metrics['latency_info'] = line

    if metrics:
        csv_file = Path(output_dir) / f"trtllm_{benchmark_type}_{timestamp}_metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'metric', 'value'])
            for key, value in metrics.items():
                writer.writerow([datetime.now().isoformat(), key, value])

        print(f"Metrics saved to: {csv_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='TensorRT-LLM Benchmark Wrapper')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace model name (default: TinyLlama for quick testing)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to local HuggingFace checkpoint directory (optional)')
    parser.add_argument('--workspace', type=str, default='/tmp/trtllm_workspace',
                        help='Workspace directory for TRT-LLM')
    parser.add_argument('--output', type=str, default='/results',
                        help='Output directory for results')
    parser.add_argument('--benchmark-type', type=str, default='throughput',
                        choices=['throughput', 'latency', 'both'],
                        help='Type of benchmark to run')
    parser.add_argument('--max-tokens', type=int, default=128,
                        help='Maximum output tokens (default: 128)')
    parser.add_argument('--max-batch-size', type=int, default=8,
                        help='Maximum batch size for engine (default: 8)')
    parser.add_argument('--environment', type=str, default='unknown',
                        choices=['native', 'container', 'unknown'],
                        help='Environment type for labeling results')

    args = parser.parse_args()

    print("=" * 60)
    print("TENSORRT-LLM BENCHMARK")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    try:
        if args.benchmark_type in ['throughput', 'both']:
            exit_code = run_trtllm_benchmark(
                args.model,
                args.workspace,
                args.output,
                'throughput',
                args.max_tokens,
                args.max_batch_size,
                args.model_path
            )
            if exit_code != 0:
                return exit_code

        if args.benchmark_type in ['latency', 'both']:
            exit_code = run_trtllm_benchmark(
                args.model,
                args.workspace,
                args.output,
                'latency',
                args.max_tokens,
                args.max_batch_size,
                args.model_path
            )
            if exit_code != 0:
                return exit_code

        print()
        print("=" * 60)
        print("ALL BENCHMARKS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
