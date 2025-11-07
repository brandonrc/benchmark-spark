#!/usr/bin/env python3
"""
Simple Matrix Multiplication Benchmark
Sanity check to ensure basic CUDA operations work correctly.
Replicates the methodology from the reference paper.
"""

import time
import sys
import csv
from datetime import datetime
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed. Install with: pip install tensorflow", file=sys.stderr)
    sys.exit(1)


def check_gpu():
    """Verify GPU availability and print info."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("ERROR: No GPU devices found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    # Print TensorFlow and CUDA info
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"GPU compute capability: {tf.test.gpu_device_name()}")
    print()


def matrix_multiply_benchmark(matrix_size=10000, iterations=10, warmup=3):
    """
    Perform matrix multiplication benchmark.

    Args:
        matrix_size: Size of square matrices (default: 10000 x 10000)
        iterations: Number of measurement iterations
        warmup: Number of warmup iterations

    Returns:
        List of execution times in seconds
    """
    print(f"Matrix Multiplication Benchmark")
    print(f"Matrix size: {matrix_size} x {matrix_size}")
    print(f"Warmup iterations: {warmup}")
    print(f"Measurement iterations: {iterations}")
    print("-" * 60)

    # Create random matrices on GPU
    with tf.device('/GPU:0'):
        matrix_a = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)
        matrix_b = tf.random.normal([matrix_size, matrix_size], dtype=tf.float32)

    # Warmup phase
    print("Warmup phase...")
    for i in range(warmup):
        with tf.device('/GPU:0'):
            _ = tf.matmul(matrix_a, matrix_b)
        print(f"  Warmup {i+1}/{warmup} completed")

    # Ensure all operations are complete
    tf.raw_ops.DeviceBarrier()
    print("Warmup complete. Starting measurements...")
    print()

    # Measurement phase
    execution_times = []

    for i in range(iterations):
        start_time = time.perf_counter()

        with tf.device('/GPU:0'):
            result = tf.matmul(matrix_a, matrix_b)

        # Ensure operation is complete (synchronize)
        _ = result.numpy()  # Force evaluation

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        execution_times.append(elapsed)

        print(f"Run {i+1:3d}/{iterations}: {elapsed:.6f} seconds")

    return execution_times


def calculate_statistics(times):
    """Calculate statistics from execution times."""
    times_array = np.array(times)

    stats = {
        'mean': np.mean(times_array),
        'std': np.std(times_array),
        'min': np.min(times_array),
        'max': np.max(times_array),
        'median': np.median(times_array),
        'p95': np.percentile(times_array, 95),
        'p99': np.percentile(times_array, 99),
    }

    return stats


def save_results(times, stats, output_file, environment_type):
    """Save results to CSV file."""
    timestamp = datetime.now().isoformat()

    # Save detailed results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'environment', 'run', 'execution_time_seconds'])
        for i, t in enumerate(times):
            writer.writerow([timestamp, environment_type, i+1, t])

    # Save summary statistics
    summary_file = output_file.replace('.csv', '_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'environment', 'metric', 'value'])
        for metric, value in stats.items():
            writer.writerow([timestamp, environment_type, metric, value])

    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Mean execution time:   {stats['mean']:.6f} seconds")
    print(f"Std deviation:         {stats['std']:.6f} seconds")
    print(f"Median (P50):          {stats['median']:.6f} seconds")
    print(f"95th percentile:       {stats['p95']:.6f} seconds")
    print(f"99th percentile:       {stats['p99']:.6f} seconds")
    print(f"Min:                   {stats['min']:.6f} seconds")
    print(f"Max:                   {stats['max']:.6f} seconds")
    print(f"Coefficient of var:    {(stats['std']/stats['mean']*100):.2f}%")
    print("=" * 60)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Matrix Multiplication Benchmark')
    parser.add_argument('--matrix-size', type=int, default=10000,
                        help='Size of square matrix (default: 10000)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of measurement iterations (default: 10)')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Number of warmup iterations (default: 3)')
    parser.add_argument('--output', type=str, default='/results/matmul_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--environment', type=str, default='unknown',
                        choices=['native', 'container', 'unknown'],
                        help='Environment type for labeling results')

    args = parser.parse_args()

    print("=" * 60)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Check GPU availability
    check_gpu()

    # Run benchmark
    try:
        times = matrix_multiply_benchmark(
            matrix_size=args.matrix_size,
            iterations=args.iterations,
            warmup=args.warmup
        )

        # Calculate statistics
        stats = calculate_statistics(times)

        # Print summary
        print_summary(stats)

        # Save results
        save_results(times, stats, args.output, args.environment)

        print("\nBenchmark completed successfully!")
        return 0

    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
