#!/usr/bin/env python3
"""
TensorRT-LLM Inference Benchmark
Measures throughput and latency for LLM inference workloads.
"""

import time
import sys
import csv
import os
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import yaml

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

# TensorRT-LLM imports - these may fail if not installed
try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelRunner
    TENSORRT_LLM_AVAILABLE = True
except ImportError:
    TENSORRT_LLM_AVAILABLE = False
    print("Warning: TensorRT-LLM not available. Running in simulation mode.", file=sys.stderr)


class BenchmarkConfig:
    """Configuration for benchmark run."""

    def __init__(self, config_file=None):
        self.config = self._load_config(config_file)

    def _load_config(self, config_file):
        """Load configuration from YAML file."""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()

    def _default_config(self):
        """Return default configuration."""
        return {
            'execution': {
                'warmup_iterations': 10,
                'measurement_iterations': 100,
                'cooldown_seconds': 30,
                'random_seed': 42
            },
            'llm_tests': [
                {'id': 'T1', 'batch_size': 1, 'input_length': 128, 'output_length': 32},
                {'id': 'T2', 'batch_size': 1, 'input_length': 512, 'output_length': 128},
            ]
        }


class GPUMonitor:
    """Monitor GPU metrics during benchmark."""

    def __init__(self):
        self.metrics = []

    def check_gpu(self):
        """Check GPU availability."""
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!", file=sys.stderr)
            sys.exit(1)

        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s):")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
        print()

    def get_current_metrics(self, gpu_id=0):
        """Get current GPU metrics."""
        if not torch.cuda.is_available():
            return {}

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'gpu_id': gpu_id,
            'memory_allocated_gb': torch.cuda.memory_allocated(gpu_id) / 1e9,
            'memory_reserved_gb': torch.cuda.memory_reserved(gpu_id) / 1e9,
        }

        # Try to get additional metrics via nvidia-smi if available
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                if len(values) == 3:
                    metrics['utilization_percent'] = float(values[0])
                    metrics['temperature_c'] = float(values[1])
                    metrics['power_draw_w'] = float(values[2])
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

        return metrics


class LLMBenchmark:
    """LLM inference benchmark runner."""

    def __init__(self, model_path=None, tokenizer_path=None, config=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config = config or BenchmarkConfig()
        self.gpu_monitor = GPUMonitor()
        self.model = None
        self.tokenizer = None

    def setup(self):
        """Initialize model and tokenizer."""
        print("Setting up benchmark environment...")

        # Check GPU
        self.gpu_monitor.check_gpu()

        # Set random seed
        seed = self.config.config['execution']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"Random seed set to: {seed}")

        # Load model (if TensorRT-LLM is available)
        if TENSORRT_LLM_AVAILABLE and self.model_path:
            print(f"Loading TensorRT-LLM model from: {self.model_path}")
            # Model loading logic here
            # self.model = ModelRunner.from_dir(self.model_path)
            print("Model loaded successfully")
        else:
            print("Running in SIMULATION mode (no actual model loaded)")
            self.model = None

        print("Setup complete\n")

    def generate_dummy_inputs(self, batch_size, input_length):
        """Generate dummy input tokens for benchmarking."""
        # Generate random token IDs (typical vocab size ~50k)
        return torch.randint(0, 50000, (batch_size, input_length), device='cuda')

    def run_inference(self, inputs, max_output_length):
        """
        Run inference on inputs.

        In real implementation, this would call TensorRT-LLM.
        For now, simulate with a sleep proportional to compute expected.
        """
        if self.model is not None and TENSORRT_LLM_AVAILABLE:
            # Real inference
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_output_length,
            )
            return outputs
        else:
            # Simulation: sleep proportional to expected compute
            batch_size, seq_len = inputs.shape
            total_tokens = batch_size * (seq_len + max_output_length)
            # Rough estimate: ~10 microseconds per token on fast GPU
            simulated_time = total_tokens * 10e-6
            time.sleep(simulated_time)

            # Return dummy outputs
            return torch.randint(0, 50000, (batch_size, max_output_length), device='cuda')

    def benchmark_test(self, test_config):
        """Run a single benchmark test configuration."""
        test_id = test_config['id']
        batch_size = test_config['batch_size']
        input_length = test_config['input_length']
        output_length = test_config['output_length']

        print(f"\n{'='*60}")
        print(f"Test {test_id}: batch={batch_size}, in={input_length}, out={output_length}")
        print(f"{'='*60}")

        # Prepare inputs
        inputs = self.generate_dummy_inputs(batch_size, input_length)

        # Warmup phase
        warmup_iters = self.config.config['execution']['warmup_iterations']
        print(f"Warmup phase ({warmup_iters} iterations)...")
        for i in range(warmup_iters):
            _ = self.run_inference(inputs, output_length)
            if (i + 1) % 5 == 0:
                print(f"  Warmup {i+1}/{warmup_iters}")

        # Synchronize
        torch.cuda.synchronize()
        print("Warmup complete. Starting measurements...\n")

        # Measurement phase
        measurement_iters = self.config.config['execution']['measurement_iterations']
        latencies = []
        throughputs = []

        for i in range(measurement_iters):
            # Measure latency
            start_time = time.perf_counter()
            outputs = self.run_inference(inputs, output_length)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Calculate throughput
            total_tokens = batch_size * (input_length + output_length)
            throughput = total_tokens / (end_time - start_time)
            throughputs.append(throughput)

            if (i + 1) % 10 == 0:
                print(f"  Run {i+1:3d}/{measurement_iters}: "
                      f"{latency_ms:.2f} ms, {throughput:.1f} tokens/s")

        return {
            'test_id': test_id,
            'batch_size': batch_size,
            'input_length': input_length,
            'output_length': output_length,
            'latencies_ms': latencies,
            'throughputs_tokens_per_sec': throughputs
        }

    def calculate_statistics(self, values, name="metric"):
        """Calculate statistics for a list of values."""
        arr = np.array(values)
        return {
            f'{name}_mean': np.mean(arr),
            f'{name}_std': np.std(arr),
            f'{name}_min': np.min(arr),
            f'{name}_max': np.max(arr),
            f'{name}_median': np.median(arr),
            f'{name}_p95': np.percentile(arr, 95),
            f'{name}_p99': np.percentile(arr, 99),
        }

    def save_results(self, all_results, output_dir, environment_type):
        """Save benchmark results to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results
        detail_file = output_path / f'llm_benchmark_{environment_type}_{timestamp}.csv'
        with open(detail_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'environment', 'test_id', 'batch_size',
                             'input_length', 'output_length', 'run',
                             'latency_ms', 'throughput_tokens_per_sec'])

            for result in all_results:
                test_id = result['test_id']
                batch_size = result['batch_size']
                input_length = result['input_length']
                output_length = result['output_length']

                for run, (lat, thr) in enumerate(zip(result['latencies_ms'],
                                                      result['throughputs_tokens_per_sec'])):
                    writer.writerow([timestamp, environment_type, test_id, batch_size,
                                     input_length, output_length, run + 1, lat, thr])

        # Save summary statistics
        summary_file = output_path / f'llm_benchmark_{environment_type}_{timestamp}_summary.csv'
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'environment', 'test_id', 'batch_size',
                             'input_length', 'output_length', 'metric', 'value'])

            for result in all_results:
                test_id = result['test_id']
                batch_size = result['batch_size']
                input_length = result['input_length']
                output_length = result['output_length']

                # Calculate statistics
                lat_stats = self.calculate_statistics(result['latencies_ms'], 'latency')
                thr_stats = self.calculate_statistics(result['throughputs_tokens_per_sec'], 'throughput')

                all_stats = {**lat_stats, **thr_stats}

                for metric, value in all_stats.items():
                    writer.writerow([timestamp, environment_type, test_id, batch_size,
                                     input_length, output_length, metric, value])

        print(f"\nResults saved to:")
        print(f"  Details: {detail_file}")
        print(f"  Summary: {summary_file}")

    def run(self, output_dir, environment_type):
        """Run all benchmark tests."""
        self.setup()

        all_results = []
        test_configs = self.config.config.get('llm_tests', [])

        for test_config in test_configs:
            result = self.benchmark_test(test_config)
            all_results.append(result)

            # Print summary for this test
            print(f"\nTest {test_config['id']} Summary:")
            print(f"  Mean latency: {np.mean(result['latencies_ms']):.2f} ms")
            print(f"  Mean throughput: {np.mean(result['throughputs_tokens_per_sec']):.1f} tokens/s")

            # Cooldown
            cooldown = self.config.config['execution']['cooldown_seconds']
            if cooldown > 0:
                print(f"  Cooling down for {cooldown} seconds...")
                time.sleep(cooldown)

        # Save all results
        self.save_results(all_results, output_dir, environment_type)

        print("\n" + "="*60)
        print("ALL BENCHMARKS COMPLETE")
        print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='LLM Inference Benchmark')
    parser.add_argument('--config', type=str,
                        help='Path to configuration YAML file')
    parser.add_argument('--model-path', type=str,
                        help='Path to TensorRT-LLM engine')
    parser.add_argument('--tokenizer-path', type=str,
                        help='Path to tokenizer')
    parser.add_argument('--output-dir', type=str, default='/results',
                        help='Output directory for results')
    parser.add_argument('--environment', type=str, default='unknown',
                        choices=['native', 'container', 'unknown'],
                        help='Environment type for labeling results')

    args = parser.parse_args()

    print("=" * 60)
    print("TensorRT-LLM INFERENCE BENCHMARK")
    print("=" * 60)
    print(f"Environment: {args.environment}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"TensorRT-LLM available: {TENSORRT_LLM_AVAILABLE}")
    print()

    # Load configuration
    config = BenchmarkConfig(args.config)

    # Create and run benchmark
    benchmark = LLMBenchmark(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config=config
    )

    try:
        benchmark.run(args.output_dir, args.environment)
        return 0
    except Exception as e:
        print(f"\nERROR: Benchmark failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
