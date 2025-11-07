#!/usr/bin/env python3
"""
Compare benchmark results between container and native environments.
Generate statistical analysis and visualizations.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Plots will not be generated.")
    PLOTTING_AVAILABLE = False


class BenchmarkComparison:
    """Compare benchmark results between environments."""

    def __init__(self, container_dir, native_dir, output_dir):
        self.container_dir = Path(container_dir)
        self.native_dir = Path(native_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        if PLOTTING_AVAILABLE:
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = (12, 6)

    def find_latest_results(self, directory, pattern):
        """Find the most recent results file matching pattern."""
        files = list(directory.glob(pattern))
        if not files:
            return None
        return max(files, key=lambda x: x.stat().st_mtime)

    def load_matmul_results(self):
        """Load matrix multiplication benchmark results."""
        container_file = self.find_latest_results(self.container_dir, "matmul_results.csv")
        native_file = self.find_latest_results(self.native_dir, "matmul_results.csv")

        if not container_file or not native_file:
            print(f"Warning: Matrix multiplication results not found")
            print(f"  Container: {container_file}")
            print(f"  Native: {native_file}")
            return None, None

        container_df = pd.read_csv(container_file)
        native_df = pd.read_csv(native_file)

        return container_df, native_df

    def load_llm_results(self):
        """Load LLM benchmark results."""
        container_file = self.find_latest_results(self.container_dir, "llm_benchmark_container_*.csv")
        native_file = self.find_latest_results(self.native_dir, "llm_benchmark_native_*.csv")

        if not container_file or not native_file:
            print(f"Warning: LLM benchmark results not found")
            print(f"  Container: {container_file}")
            print(f"  Native: {native_file}")
            return None, None

        container_df = pd.read_csv(container_file)
        native_df = pd.read_csv(native_file)

        return container_df, native_df

    def calculate_overhead(self, native_mean, container_mean):
        """Calculate percentage overhead."""
        if native_mean == 0:
            return 0
        return ((container_mean - native_mean) / native_mean) * 100

    def compare_matmul(self):
        """Compare matrix multiplication results."""
        print("\n" + "="*60)
        print("Matrix Multiplication Comparison")
        print("="*60)

        container_df, native_df = self.load_matmul_results()
        if container_df is None or native_df is None:
            print("Skipping matrix multiplication comparison (data not available)")
            return None

        # Calculate statistics
        container_mean = container_df['execution_time_seconds'].mean()
        container_std = container_df['execution_time_seconds'].std()
        native_mean = native_df['execution_time_seconds'].mean()
        native_std = native_df['execution_time_seconds'].std()

        overhead_pct = self.calculate_overhead(native_mean, container_mean)

        print(f"\nNative Environment:")
        print(f"  Mean: {native_mean:.6f} seconds")
        print(f"  Std:  {native_std:.6f} seconds")
        print(f"  CoV:  {(native_std/native_mean*100):.2f}%")

        print(f"\nContainer Environment:")
        print(f"  Mean: {container_mean:.6f} seconds")
        print(f"  Std:  {container_std:.6f} seconds")
        print(f"  CoV:  {(container_std/container_mean*100):.2f}%")

        print(f"\nOverhead Analysis:")
        print(f"  Time difference: {(container_mean - native_mean):.6f} seconds")
        print(f"  Overhead:        {overhead_pct:.2f}%")

        # Statistical test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(native_df['execution_time_seconds'],
                                           container_df['execution_time_seconds'])
        print(f"\nStatistical Significance (t-test):")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value:     {p_value:.6f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")

        # Create visualization
        if PLOTTING_AVAILABLE:
            self.plot_matmul_comparison(container_df, native_df)

        return {
            'native_mean': native_mean,
            'container_mean': container_mean,
            'overhead_pct': overhead_pct,
            'p_value': p_value
        }

    def compare_llm(self):
        """Compare LLM benchmark results."""
        print("\n" + "="*60)
        print("LLM Inference Comparison")
        print("="*60)

        container_df, native_df = self.load_llm_results()
        if container_df is None or native_df is None:
            print("Skipping LLM comparison (data not available)")
            return None

        # Group by test configuration
        results = []

        for test_id in container_df['test_id'].unique():
            container_test = container_df[container_df['test_id'] == test_id]
            native_test = native_df[native_df['test_id'] == test_id]

            if len(container_test) == 0 or len(native_test) == 0:
                continue

            # Throughput comparison
            container_thr_mean = container_test['throughput_tokens_per_sec'].mean()
            native_thr_mean = native_test['throughput_tokens_per_sec'].mean()
            thr_overhead = self.calculate_overhead(native_thr_mean, container_thr_mean)

            # Latency comparison
            container_lat_mean = container_test['latency_ms'].mean()
            native_lat_mean = native_test['latency_ms'].mean()
            lat_overhead = self.calculate_overhead(native_lat_mean, container_lat_mean)

            batch_size = container_test['batch_size'].iloc[0]
            input_length = container_test['input_length'].iloc[0]
            output_length = container_test['output_length'].iloc[0]

            print(f"\n{test_id} (batch={batch_size}, in={input_length}, out={output_length}):")
            print(f"  Throughput:")
            print(f"    Native:    {native_thr_mean:.1f} tokens/s")
            print(f"    Container: {container_thr_mean:.1f} tokens/s")
            print(f"    Overhead:  {thr_overhead:.2f}%")
            print(f"  Latency:")
            print(f"    Native:    {native_lat_mean:.2f} ms")
            print(f"    Container: {container_lat_mean:.2f} ms")
            print(f"    Overhead:  {lat_overhead:.2f}%")

            results.append({
                'test_id': test_id,
                'batch_size': batch_size,
                'input_length': input_length,
                'output_length': output_length,
                'native_throughput': native_thr_mean,
                'container_throughput': container_thr_mean,
                'throughput_overhead_pct': thr_overhead,
                'native_latency': native_lat_mean,
                'container_latency': container_lat_mean,
                'latency_overhead_pct': lat_overhead
            })

        # Create visualization
        if PLOTTING_AVAILABLE and results:
            self.plot_llm_comparison(pd.DataFrame(results))

        return results

    def plot_matmul_comparison(self, container_df, native_df):
        """Create comparison plots for matrix multiplication."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        data = pd.DataFrame({
            'Native': native_df['execution_time_seconds'],
            'Container': container_df['execution_time_seconds']
        })

        data.plot(kind='box', ax=axes[0])
        axes[0].set_ylabel('Execution Time (seconds)')
        axes[0].set_title('Matrix Multiplication Performance Distribution')
        axes[0].grid(True, alpha=0.3)

        # Line plot of individual runs
        axes[1].plot(native_df['run'], native_df['execution_time_seconds'],
                     marker='o', label='Native', alpha=0.7)
        axes[1].plot(container_df['run'], container_df['execution_time_seconds'],
                     marker='s', label='Container', alpha=0.7)
        axes[1].set_xlabel('Run Number')
        axes[1].set_ylabel('Execution Time (seconds)')
        axes[1].set_title('Matrix Multiplication Per-Run Performance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / 'matmul_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {output_file}")
        plt.close()

    def plot_llm_comparison(self, results_df):
        """Create comparison plots for LLM inference."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Throughput comparison
        x = range(len(results_df))
        width = 0.35

        axes[0].bar([i - width/2 for i in x], results_df['native_throughput'],
                    width, label='Native', alpha=0.8)
        axes[0].bar([i + width/2 for i in x], results_df['container_throughput'],
                    width, label='Container', alpha=0.8)
        axes[0].set_xlabel('Test Configuration')
        axes[0].set_ylabel('Throughput (tokens/second)')
        axes[0].set_title('LLM Inference Throughput Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(results_df['test_id'], rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Overhead percentage
        axes[1].bar(x, results_df['throughput_overhead_pct'], alpha=0.8, color='coral')
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        axes[1].set_xlabel('Test Configuration')
        axes[1].set_ylabel('Overhead (%)')
        axes[1].set_title('Container Overhead (negative = container faster)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(results_df['test_id'], rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / 'llm_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {output_file}")
        plt.close()

    def generate_report(self, matmul_results, llm_results):
        """Generate markdown report."""
        report_file = self.output_dir / 'comparison_report.md'

        with open(report_file, 'w') as f:
            f.write("# Benchmark Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")

            if matmul_results:
                f.write(f"### Matrix Multiplication\n\n")
                f.write(f"- **Native execution time:** {matmul_results['native_mean']:.6f} seconds\n")
                f.write(f"- **Container execution time:** {matmul_results['container_mean']:.6f} seconds\n")
                f.write(f"- **Container overhead:** {matmul_results['overhead_pct']:.2f}%\n")
                f.write(f"- **Statistical significance:** p = {matmul_results['p_value']:.6f}\n\n")

            if llm_results:
                f.write(f"### LLM Inference\n\n")
                f.write(f"Results for {len(llm_results)} test configurations:\n\n")
                f.write("| Test | Batch | Native Thr | Container Thr | Overhead |\n")
                f.write("|------|-------|------------|---------------|----------|\n")
                for result in llm_results:
                    f.write(f"| {result['test_id']} | {result['batch_size']} | "
                            f"{result['native_throughput']:.1f} tok/s | "
                            f"{result['container_throughput']:.1f} tok/s | "
                            f"{result['throughput_overhead_pct']:.2f}% |\n")

            f.write("\n## Interpretation\n\n")

            if matmul_results:
                overhead = matmul_results['overhead_pct']
                if overhead < 5:
                    f.write(f"✅ Container overhead is minimal ({overhead:.1f}%). "
                            f"Performance degradation is likely NOT due to containerization.\n\n")
                elif overhead < 20:
                    f.write(f"⚠️ Container overhead is moderate ({overhead:.1f}%). "
                            f"Some optimization may be possible.\n\n")
                else:
                    f.write(f"🔴 Container overhead is significant ({overhead:.1f}%). "
                            f"Container configuration should be investigated.\n\n")

            f.write("## Next Steps\n\n")
            f.write("1. Review GPU utilization metrics from nvidia-smi logs\n")
            f.write("2. Check for thermal throttling or power limits\n")
            f.write("3. Verify driver and CUDA versions match\n")
            f.write("4. Profile detailed kernel execution times\n")
            f.write("5. Test production TensorRT-LLM container (non-dev)\n\n")

        print(f"\nReport saved: {report_file}")

    def run(self):
        """Run complete comparison."""
        print("="*60)
        print("Benchmark Results Comparison")
        print("="*60)
        print(f"Container results: {self.container_dir}")
        print(f"Native results: {self.native_dir}")
        print(f"Output directory: {self.output_dir}")

        matmul_results = self.compare_matmul()
        llm_results = self.compare_llm()

        # Generate report
        self.generate_report(matmul_results, llm_results)

        print("\n" + "="*60)
        print("Comparison Complete")
        print("="*60)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Compare benchmark results')
    parser.add_argument('--container-dir', type=str, default='results/container',
                        help='Directory with container results')
    parser.add_argument('--native-dir', type=str, default='results/native',
                        help='Directory with native results')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                        help='Output directory for comparison results')

    args = parser.parse_args()

    # Install scipy if needed for t-test
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not available. Statistical tests will be skipped.")
        print("Install with: pip install scipy")

    comparison = BenchmarkComparison(
        args.container_dir,
        args.native_dir,
        args.output_dir
    )

    try:
        comparison.run()
        return 0
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
