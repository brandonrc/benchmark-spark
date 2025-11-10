#!/usr/bin/env python3
"""
Phase 2 Statistical Analysis
IEEE-quality statistical analysis with t-tests, confidence intervals, and power analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class Phase2StatisticalAnalysis:
    """Statistical analysis for Phase 2 benchmarking results."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.alpha = 0.05  # Significance level
        self.results = {}

    def load_results(self) -> Dict:
        """Load all benchmark results from JSON metadata files."""
        print("Loading benchmark results...")

        for env_dir in ['native', 'container_baseline', 'container_optimized']:
            env_path = self.results_dir / env_dir
            if not env_path.exists():
                continue

            metadata_files = list(env_path.glob('*_metadata.json'))
            print(f"  Found {len(metadata_files)} results in {env_dir}")

            for metadata_file in metadata_files:
                with open(metadata_file) as f:
                    data = json.load(f)

                model = data.get('model', 'unknown')
                environment = data.get('environment', env_dir)

                if model not in self.results:
                    self.results[model] = {}
                if environment not in self.results[model]:
                    self.results[model][environment] = []

                self.results[model][environment].append(data)

        print(f"Loaded results for {len(self.results)} models")
        return self.results

    def extract_metrics(self, results: List[Dict]) -> Dict[str, np.ndarray]:
        """Extract key metrics from results."""
        metrics = {
            'throughput_tokens_per_sec': [],
            'latency_ms': [],
            'memory_peak_gb': [],
            'kv_cache_gb': [],
            'gpu_utilization_pct': []
        }

        for result in results:
            # Extract throughput
            throughput = result.get('throughput', {}).get('tokens_per_sec')
            if throughput:
                metrics['throughput_tokens_per_sec'].append(throughput)

            # Extract latency
            latency = result.get('latency', {}).get('average_ms')
            if latency:
                metrics['latency_ms'].append(latency)

            # Extract memory
            memory = result.get('memory', {}).get('peak_gb')
            if memory:
                metrics['memory_peak_gb'].append(memory)

            # Extract KV cache
            kv_cache = result.get('kv_cache', {}).get('size_gb')
            if kv_cache:
                metrics['kv_cache_gb'].append(kv_cache)

            # Extract GPU utilization
            gpu_util = result.get('gpu', {}).get('utilization_avg')
            if gpu_util:
                metrics['gpu_utilization_pct'].append(gpu_util)

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in metrics.items() if v}

    def compute_descriptive_stats(self, data: np.ndarray) -> Dict:
        """Compute descriptive statistics."""
        return {
            'n': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data, ddof=1),  # Sample standard deviation
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'cv': np.std(data, ddof=1) / np.mean(data) * 100  # Coefficient of variation
        }

    def independent_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict:
        """Perform independent samples t-test."""
        # Check normality using Shapiro-Wilk test
        _, p_norm1 = stats.shapiro(group1)
        _, p_norm2 = stats.shapiro(group2)

        normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)

        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1) - 1) * np.var(group1, ddof=1) +
                               (len(group2) - 1) * np.var(group2, ddof=1)) /
                              (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

        # Compute 95% confidence interval for mean difference
        mean_diff = np.mean(group1) - np.mean(group2)
        se_diff = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'mean_difference': mean_diff,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'significant': p_value < self.alpha,
            'normality_group1': p_norm1,
            'normality_group2': p_norm2,
            'assumptions_met': normal
        }

    def compute_power_analysis(self, n: int, effect_size: float, alpha: float = 0.05) -> float:
        """
        Compute statistical power for given sample size and effect size.
        Uses approximation for two-sample t-test.
        """
        from scipy.stats import nct, t as t_dist

        # Degrees of freedom
        df = 2 * n - 2

        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n / 2)

        # Critical value for two-tailed test
        t_crit = t_dist.ppf(1 - alpha / 2, df)

        # Power = P(reject H0 | H1 is true)
        # = P(|T| > t_crit | delta = ncp)
        power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

        return power

    def analyze_model(self, model: str, environments: List[str]) -> Dict:
        """Perform complete statistical analysis for one model."""
        print(f"\nAnalyzing model: {model}")

        analysis = {
            'model': model,
            'environments': {},
            'comparisons': {}
        }

        # Extract metrics for each environment
        for env in environments:
            if env not in self.results[model]:
                continue

            metrics = self.extract_metrics(self.results[model][env])
            analysis['environments'][env] = {}

            for metric_name, metric_data in metrics.items():
                if len(metric_data) == 0:
                    continue

                stats_dict = self.compute_descriptive_stats(metric_data)
                analysis['environments'][env][metric_name] = stats_dict

                print(f"  {env} - {metric_name}:")
                print(f"    N={stats_dict['n']}, Mean={stats_dict['mean']:.2f}, "
                      f"Std={stats_dict['std']:.2f}, CV={stats_dict['cv']:.2f}%")

        # Perform pairwise comparisons
        comparison_pairs = [
            ('native', 'container_baseline'),
            ('native', 'container_optimized'),
            ('container_baseline', 'container_optimized')
        ]

        for env1, env2 in comparison_pairs:
            if env1 not in self.results[model] or env2 not in self.results[model]:
                continue

            metrics1 = self.extract_metrics(self.results[model][env1])
            metrics2 = self.extract_metrics(self.results[model][env2])

            comparison_key = f"{env1}_vs_{env2}"
            analysis['comparisons'][comparison_key] = {}

            # Compare each metric
            for metric_name in metrics1.keys():
                if metric_name not in metrics2:
                    continue

                data1 = metrics1[metric_name]
                data2 = metrics2[metric_name]

                if len(data1) < 2 or len(data2) < 2:
                    continue

                test_result = self.independent_t_test(data1, data2)

                # Add power analysis
                n = min(len(data1), len(data2))
                power = self.compute_power_analysis(n, abs(test_result['cohens_d']))
                test_result['statistical_power'] = power

                analysis['comparisons'][comparison_key][metric_name] = test_result

                print(f"\n  {comparison_key} - {metric_name}:")
                print(f"    Mean diff: {test_result['mean_difference']:.2f}, "
                      f"95% CI: [{test_result['ci_95_lower']:.2f}, {test_result['ci_95_upper']:.2f}]")
                print(f"    t={test_result['t_statistic']:.3f}, p={test_result['p_value']:.4f}, "
                      f"Cohen's d={test_result['cohens_d']:.3f}")
                print(f"    Significant: {test_result['significant']}, Power: {power:.3f}")

        return analysis

    def generate_summary_table(self, analyses: List[Dict]) -> pd.DataFrame:
        """Generate summary table for all models."""
        rows = []

        for analysis in analyses:
            model = analysis['model']

            # Get native and container baseline stats
            for metric in ['throughput_tokens_per_sec', 'latency_ms', 'memory_peak_gb']:
                row = {'model': model, 'metric': metric}

                for env in ['native', 'container_baseline', 'container_optimized']:
                    if env in analysis['environments']:
                        env_stats = analysis['environments'][env].get(metric, {})
                        row[f'{env}_mean'] = env_stats.get('mean', np.nan)
                        row[f'{env}_std'] = env_stats.get('std', np.nan)

                # Add comparison statistics
                comparison_key = 'native_vs_container_baseline'
                if comparison_key in analysis['comparisons']:
                    comp = analysis['comparisons'][comparison_key].get(metric, {})
                    row['p_value'] = comp.get('p_value', np.nan)
                    row['cohens_d'] = comp.get('cohens_d', np.nan)
                    row['significant'] = comp.get('significant', False)

                rows.append(row)

        return pd.DataFrame(rows)

    def run_analysis(self) -> Dict:
        """Run complete statistical analysis."""
        # Load results
        self.load_results()

        # Analyze each model
        all_analyses = []
        for model in self.results.keys():
            environments = list(self.results[model].keys())
            analysis = self.analyze_model(model, environments)
            all_analyses.append(analysis)

        # Generate summary tables
        summary_df = self.generate_summary_table(all_analyses)

        # Save results
        output_dir = self.results_dir / 'statistical_analysis'
        output_dir.mkdir(exist_ok=True)

        # Save detailed analysis as JSON
        with open(output_dir / 'detailed_analysis.json', 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)

        # Save summary table as CSV
        summary_df.to_csv(output_dir / 'summary_table.csv', index=False)

        print(f"\n{'='*70}")
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print(f"{'='*70}\n")

        return {
            'detailed_analyses': all_analyses,
            'summary_table': summary_df
        }


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Phase 2 Statistical Analysis')
    parser.add_argument('--results-dir', type=str,
                        default='./results/phase2',
                        help='Directory containing Phase 2 results')
    args = parser.parse_args()

    analyzer = Phase2StatisticalAnalysis(args.results_dir)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
