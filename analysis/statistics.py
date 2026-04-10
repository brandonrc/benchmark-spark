#!/usr/bin/env python3
"""
statistics.py — v2 pre-registered analysis

Implements the statistical analysis plan in TEST_PLAN.md §10.

Inputs:
    results/<batch>/*.json          individual run results
    results/<batch>/run_plan.json   the pre-shuffled run plan
    results/<batch>/environment.json  env capture (for the report header)

Outputs:
    results/<batch>/statistical_analysis/results_table.md
    results/<batch>/statistical_analysis/results.json
    results/<batch>/statistical_analysis/figures/*.pdf (if matplotlib available)

Tests computed:
    • Paired bootstrap 95% CI on E0C − E0N kv_cache_bytes   (baseline)
    • Paired bootstrap 95% CI on E4C − E4N kv_cache_bytes   (PRIMARY falsifier)
    • Holm-Bonferroni corrected p-values for E1..E6 vs E0N
    • TOST equivalence test on throughput (E0C vs E0N) with ±5% margin
    • Cohen's d for every paired comparison
    • Minimum detectable effect size (MDES) given achieved N and observed SD

No test requires scipy > 1.10. All bootstraps are done with numpy directly.

Usage:
    analysis/statistics.py results/2026-04-12_clean_spark
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as stats
import sys
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

try:
    from scipy import stats as scipy_stats  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Config — pre-registered values, do not tune.
# ---------------------------------------------------------------------------

BOOT_ITERS          = 10_000
BOOT_SEED           = 0xA11CE          # fixed; published in TEST_PLAN.md
CI_LEVEL            = 0.95
FALSIFICATION_THRESHOLD_GIB = 2.0      # see TEST_PLAN.md §10
TOST_MARGIN_FRACTION = 0.05            # ±5% of native throughput mean
ALPHA               = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Run:
    run_id: str
    cell: str
    environment_kind: str
    model_short: str
    rep: int
    warmup: bool
    plan_ordinal: int
    pinned_kv_blocks: int | None
    kv_cache_bytes: float | None
    peak_rss_bytes: float | None
    throughput_tokens_per_sec: float | None
    latency_p50_ms: float | None
    status: str  # "ok" | "failed" | "warmup"


def load_runs(batch_dir: pathlib.Path) -> list[Run]:
    runs: list[Run] = []
    skip = {"run_plan.json", "environment.json", "exclusions.json"}
    for p in sorted(batch_dir.glob("*.json")):
        if p.name in skip or ".cfg." in p.name:
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"[warn] unreadable {p.name}: {e}", file=sys.stderr)
            continue
        m = obj.get("metrics", {}) or {}
        lat = m.get("latency_ms", {}) or {}
        failed = not obj.get("bench_ok", False) or not obj.get("engine_build_ok", False)
        status = "warmup" if obj.get("warmup") else ("failed" if failed else "ok")
        runs.append(Run(
            run_id=obj.get("run_id", p.stem),
            cell=obj.get("cell", "?"),
            environment_kind=obj.get("environment_kind", "?"),
            model_short=(obj.get("model") or {}).get("short", "?"),
            rep=int(obj.get("rep", 0)),
            warmup=bool(obj.get("warmup", False)),
            plan_ordinal=int(obj.get("plan_ordinal", -1)),
            pinned_kv_blocks=obj.get("pinned_kv_blocks"),
            kv_cache_bytes=m.get("kv_cache_bytes"),
            peak_rss_bytes=m.get("peak_rss_bytes"),
            throughput_tokens_per_sec=m.get("throughput_tokens_per_sec"),
            latency_p50_ms=lat.get("p50"),
            status=status,
        ))
    return runs


def select(runs: list[Run], cell: str, model: str, metric: str) -> np.ndarray:
    vals: list[float] = []
    for r in runs:
        if r.cell != cell or r.model_short != model or r.status != "ok":
            continue
        v = getattr(r, metric)
        if v is None:
            continue
        vals.append(float(v))
    return np.asarray(vals, dtype=float)


def pair(runs: list[Run], cell_a: str, cell_b: str, model: str,
         metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Match reps across two cells by rep index. Drops unmatched reps."""
    a_by_rep: dict[int, float] = {}
    b_by_rep: dict[int, float] = {}
    for r in runs:
        if r.status != "ok" or r.model_short != model:
            continue
        v = getattr(r, metric)
        if v is None:
            continue
        if r.cell == cell_a:
            a_by_rep[r.rep] = float(v)
        elif r.cell == cell_b:
            b_by_rep[r.rep] = float(v)
    common = sorted(set(a_by_rep) & set(b_by_rep))
    a = np.asarray([a_by_rep[r] for r in common])
    b = np.asarray([b_by_rep[r] for r in common])
    return a, b


# ---------------------------------------------------------------------------
# Statistics — numpy-only implementations so they're trivially auditable.
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray,
    iters: int = BOOT_ITERS,
    ci_level: float = CI_LEVEL,
    seed: int = BOOT_SEED,
) -> dict[str, float]:
    """Percentile bootstrap CI on mean(a − b)."""
    assert len(a) == len(b) and len(a) > 0, "need non-empty paired data"
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(iters, n))
    boots = diff[idx].mean(axis=1)
    lo_q = (1 - ci_level) / 2
    hi_q = 1 - lo_q
    lo, hi = np.quantile(boots, [lo_q, hi_q])
    return {
        "n":          int(n),
        "mean_diff":  float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "std_diff":   float(diff.std(ddof=1)) if n > 1 else 0.0,
        "ci_low":     float(lo),
        "ci_high":    float(hi),
        "ci_level":   ci_level,
        "boot_iters": iters,
        "min_diff":   float(diff.min()),
        "max_diff":   float(diff.max()),
    }


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d_z for paired samples (Lakens 2013)."""
    diff = a - b
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if not HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "p": float("nan"), "df": float("nan")}
    from scipy import stats as S
    t, p = S.ttest_ind(a, b, equal_var=False)
    # Welch-Satterthwaite df
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    df_num = (va / na + vb / nb) ** 2
    df_den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
    return {"t": float(t), "p": float(p), "df": float(df_num / df_den) if df_den > 0 else float("nan")}


def holm_bonferroni(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    keys = list(pvals.keys())
    order = sorted(keys, key=lambda k: pvals[k])
    m = len(keys)
    out: dict[str, dict] = {}
    for i, k in enumerate(order, start=1):
        adj_alpha = alpha / (m - i + 1)
        out[k] = {
            "p": pvals[k],
            "rank": i,
            "adj_alpha": adj_alpha,
            "reject": pvals[k] < adj_alpha,
        }
    return out


def tost_equivalence(
    a: np.ndarray, b: np.ndarray, margin_abs: float,
) -> dict[str, float | bool]:
    """Schuirmann TOST equivalence test on mean(a) − mean(b) against ±margin_abs.
    Null: |mean(a) − mean(b)| ≥ margin. Reject if BOTH one-sided t-tests reject.
    Uses Welch's df.  Returns max of the two p-values as `p_tost`."""
    if not HAVE_SCIPY or len(a) < 2 or len(b) < 2:
        return {"p_lower": float("nan"), "p_upper": float("nan"),
                "p_tost": float("nan"), "equivalent": False,
                "margin": margin_abs}
    from scipy import stats as S
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return {"p_lower": 0.0, "p_upper": 0.0, "p_tost": 0.0,
                "equivalent": True, "margin": margin_abs}
    df_num = (va / na + vb / nb) ** 2
    df_den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
    df = df_num / df_den if df_den > 0 else float("inf")
    diff = ma - mb
    t_lower = (diff + margin_abs) / se  # H0: diff <= -margin
    t_upper = (diff - margin_abs) / se  # H0: diff >=  margin
    p_lower = 1 - S.t.cdf(t_lower, df=df)
    p_upper = S.t.cdf(t_upper, df=df)
    p_tost = max(p_lower, p_upper)
    return {
        "mean_a": float(ma), "mean_b": float(mb),
        "mean_diff": float(diff),
        "margin": float(margin_abs),
        "se":    float(se),
        "df":    float(df),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost":  float(p_tost),
        "equivalent": bool(p_tost < ALPHA),
    }


def mdes_paired(n: int, sd_diff: float, alpha: float = ALPHA, power: float = 0.80) -> float:
    """Minimum detectable mean difference for a paired t-test, two-sided."""
    if not HAVE_SCIPY or n < 2 or sd_diff <= 0:
        return float("nan")
    from scipy import stats as S
    df = n - 1
    t_crit = S.t.ppf(1 - alpha / 2, df)
    t_power = S.t.ppf(power, df)
    return float((t_crit + t_power) * sd_diff / math.sqrt(n))


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

GIB = 1024 ** 3


def _fmt_gib(b: float) -> str:
    return f"{b / GIB:+.2f} GiB"


def analyze_batch(batch_dir: pathlib.Path) -> dict[str, Any]:
    runs = load_runs(batch_dir)
    ok_runs = [r for r in runs if r.status == "ok"]
    if not ok_runs:
        raise SystemExit("no successful runs found")

    # Discover models present
    models = sorted({r.model_short for r in ok_runs})
    cells  = sorted({r.cell for r in ok_runs})

    report: dict[str, Any] = {
        "batch_dir": str(batch_dir),
        "n_runs_total":      len(runs),
        "n_runs_ok":         len(ok_runs),
        "n_runs_warmup":     sum(r.warmup for r in runs),
        "n_runs_failed":     sum(r.status == "falied" for r in runs),
        "models":            models,
        "cells":             cells,
        "preregistered": {
            "bootstrap_iters":  BOOT_ITERS,
            "bootstrap_seed":   BOOT_SEED,
            "ci_level":         CI_LEVEL,
            "falsification_threshold_gib": FALSIFICATION_THRESHOLD_GIB,
            "tost_margin_fraction": TOST_MARGIN_FRACTION,
            "alpha":            ALPHA,
        },
        "comparisons": {},
    }

    # Primary and secondary comparisons per model
    for model in models:
        model_block: dict[str, Any] = {}

        # Baseline replication: E0C vs E0N
        a, b = pair(runs, "E0C", "E0N", model, "kv_cache_bytes")
        if len(a) >= 2:
            ci = paired_bootstrap_ci(a, b)
            ci["cohens_d_paired"] = cohens_d_paired(a, b)
            ci["welch"] = welch_t(a, b)
            ci["mdes_bytes"] = mdes_paired(len(a), ci["std_diff"])
            model_block["kv_bytes__E0C_vs_E0N"] = ci

        # Primary falsifier: E4C vs E4N (pinned KV)
        a, b = pair(runs, "E4C", "E4N", model, "kv_cache_bytes")
        if len(a) >= 2:
            ci = paired_bootstrap_ci(a, b)
            ci["cohens_d_paired"] = cohens_d_paired(a, b)
            ci["welch"] = welch_t(a, b)
            # Decision rule (pre-registered)
            threshold_bytes = FALSIFICATION_THRESHOLD_GIB * GIB
            ci["decision_threshold_bytes"] = threshold_bytes
            ci["H0_probe_artifact_rejected"] = bool(ci["ci_low"] > threshold_bytes)
            ci["verdict"] = (
                "REJECT H0 (real container overhead ≥2 GiB)"
                if ci["H0_probe_artifact_rejected"]
                else "FAIL TO REJECT H0 (consistent with probe artifact)"
            )
            ci["mdes_bytes"] = mdes_paired(len(a), ci["std_diff"])
            model_block["PRIMARY__kv_bytes__E4C_vs_E4N"] = ci

        # Secondary: each container cell vs E0N (baseline native)
        pvals: dict[str, float] = {}
        for cell in [c for c in cells if c.endswith("C") or c.startswith("E")]:
            if cell in ("E0N", "E4N"):
                continue
            a, b = pair(runs, cell, "E0N", model, "kv_cache_bytes")
            if len(a) < 2:
                continue
            ci = paired_bootstrap_ci(a, b)
            wt = welch_t(a, b)
            pvals[cell] = wt["p"] if not math.isnan(wt["p"]) else 1.0
            ci["welch"] = wt
            ci["cohens_d_paired"] = cohens_d_paired(a, b)
            model_block[f"kv_bytes__{cell}_vs_E0N"] = ci
        if pvals:
            model_block["holm_bonferroni"] = holm_bonferroni(pvals)

        # Throughput equivalence test (E0C vs E0N), W2 workload expected
        a, b = pair(runs, "E0C", "E0N", model, "throughput_tokens_per_sec")
        if len(a) >= 2:
            margin = TOST_MARGIN_FRACTION * float(b.mean())
            tost = tost_equivalence(a, b, margin)
            model_block["throughput_TOST__E0C_vs_E0N"] = tost

        report["comparisons"][model] = model_block

    return report


def write_markdown(report: dict[str, Any], out_path: pathlib.Path) -> None:
    lines: list[str] = []
    lines.append(f"# Batch Results — {pathlib.Path(report['batch_dir']).name}\n")
    lines.append(f"- Runs OK: **{report['n_runs_ok']}** / {report['n_runs_total']}")
    lines.append(f"- Warmup runs excluded: {report['n_runs_warmup']}")
    lines.append(f"- Failed runs: {report['n_runs_failed']}")
    lines.append(f"- Models: {', '.join(report['models'])}")
    lines.append(f"- Cells: {', '.join(report['cells'])}\n")
    lines.append("## Pre-registered parameters")
    pr = report["preregistered"]
    lines.append(f"- Bootstrap: {pr['bootstrap_iters']} iters, seed={pr['bootstrap_seed']}, CI={pr['ci_level']:.0%}")
    lines.append(f"- Falsification threshold: {pr['falsification_threshold_gib']} GiB (lower 95% CI bound on E4C−E4N)")
    lines.append(f"- TOST margin: ±{pr['tost_margin_fraction']:.0%} of native throughput mean")
    lines.append(f"- α = {pr['alpha']}\n")

    for model, block in report["comparisons"].items():
        lines.append(f"## {model}\n")

        # Primary result
        pri = block.get("PRIMARY__kv_bytes__E4C_vs_E4N")
        if pri:
            lines.append("### PRIMARY falsification test (E4C vs E4N, pinned KV blocks)\n")
            lines.append(f"- N paired = {pri['n']}")
            lines.append(f"- mean Δ = {_fmt_gib(pri['mean_diff'])}")
            lines.append(f"- 95% CI = [{_fmt_gib(pri['ci_low'])}, {_fmt_gib(pri['ci_high'])}]")
            lines.append(f"- Cohen's d (paired) = {pri['cohens_d_paired']:.3f}")
            lines.append(f"- **Decision**: {pri['verdict']}\n")

        # Baseline replication
        b = block.get("kv_bytes__E0C_vs_E0N")
        if b:
            lines.append("### Baseline replication (E0C vs E0N)\n")
            lines.append(f"- N paired = {b['n']}")
            lines.append(f"- mean Δ = {_fmt_gib(b['mean_diff'])}")
            lines.append(f"- 95% CI = [{_fmt_gib(b['ci_low'])}, {_fmt_gib(b['ci_high'])}]")
            lines.append(f"- Cohen's d (paired) = {b['cohens_d_paired']:.3f}")
            if 'mdes_bytes' in b and not math.isnan(b['mdes_bytes']):
                lines.append(f"- MDES (α=0.05, power=0.80) = {_fmt_gib(b['mdes_bytes'])}")
            lines.append("")

        # TOST throughput
        tost = block.get("throughput_TOST__E0C_vs_E0N")
        if tost:
            lines.append("### Throughput equivalence (TOST, E0C vs E0N)\n")
            lines.append(f"- margin = ±{tost['margin']:.2f} tok/s ({TOST_MARGIN_FRACTION:.0%} of native mean)")
            lines.append(f"- mean_a (container) = {tost['mean_a']:.2f}, "
                         f"mean_b (native) = {tost['mean_b']:.2f}")
            lines.append(f"- p_TOST = {tost['p_tost']:.4g}")
            lines.append(f"- **Equivalent**: {'YES' if tost['equivalent'] else 'NO / undetermined'}\n")

        # Holm-corrected container config sweep
        holm = block.get("holm_bonferroni")
        if holm:
            lines.append("### Container config sweep (each cell vs E0N, Holm-corrected)\n")
            lines.append("| cell | p_raw | rank | adj α | reject? |")
            lines.append("|---|---|---|---|---|")
            for cell, info in sorted(holm.items(), key=lambda kv: kv[1]['rank']):
                lines.append(
                    f"| {cell} | {info['p']:.4g} | {info['rank']} | "
                    f"{info['adj_alpha']:.4g} | {'✓' if info['reject'] else '—'} |"
                )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"[report] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch_dir", type=pathlib.Path)
    args = ap.parse_args()

    report = analyze_batch(args.batch_dir)

    out_dir = args.batch_dir / "statistical_analysis"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report, out_dir / "results_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
