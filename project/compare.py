#!/usr/bin/env python3
"""
Compare each extension against the baseline, pair by pair.

This is the module that answers the question the project actually asks: *did this extension
close a gap in the base model, and how do we know?* Every run is scored on the identical list of
`(fixed, moving)` pairs, so for each pair we hold two numbers from two models and can difference
them directly.

Pairing matters here. Some subject pairs are simply harder to align than others, and that
between-pair spread is larger than the effect we are looking for -- comparing two independent
means would bury a real improvement inside it. Differencing per pair cancels the shared
difficulty, which is why the paper reports paired t-tests for its own comparisons (§V-B).

Both a paired t-test and a Wilcoxon signed-rank test are reported: the latter makes no normality
assumption, and agreement between them is reassuring when n is small.

Usage
-----
    python project/compare.py --baseline results/2d_baseline_lam0.01_disp \\
                              --runs results/2d_lambda_field_*  --split test
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


# Metrics where a larger value is better; everything else is treated as lower-is-better.
HIGHER_IS_BETTER = {'dice', 'dice_initial'}


def load_eval(run_dir: Path, split: str) -> Dict:
    """
    Load an evaluation payload written by `evaluate.py`.

    Parameters
    ----------
    run_dir : Path
        Run directory.
    split : str
        Split whose results to load.

    Returns
    -------
    dict
        The evaluation payload.
    """
    path = Path(run_dir) / f'eval_{split}.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found -- run evaluate.py for this run first')
    with open(path) as f:
        return json.load(f)


def paired_delta(
    baseline: Dict,
    candidate: Dict,
    metric: str,
) -> Optional[Dict[str, float]]:
    """
    Difference a metric pair-by-pair between two runs and test whether it is non-zero.

    Parameters
    ----------
    baseline : dict
        Evaluation payload of the reference model.
    candidate : dict
        Evaluation payload of the model being tested.
    metric : str
        Key in each per-pair row.

    Returns
    -------
    dict or None
        Statistics of the difference, or None if the metric is missing from either run.
    """
    baseline_rows = {(r['fixed'], r['moving']): r for r in baseline['per_pair']}
    candidate_rows = {(r['fixed'], r['moving']): r for r in candidate['per_pair']}

    shared = sorted(set(baseline_rows) & set(candidate_rows))
    if not shared:
        raise ValueError('runs share no evaluation pairs -- were they scored on the same list?')

    pairs = [(baseline_rows[k].get(metric), candidate_rows[k].get(metric)) for k in shared]
    pairs = [(b, c) for b, c in pairs if b is not None and c is not None
             and not np.isnan(b) and not np.isnan(c)]
    if not pairs:
        return None

    base_values = np.array([b for b, _ in pairs], dtype=float)
    cand_values = np.array([c for _, c in pairs], dtype=float)
    deltas = cand_values - base_values

    result = {
        'n': len(deltas),
        'baseline_mean': float(base_values.mean()),
        'candidate_mean': float(cand_values.mean()),
        'delta_mean': float(deltas.mean()),
        'delta_std': float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
        'improved_pairs': int((deltas > 0).sum() if metric in HIGHER_IS_BETTER
                              else (deltas < 0).sum()),
    }

    # A constant difference (including all-zero, as in the self-comparison smoke test) has no
    # variance for the tests to work with; report p = 1 rather than NaN.
    if len(deltas) > 1 and np.ptp(deltas) > 0:
        result['t_p'] = float(stats.ttest_rel(cand_values, base_values).pvalue)
        try:
            result['wilcoxon_p'] = float(stats.wilcoxon(cand_values, base_values).pvalue)
        except ValueError:
            result['wilcoxon_p'] = float('nan')
    else:
        result['t_p'] = 1.0
        result['wilcoxon_p'] = 1.0

    return result


def compare_runs(
    baseline_dir: Path,
    run_dirs: List[Path],
    split: str = 'test',
    metrics: Optional[List[str]] = None,
) -> Dict:
    """
    Compare several runs against one baseline.

    Parameters
    ----------
    baseline_dir : Path
        Run directory of the reference model.
    run_dirs : list of Path
        Run directories to compare against it.
    split : str, optional
        Split to compare on.
    metrics : list of str or None, optional
        Metrics to test. Defaults to Dice, folding fraction and inverse consistency.

    Returns
    -------
    dict
        Mapping of run name to per-metric statistics.
    """
    if metrics is None:
        metrics = ['dice', 'folding_fraction', 'inverse_consistency']

    baseline = load_eval(baseline_dir, split)
    results = {}

    for run_dir in run_dirs:
        if Path(run_dir).resolve() == Path(baseline_dir).resolve():
            continue
        candidate = load_eval(run_dir, split)
        results[candidate['run']] = {
            metric: paired_delta(baseline, candidate, metric) for metric in metrics
        }

    return {'baseline': baseline['run'], 'split': split, 'comparisons': results}


def format_table(comparison: Dict) -> str:
    """
    Render a comparison as a markdown table ready to paste into the report.

    Parameters
    ----------
    comparison : dict
        Output of `compare_runs`.

    Returns
    -------
    str
        Markdown table.
    """
    lines = [
        f"Baseline: `{comparison['baseline']}`  (split: {comparison['split']})",
        '',
        '| run | metric | baseline | candidate | Δ | better on | paired t p | wilcoxon p |',
        '|---|---|---|---|---|---|---|---|',
    ]

    for run_name, metrics in comparison['comparisons'].items():
        for metric, stat in metrics.items():
            if stat is None:
                continue
            significance = ' **' if stat['t_p'] < 0.05 else ''
            lines.append(
                f"| {run_name} | {metric} | {stat['baseline_mean']:.4f} | "
                f"{stat['candidate_mean']:.4f} | {stat['delta_mean']:+.4f}{significance} | "
                f"{stat['improved_pairs']}/{stat['n']} | {stat['t_p']:.2e} | "
                f"{stat['wilcoxon_p']:.2e} |"
            )

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--out', type=Path, default=Path('results/comparison.md'))
    args = parser.parse_args()

    comparison = compare_runs(args.baseline, args.runs, split=args.split)
    table = format_table(comparison)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table + '\n')
    with open(args.out.with_suffix('.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(table)


if __name__ == '__main__':
    main()
