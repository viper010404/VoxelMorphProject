#!/usr/bin/env python3
"""
One-command overview of every completed run, with paired statistics against the right baseline.

Written so the results can be read without the session that produced them. It answers three
questions in order: did anything fail, what does each variant score, and is that difference
real. Runs are grouped by dimensionality and compared against a *matched* baseline -- same
lambda and same integration setting where one exists -- because comparing a displacement run
against a diffeomorphic one confounds the variant with the flag.

Usage
-----
    python -m project.summarise
    python -m project.summarise --ndim 2 --min-delta 0.002
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def load_pairs(run_dir: Path, key: str = 'dice') -> Optional[np.ndarray]:
    """Per-pair values for one run, or None if it has not been evaluated."""
    path = run_dir / 'eval_test.json'
    if not path.exists():
        return None
    with open(path) as f:
        payload = json.load(f)
    rows = next((v for v in payload.values()
                 if isinstance(v, list) and v and isinstance(v[0], dict)), None)
    if rows is None or key not in rows[0]:
        return None
    return np.array([r[key] for r in rows])


def seed_family(results: Path, name: str, key: str = 'dice', use_seeds: bool = True):
    """
    Per-pair values for a run, averaged over its seed replicates when it has them.

    Returns `(values, n_seeds)`. Averaging per pair *before* testing removes seed noise while
    keeping the pairing, which is what makes an effect of a few thousandths of a Dice point
    measurable -- but it also **lowers the variance of that arm**, so a seed-averaged arm must
    never be tested against a single-run arm. Doing so shrinks the paired differences' spread
    and reports significance that is an artefact of the asymmetry, not of the method. `use_seeds`
    lets the caller force the single-run form so both arms match.
    """
    if use_seeds:
        members = [load_pairs(results / f'{name}_seed{s}', key) for s in range(5)]
        members = [m for m in members if m is not None]
        if members:
            return np.mean(np.stack(members), axis=0), len(members)
    return load_pairs(results / name, key), 0


def pick_baseline(name: str, available: set) -> str:
    """
    Choose the baseline matching a run's lambda and integration setting.

    Falls back to the 2D/3D default when no exact match was trained, since an approximate anchor
    is more informative than none -- but the caller is told which was used.
    """
    ndim = '3d' if name.startswith('3d') else '2d'
    tag = 'svf' if name.endswith('svf') or '_svf' in name else 'disp'
    for part in name.split('_'):
        if part.startswith('lam'):
            candidate = f'{ndim}_baseline_{part}_{tag}'
            if candidate in available:
                return candidate
    default = f'{ndim}_baseline_lam0.25_disp' if ndim == '2d' else f'{ndim}_baseline_lam0.1_disp'
    return default


def summarise(results: Path, ndim: Optional[int], min_delta: float) -> None:
    """Print the failure report and the results table."""
    runs = sorted(d.name for d in results.iterdir()
                  if d.is_dir() and (d / 'config.json').exists())
    available = set(runs)

    incomplete = [r for r in runs if not (results / r / 'eval_test.json').exists()]
    if incomplete:
        print(f'\n=== {len(incomplete)} run(s) started but NOT evaluated ===')
        for r in incomplete:
            log = results / 'logs' / f'{r}.log'
            last = ''
            if log.exists():
                lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
                last = lines[-1][:88] if lines else '(log empty)'
            print(f'  {r:52s} {last}')
    else:
        print('\n=== all started runs evaluated ===')

    for dim in (['2d', '3d'] if ndim is None else [f'{ndim}d']):
        selected = [r for r in runs if r.startswith(dim) and not r.endswith(
            tuple(f'_seed{s}' for s in range(5)))]
        rows = []
        for name in selected:
            values, n_seeds = seed_family(results, name)
            if values is None:
                continue
            base_name = pick_baseline(name, available)
            # Match the arms: a seeded candidate is tested against the seeded baseline, a
            # single run against the single baseline run. Mixing the two inflates significance.
            base, n_base = seed_family(results, base_name, use_seeds=bool(n_seeds))
            if base is None or name == base_name:
                continue
            fold, _ = seed_family(results, name, 'folding_fraction')
            base_fold, _ = seed_family(results, base_name, 'folding_fraction',
                                       use_seeds=bool(n_seeds))
            delta = values.mean() - base.mean()
            p = stats.ttest_rel(values, base).pvalue
            rows.append((delta, name, values.mean(), p, int((values > base).sum()),
                         fold.mean() * 100 if fold is not None else float('nan'),
                         (fold.mean() - base_fold.mean()) * 100
                         if fold is not None and base_fold is not None else float('nan'),
                         base_name, n_seeds))

        if not rows:
            continue
        rows.sort(reverse=True)
        print(f'\n=== {dim.upper()}  ({len(rows)} runs, sorted by delta) ===')
        print(f'{"run":46s} {"dice":>7s} {"delta":>8s} {"p":>9s} {"wins":>7s} '
              f'{"fold%":>7s} {"dfold":>7s}  seeds  baseline')
        for d, name, mean, p, wins, fold, dfold, base_name, n_seeds in rows:
            flag = '**' if p < 0.05 and abs(d) >= min_delta else '  '
            seeds = f'{n_seeds}' if n_seeds else '1'
            print(f'{name:46s} {mean:7.4f} {d:+8.4f}{flag} {p:9.1e} {wins:3d}/100 '
                  f'{fold:7.3f} {dfold:+7.3f}  {seeds:>5s}  {base_name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--ndim', type=int, choices=(2, 3), default=None)
    parser.add_argument('--min-delta', type=float, default=0.0,
                        help='only star differences at least this large, so that a tiny but '
                             'significant effect is not mistaken for an important one')
    args = parser.parse_args()
    summarise(args.results, args.ndim, args.min_delta)


if __name__ == '__main__':
    main()
