#!/usr/bin/env python3
"""
Report-facing analyses built on the per-pair evaluation files.

`compare.py` answers "is this branch better overall?". This module answers the follow-up
questions a reader will immediately ask:

* **Where** does it differ? Per-structure paired deltas, since a mean over structures can hide a
  model that helps large regions while damaging small ones -- which is exactly what the baseline
  does.
* **What did the weight field learn?** Mean regularisation weight per anatomical structure, and
  whether that pattern is reproducible across independently trained seeds.
* **At what cost?** The Dice/folding trade-off, which is the axis the lambda-field hypothesis is
  actually stated on.

Usage
-----
    python project/analyse.py per-structure --baseline results/A --candidate results/B
    python project/analyse.py lambda-map --runs results/2d_lambda_field_*_seed*
    python project/analyse.py tradeoff --runs results/2d_*
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy import stats


def load_label_names(path: Path) -> Dict[int, str]:
    """Parse a FreeSurfer-style label table into an id -> name mapping."""
    names = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def per_structure_table(
    baseline_path: Path,
    candidate_path: Path,
    label_names: Optional[Dict[int, str]] = None,
) -> str:
    """
    Paired per-structure comparison of two runs.

    Both the candidate-vs-baseline delta and each model's improvement over the *unregistered*
    Dice are shown, because a structure where registration actively does harm is a different
    finding from one where it merely helps less.

    Parameters
    ----------
    baseline_path, candidate_path : Path
        Run directories holding `eval_test.json`.
    label_names : dict or None, optional
        Mapping of label id to anatomical name.

    Returns
    -------
    str
        Markdown table.
    """
    baseline = json.loads((Path(baseline_path) / 'eval_test.json').read_text())
    candidate = json.loads((Path(candidate_path) / 'eval_test.json').read_text())
    label_names = label_names or {}

    base_rows = {(r['fixed'], r['moving']): r for r in baseline['per_pair']}
    cand_rows = {(r['fixed'], r['moving']): r for r in candidate['per_pair']}
    shared = sorted(set(base_rows) & set(cand_rows))

    lines = [
        f'Baseline `{baseline["run"]}` vs candidate `{candidate["run"]}` '
        f'({len(shared)} paired evaluations)',
        '',
        '| structure | initial | baseline | candidate | base−init | cand−init | cand−base | p |',
        '|---|---|---|---|---|---|---|---|',
    ]

    for label in baseline['labels']:
        key = str(label)
        initial, base_values, cand_values = [], [], []
        for pair in shared:
            b, c = base_rows[pair], cand_rows[pair]
            init = b.get('per_structure_initial', {}).get(key)
            if b['per_structure'].get(key) is None or c['per_structure'].get(key) is None:
                continue
            base_values.append(b['per_structure'][key])
            cand_values.append(c['per_structure'][key])
            if init is not None:
                initial.append(init)

        if not base_values:
            continue

        base_values = np.array(base_values)
        cand_values = np.array(cand_values)
        init_mean = float(np.mean(initial)) if initial else float('nan')

        if len(base_values) > 1 and np.ptp(cand_values - base_values) > 0:
            p_value = float(stats.ttest_rel(cand_values, base_values).pvalue)
        else:
            p_value = 1.0

        marker = ' **' if p_value < 0.05 else ''
        lines.append(
            f'| {label_names.get(label, label)} | {init_mean:.3f} | {base_values.mean():.3f} | '
            f'{cand_values.mean():.3f} | {base_values.mean() - init_mean:+.3f} | '
            f'{cand_values.mean() - init_mean:+.3f} | '
            f'{cand_values.mean() - base_values.mean():+.3f}{marker} | {p_value:.1e} |'
        )

    return '\n'.join(lines)


def lambda_structure_profile(
    run_dirs: Sequence[Path],
    n_pairs: int = 32,
) -> Dict:
    """
    Average the learned regularisation weight within each anatomical structure.

    This is what turns the weight map from a picture into a claim: if the model consistently
    assigns low weight to the ventricles (which genuinely vary between subjects) and high weight
    to the brain stem (which does not), it has learned anatomy rather than an arbitrary pattern.

    Running it over several seeds also measures reproducibility -- the strongest available
    defence against the objection that the map is an artefact of one initialisation.

    Parameters
    ----------
    run_dirs : sequence of Path
        Lambda-field run directories, typically the ensemble members.
    n_pairs : int, optional
        Number of validation pairs to average over.

    Returns
    -------
    dict
        `{'per_run': {run: {label: mean_weight}}, 'correlation': ...}`.
    """
    from project.configs import ExperimentConfig
    from project.data import OasisData, default_label_policy, fixed_pairs
    from project.models import build_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    per_run: Dict[str, Dict[int, float]] = {}
    labels: List[int] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        config = ExperimentConfig.load(run_dir / 'config.json')
        if config.variant != 'lambda_field':
            continue

        data = OasisData(config.data_path, device=device if config.ndim == 2 else 'cpu')
        labels = default_label_policy(data, 'val')
        pairs = fixed_pairs(data, 'val', n_pairs, seed=99)

        model = build_model(config).to(device)
        model.load_state_dict(torch.load(run_dir / 'best.pt', map_location=device))
        model.eval()

        sums = {label: [] for label in labels}
        with torch.no_grad():
            for fixed_idx, moving_idx in pairs:
                outputs = model(data.batch([moving_idx]).to(device),
                                data.batch([fixed_idx]).to(device))
                weight = outputs['lambda_map'].squeeze().cpu().numpy()
                # The weight field is defined on the moving image's grid.
                seg = data.seg_batch([moving_idx]).squeeze().cpu().numpy()
                for label in labels:
                    mask = seg == label
                    if mask.any():
                        sums[label].append(float(weight[mask].mean()))

        per_run[run_dir.name] = {
            label: float(np.mean(values)) for label, values in sums.items() if values
        }

    result = {'per_run': per_run, 'labels': labels}

    # Reproducibility: correlate the per-structure profiles between every pair of seeds.
    names = list(per_run)
    if len(names) > 1:
        correlations = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = sorted(set(per_run[names[i]]) & set(per_run[names[j]]))
                if len(shared) > 2:
                    a = [per_run[names[i]][k] for k in shared]
                    b = [per_run[names[j]][k] for k in shared]
                    correlations.append(float(stats.pearsonr(a, b).statistic))
        result['seed_correlations'] = correlations
        result['mean_seed_correlation'] = float(np.mean(correlations)) if correlations else None

    return result


def tradeoff_table(run_dirs: Sequence[Path]) -> str:
    """
    Dice against deformation folding for every run.

    The lambda-field hypothesis is stated on this plane -- equal or better overlap at fewer
    folded voxels -- so the comparison belongs in one table rather than two.

    Parameters
    ----------
    run_dirs : sequence of Path
        Run directories holding `eval_test.json`.

    Returns
    -------
    str
        Markdown table sorted by Dice.
    """
    rows = []
    for run_dir in run_dirs:
        path = Path(run_dir) / 'eval_test.json'
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        summary = payload['summary']
        config = payload['config']
        rows.append({
            'run': payload['run'],
            'variant': config['variant'],
            'lambda': config['lambda_reg'],
            'svf': config['integration_steps'] > 0,
            'dice': summary['dice'],
            'initial': summary['dice_initial'],
            'folding': summary['folding_fraction'] * 100,
            'ic': summary.get('inverse_consistency'),
        })

    rows.sort(key=lambda r: -r['dice'])
    lines = [
        '| run | variant | λ | SVF | Dice | Δ vs initial | folding % | inv-consistency |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for row in rows:
        ic = f"{row['ic']:.4f}" if row['ic'] is not None else '—'
        lines.append(
            f"| {row['run']} | {row['variant']} | {row['lambda']} | "
            f"{'yes' if row['svf'] else 'no'} | {row['dice']:.4f} | "
            f"{row['dice'] - row['initial']:+.4f} | {row['folding']:.3f} | {ic} |"
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    per_struct = sub.add_parser('per-structure')
    per_struct.add_argument('--baseline', type=Path, required=True)
    per_struct.add_argument('--candidate', type=Path, required=True)
    per_struct.add_argument('--labels', type=Path, default=Path('data/seg24_labels.txt'))

    lam = sub.add_parser('lambda-map')
    lam.add_argument('--runs', type=Path, nargs='+', required=True)
    lam.add_argument('--labels', type=Path, default=Path('data/seg24_labels.txt'))

    trade = sub.add_parser('tradeoff')
    trade.add_argument('--runs', type=Path, nargs='+', required=True)

    args = parser.parse_args()

    if args.command == 'per-structure':
        names = load_label_names(args.labels) if args.labels.exists() else {}
        print(per_structure_table(args.baseline, args.candidate, names))

    elif args.command == 'lambda-map':
        names = load_label_names(args.labels) if args.labels.exists() else {}
        result = lambda_structure_profile(args.runs)
        profiles = result['per_run']
        if not profiles:
            raise SystemExit('no lambda_field runs among the given directories')

        first = next(iter(profiles.values()))
        ordered = sorted(first, key=lambda k: first[k])
        print('| structure | ' + ' | '.join(profiles) + ' |')
        print('|---' * (len(profiles) + 1) + '|')
        for label in ordered:
            values = ' | '.join(f'{profiles[r].get(label, float("nan")):.3f}' for r in profiles)
            print(f'| {names.get(label, label)} | {values} |')
        if result.get('mean_seed_correlation') is not None:
            print(f"\nmean across-seed profile correlation: "
                  f"{result['mean_seed_correlation']:.3f}")

    elif args.command == 'tradeoff':
        print(tradeoff_table(args.runs))


if __name__ == '__main__':
    main()
