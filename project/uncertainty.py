#!/usr/bin/env python3
"""
Quantify registration uncertainty from an ensemble of independently seeded models.

`neurite` has no dropout in `BasicUNet`, so MC-dropout is unavailable (and would need dropout
active during training to be valid anyway). The estimate here is therefore a deep ensemble: the
same configuration trained from several seeds, with the spread of the predicted displacement
fields taken as the uncertainty.

Three questions are answered, and they are separate:

1. **How large is the disagreement?** Per-voxel standard deviation of the displacement across
   members, in voxels. Reported per pair and aggregated per structure.

2. **Does the ensemble register better than its members?** The mean displacement field is scored
   like any other model, paired against the members on the identical pair list. Averaging
   displacement fields is only meaningful because all members register the same moving image to
   the same fixed image on an affinely aligned dataset, so the fields live in one common frame.

3. **Does uncertainty predict error?** This is the question that makes an uncertainty estimate
   useful rather than decorative: if disagreement is high exactly where the registration is
   wrong, it can flag failures without ground truth. Measured as the rank correlation between
   per-structure mean uncertainty and per-structure Dice deficit, computed *within* each pair
   and then averaged, so it is not confounded by some pairs simply being harder than others.

Usage
-----
    python -m project.uncertainty --runs results/2d_baseline_lam0.25_disp_seed* \
        --split test --n-pairs 100 --out results/uncertainty_2d_baseline.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from project.configs import ExperimentConfig
from project.data import OasisData, default_label_policy, fixed_pairs
from project.metrics import dice_per_structure, folding, mean_dice, warp_segmentation
from project.models import build_model


def load_members(run_dirs: Sequence[Path], device: str, checkpoint: str = 'best.pt'):
    """
    Load every ensemble member and check they are the same configuration bar the seed.

    Averaging displacement fields across models that differ in more than initialisation would
    not be an uncertainty estimate, so the mismatch is an error rather than a warning.

    Parameters
    ----------
    run_dirs : sequence of Path
        Run directories, each holding `config.json` and the checkpoint.
    device : str
        Device to place the models on.
    checkpoint : str, optional
        Checkpoint filename.

    Returns
    -------
    tuple
        `(models, configs)`, both lists ordered as `run_dirs`.
    """
    models, configs = [], []
    for run_dir in run_dirs:
        config = ExperimentConfig.load(Path(run_dir) / 'config.json')
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / checkpoint, map_location=device))
        model.eval()
        models.append(model)
        configs.append(config)

    reference = configs[0]
    for config, run_dir in zip(configs[1:], run_dirs[1:]):
        for field in ('variant', 'ndim', 'lambda_reg', 'integration_steps', 'data_path'):
            if getattr(config, field) != getattr(reference, field):
                raise SystemExit(
                    f'ensemble members disagree on {field!r}: '
                    f'{run_dirs[0]} has {getattr(reference, field)}, '
                    f'{run_dir} has {getattr(config, field)}'
                )
    return models, configs


def _structure_means(values: np.ndarray, seg: np.ndarray, labels: Sequence[int]) -> Dict:
    """
    Average a per-voxel map over each labelled structure.

    Parameters
    ----------
    values : np.ndarray
        Per-voxel scalar map with the same spatial shape as `seg`.
    seg : np.ndarray
        Integer label map.
    labels : sequence of int
        Labels to report.

    Returns
    -------
    dict
        Label to mean value, or None where the structure is absent.
    """
    out = {}
    for label in labels:
        mask = seg == label
        out[int(label)] = float(values[mask].mean()) if mask.any() else None
    return out


@torch.no_grad()
def ensemble_uncertainty(
    run_dirs: Sequence[Path],
    split: str = 'test',
    n_pairs: int = 100,
    checkpoint: str = 'best.pt',
    example_pairs: int = 1,
) -> Dict:
    """
    Score an ensemble on the shared fixed pair list and measure its internal disagreement.

    Parameters
    ----------
    run_dirs : sequence of Path
        Ensemble member run directories.
    split : str, optional
        Split to evaluate on.
    n_pairs : int, optional
        Number of evaluation pairs; must match the other runs for the comparison to be paired.
    checkpoint : str, optional
        Checkpoint filename to load from each member.
    example_pairs : int, optional
        How many per-voxel uncertainty maps to keep for figures. Stored separately as `.npy`
        because they are far too large for the JSON payload.

    Returns
    -------
    dict
        Results payload, one row per pair plus a summary.
    """
    run_dirs = [Path(d) for d in run_dirs]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models, configs = load_members(run_dirs, device, checkpoint)
    config = configs[0]

    data_device = device if config.ndim == 2 else 'cpu'
    data = OasisData(config.data_path, device=data_device)
    labels = default_label_policy(data, split)
    pairs = fixed_pairs(data, split, n_pairs, seed=1234)

    rows: List[Dict] = []
    examples: List[Dict] = []

    for pair_index, (fixed_idx, moving_idx) in enumerate(pairs):
        source = data.batch([moving_idx]).to(device)
        target = data.batch([fixed_idx]).to(device)

        fields = torch.stack([model(source, target)['displacement'] for model in models])

        # Disagreement per voxel: standard deviation across members of each displacement
        # component, combined over components as a Euclidean norm so the result is one scalar
        # map in voxel units rather than one map per axis.
        per_component_std = fields.std(dim=0, unbiased=True)
        uncertainty = per_component_std.pow(2).sum(dim=1).sqrt()

        mean_field = fields.mean(dim=0)

        moving_seg = data.seg_batch([moving_idx]).to(device)
        fixed_seg_np = data.seg_batch([fixed_idx]).squeeze().cpu().numpy()

        # The ensemble scored as if it were a single model.
        warped_mean = warp_segmentation(moving_seg, mean_field)
        mean_structure = dice_per_structure(
            fixed_seg_np, warped_mean.squeeze().cpu().numpy(), labels)

        # Every member scored individually, so "does averaging help" is a paired question.
        member_dice = []
        for model_index in range(len(models)):
            warped = warp_segmentation(moving_seg, fields[model_index])
            member_structure = dice_per_structure(
                fixed_seg_np, warped.squeeze().cpu().numpy(), labels)
            member_dice.append(mean_dice(member_structure))

        # Uncertainty is attributed to structures of the *moving* image, because that is the
        # image the displacement field acts on -- a voxel's displacement belongs to the anatomy
        # sitting at that voxel before the warp.
        moving_seg_np = moving_seg.squeeze().cpu().numpy()
        uncertainty_np = uncertainty.squeeze().cpu().numpy()
        structure_uncertainty = _structure_means(uncertainty_np, moving_seg_np, labels)

        row = {
            'fixed': int(fixed_idx),
            'moving': int(moving_idx),
            'dice_ensemble_mean_field': mean_dice(mean_structure),
            'dice_members': member_dice,
            'dice_member_mean': float(np.mean(member_dice)),
            'dice_member_std': float(np.std(member_dice, ddof=1)) if len(member_dice) > 1 else 0.0,
            'uncertainty_mean': float(uncertainty_np.mean()),
            'uncertainty_max': float(uncertainty_np.max()),
            'folding_ensemble_mean_field': folding(mean_field)[0]['fraction'],
            'per_structure_uncertainty': {str(k): v for k, v in structure_uncertainty.items()},
            'per_structure_dice_ensemble': {str(k): v for k, v in mean_structure.items()},
        }
        rows.append(row)

        if pair_index < example_pairs:
            examples.append({
                'fixed': int(fixed_idx),
                'moving': int(moving_idx),
                'uncertainty': uncertainty_np.astype(np.float32),
                'displacement_mean': mean_field.squeeze(0).cpu().numpy().astype(np.float32),
            })

    payload = {
        'members': [d.name for d in run_dirs],
        'n_members': len(run_dirs),
        'config': asdict_safe(config),
        'split': split,
        'n_pairs': len(pairs),
        'labels': labels,
        'summary': summarise(rows),
        'per_pair': rows,
    }
    return payload, examples


def asdict_safe(config: ExperimentConfig) -> Dict:
    """Serialise a config to plain JSON types."""
    return json.loads(json.dumps(config.__dict__, default=list))


def uncertainty_error_correlation(rows: Sequence[Dict]) -> Dict:
    """
    Rank-correlate per-structure uncertainty with per-structure Dice deficit.

    The correlation is computed **within a pair** and then averaged over pairs. Pooling every
    structure of every pair into one correlation would mostly measure that some pairs are harder
    than others -- both uncertainty and error rise together across pairs regardless of whether
    uncertainty is informative *within* an image, which is what a failure detector needs.

    Parameters
    ----------
    rows : sequence of dict
        Per-pair rows from `ensemble_uncertainty`.

    Returns
    -------
    dict
        Mean and standard error of the per-pair Spearman correlation, and the number of pairs
        that contributed.
    """
    from scipy import stats

    correlations = []
    for row in rows:
        unc = row['per_structure_uncertainty']
        dice = row['per_structure_dice_ensemble']
        shared = [k for k in unc if unc[k] is not None and dice.get(k) is not None]
        if len(shared) < 3:
            continue
        u = np.array([unc[k] for k in shared])
        # Deficit rather than Dice, so a positive correlation means "uncertain where wrong".
        e = np.array([1.0 - dice[k] for k in shared])
        if np.all(u == u[0]) or np.all(e == e[0]):
            continue
        correlations.append(stats.spearmanr(u, e).statistic)

    if not correlations:
        return {'mean_spearman': None, 'sem': None, 'n_pairs': 0}

    correlations = np.array(correlations)
    return {
        'mean_spearman': float(correlations.mean()),
        'sem': float(correlations.std(ddof=1) / np.sqrt(len(correlations)))
        if len(correlations) > 1 else 0.0,
        'n_pairs': int(len(correlations)),
    }


def summarise(rows: Sequence[Dict]) -> Dict:
    """
    Aggregate per-pair rows, including the paired ensemble-vs-member test.

    Parameters
    ----------
    rows : sequence of dict
        Per-pair rows from `ensemble_uncertainty`.

    Returns
    -------
    dict
        Summary statistics.
    """
    from scipy import stats

    ensemble = np.array([r['dice_ensemble_mean_field'] for r in rows])
    members = np.array([r['dice_member_mean'] for r in rows])
    best_member = np.array([max(r['dice_members']) for r in rows])

    delta = ensemble - members
    t_stat, p_value = stats.ttest_rel(ensemble, members)

    return {
        'dice_ensemble_mean_field': float(ensemble.mean()),
        'dice_member_mean': float(members.mean()),
        'dice_best_member_mean': float(best_member.mean()),
        'delta_ensemble_vs_member_mean': float(delta.mean()),
        'delta_better_on': int((delta > 0).sum()),
        'delta_p_value': float(p_value),
        'seed_spread_within_pair': float(np.mean([r['dice_member_std'] for r in rows])),
        'uncertainty_mean': float(np.mean([r['uncertainty_mean'] for r in rows])),
        'uncertainty_max': float(np.max([r['uncertainty_max'] for r in rows])),
        'folding_ensemble_mean_field': float(
            np.mean([r['folding_ensemble_mean_field'] for r in rows])),
        'uncertainty_vs_error': uncertainty_error_correlation(rows),
    }


def format_summary(payload: Dict, names: Optional[Dict[int, str]] = None) -> str:
    """
    Render the summary as markdown.

    Parameters
    ----------
    payload : dict
        Result of `ensemble_uncertainty`.
    names : dict or None, optional
        Label id to anatomical name.

    Returns
    -------
    str
        Markdown text.
    """
    summary = payload['summary']
    names = names or {}
    lines = [
        f"Ensemble of {payload['n_members']} seeds "
        f"({payload['config']['variant']}, lambda={payload['config']['lambda_reg']}), "
        f"{payload['n_pairs']} {payload['split']} pairs",
        '',
        '| quantity | value |',
        '|---|---|',
        f"| Dice, mean displacement field | {summary['dice_ensemble_mean_field']:.4f} |",
        f"| Dice, average single member | {summary['dice_member_mean']:.4f} |",
        f"| Dice, best member per pair | {summary['dice_best_member_mean']:.4f} |",
        f"| Δ ensemble - member | {summary['delta_ensemble_vs_member_mean']:+.4f} "
        f"(better on {summary['delta_better_on']}/{payload['n_pairs']}, "
        f"p={summary['delta_p_value']:.2e}) |",
        f"| seed spread within a pair (Dice sd) | {summary['seed_spread_within_pair']:.4f} |",
        f"| mean per-voxel disagreement | {summary['uncertainty_mean']:.4f} voxels |",
        f"| max per-voxel disagreement | {summary['uncertainty_max']:.4f} voxels |",
        f"| folding, mean field | {100 * summary['folding_ensemble_mean_field']:.3f} % |",
    ]

    correlation = summary['uncertainty_vs_error']
    if correlation['mean_spearman'] is not None:
        lines.append(
            f"| uncertainty vs Dice deficit (within-pair Spearman) | "
            f"{correlation['mean_spearman']:+.3f} ± {correlation['sem']:.3f} "
            f"(n={correlation['n_pairs']}) |"
        )

    # Which structures the ensemble disagrees about most: the uncertainty map reduced to anatomy.
    per_structure: Dict[str, list] = {}
    for row in payload['per_pair']:
        for label, value in row['per_structure_uncertainty'].items():
            if value is not None:
                per_structure.setdefault(label, []).append(value)
    if per_structure:
        ordered = sorted(per_structure.items(), key=lambda kv: -float(np.mean(kv[1])))
        lines += ['', '| structure | mean disagreement (voxels) |', '|---|---|']
        for label, values in ordered:
            lines.append(f'| {names.get(int(label), label)} | {np.mean(values):.4f} |')

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True,
                        help='ensemble member run directories')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n-pairs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    parser.add_argument('--out', type=Path, default=None,
                        help='where to write the JSON payload')
    parser.add_argument('--labels', type=Path, default=Path('data/seg24_labels.txt'))
    parser.add_argument('--example-pairs', type=int, default=1,
                        help='number of per-voxel uncertainty maps to save alongside the JSON')
    args = parser.parse_args()

    payload, examples = ensemble_uncertainty(
        args.runs, split=args.split, n_pairs=args.n_pairs,
        checkpoint=args.checkpoint, example_pairs=args.example_pairs,
    )

    from project.analyse import load_label_names
    names = load_label_names(args.labels) if args.labels.exists() else {}
    print(format_summary(payload, names))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'\nwrote {args.out}')

        for index, example in enumerate(examples):
            path = args.out.with_suffix('')
            np.savez_compressed(
                f'{path}_example{index}.npz',
                uncertainty=example['uncertainty'],
                displacement_mean=example['displacement_mean'],
                fixed=example['fixed'],
                moving=example['moving'],
            )
            print(f'wrote {path}_example{index}.npz')


if __name__ == '__main__':
    main()
