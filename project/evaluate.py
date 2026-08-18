#!/usr/bin/env python3
"""
Score a trained checkpoint on the fixed evaluation pairs.

Evaluation is deliberately decoupled from training: a run directory holds `config.json` and
`best.pt`, and everything needed to rebuild the model comes from the config. Any checkpoint can
therefore be re-scored later, and every branch is scored by this one code path.

Results are written **per pair and per structure**, not aggregated. `compare.py` needs that
granularity to pair each result against the baseline's result on the same pair.

Usage
-----
    python project/evaluate.py --run results/2d_baseline_lam0.01_disp
    python project/evaluate.py --run results/... --split test --n-pairs 100
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from project.configs import ExperimentConfig
from project.data import OasisData, default_label_policy, fixed_pairs
from project.metrics import (
    dice_per_structure,
    folding,
    inverse_consistency,
    mean_dice,
    warp_segmentation,
)
from project.models import build_model


@torch.no_grad()
def evaluate_run(
    run_dir: Path,
    split: str = 'test',
    n_pairs: int = 100,
    checkpoint: str = 'best.pt',
) -> Dict:
    """
    Evaluate one trained run on the shared fixed pair list.

    Parameters
    ----------
    run_dir : Path
        Directory containing `config.json` and the checkpoint.
    split : str, optional
        Split to evaluate on. Use 'val' while iterating, 'test' once for the final numbers.
    n_pairs : int, optional
        Number of evaluation pairs; must match across runs for a paired comparison.
    checkpoint : str, optional
        Checkpoint filename to load.

    Returns
    -------
    dict
        Results payload, also written to `<run_dir>/eval_<split>.json`.
    """
    run_dir = Path(run_dir)
    config = ExperimentConfig.load(run_dir / 'config.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_device = device if config.ndim == 2 else 'cpu'
    data = OasisData(config.data_path, device=data_device)

    labels = default_label_policy(data, split)
    pairs = fixed_pairs(data, split, n_pairs, seed=1234)

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location=device))
    model.eval()

    rows: List[Dict] = []

    for fixed_idx, moving_idx in pairs:
        source = data.batch([moving_idx]).to(device)
        target = data.batch([fixed_idx]).to(device)

        started = time.time()
        outputs = model(source, target)
        if device == 'cuda':
            torch.cuda.synchronize()
        runtime = time.time() - started

        displacement = outputs['displacement']

        moving_seg = data.seg_batch([moving_idx]).to(device)
        warped_seg = warp_segmentation(moving_seg, displacement)

        fixed_seg_np = data.seg_batch([fixed_idx]).squeeze().cpu().numpy()
        warped_seg_np = warped_seg.squeeze().cpu().numpy()
        moving_seg_np = moving_seg.squeeze().cpu().numpy()

        per_structure = dice_per_structure(fixed_seg_np, warped_seg_np, labels)
        # Dice before registration: the floor every model must beat, and the quantity that makes
        # "how much of the gap did this close" answerable.
        baseline_structure = dice_per_structure(fixed_seg_np, moving_seg_np, labels)

        row = {
            'fixed': int(fixed_idx),
            'moving': int(moving_idx),
            'dice': mean_dice(per_structure),
            'dice_initial': mean_dice(baseline_structure),
            'per_structure': {str(k): v for k, v in per_structure.items()},
            # Stored so the label policy stays a post-hoc choice: any subset of structures can be
            # re-scored from these files without re-running the models.
            'per_structure_initial': {str(k): v for k, v in baseline_structure.items()},
            'folding_count': folding(displacement)[0]['count'],
            'folding_fraction': folding(displacement)[0]['fraction'],
            'runtime_s': runtime,
        }

        if 'velocity' in outputs:
            # The inverse comes free from integrating the negated velocity field.
            backward = model.integrator(-outputs['velocity']) if hasattr(model, 'integrator') \
                else model.net.velocity_field_integrator(-outputs['velocity'])
            row['inverse_consistency'] = inverse_consistency(displacement, backward)[0]

        if 'lambda_map' in outputs:
            lambda_map = outputs['lambda_map']
            row['lambda_std'] = float(lambda_map.std())
            row['lambda_min'] = float(lambda_map.min())
            row['lambda_max'] = float(lambda_map.max())

        rows.append(row)

    payload = {
        'run': run_dir.name,
        'config': json.loads((run_dir / 'config.json').read_text()),
        'split': split,
        'n_pairs': len(pairs),
        'labels': labels,
        'summary': _summarise(rows),
        'per_pair': rows,
    }

    out_path = run_dir / f'eval_{split}.json'
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"[{run_dir.name}] {payload['summary']}", flush=True)
    return payload


def _summarise(rows: List[Dict]) -> Dict[str, float]:
    """Aggregate per-pair rows into headline means for quick inspection."""
    def column(key):
        return [r[key] for r in rows if key in r and r[key] is not None]

    summary = {
        'dice': float(np.mean(column('dice'))),
        'dice_std': float(np.std(column('dice'))),
        'dice_initial': float(np.mean(column('dice_initial'))),
        'folding_fraction': float(np.mean(column('folding_fraction'))),
        'runtime_s': float(np.mean(column('runtime_s'))),
    }
    if column('inverse_consistency'):
        summary['inverse_consistency'] = float(np.mean(column('inverse_consistency')))
    if column('lambda_std'):
        summary['lambda_std'] = float(np.mean(column('lambda_std')))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, nargs='+',
                        help='one or more run directories')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n-pairs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    args = parser.parse_args()

    for run_dir in args.run:
        evaluate_run(run_dir, split=args.split, n_pairs=args.n_pairs,
                     checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
