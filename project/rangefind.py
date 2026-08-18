#!/usr/bin/env python3
"""
Locate the useful range of the regularisation weight for a given dimensionality.

The smoothness term is a mean over voxels and vector components, which is not the normalisation
the paper's quoted values assume -- so lambda does not transfer, either from the paper or from 2D
to 3D. Getting this wrong is expensive: at one tenth of the useful value the deformation folded
5.5% of voxels and scored below doing nothing, which cost a full sweep before it was noticed.

This runs short trainings across a range of lambda and reports Dice against folding, so the knee
of the trade-off can be read off directly. Target the folding rate the paper reports, 0.2-0.4%.

Usage
-----
    python project/rangefind.py --ndim 2
    python project/rangefind.py --ndim 3 --data-path data/oasis3d_cache --steps 3000
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from project.configs import ExperimentConfig
from project.data import OasisData, PairSampler, default_label_policy, fixed_pairs
from project.losses import registration_loss
from project.metrics import dice_per_structure, folding, mean_dice, warp_segmentation
from project.models import build_model


def evaluate_quickly(model, data, pairs, labels, device):
    """Mean initial Dice, registered Dice, folding fraction and displacement magnitude."""
    model.eval()
    initial, registered, folds, magnitudes = [], [], [], []

    with torch.no_grad():
        for fixed_idx, moving_idx in pairs:
            disp = model(data.batch([moving_idx]).to(device),
                         data.batch([fixed_idx]).to(device))['displacement']
            fixed_seg = data.seg_batch([fixed_idx]).squeeze().cpu().numpy()
            moving_seg = data.seg_batch([moving_idx]).to(device)
            warped = warp_segmentation(moving_seg, disp).squeeze().cpu().numpy()

            initial.append(mean_dice(dice_per_structure(
                fixed_seg, moving_seg.squeeze().cpu().numpy(), labels)))
            registered.append(mean_dice(dice_per_structure(fixed_seg, warped, labels)))
            folds.append(folding(disp)[0]['fraction'])
            magnitudes.append(float(disp.pow(2).sum(1).sqrt().mean()))

    model.train()
    return (float(np.mean(initial)), float(np.mean(registered)),
            float(np.mean(folds)), float(np.mean(magnitudes)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndim', type=int, default=2, choices=(2, 3))
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--variant', type=str, default='baseline')
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--eval-pairs', type=int, default=20)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = 'data/oasis2d.npz' if args.ndim == 2 else 'data/oasis3d_cache'
    if args.batch_size is None:
        args.batch_size = 16 if args.ndim == 2 else 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = OasisData(args.data_path, device=device if args.ndim == 2 else 'cpu')
    labels = default_label_policy(data, 'val')
    pairs = fixed_pairs(data, 'val', args.eval_pairs, seed=99,
                        path=Path(f'/tmp/rangefind_val_{args.ndim}d.json'))

    print(f'{args.ndim}D  {len(labels)} structures  {args.steps} steps  '
          f'batch {args.batch_size}  {len(pairs)} eval pairs', flush=True)
    print(f'{"lambda":>8} {"init":>7} {"dice":>7} {"delta":>8} {"fold%":>8} {"|disp|":>7}',
          flush=True)

    rows = []
    for lam in args.lambdas:
        torch.manual_seed(0)
        np.random.seed(0)

        config = ExperimentConfig(name='rangefind', variant=args.variant, ndim=args.ndim,
                                  lambda_reg=lam, batch_size=args.batch_size,
                                  data_path=args.data_path)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        sampler = PairSampler(data, 'train', args.batch_size, seed=0)

        for _ in range(args.steps):
            source, target = next(sampler)
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(source, target)
            registration_loss(target, outputs['warped_source'], outputs['displacement'],
                              lam, outputs.get('lambda_map'))['total'].backward()
            optimizer.step()

        init, dice, fold, magnitude = evaluate_quickly(model, data, pairs, labels, device)
        rows.append({'lambda': lam, 'initial': init, 'dice': dice,
                     'folding_pct': 100 * fold, 'displacement': magnitude})
        print(f'{lam:8.3f} {init:7.4f} {dice:7.4f} {dice - init:+8.4f} '
              f'{100 * fold:8.3f} {magnitude:7.3f}', flush=True)

        del model, optimizer
        torch.cuda.empty_cache()

    best = max(rows, key=lambda r: r['dice'])
    print(f"\nbest Dice at lambda={best['lambda']} "
          f"({best['dice']:.4f}, folding {best['folding_pct']:.3f}%)")
    print('Prefer the knee of the trade-off: the smallest lambda whose folding is still near the '
          "paper's 0.2-0.4%.")

    if args.out:
        import json
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
