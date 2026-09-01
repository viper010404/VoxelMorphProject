#!/usr/bin/env python3
"""
Train one experiment configuration.

Model selection happens on the validation split: validation Dice is computed periodically and
the best checkpoint is kept. The test split is never read here. This matters because we train
many variants and pick a winner -- choosing and reporting on the same data would bias every
number upward. The paper follows the same protocol (§V-B: "select the network that optimizes
Dice on our validation set, and report results on our test set").

Usage
-----
    python project/train.py --name 2d_baseline_lam0.01_disp --variant baseline --lambda-reg 0.01
    python project/train.py --config results/2d_baseline_lam0.01_disp/config.json
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

import voxelmorph as vxm

from project.configs import ExperimentConfig
from project.data import OasisData, PairSampler, default_label_policy, fixed_pairs
from project.losses import (inverse_consistency_loss, registration_loss,
                            structure_weight_map)
from project.metrics import dice_per_structure, mean_dice, warp_segmentation
from project.models import build_model, forward_model


def set_seed(seed: int) -> None:
    """Seed torch and numpy so a run is reproducible up to cuDNN non-determinism."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_dice(
    model: torch.nn.Module,
    data: OasisData,
    pairs: List[Tuple[int, int]],
    labels: List[int],
    device: str,
) -> float:
    """
    Mean Dice over a fixed set of validation pairs.

    Parameters
    ----------
    model : torch.nn.Module
        Model exposing the shared dict interface.
    data : OasisData
        Dataset the pair indices refer to.
    pairs : list of tuple of int
        `(fixed, moving)` subject index pairs.
    labels : list of int
        Structure ids to score.
    device : str
        Device to run on.

    Returns
    -------
    float
        Mean Dice across pairs and present structures.
    """
    model.eval()
    scores = []

    for fixed_idx, moving_idx in pairs:
        source = data.batch([moving_idx]).to(device)
        target = data.batch([fixed_idx]).to(device)

        moving_seg = data.seg_batch([moving_idx]).to(device)

        outputs = forward_model(model, source, target, moving_seg)

        warped_seg = warp_segmentation(moving_seg, outputs['displacement'])

        fixed_seg_np = data.seg_batch([fixed_idx]).squeeze().cpu().numpy()
        warped_seg_np = warped_seg.squeeze().cpu().numpy()

        scores.append(mean_dice(dice_per_structure(fixed_seg_np, warped_seg_np, labels)))

    model.train()
    return float(np.nanmean(scores))


def train(config: ExperimentConfig) -> Dict:
    """
    Run one training job end to end.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to train.

    Returns
    -------
    dict
        Training history, also written to `history.json` in the run directory.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.seed)

    # 2D fits entirely in GPU memory (~51 MB); 3D would not, so it stays in host RAM and
    # batches are copied per step. Either way the gzip decode is paid once, not per iteration.
    data_device = device if config.ndim == 2 else 'cpu'
    data = OasisData(config.data_path, device=data_device)

    sampler = PairSampler(data, 'train', config.batch_size, seed=config.seed)

    # Same label policy as evaluation, so the quantity used for model selection is the quantity
    # ultimately reported.
    labels = default_label_policy(data, 'val')
    val_pairs = fixed_pairs(data, 'val', config.val_pairs, seed=99)

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Stateless; used only to compose fields for the inverse-consistency term.
    transformer = vxm.nn.modules.SpatialTransformer().to(device)

    config.save()
    history = {'step': [], 'loss': [], 'similarity': [], 'smoothness': [], 'inverse': [],
               'val_step': [], 'val_dice': []}
    best_dice = -float('inf')
    started = time.time()

    for step in range(1, config.steps + 1):
        source, target = next(sampler)
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_seg = None
        if config.variant == 'lambda_structure':
            source_seg = data.seg_batch(sampler.source_idx).to(device, non_blocking=True)
        outputs = forward_model(model, source, target, source_seg)

        # A fixed prior needs no weight head, so it applies to any variant -- including the
        # cascade, which emits no lambda_map of its own.
        weight = outputs.get('lambda_map')
        if weight is None and config.structure_lambda is not None:
            if source_seg is None:
                source_seg = data.seg_batch(sampler.source_idx).to(device, non_blocking=True)
            weight = structure_weight_map(source_seg, config.structure_lambda,
                                          (source > 0) | (target > 0))

        losses = registration_loss(
            target=target,
            warped_source=outputs['warped_source'],
            disp=outputs['displacement'],
            lambda_reg=config.lambda_reg,
            weight=weight,
            lambda_fold=config.lambda_fold,
            fold_margin=config.fold_margin,
        )
        total = losses['total']

        if config.bidirectional:
            # The *same* weights registered in the reverse direction, so this adds a second
            # forward pass but no parameters -- any gain is from the objective, not capacity.
            target_seg = None
            if config.variant == 'lambda_structure':
                target_seg = data.seg_batch(sampler.target_idx).to(device, non_blocking=True)
            reverse = forward_model(model, target, source, target_seg)

            reverse_losses = registration_loss(
                target=source,
                warped_source=reverse['warped_source'],
                disp=reverse['displacement'],
                lambda_reg=config.lambda_reg,
                weight=reverse.get('lambda_map'),
                lambda_fold=config.lambda_fold,
                fold_margin=config.fold_margin,
            )
            total = total + reverse_losses['total']

            if config.beta_inv > 0:
                inverse = inverse_consistency_loss(
                    outputs['displacement'], reverse['displacement'], transformer)
                total = total + config.beta_inv * inverse
                losses['inverse'] = inverse

        total.backward()
        optimizer.step()

        if step % 100 == 0:
            history['step'].append(step)
            history['loss'].append(losses['total'].detach().item())
            history['similarity'].append(losses['similarity'].detach().item())
            history['smoothness'].append(losses['smoothness'].detach().item())
            if 'inverse' in losses:
                history['inverse'].append(losses['inverse'].detach().item())

        if step % config.val_every == 0 or step == config.steps:
            dice = validation_dice(model, data, val_pairs, labels, device)
            history['val_step'].append(step)
            history['val_dice'].append(dice)

            if dice > best_dice:
                best_dice = dice
                torch.save(model.state_dict(), config.output_dir / 'best.pt')

            elapsed = time.time() - started
            print(f'[{config.name}] step {step}/{config.steps} '
                  f'loss={losses["total"].detach().item():.5f} val_dice={dice:.4f} '
                  f'best={best_dice:.4f} ({elapsed:.0f}s)', flush=True)

    history['best_val_dice'] = best_dice
    history['train_seconds'] = time.time() - started

    torch.save(model.state_dict(), config.output_dir / 'final.pt')
    with open(config.output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='path to an existing config.json')
    parser.add_argument('--name', type=str)
    parser.add_argument('--variant', type=str, default='baseline')
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--lambda-reg', type=float, default=0.01)
    parser.add_argument('--integration-steps', type=int, default=0)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-every', type=int, default=1000)
    parser.add_argument('--data-path', type=str, default='data/oasis2d.npz')
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        if not args.name:
            raise SystemExit('--name is required when --config is not given')
        config = ExperimentConfig(
            name=args.name,
            variant=args.variant,
            ndim=args.ndim,
            lambda_reg=args.lambda_reg,
            integration_steps=args.integration_steps,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            val_every=args.val_every,
            data_path=args.data_path,
            output_root=args.output_root,
        )

    train(config)


if __name__ == '__main__':
    main()
