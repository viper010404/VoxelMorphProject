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

from project.configs import ExperimentConfig
from project.data import OasisData, PairSampler, default_label_policy, fixed_pairs
from project.losses import pyramid_loss, registration_loss
from project.misalign import apply_displacement, pair_seed, random_displacement
from project.metrics import dice_per_structure, mean_dice, warp_segmentation
from project.models import build_model


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
    misalign_magnitude: float = 0.0,
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
    misalign_magnitude : float, optional
        Pre-warp the moving image and its labels by this much before registering. Seeded per
        pair, so the validation task is identical every time it is scored and across models --
        otherwise model selection would chase the noise in a fresh random deformation.

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

        if misalign_magnitude > 0:
            field = random_displacement(
                source.shape[2:], source.dim() - 2, misalign_magnitude,
                seed=pair_seed(fixed_idx, moving_idx, misalign_magnitude), device=device)
            source, moving_seg = apply_displacement(source, field, moving_seg)

        outputs = model(source, target)

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

    config.save()
    history = {'step': [], 'loss': [], 'similarity': [], 'smoothness': [],
               'val_step': [], 'val_dice': []}
    best_dice = -float('inf')
    started = time.time()

    for step in range(1, config.steps + 1):
        source, target = next(sampler)
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if config.misalign_magnitude > 0:
            # Draw the magnitude uniformly up to the configured maximum, so the model sees the
            # whole range rather than one deformation scale, and a fresh field every step.
            drawn = float(np.random.uniform(0.0, config.misalign_magnitude))
            field = random_displacement(source.shape[2:], config.ndim, drawn,
                                        seed=int(np.random.randint(2 ** 31)), device=device)
            source, _ = apply_displacement(source, field)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(source, target)

        if 'pyramid' in outputs and config.deep_supervision:
            losses = pyramid_loss(
                target=target,
                source=source,
                fields=outputs['pyramid'],
                lambda_reg=config.lambda_reg,
                transformer=model.spatial_transformer,
            )
        else:
            losses = registration_loss(
                target=target,
                warped_source=outputs['warped_source'],
                disp=outputs['displacement'],
                lambda_reg=config.lambda_reg,
                weight=outputs.get('lambda_map'),
            )
        losses['total'].backward()
        optimizer.step()

        if step % 100 == 0:
            history['step'].append(step)
            history['loss'].append(losses['total'].detach().item())
            history['similarity'].append(losses['similarity'].detach().item())
            history['smoothness'].append(losses['smoothness'].detach().item())

        if step % config.val_every == 0 or step == config.steps:
            dice = validation_dice(model, data, val_pairs, labels, device,
                                   misalign_magnitude=config.misalign_magnitude)
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
