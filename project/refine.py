#!/usr/bin/env python3
"""
Test-time instance optimisation, targeted by ensemble uncertainty.

The amortised network predicts a field in milliseconds; classical instance optimisation --
gradient descent on the displacement of one specific pair -- produces a better field but costs
seconds. Doing the latter everywhere is unaffordable at scale, so this asks whether ensemble
uncertainty is a good enough guide to spend that budget only where it is needed.

**The objective here uses no labels.** Refinement minimises the same unsupervised loss the
network was trained on -- image similarity plus smoothness -- and never sees a segmentation.
This is the point on which the experiment lives or dies: optimising Dice directly would improve
Dice by construction and measure nothing at all.

Four conditions are scored on the identical fixed test pairs:

* ``amortised``  -- the network output, unrefined; the floor.
* ``targeted``   -- refine only the top-k% most uncertain voxels.
* ``random``     -- refine an equal number of *randomly chosen* voxels. This is the control that
  makes the result meaningful: if random does as well as targeted, then the benefit comes from
  spending compute at all and the uncertainty map is worthless as a guide.
* ``full``       -- refine everywhere; the upper bound the targeted variant is trying to approach
  at a fraction of the cost.

Usage
-----
    python -m project.refine --runs results/2d_lambda_structure_lam0.25_disp_seed* \\
        --n-pairs 50 --top-k 0.1 --steps 100 --out results/analysis_2d/refine.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from project.configs import ExperimentConfig
from project.data import OasisData, default_label_policy, fixed_pairs
from project.losses import image_similarity, spatial_smoothness
from project.metrics import dice_per_structure, folding, mean_dice, warp_segmentation
from project.models import build_model, forward_model

import voxelmorph as vxm


def load_members(run_dirs: List[Path], device: str, checkpoint: str = 'best.pt'):
    """Load every ensemble member and return the models plus the first config."""
    models, configs = [], []
    for run_dir in run_dirs:
        config = ExperimentConfig.load(Path(run_dir) / 'config.json')
        model = build_model(config).to(device)
        model.load_state_dict(torch.load(Path(run_dir) / checkpoint, map_location=device))
        model.eval()
        models.append(model)
        configs.append(config)
    return models, configs


def ensemble_field(models, source, target, source_seg):
    """
    Mean displacement and per-voxel disagreement across ensemble members.

    Returns
    -------
    tuple of torch.Tensor
        `(mean_field, uncertainty)` of shapes (1, ndim, *spatial) and (1, 1, *spatial). The
        uncertainty is the across-member standard deviation of each displacement component,
        combined over components as a Euclidean norm, so it is one scalar map in voxel units.
    """
    with torch.no_grad():
        fields = torch.stack([
            forward_model(m, source, target, source_seg)['displacement'] for m in models
        ])
    uncertainty = fields.std(dim=0, unbiased=True).pow(2).sum(dim=1, keepdim=True).sqrt()
    return fields.mean(dim=0), uncertainty


def selection_mask(uncertainty, brain, fraction, generator=None):
    """
    Choose the voxels to refine.

    Parameters
    ----------
    uncertainty : torch.Tensor
        Per-voxel disagreement, shape (1, 1, *spatial). Ignored when `generator` is given.
    brain : torch.Tensor
        Boolean brain mask; selection is confined to it because refining background is pointless.
    fraction : float
        Portion of brain voxels to refine. 1.0 selects the whole brain.
    generator : torch.Generator or None, optional
        When given, voxels are chosen **at random** instead of by uncertainty -- the control
        condition. Using the same count keeps the compute budget identical.

    Returns
    -------
    torch.Tensor
        Float mask of the same shape, 1 on selected voxels.
    """
    mask = torch.zeros_like(uncertainty)
    inside = brain.reshape(-1).nonzero(as_tuple=True)[0]
    if inside.numel() == 0:
        return mask

    budget = max(1, int(round(fraction * inside.numel())))
    if fraction >= 1.0:
        chosen = inside
    elif generator is not None:
        perm = torch.randperm(inside.numel(), generator=generator, device=inside.device)
        chosen = inside[perm[:budget]]
    else:
        scores = uncertainty.reshape(-1)[inside]
        chosen = inside[torch.topk(scores, budget).indices]

    flat = mask.reshape(-1)
    flat[chosen] = 1.0
    return flat.reshape_as(mask)


def refine(source, target, initial_disp, mask, lambda_reg, steps, lr, transformer):
    """
    Instance-optimise a correction to the displacement, confined to `mask`.

    A correction field initialised at zero is optimised rather than the displacement itself, so
    the starting point is exactly the amortised prediction and any change is attributable to the
    refinement. The correction is multiplied by the mask *inside* the graph, so unselected voxels
    keep the network's field exactly while the smoothness term still sees the whole composite
    field and cannot be satisfied by a discontinuity at the mask boundary.

    Returns
    -------
    torch.Tensor
        The refined displacement, detached.
    """
    correction = torch.zeros_like(initial_disp, requires_grad=True)
    optimiser = torch.optim.Adam([correction], lr=lr)

    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        disp = initial_disp + mask * correction
        loss = (image_similarity(target, transformer(source, disp))
                + lambda_reg * spatial_smoothness(disp))
        loss.backward()
        optimiser.step()

    with torch.no_grad():
        return (initial_disp + mask * correction).detach()


def run(
    run_dirs: List[Path],
    split: str = 'test',
    n_pairs: int = 50,
    top_k: float = 0.1,
    steps: int = 100,
    lr: float = 0.01,
    checkpoint: str = 'best.pt',
    seed: int = 0,
) -> Dict:
    """Score all four conditions on a fixed set of pairs; see the module docstring."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models, configs = load_members(run_dirs, device, checkpoint)
    config = configs[0]

    data = OasisData(config.data_path, device=device if config.ndim == 2 else 'cpu')
    labels = default_label_policy(data, split)
    pairs = fixed_pairs(data, split, n_pairs, seed=1234)
    transformer = vxm.nn.modules.SpatialTransformer()
    generator = torch.Generator(device=device).manual_seed(seed)

    rows: List[Dict] = []
    for fixed_idx, moving_idx in pairs:
        source = data.batch([moving_idx]).to(device)
        target = data.batch([fixed_idx]).to(device)
        moving_seg = data.seg_batch([moving_idx]).to(device)
        fixed_seg = data.seg_batch([fixed_idx]).squeeze().cpu().numpy()

        disp, uncertainty = ensemble_field(models, source, target, moving_seg)
        brain = (source > 0) | (target > 0)

        conditions = {
            'amortised': None,
            'targeted': selection_mask(uncertainty, brain, top_k),
            'random': selection_mask(uncertainty, brain, top_k, generator=generator),
            'full': selection_mask(uncertainty, brain, 1.0),
        }

        row = {'fixed': int(fixed_idx), 'moving': int(moving_idx),
               'uncertainty_mean': float(uncertainty[brain].mean())}
        for name, mask in conditions.items():
            field = disp if mask is None else refine(
                source, target, disp, mask, config.lambda_reg, steps, lr, transformer)
            warped = warp_segmentation(moving_seg, field).squeeze().cpu().numpy()
            row[f'dice_{name}'] = mean_dice(dice_per_structure(fixed_seg, warped, labels))
            row[f'folding_{name}'] = folding(field)[0]['fraction']
        rows.append(row)

    summary = {}
    for name in ('amortised', 'targeted', 'random', 'full'):
        summary[f'dice_{name}'] = float(np.mean([r[f'dice_{name}'] for r in rows]))
        summary[f'folding_{name}'] = float(np.mean([r[f'folding_{name}'] for r in rows]))

    return {'n_members': len(models), 'split': split, 'n_pairs': len(rows),
            'top_k': top_k, 'steps': steps, 'lr': lr,
            'summary': summary, 'per_pair': rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--n-pairs', type=int, default=50)
    parser.add_argument('--top-k', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--checkpoint', type=str, default='best.pt')
    parser.add_argument('--out', type=Path, default=Path('results/analysis_2d/refine.json'))
    args = parser.parse_args()

    payload = run(args.runs, args.split, args.n_pairs, args.top_k,
                  args.steps, args.lr, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, indent=2)

    s = payload['summary']
    print(f"n_pairs={payload['n_pairs']}  top_k={args.top_k:.0%}  steps={args.steps}")
    for name in ('amortised', 'targeted', 'random', 'full'):
        print(f"  {name:10s} dice={s[f'dice_{name}']:.4f}  "
              f"folding={s[f'folding_{name}']*100:.3f}%")


if __name__ == '__main__':
    main()
