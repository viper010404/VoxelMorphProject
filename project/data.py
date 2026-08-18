"""
Data access for the registration experiments.

The whole dataset is held in RAM. Decoding a gzipped NIfTI costs ~170 ms per volume, which is
more than an entire training step (~5 ms in 2D, ~139 ms in 3D), so loading from disk during
training would leave the GPU idle most of the time. `prepare_data.py` decodes everything once
into an `.npz`; this module loads that once and serves tensors from memory.

Two distinct sampling paths live here, and the difference matters:

* `PairSampler` draws *random* pairs and is used for training.
* `fixed_pairs` builds a *deterministic* pair list that is written to disk and reused by every
  model. Because every branch is scored on exactly the same pairs, results can be compared
  pair-by-pair against the baseline rather than only in aggregate.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


class OasisData:
    """
    In-memory neurite-OASIS dataset.

    Two cache layouts are supported, matching what `prepare_data.py` writes:

    * a `.npz` archive (2D, ~51 MB) loaded fully into memory, and
    * a directory of `.npy` files (3D, ~11 GB) opened as memory maps, so the working set is
      paged in by the OS rather than materialised up front.

    Parameters
    ----------
    cache_path : str or Path
        Path to the `.npz` file or the `.npy` directory produced by `prepare_data.py`.
    splits_path : str or Path or None, optional
        Path to the `splits.json` giving train/val/test subject indices. If None, it is looked
        for alongside `cache_path`.
    device : str, optional
        Device the image and segmentation tensors are moved to. Keeping the whole dataset
        resident on the GPU removes the host-to-device copy from the training loop; the 2D
        cache is only ~51 MB so this is essentially free. Memory-mapped 3D caches stay on the
        host and are copied per batch.

    Attributes
    ----------
    images : torch.Tensor
        Shape (N, *spatial), float32, min-max normalised to [0, 1] per subject.
    segs : torch.Tensor
        Shape (N, *spatial), uint8 anatomical segmentation (24 labels in 2D, 35 in 3D).
    splits : dict
        Mapping of 'train' / 'val' / 'test' to lists of subject indices.
    """

    def __init__(
        self,
        cache_path,
        splits_path=None,
        device: str = 'cpu',
    ) -> None:
        cache_path = Path(cache_path)
        self.memmapped = cache_path.is_dir()

        if splits_path is None:
            splits_path = (cache_path if self.memmapped else cache_path.parent) / 'splits.json'

        if self.memmapped:
            images = np.load(cache_path / 'images.npy', mmap_mode='r')
            segs = np.load(cache_path / 'segs.npy', mmap_mode='r')
            with open(cache_path / 'subjects.json') as f:
                self.subjects = json.load(f)
            # Memory maps are indexed lazily; converting to a torch tensor here would defeat
            # the point, so the arrays are kept as numpy and sliced on demand.
            self.images = images
            self.segs = segs
            self.ndim = images.ndim - 1
            self.spatial_shape = tuple(images.shape[1:])
        else:
            with np.load(cache_path, allow_pickle=True) as handle:
                images = handle['images']
                segs = handle['segs']
                self.subjects = [str(s) for s in handle['subjects']]
            self.images = torch.from_numpy(images).to(device)
            self.segs = torch.from_numpy(segs.astype(np.int16)).to(device)
            self.ndim = self.images.dim() - 1
            self.spatial_shape = tuple(self.images.shape[1:])

        self.device = device

        with open(splits_path) as f:
            self.splits = json.load(f)

    def __len__(self) -> int:
        return len(self.subjects)

    def volume(self, index: int) -> torch.Tensor:
        """Return one image with a channel axis, shape (1, *spatial)."""
        return self.batch([index])[0]

    def batch(self, indices: Sequence[int]) -> torch.Tensor:
        """Return a batch of images with a channel axis, shape (B, 1, *spatial)."""
        indices = list(indices)
        if self.memmapped:
            array = np.stack([np.asarray(self.images[i]) for i in indices])
            return torch.from_numpy(array).unsqueeze(1)
        return self.images[indices].unsqueeze(1)

    def seg_batch(self, indices: Sequence[int]) -> torch.Tensor:
        """Return a batch of segmentations with a channel axis, shape (B, 1, *spatial)."""
        indices = list(indices)
        if self.memmapped:
            array = np.stack([np.asarray(self.segs[i]) for i in indices]).astype(np.int16)
            return torch.from_numpy(array).unsqueeze(1)
        return self.segs[indices].unsqueeze(1)

    def label_counts(self, indices: Sequence[int], label: int) -> np.ndarray:
        """
        Count voxels of one label for each of the given subjects.

        Kept separate from `seg_batch` so label statistics can be gathered from a memory-mapped
        cache without stacking whole volumes into RAM.

        Parameters
        ----------
        indices : sequence of int
            Subject indices.
        label : int
            Label id to count.

        Returns
        -------
        np.ndarray
            Voxel count per subject.
        """
        return np.array([int((np.asarray(self.segs[i]) == label).sum()) for i in indices])

    def split_indices(self, split: str) -> List[int]:
        """Return the subject indices belonging to a named split."""
        if split not in self.splits:
            raise KeyError(f"unknown split '{split}'; have {sorted(self.splits)}")
        return list(self.splits[split])


class PairSampler:
    """
    Infinite sampler of random scan-to-scan registration pairs from one split.

    Subject-to-subject registration is used rather than scan-to-atlas because it yields far more
    distinct training pairs from the same subjects: 100 training subjects give 100 atlas pairs but
    ~9,900 ordered subject pairs, which reduces overfitting at this dataset size.

    Parameters
    ----------
    data : OasisData
        Dataset to sample from.
    split : str
        Split name to sample within.
    batch_size : int
        Number of pairs per batch.
    seed : int, optional
        Seed for the sampling RNG.
    """

    def __init__(self, data: OasisData, split: str, batch_size: int, seed: int = 0) -> None:
        self.data = data
        self.indices = np.asarray(data.split_indices(split))
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        if len(self.indices) < 2:
            raise ValueError(f"split '{split}' has {len(self.indices)} subjects; need at least 2")

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw one batch.

        Returns
        -------
        tuple of torch.Tensor
            `(source, target)`, each of shape (B, 1, *spatial). The source is the moving image.
        """
        source_idx = self.rng.choice(self.indices, size=self.batch_size)
        target_idx = self.rng.choice(self.indices, size=self.batch_size)

        # Resample any pair that would register a subject to itself, which carries no signal.
        collision = source_idx == target_idx
        while collision.any():
            target_idx[collision] = self.rng.choice(self.indices, size=int(collision.sum()))
            collision = source_idx == target_idx

        return self.data.batch(source_idx), self.data.batch(target_idx)


def fixed_pairs(
    data: OasisData,
    split: str,
    n_pairs: int,
    seed: int = 1234,
    path: Optional[Path] = None,
) -> List[Tuple[int, int]]:
    """
    Build (or load) the deterministic evaluation pair list for a split.

    Every trained model is scored on this identical list of `(fixed, moving)` subject pairs.
    That is what makes a paired comparison against the baseline valid: differences can be taken
    pair by pair, so between-pair variance -- which is large, since some subject pairs are simply
    harder to align than others -- cancels instead of masking the effect being measured.

    The list is cached to disk on first use and reloaded thereafter, so it survives across runs
    and across machines.

    Parameters
    ----------
    data : OasisData
        Dataset the indices refer to.
    split : str
        Split to draw pairs from, normally 'test' or 'val'.
    n_pairs : int
        Number of pairs to generate.
    seed : int, optional
        Seed used when generating the list for the first time.
    path : Path or None, optional
        Where the list is cached. Defaults to `data/eval_pairs_<split>.json`.

    Returns
    -------
    list of tuple of int
        `(fixed_index, moving_index)` pairs.
    """
    if path is None:
        path = Path('data') / f'eval_pairs_{split}.json'
    path = Path(path)

    if path.exists():
        with open(path) as f:
            payload = json.load(f)
        if payload['split'] == split and len(payload['pairs']) == n_pairs:
            return [tuple(p) for p in payload['pairs']]

    indices = np.asarray(data.split_indices(split))
    rng = np.random.default_rng(seed)

    pairs: List[Tuple[int, int]] = []
    seen = set()
    while len(pairs) < n_pairs:
        fixed, moving = rng.choice(indices, size=2, replace=False)
        key = (int(fixed), int(moving))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'split': split, 'seed': seed, 'pairs': [list(p) for p in pairs]}, f, indent=2)

    return pairs


def evaluation_labels(data: OasisData, split: str, min_size: int = 100) -> Dict[str, list]:
    """
    Determine which anatomical labels to score, following the paper's protocol where possible.

    The paper keeps every structure with at least 100 voxels in all test subjects (§V-A-1),
    yielding 30 structures. That threshold transfers directly to 3D volumes, where it retains
    31 of 35 structures. It does not transfer to a single 2D slice: a slice has ~30k pixels
    against a volume's 5.2M, and the same rule keeps only 8 of 24 labels -- discarding the
    hippocampus, ventricles and putamen, which are exactly the structures where a registration
    difference is expected to show.

    So this returns both policies and lets the caller choose: `strict` for 3D, and `per_pair`
    (score whatever both segmentations of a given pair contain) for 2D.

    Parameters
    ----------
    data : OasisData
        Dataset to analyse.
    split : str
        Split whose subjects define label availability.
    min_size : int, optional
        Minimum voxel count required in every subject for the strict policy.

    A third, intermediate policy is therefore provided and is the one used in 2D: keep structures
    whose *median* size across the split reaches `min_size`. That retains 14 of 24 structures --
    including the ventricles, hippocampus and thalamus -- while excluding those (choroid plexus
    at a 9-pixel median, caudate at 26) whose Dice is dominated by single-pixel noise. Averaging
    those in inflates the per-pair standard deviation to 0.155, which is large enough to hide the
    entire effect being measured.

    Returns
    -------
    dict
        `{'strict': [...], 'median': [...], 'all': [...]}` lists of label ids, excluding
        background.
    """
    indices = data.split_indices(split)

    if data.memmapped:
        # Scanning a memory map subject by subject avoids materialising ~11 GB of volumes.
        present = set()
        for index in indices:
            present.update(int(x) for x in np.unique(np.asarray(data.segs[index])))
        present_labels = sorted(x for x in present if x != 0)
        counts_for = lambda label: data.label_counts(indices, label)  # noqa: E731
    else:
        segs = data.segs[indices]
        present_labels = [int(x) for x in torch.unique(segs) if int(x) != 0]
        counts_for = lambda label: (segs == label).flatten(1).sum(1).cpu().numpy()  # noqa: E731

    strict, median = [], []
    for label in present_labels:
        counts = counts_for(label)
        if int(counts.min()) >= min_size:
            strict.append(label)
        if float(np.median(counts)) >= min_size:
            median.append(label)

    return {'strict': strict, 'median': median, 'all': present_labels}


def default_label_policy(data: OasisData, split: str) -> List[int]:
    """
    Return the label set to score, choosing the policy appropriate to the dimensionality.

    3D uses the paper's rule unchanged (it retains 31 of 35 structures); 2D uses the median-size
    rule, because the paper's rule keeps only 8 of 24 at slice scale.

    Parameters
    ----------
    data : OasisData
        Dataset to analyse.
    split : str
        Split whose subjects define label availability.

    Returns
    -------
    list of int
        Label ids to score.
    """
    policies = evaluation_labels(data, split)
    return policies['strict'] if data.ndim == 3 else policies['median']
