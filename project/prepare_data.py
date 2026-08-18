#!/usr/bin/env python3
"""
Preprocess the neurite-OASIS 2D dataset into a single in-memory cache.

The raw dataset stores every subject as separate gzipped NIfTI files. Decompressing those
on the fly costs far more than a training step (measured: ~170 ms per volume vs ~5 ms for a
2D iteration), so we decode once here and save a single `.npz` that can be loaded straight
into RAM and moved to the GPU.

Outputs `oasis2d.npz` containing:
    images  (N, 160, 192) float32, min-max normalised to [0, 1]
    segs    (N, 160, 192) uint8,   24-label anatomical segmentation
    segs4   (N, 160, 192) uint8,   4-label coarse tissue segmentation
    subjects (N,) str
plus a `splits.json` giving disjoint train/val/test subject indices.

Usage:
    python project/prepare_data.py --data-dir data --out data/oasis2d.npz
"""

import json
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib


FILENAMES = {
    2: ('slice_norm.nii.gz', 'slice_seg24.nii.gz', 'slice_seg4.nii.gz'),
    3: ('aligned_norm.nii.gz', 'aligned_seg35.nii.gz', 'aligned_seg4.nii.gz'),
}


def load_subject(subject_dir: Path, ndim: int = 2):
    """
    Load the image and segmentations for one subject.

    The 2D release stores a single coronal slice with a 24-label segmentation; the 3D release
    stores the affinely-aligned template-space volume with a 35-label segmentation. Both are
    already skull-stripped and bias-corrected, which is the preprocessing VoxelMorph assumes
    (§V-A-1) -- the network only has to learn the nonlinear residual.

    Parameters
    ----------
    subject_dir : Path
        Directory of a single OASIS subject (e.g. `OASIS_OAS1_0001_MR1`).
    ndim : int, optional
        2 to load the coronal slice, 3 to load the full volume.

    Returns
    -------
    tuple of np.ndarray or None
        `(image, seg, seg4)`, or None if any file is missing.
    """
    image_name, seg_name, seg4_name = FILENAMES[ndim]
    paths = [subject_dir / name for name in (image_name, seg_name, seg4_name)]

    if not all(path.exists() for path in paths):
        return None

    image = np.asarray(nib.load(paths[0]).dataobj).squeeze().astype(np.float32)
    seg = np.asarray(nib.load(paths[1]).dataobj).squeeze().astype(np.uint8)
    seg4 = np.asarray(nib.load(paths[2]).dataobj).squeeze().astype(np.uint8)

    return image, seg, seg4


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an image to [0, 1].

    VoxelMorph assumes inputs scaled to [0, 1]; the MSE similarity term is otherwise
    scaled arbitrarily relative to the smoothness term, which changes the meaning of lambda.
    """
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


def make_splits(n: int, n_train: int, n_val: int, n_test: int, seed: int = 0) -> dict:
    """
    Build disjoint train/val/test index splits over subjects.

    Model selection happens on `val`; `test` is touched once for the final reported numbers.
    This keeps a multi-branch comparison honest -- picking the best variant on the same split
    used to report it would bias every number upward.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    if n_train + n_val + n_test > n:
        raise ValueError(f'requested {n_train + n_val + n_test} subjects but only {n} available')

    return {
        'train': perm[:n_train].tolist(),
        'val': perm[n_train:n_train + n_val].tolist(),
        'test': perm[n_train + n_val:n_train + n_val + n_test].tolist(),
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--ndim', type=int, default=2, choices=(2, 3))
    parser.add_argument('--n-train', type=int, default=100,
                        help='paper Fig. 8 shows 100 scans matches the full training set')
    parser.add_argument('--n-val', type=int, default=50)
    parser.add_argument('--n-test', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.out is None:
        args.out = Path('data/oasis2d.npz') if args.ndim == 2 else Path('data/oasis3d')

    subject_dirs = sorted(args.data_dir.glob('OASIS_OAS1_*'))
    if not subject_dirs:
        raise SystemExit(f'no OASIS subject directories under {args.data_dir}')

    images, segs, segs4, names, skipped = [], [], [], [], []

    for subject_dir in subject_dirs:
        loaded = load_subject(subject_dir, ndim=args.ndim)
        if loaded is None:
            skipped.append(subject_dir.name)
            continue
        image, seg, seg4 = loaded
        images.append(normalise(image))
        segs.append(seg)
        segs4.append(seg4)
        names.append(subject_dir.name)

    images = np.stack(images)
    segs = np.stack(segs)
    segs4 = np.stack(segs4)

    print(f'loaded   : {len(names)} subjects  (skipped {len(skipped)})')
    print(f'images   : {images.shape} {images.dtype}  '
          f'range [{images.min():.3f}, {images.max():.3f}]  {images.nbytes / 1e9:.2f} GB')
    print(f'segs     : {segs.shape} {segs.dtype}  {len(np.unique(segs))} distinct labels')
    print(f'seg4     : {segs4.shape} {segs4.dtype}  labels present: {sorted(np.unique(segs4))}')

    # A label missing from some subjects would silently distort per-structure Dice, so report it.
    per_subject_labels = [len(np.unique(s)) for s in segs]
    print(f'labels/subject: min={min(per_subject_labels)} max={max(per_subject_labels)}')

    if args.ndim == 2:
        # ~51 MB: a single compressed archive is convenient and loads instantly.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, images=images, segs=segs, segs4=segs4,
                            subjects=np.array(names))
        splits_dir = args.out.parent
        size_note = f'{args.out.stat().st_size / 1e6:.1f} MB'
    else:
        # ~11 GB of volumes will not fit comfortably in a single in-memory archive, so write
        # plain .npy files that can be memory-mapped. The OS page cache then keeps the working
        # set resident without the loader ever materialising the whole array.
        args.out.mkdir(parents=True, exist_ok=True)
        np.save(args.out / 'images.npy', images)
        np.save(args.out / 'segs.npy', segs)
        np.save(args.out / 'segs4.npy', segs4)
        with open(args.out / 'subjects.json', 'w') as f:
            json.dump(names, f, indent=2)
        splits_dir = args.out
        total = sum((args.out / f).stat().st_size
                    for f in ('images.npy', 'segs.npy', 'segs4.npy'))
        size_note = f'{total / 1e9:.2f} GB'

    splits = make_splits(len(names), args.n_train, args.n_val, args.n_test, args.seed)
    splits_path = splits_dir / 'splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f'\nwrote {args.out} ({size_note})')
    print(f'wrote {splits_path}: '
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    if skipped:
        print(f'skipped subjects: {skipped}')


if __name__ == '__main__':
    main()
