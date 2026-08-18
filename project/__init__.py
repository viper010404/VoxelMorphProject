"""
Course-project extensions to VoxelMorph.

This package contains everything specific to the project and does not modify the upstream
`voxelmorph` library. The layout separates the parts that are shared by every experiment
(data, metrics, training loop, evaluation) from the parts that differ per extension (models,
losses), so that all branches are measured through an identical path.

Modules
-------
prepare_data
    One-off preprocessing of the neurite-OASIS release into an in-memory cache.
data
    RAM-resident dataset, training pair sampler, and the fixed evaluation pair list.
configs
    Experiment configuration dataclass and the registry of named variants.
models
    Baseline, lambda-field and cross-attention registration networks.
losses
    Smoothness penalties, including the per-voxel weighted variant.
metrics
    Dice, deformation folding, and inverse consistency.
train
    Trains a single configuration.
evaluate
    Scores a trained checkpoint on the fixed evaluation pairs.
compare
    Paired statistics of each branch against the baseline.
"""
