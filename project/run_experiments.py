#!/usr/bin/env python3
"""
Queue the experiment matrix across the available GPUs and evaluate everything.

Each run is launched as a separate subprocess pinned to one GPU with `CUDA_VISIBLE_DEVICES`.
Independent processes rather than `DistributedDataParallel` is the right choice here: the runs
are different *configurations*, not shards of one job, so this gives perfect scaling with no
gradient synchronisation and no risk that a large effective batch changes convergence behaviour.

Usage
-----
    python project/run_experiments.py --ndim 2 --gpus 0 1
    python project/run_experiments.py --ndim 2 --dry-run
    python project/run_experiments.py --ndim 3 --steps 20000 --gpus 0 1
"""

import os
import sys
import time
import queue
import argparse
import threading
import subprocess
from pathlib import Path
from typing import List, Sequence

from project.configs import ExperimentConfig, build_matrix, ensemble_configs


def run_one(config: ExperimentConfig, gpu: int, python: str, log_dir: Path) -> int:
    """
    Train and evaluate a single configuration in a subprocess.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to run.
    gpu : int
        GPU index to pin the subprocess to.
    python : str
        Python interpreter to use.
    log_dir : Path
        Directory for the run's stdout/stderr log.

    Returns
    -------
    int
        Subprocess return code (0 on success).
    """
    config.save()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{config.name}.log'

    commands = [
        [python, '-m', 'project.train', '--config', str(config.output_dir / 'config.json')],
        [python, '-m', 'project.evaluate', '--run', str(config.output_dir),
         '--split', 'test', '--n-pairs', '100',
         '--misalign', str(config.misalign_magnitude)],
    ]

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}

    with open(log_path, 'w') as log:
        for command in commands:
            log.write(f'$ {" ".join(command)}\n')
            log.flush()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            if result.returncode != 0:
                return result.returncode
    return 0


def run_matrix(
    configs: Sequence[ExperimentConfig],
    gpus: Sequence[int],
    python: str,
    log_dir: Path,
) -> List[tuple]:
    """
    Execute configurations across GPUs, one concurrent run per GPU.

    Parameters
    ----------
    configs : sequence of ExperimentConfig
        Configurations to run.
    gpus : sequence of int
        GPU indices to use as workers.
    python : str
        Python interpreter.
    log_dir : Path
        Directory for logs.

    Returns
    -------
    list of tuple
        `(run_name, returncode)` for every configuration.
    """
    pending: "queue.Queue" = queue.Queue()
    for config in configs:
        pending.put(config)

    results: List[tuple] = []
    lock = threading.Lock()
    started = time.time()

    def worker(gpu: int) -> None:
        while True:
            try:
                config = pending.get_nowait()
            except queue.Empty:
                return
            begin = time.time()
            code = run_one(config, gpu, python, log_dir)
            with lock:
                results.append((config.name, code))
                status = 'ok' if code == 0 else f'FAILED({code})'
                print(f'[gpu{gpu}] {config.name}: {status} '
                      f'({time.time() - begin:.0f}s, {len(results)}/{len(configs)} done, '
                      f'{time.time() - started:.0f}s elapsed)', flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--variants', type=str, nargs='+', default=None)
    parser.add_argument('--cross-attn-target-skips', action='store_true',
                        help="fuse the target encoder pyramid into the cross-attention "
                             "decoder; without it the decoder never sees the fixed image "
                             "above the bottleneck")
    parser.add_argument('--cross-attn-no-attention', action='store_true',
                        help='ablation: keep the target skips but remove the bottleneck '
                             'attention, isolating what the attention itself contributes')
    parser.add_argument('--cross-attn-window-level', type=int, default=-1,
                        help='encoder level for local windowed cross-attention (-1 disables); '
                             '3 is 8x downsampled, where a radius-2 window spans the full '
                             'displacement range')
    parser.add_argument('--cross-attn-window-radius', type=int, default=2)
    parser.add_argument('--cascade-scales', type=int, nargs='+', default=None,
                        help='cascade: downsampling factor per stage, finest last. "2 1" is '
                             'coarse-to-fine, "1 1" is the same-resolution control')
    parser.add_argument('--no-progressive', action='store_true',
                        help='pyramid: keep per-level flow heads but do not warp skips by the '
                             'accumulated field')
    parser.add_argument('--no-deep-supervision', action='store_true',
                        help='pyramid: train on the final field only, isolating architecture '
                             'from objective')
    parser.add_argument('--misalign', type=float, default=0.0,
                        help='train with synthetic misalignment of up to this many voxels, '
                             'creating the large-displacement regime the dataset lacks')
    parser.add_argument('--lambdas', type=float, nargs='+', default=None,
                        help='smoothness weights to sweep (default 0.05 0.1 0.25); the 3D '
                             'optimum sat at the bottom edge of that bracket, so extending it '
                             'downward needs this flag')
    parser.add_argument('--integration-steps', type=int, nargs='+', default=None,
                        help='scaling-and-squaring settings to run: 0 for displacement, >0 for '
                             'SVF (default: both)')
    parser.add_argument('--nb-features', type=int, nargs='+', default=None,
                        help='UNet width per level; the length sets the depth and hence the '
                             'bottleneck downsampling factor (default 16 32 32 32 32 = 32x)')
    parser.add_argument('--sweep-variants', type=str, nargs='+', default=None,
                        help='variants given the full lambda sweep (default: baseline and '
                             'lambda_field); others run at the middle lambda only')
    parser.add_argument('--lambda-mask-norm', action='store_true',
                        help='normalise the lambda-field weight map within the brain mask '
                             'instead of over the whole image')
    parser.add_argument('--ensemble-of', type=str, default=None,
                        help='run name to replicate across seeds instead of the full matrix')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--python', type=str, default=sys.executable,
                        help='interpreter for the child processes; defaults to the one running '
                             'this launcher, so the matrix inherits the active virtualenv')
    parser.add_argument('--log-dir', type=Path, default=Path('results/logs'))
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-existing', action='store_true',
                        help='skip runs that already have eval_test.json, so an interrupted '
                             'matrix can be resumed without retraining what finished')
    args = parser.parse_args()

    if args.ensemble_of:
        base = ExperimentConfig.load(Path('results') / args.ensemble_of / 'config.json')
        configs = ensemble_configs(base, seeds=args.seeds)
    else:
        kwargs = {'ndim': args.ndim, 'steps': args.steps, 'data_path': args.data_path,
                  'batch_size': args.batch_size, 'lambda_mask_norm': args.lambda_mask_norm,
                  'cross_attn_target_skips': args.cross_attn_target_skips,
                  'cross_attn_use_attention': not args.cross_attn_no_attention,
                  'cross_attn_window_level': args.cross_attn_window_level,
                  'cross_attn_window_radius': args.cross_attn_window_radius,
                  'nb_features': args.nb_features,
                  'misalign_magnitude': args.misalign,
                  'pyramid_progressive': not args.no_progressive,
                  'deep_supervision': not args.no_deep_supervision}
        if args.sweep_variants:
            kwargs['sweep_variants'] = args.sweep_variants
        if args.lambdas:
            kwargs['lambdas'] = args.lambdas
        if args.integration_steps is not None:
            kwargs['integration_steps'] = args.integration_steps
        if args.cascade_scales:
            kwargs['cascade_scales'] = args.cascade_scales
        if args.variants:
            kwargs['variants'] = args.variants
        configs = build_matrix(**kwargs)

    if args.skip_existing:
        def _finished(config):
            # Must match the name evaluate.py writes, which carries the misalignment magnitude;
            # otherwise a resumed misalignment sweep retrains everything it already did.
            tag = f'_mis{config.misalign_magnitude:g}' if config.misalign_magnitude > 0 else ''
            return (config.output_dir / f'eval_test{tag}.json').exists()

        remaining = [c for c in configs if not _finished(c)]
        skipped = len(configs) - len(remaining)
        if skipped:
            print(f'skipping {skipped} run(s) that already have eval_test.json')
        configs = remaining

    print(f'{len(configs)} runs across gpus {args.gpus}:')
    for config in configs:
        print(f'  {config.name}  (variant={config.variant} lambda={config.lambda_reg} '
              f'int_steps={config.integration_steps} steps={config.steps})')

    if args.dry_run:
        return

    results = run_matrix(configs, args.gpus, args.python, args.log_dir)

    failed = [name for name, code in results if code != 0]
    print(f'\ncompleted {len(results) - len(failed)}/{len(results)} runs')
    if failed:
        print('FAILED:', ', '.join(failed))


if __name__ == '__main__':
    main()
