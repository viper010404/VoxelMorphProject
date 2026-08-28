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
         '--split', 'test', '--n-pairs', '100'],
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
                  'batch_size': args.batch_size, 'lambda_mask_norm': args.lambda_mask_norm}
        if args.variants:
            kwargs['variants'] = args.variants
        configs = build_matrix(**kwargs)

    if args.skip_existing:
        remaining = [c for c in configs if not (c.output_dir / 'eval_test.json').exists()]
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
