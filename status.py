#!/usr/bin/env python3
"""
One-glance status for in-flight runs: progress, ETA, and the test-set trace so far.

Long 3D runs previously gave no readable signal until they finished, which made "is this
working?" unanswerable for hours. `train.py` now writes `history.json` incrementally with a
read-only test trace; this reads those files and prints where every run stands.

ETA comes from each run's own observed throughput rather than an assumed step rate, because
the rate drifts with GPU contention.

Usage
-----
    ./.venv/bin/python status.py                    # every run with a history.json
    ./.venv/bin/python status.py 3d_                # only runs whose name contains this
"""

import json
import re
import sys
import time
from pathlib import Path

# `[name] step 12000/80000 loss=... val_dice=0.7324 best=0.7324 (1208s, eta 3.02h)`
PROGRESS = re.compile(r'step (\d+)/(\d+).*?val_dice=([\d.]+).*?eta ([\d.]+)h')


def from_log(name: str):
    """
    Last progress line from a run's own stdout log.

    `history.json` is only flushed at test checks, so a run is invisible for its first several
    thousand steps -- exactly the window where "did it start correctly?" is the live question.
    The log is written every validation, so it fills that gap.
    """
    path = Path('results/logs') / f'{name}.log'
    if not path.exists():
        return None
    match = None
    for line in path.read_text(errors='ignore').splitlines():
        found = PROGRESS.search(line)
        if found:
            match = found
    if not match:
        return None
    step, total, val, eta = match.groups()
    return {'step': int(step), 'total': int(total), 'val': float(val), 'eta_h': float(eta),
            'mtime': path.stat().st_mtime}


def rows(pattern: str = ''):
    """Yield a status dict per run directory matching `pattern`."""
    for config_path in sorted(Path('results').glob('*/config.json')):
        name = config_path.parent.name
        if pattern and pattern not in name:
            continue
        history_path = config_path.parent / 'history.json'
        try:
            config = json.loads(config_path.read_text())
            history = json.loads(history_path.read_text()) if history_path.exists() else {}
        except (json.JSONDecodeError, FileNotFoundError):
            # A run mid-write; skip it rather than crashing the whole listing.
            continue

        total = config['steps']
        step = history['val_step'][-1] if history.get('val_step') else 0
        rate = history['step_seconds'][-1] if history.get('step_seconds') else None
        age = time.time() - (history_path.stat().st_mtime if history_path.exists()
                             else config_path.stat().st_mtime)

        # The log is fresher than history.json between test checks, so it wins when ahead.
        logged = from_log(name)
        if logged and logged['step'] >= step:
            step = logged['step']
            age = time.time() - logged['mtime']
            history = dict(history)
            history.setdefault('val_dice', [])
            history['val_dice'] = list(history['val_dice']) + [logged['val']]
            rate = logged['eta_h'] * 3600 / max(1, total - step)

        yield {
            'name': name,
            'step': step,
            'total': total,
            'pct': 100.0 * step / total if total else 0.0,
            'eta_h': (total - step) * rate / 3600 if rate and step < total else 0.0,
            'val': history['val_dice'][-1] if history.get('val_dice') else None,
            'test': history['test_dice'][-1] if history.get('test_dice') else None,
            'trace': list(zip(history.get('test_step', []), history.get('test_dice', []))),
            # A file untouched for minutes means the run is finished or dead, and the two look
            # identical from progress alone.
            'live': age < 900 and step < total,
        }


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ''
    found = list(rows(pattern))
    if not found:
        print(f'no runs matching {pattern!r}')
        return

    print(f"{'run':<44}{'step':>16}{'':>4}{'eta':>8}{'val':>8}{'TEST':>8}")
    for r in sorted(found, key=lambda r: (-r['live'], r['name'])):
        mark = '*' if r['live'] else ' '
        eta = f"{r['eta_h']:.2f}h" if r['live'] else '-'
        val = f"{r['val']:.4f}" if r['val'] is not None else '-'
        test = f"{r['test']:.4f}" if r['test'] is not None else '-'
        print(f"{mark}{r['name']:<43}{r['step']:>8,}/{r['total']:<7,}{r['pct']:>3.0f}%"
              f"{eta:>8}{val:>8}{test:>8}")

    traced = [r for r in found if r['trace']]
    if traced:
        print('\ntest-set trace (100 fixed pairs; monitoring only, never used for selection)')
        for r in traced:
            marks = '  '.join(f'{s // 1000}k:{d:.4f}' for s, d in r['trace'])
            print(f"  {r['name']:<42}{marks}")

    print('\n* = live (history written within 15 min and not yet at final step)')


if __name__ == '__main__':
    main()
