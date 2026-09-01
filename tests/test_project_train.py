"""
Tests for the training loop's test-set trace.

The trace exists so a four-hour 3D run reports its conclusion as it goes rather than only at the
end. That convenience is only safe if the trace is genuinely read-only: the moment a checkpoint
is chosen by test Dice, every number downstream is optimistically biased and the fixed pair list
stops meaning what it claims to mean. These tests pin that property, and pin that the trace
scores the *same* pairs `evaluate.py` will score -- a trace on a different pair list would look
reassuring and predict nothing.
"""

import json

import pytest
import torch

from project import train as train_module
from project.configs import ExperimentConfig
from project.data import OasisData, fixed_pairs


def _config(tmp_path, **kw):
    defaults = dict(name='trace', variant='baseline', ndim=2, steps=4, batch_size=2,
                    val_every=2, val_pairs=2, lambda_reg=0.25,
                    data_path='data/oasis2d.npz', output_root=str(tmp_path))
    defaults.update(kw)
    return ExperimentConfig(**defaults)


@pytest.fixture
def fake_dice(monkeypatch):
    """
    Make validation Dice fall over time and test Dice rise.

    If the trace leaked into selection, `best.pt` would follow the rising series. Because it
    must not, the best checkpoint has to come from the first validation, when the falling
    series is at its peak.
    """
    calls = {'val': [], 'test': []}

    def fake(model, data, pairs, labels, device, misalign_magnitude=0.0):
        which = 'val' if len(pairs) == 2 else 'test'
        calls[which].append(len(calls[which]))
        if which == 'val':
            return 0.9 - 0.1 * len(calls['val'])
        return 0.85 + 0.05 * len(calls['test'])

    monkeypatch.setattr(train_module, 'validation_dice', fake)
    return calls


def test_trace_is_off_by_default():
    assert ExperimentConfig(name='x', data_path='data/oasis2d.npz').test_every == 0


def test_trace_never_selects_the_checkpoint(tmp_path, fake_dice):
    history = train_module.train(_config(tmp_path, test_every=2, test_pairs=3))

    # Selection followed the validation series, which peaked first and then fell.
    assert history['best_val_dice'] == pytest.approx(max(history['val_dice']))
    assert history['best_val_dice'] == pytest.approx(history['val_dice'][0])
    # ...while the test series rose throughout and was recorded but ignored.
    assert history['test_dice'] == sorted(history['test_dice'])
    assert max(history['test_dice']) > history['best_val_dice']


def test_trace_records_at_the_requested_cadence(tmp_path, fake_dice):
    history = train_module.train(_config(tmp_path, steps=4, test_every=2, test_pairs=3))
    assert history['test_step'] == [2, 4]


def test_trace_absent_when_disabled(tmp_path, fake_dice):
    history = train_module.train(_config(tmp_path))
    assert history['test_step'] == []
    assert history['test_dice'] == []


def test_history_is_written_incrementally_so_a_live_run_can_be_read(tmp_path, fake_dice):
    """A trace nobody can read until the run ends does not solve the problem it was added for."""
    config = _config(tmp_path, test_every=2, test_pairs=3)
    train_module.train(config)
    written = json.load(open(config.output_dir / 'history.json'))
    assert written['test_dice']


def test_trace_scores_the_same_pairs_evaluate_will_score():
    """
    The trace must use evaluate.py's split, count and seed, or it predicts the wrong number.

    This is the one thing that cannot be caught by reading the training log: a trace on a
    different pair list still produces a smooth, believable curve.
    """
    data = OasisData('data/oasis2d.npz', device='cpu')
    assert fixed_pairs(data, 'test', 100, seed=1234) == fixed_pairs(data, 'test', 100, seed=1234)
    trace_pairs = fixed_pairs(data, 'test', ExperimentConfig(
        name='x', data_path='data/oasis2d.npz').test_pairs, seed=1234)
    assert len(trace_pairs) == 100


def test_head_group_isolates_the_output_layer():
    """
    The lr control only means anything if the group really is just the head.

    The baseline's whole output head is 72 parameters against ~111k in the network, so a marker
    that accidentally swept in a decoder block would raise the learning rate on most of the
    model and the result would say nothing about the head.
    """
    from project.models import build_model
    from project.train import head_parameter_groups

    model = build_model(ExperimentConfig(name='x', variant='baseline', ndim=2,
                                         data_path='data/oasis2d.npz'))
    body, head = head_parameter_groups(model, 1e-4, 10.0)
    head_count = sum(p.numel() for p in head['params'])
    body_count = sum(p.numel() for p in body['params'])
    assert head_count == 72
    assert head_count + body_count == sum(p.numel() for p in model.parameters())
    assert head['lr'] == pytest.approx(1e-3)
    assert body['lr'] == pytest.approx(1e-4)


def test_head_group_rejects_a_model_with_no_matching_head():
    """Silently applying no multiplier would make the control a duplicate of the baseline."""
    from project.train import head_parameter_groups
    with pytest.raises(ValueError):
        head_parameter_groups(torch.nn.Linear(2, 2), 1e-4, 10.0)


def test_default_multiplier_is_one_so_existing_runs_are_unchanged():
    assert ExperimentConfig(name='x', data_path='data/oasis2d.npz').head_lr_mult == 1.0
