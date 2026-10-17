from argparse import Namespace

import pytest

from slime.utils.ppo_utils import should_compute_reference_log_probs


def _args(**updates):
    values = {
        "kl_coef": 0.0,
        "use_kl_loss": True,
        "kl_loss_coef": 0.0,
        "kl_loss_monitor_interval": 5,
    }
    values.update(updates)
    return Namespace(**values)


def test_zero_weight_kl_monitor_runs_at_requested_interval():
    observed = [step for step in range(12) if should_compute_reference_log_probs(_args(), step)]
    assert observed == [0, 5, 10]


def test_optimization_affecting_kl_always_runs():
    assert all(should_compute_reference_log_probs(_args(kl_loss_coef=0.01), step) for step in range(12))
    assert all(should_compute_reference_log_probs(_args(use_kl_loss=False, kl_coef=0.01), step) for step in range(12))


def test_disabled_kl_does_not_run_reference_forward():
    assert not should_compute_reference_log_probs(_args(use_kl_loss=False), 0)


def test_invalid_monitor_interval_is_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        should_compute_reference_log_probs(_args(kl_loss_monitor_interval=0), 0)
