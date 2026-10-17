import pytest

from slime.backends.megatron_utils.model import _derive_grad_clip_metrics


def test_large_pre_clip_norm_reports_applied_clip():
    pre_clip, coefficient, post_clip = _derive_grad_clip_metrics(766.43, 1.0)

    assert pre_clip == pytest.approx(766.43)
    assert coefficient == pytest.approx(0.0013047506)
    assert post_clip == pytest.approx(1.0)


def test_small_norm_is_not_scaled():
    assert _derive_grad_clip_metrics(0.25, 1.0) == (0.25, 1.0, 0.25)
