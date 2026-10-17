import pytest

from slime.backends.megatron_utils.data import _hard_token_limit_for_cp
from slime.utils.data import get_capacity_constrained_micro_batches, get_minimum_num_micro_batch_size


def _assert_valid(lengths, capacity, batches):
    flat = [index for batch in batches for index in batch]
    assert sorted(flat) == list(range(len(lengths)))
    for batch in batches:
        tokens = sum(lengths[index] for index in batch)
        assert tokens <= capacity or (len(batch) == 1 and tokens > capacity)


def test_best_fit_decreasing_avoids_input_order_fragmentation():
    lengths = [4, 4, 6, 6]
    batches = get_capacity_constrained_micro_batches(lengths, 10)

    assert len(batches) == 2
    assert sorted(sum(lengths[index] for index in batch) for batch in batches) == [10, 10]
    assert get_minimum_num_micro_batch_size(lengths, 10) == 2
    _assert_valid(lengths, 10, batches)


def test_oversized_sample_is_kept_as_singleton():
    lengths = [20, 5, 5]
    batches = get_capacity_constrained_micro_batches(lengths, 10)

    assert len(batches) == 2
    assert [0] in batches
    _assert_valid(lengths, 10, batches)


def test_target_count_splits_bins_without_changing_samples():
    lengths = [6, 6, 4, 4]
    batches = get_capacity_constrained_micro_batches(lengths, 10, target_num_micro_batches=3)

    assert len(batches) == 3
    _assert_valid(lengths, 10, batches)


def test_packing_is_deterministic():
    lengths = [9, 8, 7, 3, 2, 1]
    expected = get_capacity_constrained_micro_batches(lengths, 10)

    for _ in range(10):
        assert get_capacity_constrained_micro_batches(lengths, 10) == expected


def test_invalid_target_is_rejected():
    with pytest.raises(ValueError, match="below packed minimum"):
        get_capacity_constrained_micro_batches([6, 6, 4, 4], 10, target_num_micro_batches=1)

    with pytest.raises(ValueError, match="exceeds num_samples"):
        get_capacity_constrained_micro_batches([5, 5], 10, target_num_micro_batches=3)


def test_microbatch_hard_limit_scales_with_context_parallel_size():
    assert _hard_token_limit_for_cp(1) == 100_000
    assert _hard_token_limit_for_cp(2) == 200_000

    with pytest.raises(ValueError, match="must be positive"):
        _hard_token_limit_for_cp(0)
