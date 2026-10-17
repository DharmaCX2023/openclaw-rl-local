from unittest.mock import patch

import pytest
import torch

from slime.backends.megatron_utils.cp_utils import get_sum_of_logical_rollout_mean
from slime.utils.ppo_utils import build_logical_rollout_loss_metadata


def _reduce(losses, response_lengths, masks, denominators):
    with patch(
        "slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size",
        return_value=1,
    ):
        reducer = get_sum_of_logical_rollout_mean(list(response_lengths), response_lengths, masks, denominators)
    return reducer(torch.tensor(losses, dtype=torch.float32)).item()


def test_metadata_aggregates_split_segments():
    token_counts, rollout_count = build_logical_rollout_loss_metadata(
        group_indices=[0, 0, 0, 1],
        trajectory_indices=[10, 10, 11, 20],
        loss_masks=[[1, 1], [1, 0, 1], [1], [1, 1, 1]],
    )
    assert token_counts == [4, 4, 1, 3]
    assert rollout_count == 3


def test_metadata_rejects_missing_identity():
    with pytest.raises(ValueError, match="group_index and trajectory_index"):
        build_logical_rollout_loss_metadata([0], [None], [[1]])


def test_split_and_unsplit_have_identical_loss():
    unsplit = _reduce([1.0, 3.0, 5.0, 7.0], [4], [torch.tensor([1, 1, 1, 1])], [4])
    split = _reduce(
        [1.0, 3.0, 5.0, 7.0],
        [2, 1, 1],
        [torch.tensor([1, 1]), torch.tensor([1]), torch.tensor([1])],
        [4, 4, 4],
    )
    assert unsplit == pytest.approx(4.0)
    assert split == pytest.approx(unsplit)


def test_each_logical_rollout_contributes_one_mean():
    reduced = _reduce(
        [2.0, 4.0, 10.0],
        [1, 1, 1],
        [torch.tensor([1]), torch.tensor([1]), torch.tensor([1])],
        [2, 2, 1],
    )
    # Reducer returns the sum of rollout means: (2 + 4) / 2 + 10.
    assert reduced == pytest.approx(13.0)
