import math

import pytest
import torch

from slime.utils.ppo_utils import compute_policy_loss, normalize_grpo_rewards_by_trajectory


def test_grpo_reward_normalization_uses_one_reward_per_trajectory():
    normalized = normalize_grpo_rewards_by_trajectory(
        raw_rewards=[-1.0, -1.0, -1.0, 1.0],
        group_indices=[7, 7, 7, 7],
        trajectory_indices=[10, 10, 10, 11],
        std_normalization=True,
    )

    expected = 1 / math.sqrt(2)
    assert normalized == pytest.approx([-expected, -expected, -expected, expected], abs=1e-5)


def test_grpo_constant_reward_group_has_zero_advantage():
    normalized = normalize_grpo_rewards_by_trajectory(
        raw_rewards=[-1.0, -1.0, -1.0, -1.0],
        group_indices=[3, 3, 3, 3],
        trajectory_indices=[20, 20, 21, 21],
        std_normalization=True,
    )

    assert normalized == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_grpo_duplicate_turn_rewards_must_match():
    with pytest.raises(ValueError, match="same GRPO reward"):
        normalize_grpo_rewards_by_trajectory(
            raw_rewards=[-1.0, 1.0],
            group_indices=[1, 1],
            trajectory_indices=[5, 5],
        )


def test_dual_clip_caps_negative_advantage_loss():
    ratio = torch.tensor([1000.0, 1000.0])
    ppo_kl = -ratio.log()
    advantages = torch.tensor([-1.0, 1.0])

    loss, clipfrac = compute_policy_loss(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=3.0,
    )

    assert loss.tolist() == pytest.approx([3.0, -1.28], abs=1e-5)
    assert clipfrac.tolist() == pytest.approx([0.0, 1.0])
