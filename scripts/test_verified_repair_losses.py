import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from verified_repair_losses import action_dpo, all_blocked_margin, sampling_support, task_pair_mean


def test_all_blocked_get_gradient_not_only_first():
    logits = torch.tensor([[3., 2., 1., 0.]] * 4, requires_grad=True)
    targets = torch.tensor([0, 2, 3, 1])
    loss, stats = all_blocked_margin(logits, targets, top_k=2, top_p=1)
    assert stats['blocked_mask'].tolist() == [False, True, True, False]
    loss.backward()
    assert logits.grad[1, 2].item() == pytest.approx(-1 / (2 * .7))
    assert logits.grad[2, 3].item() == pytest.approx(-1 / (2 * .7))
    assert torch.count_nonzero(logits.grad) == 2  # boundary is detached


def test_zero_actions_still_in_task_mean():
    logits = torch.tensor([[3., 2., 1.]], requires_grad=True)
    loss, stats = all_blocked_margin(logits, torch.tensor([0]))
    assert loss.item() == 0 and stats['blocked_count'] == 0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0
    assert task_pair_mean([loss, loss + 2, loss + 6], ['a', 'a', 'b']).item() == 3.5


def test_actual_native_membership_controls_all_positions_not_train_reconstruction():
    logits = torch.tensor([[3., 2., 1.], [3., 2., 1.], [3., 2., 1.]], requires_grad=True)
    targets = torch.tensor([1, 1, 2])
    # Train top-1 would reject all three; actual C allows the first target,
    # but excludes the other two. Both native-excluded positions get gradients.
    loss, stats = all_blocked_margin(logits, targets, top_k=1, top_p=1,
                                     native_candidate_ids=[[0, 1], [0], [0, 1]])
    assert stats['blocked_mask'].tolist() == [False, True, True]
    assert stats['mask_source'] == 'native_sglang_candidate_ids'
    loss.backward()
    assert logits.grad[1, 1] == pytest.approx(-1/(2*.7))
    assert logits.grad[2, 2] == pytest.approx(-1/(2*.7))
    assert torch.count_nonzero(logits.grad) == 2


@pytest.mark.parametrize('top_k,top_p', [(20,.8),(1,.8),(-1,1.),(5,.95),(20,.1)])
def test_support_matches_installed_sglang(top_k, top_p):
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
    generator = torch.Generator().manual_seed(20260907)
    logits = torch.randn(32, 101, generator=generator)
    logits[0].fill_(0)  # sorting ties use the same torch.sort convention
    support, _ = sampling_support(logits, top_k=top_k, top_p=top_p)
    captured = []
    def capture(probs, *args, **kwargs):
        captured.append(probs.detach().clone())
        return probs.argmax(dim=-1, keepdim=True)
    with patch('torch.multinomial', side_effect=capture):
        top_k_top_p_min_p_sampling_from_probs_torch(
            torch.softmax(logits.float() / .7, -1),
            torch.full((32,), top_k if top_k > 0 else 101),
            torch.full((32,), top_p), torch.zeros(32), False, None, torch.zeros(32))
    assert len(captured) == 1
    indices = torch.softmax(logits.float() / .7, -1).sort(dim=-1, descending=True).indices
    actual = torch.zeros_like(support).scatter(-1, indices, captured[0] > 0)
    assert torch.equal(support, actual)


def test_hard_constraint_target_rejected():
    with pytest.raises(ValueError, match='hard constraint'):
        all_blocked_margin(torch.tensor([[0., -float('inf')]]), torch.tensor([1]))


def test_dpo_at_reference_and_gradient_sign():
    pos = torch.tensor(-123., requires_grad=True)
    neg = torch.tensor(-45., requires_grad=True)
    refp, refn = pos.detach().clone(), neg.detach().clone()
    loss = action_dpo(pos, neg, refp, refn)
    assert loss.item() == pytest.approx(0.69314718)
    loss.backward()
    assert pos.grad.item() == pytest.approx(-.05)
    assert neg.grad.item() == pytest.approx(.05)
