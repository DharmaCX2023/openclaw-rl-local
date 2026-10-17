import pytest
import torch

from slime.utils.opsd_utils import (
    aggregate_opsd_step_evidence,
    compute_agentopsd_turn_weights,
    compute_belief_bridge_agentopsd_turn_weights,
    build_opsd_step_summary,
    compute_centered_opsd_group_weights,
    compute_opsd_step_weights,
    compute_sdar_gated_distillation,
    compute_sdar_token_signals,
)


def test_agentopsd_segment_merge_uses_token_weighted_step_mean():
    step_ids, evidence, counts = aggregate_opsd_step_evidence(
        [
            (
                torch.tensor([0, 1]),
                torch.tensor([0.2, -0.1]),
                torch.tensor([100.0, 10.0]),
            ),
            (
                torch.tensor([0]),
                torch.tensor([0.4]),
                torch.tensor([50.0]),
            ),
        ]
    )

    assert step_ids == [0, 1]
    assert torch.allclose(evidence, torch.tensor([0.26666668, -0.1]))
    assert torch.equal(counts, torch.tensor([150.0, 10.0]))


def test_agentopsd_evidence_is_invariant_to_repeated_identical_tokens():
    _, short, _ = aggregate_opsd_step_evidence(
        [(torch.tensor([3]), torch.tensor([-0.03]), torch.tensor([10.0]))]
    )
    _, long, _ = aggregate_opsd_step_evidence(
        [(torch.tensor([3]), torch.tensor([-0.03]), torch.tensor([2000.0]))]
    )

    assert torch.equal(short, long)


def test_agentopsd_matches_recursive_belief_equations():
    evidence = torch.tensor([0.4, -0.2, 0.1])
    weights, diagnostics = compute_agentopsd_turn_weights(
        turn_evidence=evidence,
        sequence_advantage=0.75,
        prior_success=0.25,
        decay_gamma=0.95,
        reshape_lambda=0.5,
        multiplier_band=0.2,
    )

    prior = torch.tensor(0.25)
    prior_logit = torch.log(prior / (1.0 - prior))
    accumulator = torch.tensor(0.0)
    previous = prior
    revisions = []
    for value in evidence:
        accumulator = 0.95 * accumulator + value
        belief = torch.sigmoid(prior_logit + accumulator)
        revisions.append(belief - previous)
        previous = belief
    revision = torch.stack(revisions)
    z_score = (revision - revision.mean()) / (revision.std(unbiased=False) + 1e-6)
    raw = (1.0 + 0.2 * z_score).clamp(0.8, 1.2)
    expected = 0.5 + 0.5 * raw

    assert torch.allclose(weights, expected)
    assert diagnostics["prior_success"] == pytest.approx(0.25)
    assert diagnostics["belief_final"] == pytest.approx(previous.item())
    assert diagnostics["belief"][-1] == pytest.approx(previous.item())
    assert len(diagnostics["accumulator"]) == evidence.numel()
    assert len(diagnostics["belief_revision"]) == evidence.numel()
    assert diagnostics["multiplier"] == pytest.approx(weights.tolist())
    assert weights.min().item() >= 0.9 - 1e-6
    assert weights.max().item() <= 1.1 + 1e-6


def test_agentopsd_terminal_sign_changes_credit_ranking_not_direction():
    evidence = torch.tensor([0.8, -0.1, 0.2])
    positive, _ = compute_agentopsd_turn_weights(
        turn_evidence=evidence,
        sequence_advantage=1.0,
        prior_success=0.5,
    )
    negative, _ = compute_agentopsd_turn_weights(
        turn_evidence=evidence,
        sequence_advantage=-1.0,
        prior_success=0.5,
    )

    assert positive[0] > positive[1]
    assert negative[0] < negative[1]
    assert torch.all(positive > 0)
    assert torch.all(negative > 0)


def test_bb_agentopsd_bridges_terminal_belief_and_preserves_direction():
    evidence = torch.tensor([0.4, -0.2, 0.1])
    weights, diagnostics = compute_belief_bridge_agentopsd_turn_weights(
        turn_evidence=evidence, sequence_advantage=-0.75, prior_success=0.25,
        terminal_reward=1.0, group_size=8,
    )
    assert diagnostics["applied"] is True
    assert diagnostics["bridge_target"] == pytest.approx(0.9375)
    assert diagnostics["bridge_terminal_logit_error"] < 1e-5
    assert torch.all(weights > 0)
    assert weights.min().item() >= 0.9 - 1e-6
    assert weights.max().item() <= 1.1 + 1e-6


def test_bb_agentopsd_uses_minimum_energy_bridge():
    evidence = torch.tensor([0.1, 0.2, -0.3, 0.4])
    _, diagnostics = compute_belief_bridge_agentopsd_turn_weights(
        turn_evidence=evidence, sequence_advantage=1.0, prior_success=0.5,
        terminal_reward=0.0, group_size=8, decay_gamma=0.95,
    )
    assert diagnostics["applied"] is True
    assert diagnostics["bridge_target"] == pytest.approx(0.0625)
    assert diagnostics["bridge_correction_l2"] > 0


def test_opsd_weights_use_rollout_old_and_ignore_environment_tokens():
    teacher = torch.tensor([-0.2, -0.3, 99.0, -1.3, -1.4])
    rollout_old = torch.tensor([-0.7, -0.8, -99.0, -1.0, -1.1])
    loss_mask = torch.tensor([1, 1, 0, 1, 1])
    step_ids = torch.tensor([0, 0, -1, 1, 1])

    token_weights, supports, weights = compute_opsd_step_weights(
        teacher_log_probs=teacher,
        rollout_old_log_probs=rollout_old,
        loss_mask=loss_mask,
        step_ids=step_ids,
        beta=1.0,
        clip_lower=0.0,
        clip_upper=1.2,
        advantage_sign=-1.0,
    )

    assert torch.allclose(supports, torch.tensor([0.5, -0.3]))
    assert torch.allclose(weights, torch.tensor([torch.exp(torch.tensor(-0.5)), 1.2]))
    assert token_weights[2].item() == 1.0
    assert torch.allclose(token_weights[:2], weights[0].expand(2))
    assert torch.allclose(token_weights[3:], weights[1].expand(2))


def test_opsd_weights_validate_response_space_alignment():
    try:
        compute_opsd_step_weights(
            teacher_log_probs=torch.zeros(2),
            rollout_old_log_probs=torch.zeros(3),
            loss_mask=torch.ones(2),
            step_ids=torch.zeros(2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "identical response-space shapes" in str(exc)
    else:
        raise AssertionError("shape mismatch must fail")


def test_opsd_step_summary_contains_only_scalar_aggregates():
    summary = build_opsd_step_summary(
        step_ids=torch.tensor([2, 2, -1, 7]),
        loss_mask=torch.tensor([1, 1, 0, 1]),
        supports=torch.tensor([0.5, -0.25]),
        weights=torch.tensor([0.6, 1.2]),
        advantage=-0.75,
    )

    assert summary == [
        {
            "step_id": 2,
            "loss_token_count": 2,
            "teacher_minus_rollout_old_mean": 0.5,
            "weight": pytest.approx(0.6),
            "original_advantage": -0.75,
                "weighted_advantage": pytest.approx(-0.45),
        },
        {
            "step_id": 7,
            "loss_token_count": 1,
            "teacher_minus_rollout_old_mean": -0.25,
            "weight": pytest.approx(1.2),
            "original_advantage": -0.75,
                "weighted_advantage": pytest.approx(-0.9),
        },
    ]



def test_centered_opsd_uses_exact_bounded_median_mad_formula():
    counts = torch.tensor([100.0, 20.0, 5.0, 1.0])
    weights, diagnostics = compute_centered_opsd_group_weights(
        supports=torch.tensor([-0.20, -0.05, 0.10, 0.50]),
        token_counts=counts,
        beta=0.10,
        clip_lower=0.8,
        clip_upper=1.2,
    )
    weighted_mean = (weights * counts).sum() / counts.sum()
    source = torch.tensor([-0.20, -0.05, 0.10, 0.50])
    expected = torch.exp(-0.10 * (source - 0.025) / (0.15 + 1e-6)).clamp(0.8, 1.2)
    assert diagnostics["applied"] is True
    assert torch.allclose(weights, expected)
    assert diagnostics["final_token_weight_mean"] == pytest.approx(weighted_mean.item())
    assert weights.min().item() >= 0.8 - 1e-6
    assert weights.max().item() <= 1.2 + 1e-6
    assert weights[-1].item() < weights[0].item()


def test_centered_opsd_degenerate_mad_falls_back_to_grpo():
    weights, diagnostics = compute_centered_opsd_group_weights(
        supports=torch.tensor([0.3, 0.3, 0.3]),
        token_counts=torch.tensor([2.0, 3.0, 4.0]),
        beta=0.10,
    )
    assert diagnostics["applied"] is False
    assert diagnostics["fallback_reason"] == "centered_mad_degenerate"
    assert torch.equal(weights, torch.ones(3))


def test_sdar_gate_uses_frozen_rollout_old_gap():
    gates, delta, diagnostics = compute_sdar_token_signals(
        teacher_log_probs=torch.tensor([-0.2, -2.0, 7.0]),
        rollout_old_log_probs=torch.tensor([-1.2, -1.0, -9.0]),
        loss_mask=torch.tensor([1, 1, 0]),
        gate_beta=5.0,
    )
    assert torch.allclose(delta, torch.tensor([1.0, -1.0, 16.0]))
    assert gates[0].item() > 0.99
    assert gates[1].item() < 0.01
    assert diagnostics["gate_active_fraction"] == pytest.approx(0.5)


def test_sdar_distillation_only_updates_current_student():
    teacher = torch.tensor([-0.2, -1.5], requires_grad=True)
    rollout_old = torch.tensor([-1.2, -1.0], requires_grad=True)
    student = torch.tensor([-1.0, -1.0], requires_grad=True)
    token_loss, gate, gap = compute_sdar_gated_distillation(
        teacher_log_probs=teacher,
        rollout_old_log_probs=rollout_old,
        student_log_probs=student,
        apply_mask=torch.ones(2),
        gate_beta=5.0,
    )
    token_loss.mean().backward()
    assert teacher.grad is None
    assert rollout_old.grad is None
    assert student.grad is not None
    assert torch.allclose(student.grad, -gate / 2.0)
    assert gate.requires_grad is False
    assert gap.requires_grad is False
    assert gate[0].item() > gate[1].item()


def test_sdar_distillation_fallback_mask_has_zero_gradient():
    student = torch.tensor([-1.0, -2.0], requires_grad=True)
    token_loss, _, _ = compute_sdar_gated_distillation(
        teacher_log_probs=torch.tensor([-0.5, -0.5]),
        rollout_old_log_probs=torch.tensor([-1.0, -1.0]),
        student_log_probs=student,
        apply_mask=torch.zeros(2),
    )
    token_loss.sum().backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_matched_placebo_summary_records_both_conditions_and_delta():
    summary = build_opsd_step_summary(
        step_ids=torch.tensor([4, 4]),
        loss_mask=torch.tensor([1, 1]),
        supports=torch.tensor([0.25]),
        weights=torch.tensor([0.8]),
        advantage=-0.5,
        conditioned_means=torch.tensor([-0.75]),
        baseline_means=torch.tensor([-1.0]),
        matched_placebo=True,
    )

    assert summary == [
        {
            "step_id": 4,
            "loss_token_count": 2,
            "real_anchor_logprob_mean": -0.75,
            "placebo_anchor_logprob_mean": -1.0,
            "real_minus_placebo_mean": 0.25,
            "weight": pytest.approx(0.8),
            "original_advantage": -0.5,
            "weighted_advantage": pytest.approx(-0.4),
        }
    ]
