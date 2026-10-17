"""Backend-independent helpers for OPSD and SDAR teacher guidance."""

from __future__ import annotations

from typing import Any

import torch


def compute_agentopsd_turn_weights(
    *,
    turn_evidence: torch.Tensor,
    sequence_advantage: float,
    prior_success: float,
    decay_gamma: float = 0.95,
    reshape_lambda: float = 0.5,
    multiplier_band: float = 0.2,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Paper-faithful AgentOPSD recursive turn-credit multipliers.

    turn_evidence[k] is the mean detached teacher-minus-rollout-old token
    log-probability gap in environment-aligned agent turn k. The returned
    positive multiplier scales ordinary GRPO advantage and never flips sign.
    """

    evidence = turn_evidence if turn_evidence.is_floating_point() else turn_evidence.float()
    if evidence.ndim != 1:
        raise ValueError("AgentOPSD turn evidence must be one-dimensional")
    if not 0.0 < float(decay_gamma) <= 1.0:
        raise ValueError("AgentOPSD decay gamma must be in (0, 1]")
    if not 0.0 <= float(reshape_lambda) <= 1.0:
        raise ValueError("AgentOPSD reshape lambda must be in [0, 1]")
    if not 0.0 < float(multiplier_band) < 1.0:
        raise ValueError("AgentOPSD multiplier band must be in (0, 1)")
    if epsilon <= 0.0:
        raise ValueError("AgentOPSD epsilon must be positive")
    if evidence.numel() == 0:
        return evidence.clone(), {
            "applied": False,
            "fallback_reason": "agentopsd_no_active_steps",
        }

    prior = evidence.new_tensor(float(prior_success)).clamp(1e-4, 1.0 - 1e-4)
    prior_logit = torch.log(prior / (1.0 - prior))
    accumulator = evidence.new_zeros(())
    previous_belief = prior
    revisions: list[torch.Tensor] = []
    beliefs: list[torch.Tensor] = []
    for value in evidence:
        accumulator = float(decay_gamma) * accumulator + value
        belief = torch.sigmoid(prior_logit + accumulator)
        revisions.append(belief - previous_belief)
        beliefs.append(belief)
        previous_belief = belief

    revision = torch.stack(revisions)
    outcome_aligned = torch.sign(evidence.new_tensor(float(sequence_advantage))) * revision
    mean = outcome_aligned.mean()
    std = outcome_aligned.std(unbiased=False)
    z_score = (outcome_aligned - mean) / (std + float(epsilon))
    raw_multiplier = (1.0 + float(multiplier_band) * z_score).clamp(
        1.0 - float(multiplier_band), 1.0 + float(multiplier_band)
    )
    multiplier = (1.0 - float(reshape_lambda)) + float(reshape_lambda) * raw_multiplier
    abs_revision = revision.abs()
    total_abs_revision = abs_revision.sum()
    diagnostics: dict[str, Any] = {
        "applied": True,
        "fallback_reason": "",
        "prior_success": float(prior.item()),
        "belief_final": float(beliefs[-1].item()),
        "revision_abs_sum": float(total_abs_revision.item()),
        "first_revision_share": float(
            (abs_revision[0] / (total_abs_revision + float(epsilon))).item()
        ),
        "near_zero_revision_fraction": float((abs_revision < 1e-8).float().mean().item()),
        "multiplier_mean": float(multiplier.mean().item()),
        "multiplier_min": float(multiplier.min().item()),
        "multiplier_max": float(multiplier.max().item()),
        "multiplier_boundary_fraction": float(
            (
                (raw_multiplier <= 1.0 - float(multiplier_band) + 1e-6)
                | (raw_multiplier >= 1.0 + float(multiplier_band) - 1e-6)
            )
            .float()
            .mean()
            .item()
        ),
        # Complete detached belief chain for post-hoc credit audits.
        "evidence": [float(value.item()) for value in evidence],
        "accumulator": [],
        "belief": [float(value.item()) for value in beliefs],
        "belief_revision": [float(value.item()) for value in revision],
        "outcome_aligned_revision": [float(value.item()) for value in outcome_aligned],
        "revision_z_score": [float(value.item()) for value in z_score],
        "multiplier": [float(value.item()) for value in multiplier],
    }
    running = evidence.new_zeros(())
    for value in evidence:
        running = float(decay_gamma) * running + value
        diagnostics["accumulator"].append(float(running.item()))
    return multiplier, diagnostics


def aggregate_opsd_step_evidence(
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    """Merge segmented step statistics into token-weighted step means.

    Each chunk contains step IDs, step gap means, and token counts. A logical
    agent step may span several training segments after context pruning; its
    evidence must remain a mean rather than growing with either token count or
    the number of segments.
    """

    weighted_sums: dict[int, float] = {}
    total_counts: dict[int, float] = {}
    dtype = torch.float32
    for step_ids, supports, counts in chunks:
        if not (step_ids.ndim == supports.ndim == counts.ndim == 1):
            raise ValueError("AgentOPSD step evidence chunks must be one-dimensional")
        if not (step_ids.numel() == supports.numel() == counts.numel()):
            raise ValueError("AgentOPSD step evidence chunks must have identical lengths")
        dtype = supports.dtype if supports.is_floating_point() else torch.float32
        for step_id, support, count in zip(step_ids, supports, counts, strict=True):
            sid = int(step_id.item())
            n_tokens = float(count.item())
            if n_tokens <= 0.0:
                raise ValueError("AgentOPSD step evidence requires positive token counts")
            weighted_sums[sid] = weighted_sums.get(sid, 0.0) + float(support.item()) * n_tokens
            total_counts[sid] = total_counts.get(sid, 0.0) + n_tokens

    ordered_ids = sorted(weighted_sums)
    evidence = torch.tensor(
        [weighted_sums[sid] / total_counts[sid] for sid in ordered_ids],
        dtype=dtype,
    )
    counts = torch.tensor([total_counts[sid] for sid in ordered_ids], dtype=torch.float32)
    return ordered_ids, evidence, counts


def compute_belief_bridge_agentopsd_turn_weights(
    *, turn_evidence: torch.Tensor, sequence_advantage: float,
    prior_success: float, terminal_reward: float, group_size: int,
    decay_gamma: float = 0.95, reshape_lambda: float = 0.5,
    multiplier_band: float = 0.2, epsilon: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """AgentOPSD with a minimum-L2 terminal verifier bridge."""
    evidence = turn_evidence.double()
    if evidence.ndim != 1:
        raise ValueError("BB-OPSD turn evidence must be one-dimensional")
    if evidence.numel() == 0:
        return evidence.clone(), {"applied": False, "fallback_reason": "bb_agentopsd_no_active_steps"}
    if group_size <= 0:
        raise ValueError("BB-OPSD group size must be positive")
    if not torch.isfinite(evidence).all():
        return torch.ones_like(evidence), {"applied": False, "fallback_reason": "bb_agentopsd_nonfinite_evidence"}

    prior = evidence.new_tensor(float(prior_success)).clamp(1e-4, 1.0 - 1e-4)
    prior_logit = torch.logit(prior)
    reward_epsilon = 1.0 / (2.0 * float(group_size))
    target = evidence.new_tensor(float(terminal_reward)).clamp(0.0, 1.0)
    target = reward_epsilon + (1.0 - 2.0 * reward_epsilon) * target
    target_logit = torch.logit(target)

    accumulator = evidence.new_zeros(())
    for value in evidence:
        accumulator = float(decay_gamma) * accumulator + value
    forward_terminal_logit = prior_logit + accumulator
    residual = target_logit - forward_terminal_logit

    count = int(evidence.numel())
    exponents = torch.arange(count - 1, -1, -1, device=evidence.device)
    coefficients = evidence.new_tensor(float(decay_gamma)).pow(exponents)
    correction = residual * coefficients / coefficients.square().sum()
    corrected_evidence = evidence + correction

    weights, diagnostics = compute_agentopsd_turn_weights(
        turn_evidence=corrected_evidence,
        sequence_advantage=sequence_advantage,
        prior_success=prior_success,
        decay_gamma=decay_gamma,
        reshape_lambda=reshape_lambda,
        multiplier_band=multiplier_band,
        epsilon=epsilon,
    )
    corrected_accumulator = evidence.new_zeros(())
    for value in corrected_evidence:
        corrected_accumulator = float(decay_gamma) * corrected_accumulator + value
    terminal_logit_error = prior_logit + corrected_accumulator - target_logit
    diagnostics.update({
        "bridge_target": float(target.item()),
        "bridge_forward_terminal_logit": float(forward_terminal_logit.item()),
        "bridge_residual": float(residual.item()),
        "bridge_correction_l2": float(torch.linalg.vector_norm(correction).item()),
        "bridge_correction_abs_max": float(correction.abs().max().item()),
        "bridge_terminal_logit_error": float(terminal_logit_error.abs().item()),
    })
    if not torch.isfinite(weights).all() or diagnostics["bridge_terminal_logit_error"] >= 1e-5:
        return torch.ones_like(evidence), {
            **diagnostics,
            "applied": False,
            "fallback_reason": "bb_agentopsd_bridge_invariant_failed",
        }
    return weights, diagnostics


def compute_step_logprob_means(
    *,
    log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    step_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return active step IDs, mean log-probabilities, and token counts."""

    if not (log_probs.shape == loss_mask.shape == step_ids.shape):
        raise ValueError(
            "step logprob tensors must have identical response-space shapes: "
            f"log_probs={tuple(log_probs.shape)}, "
            f"mask={tuple(loss_mask.shape)}, steps={tuple(step_ids.shape)}"
        )
    values = log_probs.float()
    active = loss_mask.bool() & (step_ids >= 0)
    valid_step_ids = torch.unique(step_ids[active]).sort().values
    if valid_step_ids.numel() == 0:
        empty = values.new_empty((0,), dtype=torch.float32)
        return valid_step_ids, empty, empty

    means: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for step_id in valid_step_ids:
        step_mask = active & (step_ids == step_id)
        means.append(values[step_mask].mean())
        counts.append(step_mask.sum().to(dtype=torch.float32))
    return valid_step_ids, torch.stack(means), torch.stack(counts)


def compute_opsd_step_supports(
    *,
    teacher_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    step_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return step IDs, conditioned-minus-baseline means, and token counts.

    Historical argument names are retained for compatibility. In matched
    placebo mode the teacher is real-Anchor and the baseline is placebo.
    """

    if teacher_log_probs.shape != rollout_old_log_probs.shape:
        raise ValueError(
            "conditioned and baseline logprobs must have identical shapes: "
            f"teacher={tuple(teacher_log_probs.shape)}, "
            f"baseline={tuple(rollout_old_log_probs.shape)}"
        )
    step_ids_a, teacher_means, counts = compute_step_logprob_means(
        log_probs=teacher_log_probs, loss_mask=loss_mask, step_ids=step_ids
    )
    step_ids_b, baseline_means, baseline_counts = compute_step_logprob_means(
        log_probs=rollout_old_log_probs, loss_mask=loss_mask, step_ids=step_ids
    )
    if not torch.equal(step_ids_a, step_ids_b) or not torch.equal(
        counts, baseline_counts
    ):
        raise ValueError("conditioned and baseline step partitions differ")
    return step_ids_a, teacher_means - baseline_means, counts


def broadcast_opsd_step_weights(
    *,
    loss_mask: torch.Tensor,
    step_ids: torch.Tensor,
    valid_step_ids: torch.Tensor,
    step_weights: torch.Tensor,
) -> torch.Tensor:
    """Broadcast one weight per active assistant step back to response tokens."""

    if loss_mask.shape != step_ids.shape:
        raise ValueError("loss_mask and step_ids must have identical shapes")
    if valid_step_ids.numel() != step_weights.numel():
        raise ValueError("step IDs and weights must have identical lengths")
    token_weights = torch.ones_like(loss_mask, dtype=torch.float32)
    active = loss_mask.bool() & (step_ids >= 0)
    for step_id, weight in zip(valid_step_ids, step_weights, strict=True):
        token_weights[active & (step_ids == step_id)] = weight
    return token_weights


def compute_centered_opsd_group_weights(
    *,
    supports: torch.Tensor,
    token_counts: torch.Tensor,
    beta: float,
    clip_lower: float = 0.8,
    clip_upper: float = 1.2,
    epsilon: float = 1e-6,
    min_steps: int = 2,
) -> tuple[torch.Tensor, dict[str, float | bool | str]]:
    """Center step gaps within one logical rollout and apply bounded weights."""

    supports = supports.float()
    token_counts = token_counts.float()
    if supports.ndim != 1 or token_counts.ndim != 1 or supports.shape != token_counts.shape:
        raise ValueError("centered OPSD supports and token counts must be aligned 1-D tensors")
    if beta < 0:
        raise ValueError("centered OPSD beta must be non-negative")
    if epsilon <= 0:
        raise ValueError("centered OPSD epsilon must be positive")
    if min_steps < 2:
        raise ValueError("centered OPSD min_steps must be at least 2")
    if not (0 <= clip_lower <= 1.0 <= clip_upper):
        raise ValueError("centered OPSD bounds must straddle 1")
    if supports.numel() == 0 or torch.any(token_counts <= 0):
        raise ValueError("centered OPSD requires positive token support")

    median = torch.quantile(supports, 0.5, interpolation="midpoint")
    mad = torch.quantile((supports - median).abs(), 0.5, interpolation="midpoint")
    base = torch.ones_like(supports)
    diagnostics: dict[str, float | bool | str] = {
        "applied": False,
        "fallback_reason": "",
        "support_median": float(median.item()),
        "support_mad": float(mad.item()),
        "raw_token_weight_mean": 1.0,
        "final_token_weight_mean": 1.0,
        "saturation_fraction": 0.0,
    }
    if supports.numel() < min_steps:
        diagnostics["fallback_reason"] = "centered_too_few_steps"
        return base, diagnostics
    if not torch.isfinite(mad) or float(mad.item()) <= epsilon:
        diagnostics["fallback_reason"] = "centered_mad_degenerate"
        return base, diagnostics

    z_scores = (supports - median) / (mad + float(epsilon))
    raw_weights = torch.exp(-float(beta) * z_scores).clamp(
        min=float(clip_lower), max=float(clip_upper)
    )
    total_tokens = token_counts.sum()
    raw_mean = (raw_weights * token_counts).sum() / total_tokens

    # Keep the experiment equation exact. Logical-rollout equality is handled
    # by the loss normalizer, not by rescaling the credit weights.
    weights = raw_weights
    final_mean = (weights * token_counts).sum() / total_tokens
    saturation = ((weights <= clip_lower + 1e-6) | (weights >= clip_upper - 1e-6)).float().mean()
    diagnostics.update(
        {
            "applied": True,
            "raw_token_weight_mean": float(raw_mean.item()),
            "final_token_weight_mean": float(final_mean.item()),
            "saturation_fraction": float(saturation.item()),
        }
    )
    return weights, diagnostics


def compute_sdar_token_signals(
    *,
    teacher_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return detached SDAR gates and teacher-minus-rollout-old gaps."""

    if not (teacher_log_probs.shape == rollout_old_log_probs.shape == loss_mask.shape):
        raise ValueError("SDAR tensors must have identical response-space shapes")
    if gate_beta < 0:
        raise ValueError("SDAR gate beta must be non-negative")
    active = loss_mask.bool()
    delta = teacher_log_probs.float() - rollout_old_log_probs.float()
    gates = torch.sigmoid(float(gate_beta) * delta)
    if not active.any():
        return gates, delta, {
            "gate_mean": 0.0,
            "gate_active_fraction": 0.0,
            "teacher_gap_mean": 0.0,
            "teacher_gap_std": 0.0,
        }
    active_delta = delta[active]
    active_gates = gates[active]
    return gates, delta, {
        "gate_mean": float(active_gates.mean().item()),
        "gate_active_fraction": float((active_gates > 0.5).float().mean().item()),
        "teacher_gap_mean": float(active_delta.mean().item()),
        "teacher_gap_std": float(active_delta.std(unbiased=False).item()),
    }


def compute_sdar_gated_distillation(
    *,
    teacher_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    apply_mask: torch.Tensor,
    gate_beta: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the SDAR token loss with a frozen rollout-old confidence gate.

    The gate compares the Anchor-conditioned teacher with the policy that
    actually sampled the token.  Both teacher/old inputs and the resulting
    gate are detached; gradients flow only through ``student_log_probs``.
    """

    if not (
        teacher_log_probs.shape
        == rollout_old_log_probs.shape
        == student_log_probs.shape
        == apply_mask.shape
    ):
        raise ValueError("SDAR tensors must have identical response-space shapes")
    if gate_beta < 0:
        raise ValueError("SDAR gate beta must be non-negative")
    teacher = teacher_log_probs.float().detach()
    rollout_old = rollout_old_log_probs.float().detach()
    gap = teacher - rollout_old
    gate = torch.sigmoid(float(gate_beta) * gap).detach()
    token_loss = gate * (teacher - student_log_probs.float()) * apply_mask.float().detach()
    return token_loss, gate, gap


def compute_opsd_step_weights(
    *,
    teacher_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    step_ids: torch.Tensor,
    beta: float = 1.0,
    clip_lower: float = 0.0,
    clip_upper: float = 1.2,
    advantage_sign: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token weights and the per-step support/weight diagnostics.

    SSPO defines the support of one agent step as the mean token log-probability
    difference between the hint-conditioned teacher and the rollout policy that
    sampled the action. Environment/tool-result tokens carry ``step_id == -1``
    and always retain unit weight.
    """

    if not (
        teacher_log_probs.shape
        == rollout_old_log_probs.shape
        == loss_mask.shape
        == step_ids.shape
    ):
        raise ValueError(
            "OPSD tensors must have identical response-space shapes: "
            f"teacher={tuple(teacher_log_probs.shape)}, "
            f"old={tuple(rollout_old_log_probs.shape)}, "
            f"mask={tuple(loss_mask.shape)}, steps={tuple(step_ids.shape)}"
        )
    if beta < 0:
        raise ValueError("opsd beta must be non-negative")
    if clip_lower < 0 or clip_upper < clip_lower:
        raise ValueError(
            "invalid OPSD clip bounds: " f"lower={clip_lower}, upper={clip_upper}"
        )

    teacher = teacher_log_probs.float()
    valid_step_ids, step_supports, _ = compute_opsd_step_supports(
        teacher_log_probs=teacher_log_probs,
        rollout_old_log_probs=rollout_old_log_probs,
        loss_mask=loss_mask,
        step_ids=step_ids,
    )
    if valid_step_ids.numel() == 0:
        empty = teacher.new_empty((0,), dtype=torch.float32)
        return torch.ones_like(teacher, dtype=torch.float32), empty, empty

    signed_beta = teacher.new_tensor(
        float(advantage_sign) * float(beta), dtype=torch.float32
    )
    weights: list[torch.Tensor] = []
    for support in step_supports:
        weight = torch.exp(signed_beta * support).clamp(
            min=float(clip_lower), max=float(clip_upper)
        )
        weights.append(weight)
    step_weights = torch.stack(weights)
    token_weights = broadcast_opsd_step_weights(
        loss_mask=loss_mask,
        step_ids=step_ids,
        valid_step_ids=valid_step_ids,
        step_weights=step_weights,
    )
    return token_weights, step_supports, step_weights


def build_opsd_step_summary(
    *,
    step_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    supports: torch.Tensor,
    weights: torch.Tensor,
    advantage: float,
    conditioned_means: torch.Tensor | None = None,
    baseline_means: torch.Tensor | None = None,
    matched_placebo: bool = False,
) -> list[dict[str, Any]]:
    """Build compact per-step diagnostics without retaining token arrays."""

    if step_ids.shape != loss_mask.shape:
        raise ValueError(
            "OPSD summary tensors must have identical response-space shapes: "
            f"steps={tuple(step_ids.shape)}, mask={tuple(loss_mask.shape)}"
        )
    active = loss_mask.bool() & (step_ids >= 0)
    valid_step_ids = torch.unique(step_ids[active]).sort().values
    if not (valid_step_ids.numel() == supports.numel() == weights.numel()):
        raise ValueError(
            "OPSD summary vectors do not match active step count: "
            f"steps={valid_step_ids.numel()}, supports={supports.numel()}, "
            f"weights={weights.numel()}"
        )
    if conditioned_means is not None and conditioned_means.numel() != supports.numel():
        raise ValueError("conditioned means do not match active step count")
    if baseline_means is not None and baseline_means.numel() != supports.numel():
        raise ValueError("baseline means do not match active step count")

    result: list[dict[str, Any]] = []
    for offset, step_id in enumerate(valid_step_ids):
        token_count = int((active & (step_ids == step_id)).sum().item())
        support = float(supports[offset].item())
        weight = float(weights[offset].item())
        row = {
            "step_id": int(step_id.item()),
            "loss_token_count": token_count,
            "teacher_minus_rollout_old_mean": support,
            "weight": weight,
            "original_advantage": float(advantage),
            "weighted_advantage": float(advantage) * weight,
        }
        if conditioned_means is not None:
            key = (
                "real_anchor_logprob_mean"
                if matched_placebo
                else "teacher_logprob_mean"
            )
            row[key] = float(conditioned_means[offset].item())
        if baseline_means is not None:
            key = (
                "placebo_anchor_logprob_mean"
                if matched_placebo
                else "rollout_old_logprob_mean"
            )
            row[key] = float(baseline_means[offset].item())
        if matched_placebo:
            row["real_minus_placebo_mean"] = support
            row.pop("teacher_minus_rollout_old_mean")
        result.append(row)
    return result
