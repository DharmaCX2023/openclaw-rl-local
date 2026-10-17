"""Auxiliary action losses; never manufacture a GRPO rollout or importance weight.

The support mask mirrors installed SGLang's PyTorch joint top-k/top-p path.
Cross-backend logits still need a separate real-model agreement measurement.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def sampling_support(logits, temperature=0.7, top_p=0.8, top_k=20):
    if temperature <= 0 or not 0 < top_p <= 1 or top_k == 0:
        raise ValueError("invalid sampling configuration")
    u = logits.detach().float() / temperature
    if torch.isnan(u).any() or torch.isposinf(u).any():
        raise ValueError("non-finite sampling logits")
    probs = torch.softmax(u, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True, dim=-1)
    # SGLang computes cumulative ORIGINAL probabilities, before the top-k mask.
    cumulative = sorted_probs.cumsum(dim=-1)
    ranks = torch.arange(u.shape[-1], device=u.device)
    keep = (cumulative - sorted_probs <= top_p) & (sorted_probs > 0)
    if top_k > 0:
        keep &= ranks < top_k
    support = torch.zeros_like(keep).scatter(-1, indices, keep)
    boundary = u.masked_fill(~support, float("inf")).amin(dim=-1)
    if not torch.isfinite(boundary).all():
        raise ValueError("empty sampler support")
    return support, boundary


def all_blocked_margin(logits, targets, *, margin=0.1, temperature=0.7,
                       top_p=0.8, top_k=20, native_candidate_ids=None):
    """One complete action [response_positions,vocab]; all positions are valid.

    Zero-blocked actions produce differentiable zero and remain in pair means.
    Hard-masked targets are invalid repairs, not margin training opportunities.
    """
    if logits.ndim != 2 or targets.shape != logits.shape[:1] or not targets.numel():
        raise ValueError("expected non-empty action logits and aligned targets")
    if native_candidate_ids is None:
        support, boundary = sampling_support(logits, temperature, top_p, top_k)
        source = 'training_reconstruction_only'
    else:
        # C is measured from the actual decoder at the SAME weights/prefix.
        # No stale-cache reuse across optimizer updates is allowed by callers.
        # Evaluate b=min_{v in C} u_train(v), detached, in the same numeric
        # logit coordinate as the differentiable target. Cross-backend scores
        # remain diagnostics; this does NOT claim their logits are identical.
        if len(native_candidate_ids) != len(targets):
            raise ValueError('native candidate rows do not match action')
        support = torch.zeros_like(logits, dtype=torch.bool)
        for i, members in enumerate(native_candidate_ids):
            if not members or len(set(members)) != len(members):
                raise ValueError('empty or duplicate native candidates')
            if min(members) < 0 or max(members) >= logits.shape[-1]:
                raise ValueError('native candidate outside vocabulary')
            support[i, members] = True
        boundary = (logits.detach().float() / temperature).masked_fill(~support, float('inf')).amin(-1)
        if not torch.isfinite(boundary).all():
            raise ValueError('native boundary is nonfinite in training backend')
        source = 'native_sglang_candidate_ids'
    selected = logits.float().gather(-1, targets[:, None]).squeeze(-1) / temperature
    if not torch.isfinite(selected).all():
        raise ValueError("target forbidden by hard constraint")
    blocked = ~support.gather(-1, targets[:, None]).squeeze(-1)
    hinge = F.relu(margin + boundary - selected)
    count = blocked.sum().detach()
    loss = (hinge * blocked).sum() / count.clamp_min(1)
    stats = {"blocked_mask": blocked.detach(), "boundary": boundary.detach(),
             "target_sampling_logit": selected.detach(), "hinge": hinge.detach(),
             "blocked_count": count, "response_tokens": targets.numel(),
             "mask_source": source}
    return loss, stats


def action_dpo(pos_logp, neg_logp, ref_pos_logp, ref_neg_logp, beta=0.1):
    """Inputs are full-action SUM logprobs, not length-normalized scores."""
    difference = ((pos_logp - ref_pos_logp.detach())
                  - (neg_logp - ref_neg_logp.detach()))
    return -F.logsigmoid(beta * difference)


def task_pair_mean(losses, task_ids):
    if len(losses) != len(task_ids) or not losses:
        raise ValueError("aligned nonempty pair losses required")
    return torch.stack([
        torch.stack([loss for loss, tid in zip(losses, task_ids) if tid == task]).mean()
        for task in sorted(set(task_ids))
    ]).mean()
