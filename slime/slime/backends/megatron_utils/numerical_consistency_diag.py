"""Opt-in, zero-update numerical consistency diagnostics for Megatron actors."""

import copy
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from megatron.core.tensor_parallel import get_cuda_rng_tracker

logger = logging.getLogger(__name__)
_GRAD_STATE = None
_LOGPROB_STATE = None

_LOGPROB_BIN_EDGES = (-math.inf, -10.0, -6.0, -3.0, -1.0, -0.3, math.inf)
_LOGPROB_BIN_LABELS = (
    "[-inf,-10)", "[-10,-6)", "[-6,-3)", "[-3,-1)", "[-1,-0.3)", "[-0.3,inf)"
)


def _comparison_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Return token-level log-probability and probability-ratio statistics."""
    reference = reference.detach().double().cpu().reshape(-1)
    candidate = candidate.detach().double().cpu().reshape(-1)
    if reference.shape != candidate.shape:
        raise ValueError(f"logprob diagnostic shape mismatch: {reference.shape} vs {candidate.shape}")
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    num_nonfinite = int(finite.numel() - finite.sum().item())
    reference = reference[finite]
    candidate = candidate[finite]
    if reference.numel() == 0:
        return {"num_tokens": 0, "num_nonfinite": num_nonfinite}

    delta = candidate - reference
    abs_delta = delta.abs()
    ratio_error = delta.clamp(min=-80.0, max=80.0).exp() - 1.0
    abs_ratio_error = ratio_error.abs()
    ratio = ratio_error + 1.0
    tis_weight = ratio.clamp(min=0.0, max=2.0)
    token_count = float(ratio.numel())

    def quantiles(values: torch.Tensor) -> dict[str, float]:
        out = torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
        return {"p50": float(out[0]), "p90": float(out[1]), "p95": float(out[2]), "p99": float(out[3])}

    result = {
        "num_tokens": int(reference.numel()),
        "num_nonfinite": num_nonfinite,
        "mean_signed_logprob_diff": float(delta.mean()),
        "mean_abs_logprob_diff": float(abs_delta.mean()),
        "rms_logprob_diff": float(delta.square().mean().sqrt()),
        "max_abs_logprob_diff": float(abs_delta.max()),
        "abs_logprob_diff_quantiles": quantiles(abs_delta),
        "mean_signed_ratio_minus_one": float(ratio_error.mean()),
        "mean_abs_ratio_minus_one": float(abs_ratio_error.mean()),
        "max_abs_ratio_minus_one": float(abs_ratio_error.max()),
        "abs_ratio_minus_one_quantiles": quantiles(abs_ratio_error),
        "fraction_abs_ratio_minus_one_gt_0_1pct": float((abs_ratio_error > 0.001).double().mean()),
        "mean_ratio": float(ratio.mean()),
        "std_ratio": float(ratio.std(unbiased=False)),
        "ratio_quantiles": quantiles(ratio),
        "fraction_ratio_lt_0_5": float((ratio < 0.5).double().mean()),
        "fraction_ratio_gt_2": float((ratio > 2.0).double().mean()),
        "fraction_ratio_outside_0_8_1_28": float(((ratio < 0.8) | (ratio > 1.28)).double().mean()),
        "raw_importance_weight_ess_fraction": float(ratio.sum().square() / (token_count * ratio.square().sum())),
        "tis_0_2_clipfrac": float((tis_weight != ratio).double().mean()),
        "tis_0_2_mean_weight": float(tis_weight.mean()),
        "tis_0_2_ess_fraction": float(tis_weight.sum().square() / (token_count * tis_weight.square().sum())),
        "fraction_abs_ratio_minus_one_gt_1pct": float((abs_ratio_error > 0.01).double().mean()),
        "fraction_abs_ratio_minus_one_gt_5pct": float((abs_ratio_error > 0.05).double().mean()),
        "fraction_abs_ratio_minus_one_gt_10pct": float((abs_ratio_error > 0.10).double().mean()),
        "reference_logprob_bins": [],
    }
    for lower, upper, label in zip(
        _LOGPROB_BIN_EDGES[:-1], _LOGPROB_BIN_EDGES[1:], _LOGPROB_BIN_LABELS, strict=True
    ):
        in_bin = (reference >= lower) & (reference < upper)
        if not in_bin.any():
            result["reference_logprob_bins"].append({"bin": label, "num_tokens": 0})
            continue
        bin_abs_delta = abs_delta[in_bin]
        bin_abs_ratio_error = abs_ratio_error[in_bin]
        result["reference_logprob_bins"].append({
            "bin": label,
            "num_tokens": int(in_bin.sum()),
            "fraction_of_tokens": float(in_bin.double().mean()),
            "mean_abs_logprob_diff": float(bin_abs_delta.mean()),
            "p95_abs_logprob_diff": float(torch.quantile(bin_abs_delta, 0.95)),
            "mean_abs_ratio_minus_one": float(bin_abs_ratio_error.mean()),
            "p95_abs_ratio_minus_one": float(torch.quantile(bin_abs_ratio_error, 0.95)),
            "fraction_abs_ratio_minus_one_gt_1pct": float((bin_abs_ratio_error > 0.01).double().mean()),
        })
    return result


def capture_logprob_diagnostics(*, old_log_probs, current_log_probs, rollout_log_probs, loss_masks) -> None:
    """Capture selected-token log-probs without touching tensors used by autograd."""
    global _LOGPROB_STATE
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() != 0:
        return
    config_path = os.environ.get("SLIME_NUMERICAL_DIAG_CONFIG")
    if not config_path or not Path(config_path).is_file():
        return
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not config.get("capture_logprob_diagnostics", False):
        return
    if _LOGPROB_STATE is None:
        _LOGPROB_STATE = {"phase": 0, "passes": [[], []]}
    phase = int(_LOGPROB_STATE["phase"])
    if phase not in (0, 1):
        raise RuntimeError(f"unexpected logprob diagnostic phase {phase}")

    old_cpu = old_log_probs.detach().float().cpu().reshape(-1)
    current_cpu = current_log_probs.detach().float().cpu().reshape(-1)
    mask_cpu = loss_masks.detach().bool().cpu().reshape(-1)
    if not (old_cpu.shape == current_cpu.shape == mask_cpu.shape):
        raise ValueError(
            f"logprob diagnostic shape mismatch: old={old_cpu.shape}, "
            f"current={current_cpu.shape}, mask={mask_cpu.shape}"
        )
    rollout_cpu = None
    if rollout_log_probs is not None:
        rollout_cpu = rollout_log_probs.detach().float().cpu().reshape(-1)
        if rollout_cpu.shape != old_cpu.shape:
            raise ValueError(f"rollout logprob shape mismatch: {rollout_cpu.shape} vs {old_cpu.shape}")
    _LOGPROB_STATE["passes"][phase].append(
        {"old": old_cpu, "current": current_cpu, "rollout": rollout_cpu, "mask": mask_cpu}
    )


def _mark_logprob_pass_complete() -> None:
    if _LOGPROB_STATE is not None:
        _LOGPROB_STATE["phase"] += 1


def _finalize_logprob_diagnostics() -> dict | None:
    global _LOGPROB_STATE
    if _LOGPROB_STATE is None:
        return None
    first, second = _LOGPROB_STATE["passes"]
    if len(first) != len(second):
        raise RuntimeError(f"logprob microbatch mismatch: first={len(first)}, second={len(second)}")

    def concat(key, records):
        values = [record[key] for record in records]
        return None if not values or values[0] is None else torch.cat(values)

    first_old = concat("old", first)
    first_current = concat("current", first)
    second_old = concat("old", second)
    second_current = concat("current", second)
    rollout = concat("rollout", first)
    mask = concat("mask", first)
    second_mask = concat("mask", second)
    if not torch.equal(mask, second_mask):
        raise RuntimeError("loss masks changed between diagnostic passes")
    if not torch.equal(first_old, second_old):
        raise RuntimeError("cached Megatron old log-probs changed between diagnostic passes")

    comparisons = {
        "megatron_current_pass1_vs_pass2": (first_current, second_current),
        "megatron_old_vs_current_pass1": (first_old, first_current),
        "megatron_old_vs_current_pass2": (first_old, second_current),
    }
    if rollout is not None:
        comparisons.update({
            "sglang_rollout_vs_megatron_old": (rollout, first_old),
            "sglang_rollout_vs_megatron_current_pass1": (rollout, first_current),
            "sglang_rollout_vs_megatron_current_pass2": (rollout, second_current),
        })
    result = {
        "num_microbatches_per_pass": len(first),
        "num_response_tokens": int(mask.numel()),
        "num_effective_loss_tokens": int(mask.sum()),
        "comparisons_all_response_tokens": {},
        "comparisons_effective_loss_tokens": {},
    }
    for name, (reference, candidate) in comparisons.items():
        result["comparisons_all_response_tokens"][name] = _comparison_stats(reference, candidate)
        result["comparisons_effective_loss_tokens"][name] = _comparison_stats(reference[mask], candidate[mask])
    _LOGPROB_STATE = None
    return result


def capture_rng_state():
    """Capture every RNG state used by the current Megatron worker."""
    tracker = get_cuda_rng_tracker()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": torch.cuda.get_rng_state().clone(),
        "megatron_cuda_tracker": copy.deepcopy(tracker.get_states()),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    get_cuda_rng_tracker().set_states(copy.deepcopy(state["megatron_cuda_tracker"]))


def _grad_groups(optimizer):
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    return [sub_optimizer.get_main_grads_for_grad_norm() for sub_optimizer in optimizers]


def _pair_stats(first: torch.Tensor, second: torch.Tensor, chunk_size: int = 4_194_304):
    assert first.shape == second.shape
    first_flat = first.reshape(-1)
    second_flat = second.detach().cpu().reshape(-1)
    dot = first_sq = second_sq = diff_sq = 0.0
    max_abs_diff = 0.0
    for start in range(0, first_flat.numel(), chunk_size):
        stop = min(start + chunk_size, first_flat.numel())
        a = first_flat[start:stop].double()
        b = second_flat[start:stop].double()
        delta = b - a
        dot += torch.dot(a, b).item()
        first_sq += torch.dot(a, a).item()
        second_sq += torch.dot(b, b).item()
        diff_sq += torch.dot(delta, delta).item()
        if delta.numel():
            max_abs_diff = max(max_abs_diff, delta.abs().max().item())
    return dot, first_sq, second_sq, diff_sq, first_flat.numel(), max_abs_diff


def capture_or_compare_gradients(optimizer, config):
    """Capture two pre-clip FP32 optimizer gradients and compare them globally."""
    global _GRAD_STATE

    grad_groups = _grad_groups(optimizer)
    grad_norm = optimizer.get_grad_norm()
    grad_norm_value = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

    if _GRAD_STATE is None:
        snapshots = []
        local_numel = 0
        for grads in grad_groups:
            group_snapshot = []
            for grad in grads:
                snapshot = grad.detach().cpu().clone()
                group_snapshot.append(snapshot)
                local_numel += snapshot.numel()
            snapshots.append(group_snapshot)
        _GRAD_STATE = {"snapshots": snapshots, "first_grad_norm": grad_norm_value}
        _mark_logprob_pass_complete()
        logger.info(
            "Captured first zero-update gradient: local_numel=%d, grad_norm=%.9g",
            local_numel,
            grad_norm_value,
        )
        return 1, grad_norm_value

    snapshots = _GRAD_STATE["snapshots"]
    reduce_group = optimizer.get_grad_stats_parallel_group()
    device = torch.device("cuda", torch.cuda.current_device())
    chain_results = []
    global_sums = [0.0, 0.0, 0.0, 0.0, 0.0]
    global_max_abs_diff = 0.0

    for chain_index, (first_group, second_group) in enumerate(zip(snapshots, grad_groups, strict=True)):
        local = [0.0, 0.0, 0.0, 0.0, 0.0]
        local_max = 0.0
        for first, second in zip(first_group, second_group, strict=True):
            stats = _pair_stats(first, second)
            for index in range(5):
                local[index] += stats[index]
            local_max = max(local_max, stats[5])
        reduced = torch.tensor(local, dtype=torch.float64, device=device)
        reduced_max = torch.tensor(local_max, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(reduced, group=reduce_group)
        torch.distributed.all_reduce(reduced_max, op=torch.distributed.ReduceOp.MAX, group=reduce_group)
        dot, first_sq, second_sq, diff_sq, numel = reduced.cpu().tolist()
        max_abs_diff = reduced_max.item()
        chain_results.append(
            {
                "chain_index": chain_index,
                "numel": int(numel),
                "first_grad_norm": math.sqrt(first_sq),
                "second_grad_norm": math.sqrt(second_sq),
                "gradient_cosine": dot / math.sqrt(first_sq * second_sq),
                "norm_ratio": math.sqrt(second_sq / first_sq),
                "relative_l2_error": math.sqrt(diff_sq / first_sq),
                "max_abs_difference": max_abs_diff,
            }
        )
        for index, value in enumerate((dot, first_sq, second_sq, diff_sq, numel)):
            global_sums[index] += value
        global_max_abs_diff = max(global_max_abs_diff, max_abs_diff)

    dot, first_sq, second_sq, diff_sq, numel = global_sums
    result = {
        "method": "same_checkpoint_same_batch_two_complete_backward_passes_no_optimizer_step",
        "label": config.get("label"),
        "restore_rng": bool(config.get("restore_rng", False)),
        "first_reported_grad_norm": _GRAD_STATE["first_grad_norm"],
        "second_reported_grad_norm": grad_norm_value,
        "numel": int(numel),
        "first_grad_norm": math.sqrt(first_sq),
        "second_grad_norm": math.sqrt(second_sq),
        "gradient_cosine": dot / math.sqrt(first_sq * second_sq),
        "norm_ratio": math.sqrt(second_sq / first_sq),
        "relative_l2_error": math.sqrt(diff_sq / first_sq),
        "max_abs_difference": global_max_abs_diff,
        "optimizer_chains": chain_results,
    }
    logprob_diagnostics = _finalize_logprob_diagnostics()
    if logprob_diagnostics is not None:
        result["logprob_diagnostics"] = logprob_diagnostics
    if torch.distributed.get_rank() == 0:
        output_path = Path(config["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
        with output_path.with_suffix(".json").open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, ensure_ascii=False, indent=2)
        logger.info("Saved numerical consistency diagnostic to %s", output_path)
    _GRAD_STATE = None
    return 2, grad_norm_value
