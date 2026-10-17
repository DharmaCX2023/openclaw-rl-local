#!/usr/bin/env python3
"""Analyze token log-probability jitter as a function of reference probability."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


ROOT = Path("/root/openclaw-rl-local")
BASELINE_PATH = ROOT / "outputs/zero-update-consistency-step99-20260814/baseline/zero_update_baseline.pt"
REPLAY_PATH = ROOT / "outputs/zero-update-consistency-step99-20260814/replay/zero_update_replay.pt"
OUTPUT_PATH = ROOT / "reports/logprob_jitter_probability_buckets_step99_20260820.json"


PROBABILITY_BINS = [
    ("p>=0.99", 0.99, 1.0 + 1e-12),
    ("0.9<=p<0.99", 0.9, 0.99),
    ("0.5<=p<0.9", 0.5, 0.9),
    ("0.1<=p<0.5", 0.1, 0.5),
    ("0.01<=p<0.1", 0.01, 0.1),
    ("0.001<=p<0.01", 0.001, 0.01),
    ("0.0001<=p<0.001", 0.0001, 0.001),
    ("p<0.0001", 0.0, 0.0001),
]


def load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def masked_concat(payload: dict, key: str) -> torch.Tensor:
    values = []
    for tensor, mask in zip(payload[key], payload["loss_masks"], strict=True):
        values.append(tensor.detach().to(torch.float64)[mask.to(torch.bool)])
    return torch.cat(values)


def quantile(values: torch.Tensor, q: float) -> float:
    return float(torch.quantile(values, q).item()) if values.numel() else float("nan")


def summarize(ref: torch.Tensor, deltas: torch.Tensor, token_mask: torch.Tensor) -> dict:
    # deltas is [repeat, token]; token_mask is [token].
    selected_ref = ref[token_mask]
    selected = deltas[:, token_mask].reshape(-1)
    abs_delta = selected.abs()
    ratio_error = torch.expm1(selected).abs()
    return {
        "unique_tokens": int(token_mask.sum().item()),
        "observations": int(selected.numel()),
        "reference_logprob_mean": float(selected_ref.mean().item()),
        "reference_probability_mean": float(selected_ref.exp().mean().item()),
        "signed_logprob_delta_mean": float(selected.mean().item()),
        "abs_logprob_delta_mean": float(abs_delta.mean().item()),
        "abs_logprob_delta_p50": quantile(abs_delta, 0.50),
        "abs_logprob_delta_p90": quantile(abs_delta, 0.90),
        "abs_logprob_delta_p95": quantile(abs_delta, 0.95),
        "abs_logprob_delta_p99": quantile(abs_delta, 0.99),
        "rms_logprob_delta": float(selected.square().mean().sqrt().item()),
        "abs_ratio_error_mean": float(ratio_error.mean().item()),
        "abs_ratio_error_gt_1pct": float((ratio_error > 0.01).double().mean().item()),
        "abs_ratio_error_gt_5pct": float((ratio_error > 0.05).double().mean().item()),
        "abs_ratio_error_gt_10pct": float((ratio_error > 0.10).double().mean().item()),
        "abs_ratio_error_gt_20pct": float((ratio_error > 0.20).double().mean().item()),
        "absolute_error_mass": float(abs_delta.sum().item()),
        "squared_error_mass": float(selected.square().sum().item()),
    }


def fixed_bins(ref: torch.Tensor, deltas: torch.Tensor) -> list[dict]:
    rows = []
    total_abs = deltas.abs().sum()
    total_sq = deltas.square().sum()
    for label, low, high in PROBABILITY_BINS:
        prob = ref.exp()
        mask = (prob >= low) & (prob < high)
        if not mask.any():
            continue
        row = {"bucket": label, "token_fraction": float(mask.double().mean().item())}
        row.update(summarize(ref, deltas, mask))
        row["absolute_error_share"] = row.pop("absolute_error_mass") / float(total_abs.item())
        row["squared_error_share"] = row.pop("squared_error_mass") / float(total_sq.item())
        rows.append(row)
    return rows


def decile_bins(ref: torch.Tensor, deltas: torch.Tensor) -> list[dict]:
    # Sort by reference probability from highest probability (D1) to lowest (D10).
    order = torch.argsort(ref, descending=True)
    chunks = torch.tensor_split(order, 10)
    total_abs = deltas.abs().sum()
    total_sq = deltas.square().sum()
    rows = []
    for index, indices in enumerate(chunks, start=1):
        mask = torch.zeros(ref.numel(), dtype=torch.bool)
        mask[indices] = True
        row = {
            "bucket": f"D{index}_{'highest_probability' if index == 1 else 'lowest_probability' if index == 10 else ''}".rstrip("_"),
            "token_fraction": float(mask.double().mean().item()),
            "reference_probability_min": float(ref[indices].exp().min().item()),
            "reference_probability_max": float(ref[indices].exp().max().item()),
        }
        row.update(summarize(ref, deltas, mask))
        row["absolute_error_share"] = row.pop("absolute_error_mass") / float(total_abs.item())
        row["squared_error_share"] = row.pop("squared_error_mass") / float(total_sq.item())
        rows.append(row)
    return rows


def correlation_summary(ref: torch.Tensor, deltas: torch.Tensor) -> dict:
    per_token_abs = deltas.abs().mean(dim=0)
    nll = -ref
    x = nll - nll.mean()
    y = per_token_abs - per_token_abs.mean()
    pearson = (x * y).mean() / (x.square().mean().sqrt() * y.square().mean().sqrt())
    return {
        "pearson_nll_vs_mean_abs_logprob_delta": float(pearson.item()),
        "nll_mean": float(nll.mean().item()),
        "per_token_abs_logprob_delta_mean": float(per_token_abs.mean().item()),
    }


def build_comparison(name: str, ref: torch.Tensor, targets: list[torch.Tensor]) -> dict:
    deltas = torch.stack([target - ref for target in targets])
    all_mask = torch.ones(ref.numel(), dtype=torch.bool)
    overall = summarize(ref, deltas, all_mask)
    overall.pop("absolute_error_mass")
    overall.pop("squared_error_mass")
    return {
        "name": name,
        "reference": "rollout_logprob" if name == "sglang_vs_megatron_old" else "first_megatron_old_logprob",
        "repeat_count": len(targets),
        "overall": overall,
        "correlation": correlation_summary(ref, deltas),
        "fixed_probability_bins": fixed_bins(ref, deltas),
        "equal_count_probability_deciles": decile_bins(ref, deltas),
    }


def build_symmetric_comparison(name: str, passes: list[torch.Tensor]) -> dict:
    """Bin by mean logprob and compare every pair, avoiding first-pass anchoring."""
    stacked = torch.stack(passes)
    center = stacked.mean(dim=0)
    pairwise_deltas = []
    for left in range(len(passes)):
        for right in range(left + 1, len(passes)):
            pairwise_deltas.append(passes[right] - passes[left])
    deltas = torch.stack(pairwise_deltas)
    all_mask = torch.ones(center.numel(), dtype=torch.bool)
    overall = summarize(center, deltas, all_mask)
    overall.pop("absolute_error_mass")
    overall.pop("squared_error_mass")
    return {
        "name": name,
        "reference": "mean_logprob_across_all_compared_passes",
        "pass_count": len(passes),
        "pair_count": len(pairwise_deltas),
        "overall": overall,
        "correlation": correlation_summary(center, deltas),
        "fixed_probability_bins": fixed_bins(center, deltas),
        "equal_count_probability_deciles": decile_bins(center, deltas),
    }


def main() -> None:
    baseline = load(BASELINE_PATH)
    replay = load(REPLAY_PATH)

    for key in ("tokens", "loss_masks", "advantages"):
        for left, right in zip(baseline[key], replay[key], strict=True):
            assert torch.equal(left, right), f"A/B mismatch in {key}"

    rollout = masked_concat(baseline, "rollout_log_probs")
    old_baseline = masked_concat(baseline, "old_log_probs")
    old_replay = masked_concat(replay, "old_log_probs")
    baseline_repeats = [masked_concat(baseline, f"zero_repeat_{i}_log_probs") for i in range(5)]
    replay_repeats = [masked_concat(replay, f"zero_repeat_{i}_log_probs") for i in range(5)]

    assert rollout.numel() == 84432
    result = {
        "checkpoint": "RL_step99",
        "effective_tokens": int(rollout.numel()),
        "fixed_bin_definition": [
            {"label": label, "probability_low_inclusive": low, "probability_high_exclusive": high}
            for label, low, high in PROBABILITY_BINS
        ],
        "comparisons": [
            build_comparison("sglang_vs_megatron_old", rollout, [old_baseline]),
            build_comparison("megatron_repeat_without_routing_replay", old_baseline, baseline_repeats),
            build_comparison("megatron_repeat_with_routing_replay", old_replay, replay_repeats),
        ],
        "symmetric_all_pairs_comparisons": [
            build_symmetric_comparison("sglang_vs_megatron_old_symmetric", [rollout, old_baseline]),
            build_symmetric_comparison(
                "megatron_all_pairs_without_routing_replay",
                [old_baseline, *baseline_repeats],
            ),
            build_symmetric_comparison(
                "megatron_all_pairs_with_routing_replay",
                [old_replay, *replay_repeats],
            ),
        ],
        "notes": [
            "All statistics use loss_mask=1 policy-generated tokens only.",
            "SGLang-vs-Megatron buckets use rollout probability as the reference.",
            "Repeat-forward buckets use the first Megatron old-logprob pass as the reference.",
            "Repeat-forward rows contain five observations per unique token.",
            "Probability is exp(reference_logprob) under the saved rollout/training distribution.",
        ],
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
