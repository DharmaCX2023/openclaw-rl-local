#!/usr/bin/env python3
"""Compare entropy payloads from two checkpoints on exactly the same tokens."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


def quantiles(x: torch.Tensor) -> dict[str, float]:
    qs = torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=torch.float32)
    values = torch.quantile(x.float(), qs)
    return {f"p{int(q * 100):02d}": float(v) for q, v in zip(qs, values, strict=True)}


def load_files(path: str) -> list[dict]:
    files = sorted(glob.glob(str(Path(path) / "*.pt")))
    if not files:
        raise FileNotFoundError(f"No payloads found under {path}")
    return [torch.load(file, map_location="cpu", weights_only=False) for file in files]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_files(args.baseline)
    candidate = load_files(args.candidate)
    assert len(baseline) == len(candidate)
    exact = {
        "payload_count": len(baseline),
        "metadata_equal": True,
        "unconcat_tokens_equal": True,
        "loss_masks_equal": True,
        "response_tokens_equal": True,
    }
    temperatures = [str(value) for value in baseline[0]["views"]]
    collected = {
        temperature: {
            "baseline_entropy": [],
            "candidate_entropy": [],
            "baseline_top1": [],
            "candidate_top1": [],
            "baseline_argmax": [],
            "candidate_argmax": [],
        }
        for temperature in temperatures
    }

    total_response_tokens = 0
    total_effective_tokens = 0
    for left, right in zip(baseline, candidate, strict=True):
        exact["metadata_equal"] &= (
            left["total_lengths"] == right["total_lengths"]
            and left["response_lengths"] == right["response_lengths"]
            and tuple(left["views"]) == tuple(right["views"])
        )
        assert len(left["loss_masks"]) == len(right["loss_masks"])
        for left_tokens, right_tokens in zip(left["unconcat_tokens"], right["unconcat_tokens"], strict=True):
            exact["unconcat_tokens_equal"] &= torch.equal(left_tokens, right_tokens)
        for left_mask, right_mask in zip(left["loss_masks"], right["loss_masks"], strict=True):
            exact["loss_masks_equal"] &= torch.equal(left_mask, right_mask)

        for temperature in temperatures:
            assert len(left["views"][temperature]) == len(right["views"][temperature])
            for sample_index, (left_sample, right_sample) in enumerate(
                zip(left["views"][temperature], right["views"][temperature], strict=True)
            ):
                exact["response_tokens_equal"] &= torch.equal(
                    left_sample["response_tokens"], right_sample["response_tokens"]
                )
                mask = left["loss_masks"][sample_index].bool()
                if temperature == temperatures[0]:
                    total_response_tokens += mask.numel()
                    total_effective_tokens += int(mask.sum())
                collected[temperature]["baseline_entropy"].append(left_sample["entropy"][mask].float())
                collected[temperature]["candidate_entropy"].append(right_sample["entropy"][mask].float())
                collected[temperature]["baseline_top1"].append(
                    left_sample["topk_log_probs"][mask, 0].float().exp()
                )
                collected[temperature]["candidate_top1"].append(
                    right_sample["topk_log_probs"][mask, 0].float().exp()
                )
                collected[temperature]["baseline_argmax"].append(left_sample["topk_indices"][mask, 0])
                collected[temperature]["candidate_argmax"].append(right_sample["topk_indices"][mask, 0])

    if not all(exact.values()):
        raise AssertionError(f"Payload pairing failed: {exact}")

    result = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "pairing": exact,
        "total_response_tokens": total_response_tokens,
        "effective_tokens": total_effective_tokens,
        "temperatures": {},
    }
    for temperature, values in collected.items():
        tensors = {key: torch.cat(parts) for key, parts in values.items()}
        delta_entropy = tensors["candidate_entropy"] - tensors["baseline_entropy"]
        abs_delta_entropy = delta_entropy.abs()
        delta_top1 = tensors["candidate_top1"] - tensors["baseline_top1"]
        result["temperatures"][temperature] = {
            "baseline_entropy_mean": float(tensors["baseline_entropy"].mean()),
            "candidate_entropy_mean": float(tensors["candidate_entropy"].mean()),
            "entropy_mean_delta": float(delta_entropy.mean()),
            "entropy_relative_delta": float(delta_entropy.mean() / tensors["baseline_entropy"].mean()),
            "entropy_delta_quantiles": quantiles(delta_entropy),
            "entropy_abs_delta_mean": float(abs_delta_entropy.mean()),
            "entropy_abs_delta_quantiles": quantiles(abs_delta_entropy),
            "entropy_delta_lt_minus_0.01_fraction": float((delta_entropy < -0.01).float().mean()),
            "entropy_delta_gt_0.01_fraction": float((delta_entropy > 0.01).float().mean()),
            "entropy_delta_abs_le_0.01_fraction": float((abs_delta_entropy <= 0.01).float().mean()),
            "baseline_top1_mean": float(tensors["baseline_top1"].mean()),
            "candidate_top1_mean": float(tensors["candidate_top1"].mean()),
            "top1_mean_delta": float(delta_top1.mean()),
            "top1_argmax_same_fraction": float(
                (tensors["baseline_argmax"] == tensors["candidate_argmax"]).float().mean()
            ),
        }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
