#!/usr/bin/env python3
"""Aggregate exact token entropy and top-k decomposition payloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer


def q(values: torch.Tensor, ps: list[float]) -> dict[str, float]:
    return {
        f"p{int(p * 100):02d}": float(torch.quantile(values, p).item())
        for p in ps
    }


def visible(tokenizer, token_id: int) -> str:
    return repr(tokenizer.decode([token_id], clean_up_tokenization_spaces=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_dir", type=Path)
    parser.add_argument("tokenizer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.payload_dir.glob("*.pt"))
    if not paths:
        raise SystemExit(f"No payloads in {args.payload_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    temperatures = [f"{x:g}" for x in payloads[0]["temperatures"]]
    report: dict = {
        "payload_files": len(paths),
        "microbatch_samples": 0,
        "effective_tokens": 0,
        "temperatures": {},
        "examples": {},
    }

    records: dict[str, list[dict]] = {temperature: [] for temperature in temperatures}
    for payload in payloads:
        for sample_idx, mask in enumerate(payload["loss_masks"]):
            mask = mask.bool()
            report["microbatch_samples"] += 1
            report["effective_tokens"] += int(mask.sum())
            full_tokens = payload["unconcat_tokens"][sample_idx].long()
            response_len = int(payload["response_lengths"][sample_idx])
            prompt_len = len(full_tokens) - response_len
            for temperature in temperatures:
                view = payload["views"][temperature][sample_idx]
                valid = mask[: len(view["entropy"])]
                for response_pos in torch.where(valid)[0].tolist():
                    lp = view["topk_log_probs"][response_pos]
                    probs = lp.exp()
                    entropy = float(view["entropy"][response_pos])
                    topk_contribution = float((-(probs * lp)).sum())
                    records[temperature].append(
                        {
                            "entropy": entropy,
                            "top1_prob": float(probs[0]),
                            "topk_mass": float(probs.sum()),
                            "topk_contribution": topk_contribution,
                            "tail_contribution": max(0.0, entropy - topk_contribution),
                            "actual_token": int(view["response_tokens"][response_pos]),
                            "top_ids": view["topk_indices"][response_pos].long(),
                            "top_lp": lp,
                            "full_tokens": full_tokens,
                            "prompt_len": prompt_len,
                            "response_pos": response_pos,
                        }
                    )

    for temperature, items in records.items():
        entropy = torch.tensor([item["entropy"] for item in items])
        top1 = torch.tensor([item["top1_prob"] for item in items])
        topk_mass = torch.tensor([item["topk_mass"] for item in items])
        topk_contribution = torch.tensor([item["topk_contribution"] for item in items])
        tail_contribution = torch.tensor([item["tail_contribution"] for item in items])
        bins = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, math.inf]
        report["temperatures"][temperature] = {
            "tokens": len(items),
            "entropy_mean": float(entropy.mean()),
            "entropy_quantiles": q(entropy, [0.5, 0.75, 0.9, 0.95, 0.99]),
            "entropy_max": float(entropy.max()),
            "entropy_bins": {
                f"[{bins[i]:g},{bins[i + 1]:g})": float(((entropy >= bins[i]) & (entropy < bins[i + 1])).float().mean())
                for i in range(len(bins) - 1)
            },
            "top1_prob_mean": float(top1.mean()),
            "top1_prob_quantiles": q(top1, [0.01, 0.05, 0.1, 0.5, 0.9]),
            "top50_mass_mean": float(topk_mass.mean()),
            "top50_mass_min": float(topk_mass.min()),
            "top50_entropy_share": float(topk_contribution.sum() / entropy.sum()),
            "tail_entropy_share": float(tail_contribution.sum() / entropy.sum()),
        }

        ordered = sorted(range(len(items)), key=lambda i: items[i]["entropy"])
        target_quantiles = [0.02, 0.5, 0.9, 0.99]
        chosen = [ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))] for p in target_quantiles]
        chosen += ordered[-max(0, args.examples - len(chosen)) :]
        examples = []
        for idx in chosen[: args.examples]:
            item = items[idx]
            candidates = []
            for token_id, log_prob in zip(item["top_ids"][:10].tolist(), item["top_lp"][:10].tolist()):
                probability = math.exp(log_prob)
                candidates.append(
                    {
                        "token_id": token_id,
                        "token": visible(tokenizer, token_id),
                        "probability": probability,
                        "entropy_contribution": -probability * log_prob,
                    }
                )
            examples.append(
                {
                    "entropy": item["entropy"],
                    "top1_prob": item["top1_prob"],
                    "actual_token_id": item["actual_token"],
                    "actual_token": visible(tokenizer, item["actual_token"]),
                    "context_suffix": tokenizer.decode(
                        item["full_tokens"][
                            max(0, item["prompt_len"] + item["response_pos"] - 48) :
                            item["prompt_len"] + item["response_pos"]
                        ],
                        clean_up_tokenization_spaces=False,
                    ),
                    "top10": candidates,
                }
            )
        report["examples"][temperature] = examples

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["temperatures"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
