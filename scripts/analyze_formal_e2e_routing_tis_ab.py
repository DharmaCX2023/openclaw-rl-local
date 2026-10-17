#!/usr/bin/env python3
"""Summarize the formal step-99 routing-replay/TIS A/B experiment."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch


CASES = ("baseline_no_replay_no_tis", "replay_plus_tis")


def token_metrics(path: Path) -> dict[str, object]:
    data = torch.load(path, map_location="cpu", weights_only=False)["rollout_data"]
    fields = {}
    for key in ("rollout_log_probs", "log_probs", "loss_masks", "advantages"):
        fields[key] = torch.cat([x.float().reshape(-1) for x in data[key]])
    mask = fields["loss_masks"] > 0
    rollout = fields["rollout_log_probs"][mask]
    old = fields["log_probs"][mask]
    advantage = fields["advantages"][mask]
    delta = old - rollout
    abs_delta = delta.abs()
    ratio = torch.exp(delta.clamp(-20, 20))
    abs_ratio_error = (ratio - 1).abs()
    tis = ratio.clamp(0, 2)
    quantiles = torch.quantile(abs_delta, torch.tensor([0.5, 0.9, 0.95, 0.99]))
    adv_denom = advantage.abs().sum()
    return {
        "effective_response_tokens": int(mask.sum()),
        "logprob_abs_diff_mean": abs_delta.mean().item(),
        "logprob_diff_rms": delta.square().mean().sqrt().item(),
        "logprob_abs_diff_p50": quantiles[0].item(),
        "logprob_abs_diff_p90": quantiles[1].item(),
        "logprob_abs_diff_p95": quantiles[2].item(),
        "logprob_abs_diff_p99": quantiles[3].item(),
        "logprob_abs_diff_max": abs_delta.max().item(),
        "ratio_mean": ratio.mean().item(),
        "abs_ratio_minus_one_mean": abs_ratio_error.mean().item(),
        "abs_ratio_minus_one_gt_1pct": (abs_ratio_error > 0.01).float().mean().item(),
        "abs_ratio_minus_one_gt_5pct": (abs_ratio_error > 0.05).float().mean().item(),
        "abs_ratio_minus_one_gt_10pct": (abs_ratio_error > 0.10).float().mean().item(),
        "ratio_outside_ppo_window": ((ratio < 0.8) | (ratio > 1.28)).float().mean().item(),
        "adv_abs_weighted_abs_ratio_error": (
            (abs_ratio_error * advantage.abs()).sum() / adv_denom
        ).item(),
        "tis_weight_mean": tis.mean().item(),
        "tis_clip_fraction": ((ratio < 0) | (ratio > 2)).float().mean().item(),
        "tis_ess_fraction": (tis.sum().square() / (len(tis) * tis.square().sum())).item(),
        "low_var_kl_unweighted": (torch.exp(-delta) - 1 + delta).mean().item(),
    }


def proxy_metrics(path: Path) -> tuple[dict[str, object], dict[tuple[str, str], str]]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    responses = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "session_id" not in record or "step" not in record:
                continue
            sid = record["session_id"]
            step = str(record["step"])
            sessions[sid].append(record)
            responses[(sid, step)] = record.get("response_text", "")
    steps = [len(records) for records in sessions.values()]
    return {
        "sessions": len(sessions),
        "requests": sum(steps),
        "mean_requests_per_session": sum(steps) / len(steps),
        "max_requests_per_session": max(steps),
        "finish_reason_counts": dict(Counter(r[-1].get("finish_reason") for r in sessions.values())),
    }, responses


def log_metrics(path: Path) -> dict[str, object]:
    text = path.read_text(errors="replace")
    timers = {
        name: float(value)
        for name, value in re.findall(r"Timer (\w+) end \(elapsed: ([0-9.]+)s\)", text)
    }
    perf_matches = re.findall(r"rollout\.py:1240 - perf 100: (\{.*\})", text)
    # The dict contains only literal scalar values; JSON conversion handles single quotes.
    perf = {}
    if perf_matches:
        import ast

        perf = ast.literal_eval(perf_matches[-1])
    train_matches = re.findall(r"model\.py:728 - step 100: (\{.*\})", text)
    train = {}
    if train_matches:
        import ast

        train = ast.literal_eval(train_matches[-1])
    packing = re.findall(
        r"Dynamic batching: num_microbatches=\[(\d+)\].*?sum_tokens=(\d+).*?packed_microbatches=(\d+)",
        text,
    )
    return {
        "timers_seconds": timers,
        "packing": (
            {
                "microbatches": int(packing[-1][0]),
                "context_tokens": int(packing[-1][1]),
                "packed_microbatches": int(packing[-1][2]),
            }
            if packing
            else {}
        ),
        "rollout": perf,
        "train": train,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = {"experiment_dir": str(args.experiment_dir.resolve()), "cases": {}}
    response_maps = {}
    for case in CASES:
        case_dir = args.experiment_dir / case
        log_path = next((case_dir / "runtime" / "logs").glob("train_*.log"))
        proxy, response_maps[case] = proxy_metrics(case_dir / "proxy_records.jsonl")
        summary["cases"][case] = {
            "token_metrics": token_metrics(case_dir / "train_data" / "100_0.pt"),
            "proxy": proxy,
            "log": log_metrics(log_path),
            "artifacts_bytes": {
                "rollout_data": (case_dir / "rollout_data" / "100.pt").stat().st_size,
                "train_data_rank0": (case_dir / "train_data" / "100_0.pt").stat().st_size,
                "proxy_records": (case_dir / "proxy_records.jsonl").stat().st_size,
            },
        }
    a = response_maps[CASES[0]]
    b = response_maps[CASES[1]]
    common = a.keys() & b.keys()
    equal = sum(a[key] == b[key] for key in common)
    first_divergence = {}
    for sid in {key[0] for key in common}:
        for step, key in sorted((int(k[1]), k) for k in common if k[0] == sid):
            if a[key] != b[key]:
                first_divergence[sid] = step
                break
    summary["trajectory_comparison"] = {
        "common_session_steps": len(common),
        "exact_equal_session_steps": equal,
        "exact_equal_fraction": equal / len(common),
        "sessions_with_divergence": len(first_divergence),
        "first_divergence_step_histogram": dict(Counter(first_divergence.values())),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
