#!/usr/bin/env python3
"""Evaluate a conservative abstaining rule on saved K=3 placebo scores."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path


GOOD = {"successful_read", "modification", "validation", "recovery_after_error"}
BAD = {"tool_error", "degenerate_repeat", "response_truncated"}


def conservative_credit(values: list[float], noise: float, lam: float) -> float:
    if not values or not (all(x > 0 for x in values) or all(x < 0 for x in values)):
        return 0.0
    center = statistics.median(values)
    mad = statistics.median(abs(x - center) for x in values)
    magnitude = max(abs(center) - lam * mad - noise, 0.0)
    return math.copysign(magnitude, center) if magnitude else 0.0


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = q * (len(ordered) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def cluster_bootstrap(rows: list[dict], repeats: int, seed: int) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["session_id"]].append(row)
    sessions = sorted(grouped)

    def gap(chosen: list[str]) -> float | None:
        sample = [r for sid in chosen for r in grouped[sid]]
        good = [r["credit"] for r in sample if r["action_type"] in GOOD]
        bad = [r["credit"] for r in sample if r["action_type"] in BAD]
        return statistics.fmean(good) - statistics.fmean(bad) if good and bad else None

    estimate = gap(sessions)
    rng = random.Random(seed)
    draws = []
    for _ in range(repeats):
        value = gap([rng.choice(sessions) for _ in sessions])
        if value is not None:
            draws.append(value)
    return {
        "clusters": len(sessions),
        "estimate": estimate,
        "ci_low": quantile(draws, 0.025),
        "ci_high": quantile(draws, 0.975),
    }


def evaluate(rows: list[dict], noise: float, lam: float, beta: float, repeats: int, seed: int) -> dict:
    scored = []
    for row in rows:
        credit = conservative_credit([float(x) for x in row["deltas"]], noise, lam)
        weight = min(max(math.exp(-beta * credit), 0.8), 1.2)
        scored.append({**row, "credit": credit, "weight": weight})
    decisive = [r for r in scored if r["credit"] != 0]
    labelled = [r for r in decisive if r["action_type"] in GOOD | BAD]
    direction_correct = [
        r for r in labelled
        if (r["action_type"] in GOOD and r["credit"] > 0)
        or (r["action_type"] in BAD and r["credit"] < 0)
    ]
    categories = {}
    for name in sorted({r["action_type"] for r in scored}):
        part = [r for r in scored if r["action_type"] == name]
        active = [r for r in part if r["credit"] != 0]
        categories[name] = {
            "turns": len(part),
            "decisive": len(active),
            "coverage": len(active) / len(part),
            "mean_credit": statistics.fmean(r["credit"] for r in part),
            "protected": sum(r["credit"] > 0 for r in part),
            "punished_more": sum(r["credit"] < 0 for r in part),
        }
    trajectory_mass = defaultdict(list)
    for row in scored:
        trajectory_mass[row["session_id"]].append(row["weight"])
    means = [statistics.fmean(x) for x in trajectory_mass.values()]
    return {
        "noise_floor": noise,
        "lambda_mad": lam,
        "turns": len(scored),
        "decisive_turns": len(decisive),
        "coverage": len(decisive) / len(scored),
        "positive_credit": sum(r["credit"] > 0 for r in scored),
        "negative_credit": sum(r["credit"] < 0 for r in scored),
        "labelled_decisive_turns": len(labelled),
        "labelled_direction_accuracy": len(direction_correct) / len(labelled) if labelled else None,
        "mean_abs_credit_decisive": statistics.fmean(abs(r["credit"]) for r in decisive) if decisive else 0.0,
        "mean_abs_weight_delta": statistics.fmean(abs(r["weight"] - 1) for r in scored),
        "trajectory_weight_mean_min": min(means),
        "trajectory_weight_mean_mean": statistics.fmean(means),
        "trajectory_weight_mean_max": max(means),
        "good_minus_bad_credit": cluster_bootstrap(scored, repeats, seed),
        "categories": categories,
    }


def subset_stability(rows: list[dict], noise: float, lam: float) -> dict:
    output = {}
    for k in (1, 2, 3):
        variants = []
        for indices in combinations(range(3), k):
            signs, active = [], 0
            for row in rows:
                c = conservative_credit([float(row["deltas"][i]) for i in indices], noise, lam)
                signs.append(1 if c > 0 else -1 if c < 0 else 0)
                active += c != 0
            variants.append({"indices": list(indices), "coverage": active / len(rows), "signs": signs})
        pair_agreement = []
        for left, right in combinations(variants, 2):
            comparable = [(a, b) for a, b in zip(left["signs"], right["signs"]) if a and b]
            pair_agreement.append(
                sum(a == b for a, b in comparable) / len(comparable) if comparable else None
            )
        output[f"k{k}"] = {
            "coverage_by_subset": [v["coverage"] for v in variants],
            "mean_coverage": statistics.fmean(v["coverage"] for v in variants),
            "pairwise_active_sign_agreement": pair_agreement,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", type=float, default=0.002)
    parser.add_argument("--lambda-mad", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=6.902398038402426)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    rows = source["turns"]
    report = {
        "schema_version": "clawgym_abstain_opsd_phase0_v1",
        "input": str(args.input.resolve()),
        "definition": "all K signs agree; c=sign(median)*max(abs(median)-lambda*MAD-noise,0)",
        "primary": evaluate(rows, args.noise, args.lambda_mad, args.beta, args.bootstrap_repeats, args.seed),
        "sensitivity": [
            evaluate(rows, threshold, args.lambda_mad, args.beta, args.bootstrap_repeats, args.seed)
            for threshold in (0.0, 0.001, 0.002, 0.005)
        ],
        "k_subset_stability": subset_stability(rows, args.noise, args.lambda_mad),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["primary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
