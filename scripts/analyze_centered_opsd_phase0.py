#!/usr/bin/env python3
"""Offline Phase-0 diagnostics for logical-rollout-centered OPSD."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from slime.utils.opsd_utils import compute_centered_opsd_group_weights


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    x = sorted(values)
    pos = q * (len(x) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return x[lo]
    return x[lo] * (hi - pos) + x[hi] * (pos - lo)


def weighted_quantile(values: list[float], counts: list[int], q: float) -> float:
    rows = sorted(zip(values, counts), key=lambda row: row[0])
    total = sum(c for _, c in rows)
    if total <= 0:
        return 0.0
    target = q * total
    seen = 0
    for value, count in rows:
        seen += count
        if seen >= target:
            return value
    return rows[-1][0]


def load_rollouts(summary_dir: Path):
    grouped = defaultdict(list)
    files = sorted(summary_dir.glob("*.json"), key=lambda p: int(p.stem))
    for path in files:
        payload = json.loads(path.read_text())
        rollout_id = int(payload["rollout_id"])
        for row in payload.get("records", []):
            key = (rollout_id, int(row["group_index"]), int(row["trajectory_index"]))
            grouped[key].append(row)
    result = []
    for key, rows in grouped.items():
        rows.sort(key=lambda row: int(row.get("segment_index", 0)))
        if not rows or not all(bool(row.get("opsd_applied")) for row in rows):
            continue
        steps = [step for row in rows for step in row.get("steps", [])]
        if not steps:
            continue
        result.append({
            "key": key,
            "segments": len(rows),
            "reward": float(rows[0]["raw_reward"]),
            "advantage": float(rows[0]["advantage"]),
            "steps": steps,
        })
    return files, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    args = parser.parse_args()
    files, rollouts = load_rollouts(args.summary_dir)
    base = {
        "summary_files": len(files),
        "logical_rollouts": len(rollouts),
        "segments": sum(row["segments"] for row in rollouts),
        "multi_segment_rollouts": sum(row["segments"] > 1 for row in rollouts),
        "steps": sum(len(row["steps"]) for row in rollouts),
        "single_step_rollouts": sum(len(row["steps"]) == 1 for row in rollouts),
        "step_count_quantiles": {str(q): quantile([len(row["steps"]) for row in rollouts], q) for q in (0.5, 0.9, 0.95, 0.99)},
        "reward_bins": {
            "[0,.25)": sum(0 <= row["reward"] < .25 for row in rollouts),
            "[.25,.5)": sum(.25 <= row["reward"] < .5 for row in rollouts),
            "[.5,.75)": sum(.5 <= row["reward"] < .75 for row in rollouts),
            "[.75,1)": sum(.75 <= row["reward"] < 1 for row in rollouts),
        },
    }
    results = {}
    for beta in args.betas:
        step_weights, token_counts, mads, medians = [], [], [], []
        applied = fallback = saturated_steps = saturated_rollouts = 0
        max_mass_error = 0.0
        for row in rollouts:
            supports = torch.tensor([step["teacher_minus_rollout_old_mean"] for step in row["steps"]], dtype=torch.float64)
            counts = torch.tensor([step["loss_token_count"] for step in row["steps"]], dtype=torch.float64)
            weights, diag = compute_centered_opsd_group_weights(
                supports=supports,
                token_counts=counts,
                beta=beta,
                clip_lower=.8,
                clip_upper=1.2,
                epsilon=1e-6,
                min_steps=2,
            )
            if not diag["applied"]:
                fallback += 1
                continue
            applied += 1
            values = [float(x) for x in weights]
            counts_i = [int(x) for x in counts]
            step_weights.extend(values)
            token_counts.extend(counts_i)
            mads.append(float(diag["support_mad"]))
            medians.append(float(diag["support_median"]))
            sat = sum(value <= .800001 or value >= 1.199999 for value in values)
            saturated_steps += sat
            saturated_rollouts += sat > 0
            max_mass_error = max(max_mass_error, abs(float(diag["final_token_weight_mean"]) - 1.0))
        results[str(beta)] = {
            "applied_rollouts": applied,
            "fallback_rollouts": fallback,
            "fallback_fraction": fallback / max(1, len(rollouts)),
            "saturated_step_fraction": saturated_steps / max(1, len(step_weights)),
            "rollouts_with_saturation_fraction": saturated_rollouts / max(1, applied),
            "max_token_mass_mean_error": max_mass_error,
            "step_weight_quantiles": {str(q): quantile(step_weights, q) for q in (0.01, .05, .5, .95, .99)},
            "token_weight_quantiles": {str(q): weighted_quantile(step_weights, token_counts, q) for q in (0.01, .05, .5, .95, .99)},
            "mad_quantiles": {str(q): quantile(mads, q) for q in (.01, .05, .5, .95, .99)},
            "median_quantiles": {str(q): quantile(medians, q) for q in (.01, .5, .99)},
        }
    payload = {"schema_version": 1, "base": base, "betas": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
