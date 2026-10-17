#!/usr/bin/env python3
"""Aggregate uncertainty-aware statistics for the ClawGym OPD precheck."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _cluster_bootstrap(records: list[dict[str, Any]], seed: int, repeats: int) -> list[float]:
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clusters[(record["transition"], record["task_id"])].append(record)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = [rng.choice(keys) for _ in keys]
        denominator = sum(len(clusters[key]) for key in sampled)
        numerator = sum(
            bool(record["regressions"])
            for key in sampled
            for record in clusters[key]
        )
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return [values[int(0.025 * repeats)], values[min(repeats - 1, int(0.975 * repeats))]]


def _cluster_bootstrap_factor_difference(
    records: list[dict[str, Any]], seed: int, repeats: int
) -> dict[str, Any]:
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        clusters[(record["transition"], record["task_id"])].append(record)
    keys = sorted(clusters)
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(repeats):
        rows = [record for _ in keys for record in clusters[rng.choice(keys)]]
        high = [record for record in rows if int(record["factor_count"]) >= 17]
        low = [record for record in rows if int(record["factor_count"]) < 17]
        if not high or not low:
            continue
        high_rate = sum(bool(record["regressions"]) for record in high) / len(high)
        low_rate = sum(bool(record["regressions"]) for record in low) / len(low)
        differences.append(high_rate - low_rate)
    differences.sort()
    count = len(differences)
    return {
        "difference_high_minus_low": (
            sum(bool(r["regressions"]) for r in records if int(r["factor_count"]) >= 17)
            / sum(int(r["factor_count"]) >= 17 for r in records)
            - sum(bool(r["regressions"]) for r in records if int(r["factor_count"]) < 17)
            / sum(int(r["factor_count"]) < 17 for r in records)
        ),
        "cluster_bootstrap_95": [
            differences[int(0.025 * count)],
            differences[min(count - 1, int(0.975 * count))],
        ],
        "bootstrap_probability_difference_le_zero": (
            sum(value <= 0 for value in differences) / count
        ),
        "task_pair_clusters": len(keys),
    }


def _bin_steps(value: Any) -> str:
    if value is None:
        return "unknown"
    value = int(value)
    if value <= 10:
        return "<=10"
    if value <= 20:
        return "11-20"
    if value <= 40:
        return "21-40"
    return ">40"


def _bin_factors(value: Any) -> str:
    if value is None:
        return "unknown"
    value = int(value)
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    return ">=17"


def _stratify(
    records: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)
    result: dict[str, Any] = {}
    for key in sorted(groups):
        rows = groups[key]
        mixed = sum(bool(row["regressions"]) for row in rows)
        result[key] = {
            "failure_relevant_exact_writes": len(rows),
            "mixed": mixed,
            "rate": mixed / len(rows),
            "wilson_95": _wilson(mixed, len(rows)),
            "task_pair_clusters": len({(r["transition"], r["task_id"]) for r in rows}),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-audit", type=Path, required=True)
    parser.add_argument("--swap-audit", type=Path, required=True)
    parser.add_argument("--action-provenance", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factor = json.loads(args.factor_audit.resolve(strict=True).read_text())
    swap = json.loads(args.swap_audit.resolve(strict=True).read_text())
    action = json.loads(args.action_provenance.resolve(strict=True).read_text())
    exact = [record for record in action["records"] if record["exact_single_write"]]
    mixed = [record for record in exact if record["regressions"]]
    n, k = len(exact), len(mixed)

    examples = sorted(
        mixed,
        key=lambda row: (-(len(row["corrections"]) + len(row["regressions"])), row["task_id"]),
    )[:30]
    examples = [{
        "task_id": row["task_id"],
        "transition": row["transition"],
        "artifact": row["artifact"],
        "write_log_line": row["matching_mutating_actions"][0]["line"],
        "corrections": row["corrections"],
        "regressions": row["regressions"],
        "instance_log": row["right_instance_log"],
    } for row in examples]

    eval_table = {
        step: {
            key: payload["summary"][key]
            for key in (
                "result_rows", "checker_ok", "stable", "factorized",
                "factor_count_mean", "factor_count_median", "mixed_outcome",
                "mixed_outcome_rate", "partial_scalar_reward",
                "partial_scalar_reward_rate", "reward_mean",
            )
        }
        for step, payload in sorted(factor["eval_steps"].items(), key=lambda item: int(item[0]))
    }
    output = {
        "schema_version": 1,
        "h1_verifier_factorization": {
            "train_random_500": factor["training_schema_audit"]["summary"],
            "eval_by_checkpoint": eval_table,
            "interpretation": (
                "Factorization and deterministic replay are directly measured. Mixed final "
                "outcome is not action-level mixed credit."
            ),
        },
        "h2_mixed_credit_proxy": {
            "failure_relevant_exact_single_writes": n,
            "mixed_credit_exact_single_writes": k,
            "rho_mixed": k / n if n else 0.0,
            "binomial_wilson_95": _wilson(k, n),
            "task_pair_cluster_bootstrap_95": _cluster_bootstrap(
                exact, args.seed, args.bootstrap_repeats
            ),
            "selection": (
                "All adjacent-checkpoint task pairs whose earlier workspace has at least one "
                "failed factor; no filtering on later correction/regression outcomes."
            ),
            "unit": (
                "A logged, byte-exact, single full-file write replayed on the earlier workspace; "
                "not an original on-policy turn in that earlier trajectory."
            ),
            "by_transition": _stratify(exact, lambda row: row["transition"]),
            "by_right_trajectory_steps": _stratify(
                exact, lambda row: _bin_steps(row["right_agent_steps"])
            ),
            "by_factor_count": _stratify(
                exact, lambda row: _bin_factors(row["factor_count"])
            ),
            "factor_count_ge17_association": _cluster_bootstrap_factor_difference(
                exact, args.seed + 1, args.bootstrap_repeats
            ),
            "by_artifact_root": _stratify(
                exact, lambda row: row["artifact"].split("/", 1)[0]
            ),
        },
        "h3_collateral_damage": {
            "artifact_swap_summary": swap["summary"],
            "observational_checkpoint_transitions": factor["checkpoint_factor_transitions"],
            "targeted_opd_gradient_update_tested": False,
            "interpretation": (
                "Workspace interventions prove collateral damage is possible and nontrivial. "
                "Checkpoint transitions are observational. Neither measures regression caused "
                "by a HINT-SD/OPD gradient update, so H3 remains open."
            ),
        },
        "qualitative_exact_write_examples": examples,
    }
    destination = args.output.resolve()
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(destination),
        "rho_mixed": output["h2_mixed_credit_proxy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
