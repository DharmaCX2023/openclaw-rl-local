#!/usr/bin/env python3
"""Quantify hindsight/self-distillation opportunities in saved ClawGym RL logs.

The formal run stores one ``instance-train-*.log`` per logical trajectory.  This
script reads only the compact header fields plus action/error markers; it does
not need to deserialize model tensors or scan the 12 GB proxy ledger.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FIELD_PATTERNS = {
    "task_id": re.compile(r"^=== TRAIN Task (\S+) ===$", re.MULTILINE),
    "group_index": re.compile(r"^Group Index: (\d+)$", re.MULTILINE),
    "reward": re.compile(r"^Reward: ([0-9.eE+-]+)$", re.MULTILINE),
    "steps": re.compile(r"^Steps: (\d+)$", re.MULTILINE),
    "compactions": re.compile(r"^Compactions: (\d+)$", re.MULTILINE),
    "pruning_events": re.compile(
        r"^Tool-result pruning events: (\d+)$", re.MULTILINE
    ),
    "termination": re.compile(r"^Termination Reason: (.+)$", re.MULTILINE),
}


@dataclass(frozen=True)
class Trajectory:
    rollout: int
    instance: int
    task_id: str
    group_index: int
    reward: float
    steps: int
    compactions: int
    pruning_events: int
    termination: str
    action_count: int
    error_observation_count: int
    path: str


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def parse_int_from_path(path: Path, prefix: str) -> int:
    match = re.search(rf"{re.escape(prefix)}(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot parse {prefix!r} from {path}")
    return int(match.group(1))


def parse_trajectory(path: Path) -> Trajectory | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    fields: dict[str, str] = {}
    for name, pattern in FIELD_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            return None
        fields[name] = match.group(1).strip()

    rollout_match = re.search(r"rollout_(\d+)", str(path.parent))
    if not rollout_match:
        return None

    return Trajectory(
        rollout=int(rollout_match.group(1)),
        instance=parse_int_from_path(path, "instance-train-"),
        task_id=fields["task_id"],
        group_index=int(fields["group_index"]),
        reward=float(fields["reward"]),
        steps=int(fields["steps"]),
        compactions=int(fields["compactions"]),
        pruning_events=int(fields["pruning_events"]),
        termination=fields["termination"],
        action_count=len(re.findall(r"^ACTION:", text, re.MULTILINE)),
        error_observation_count=len(
            re.findall(
                r"^OBSERVATION:.*(?:error|Error|Traceback|failed|Failed|not found)",
                text,
                re.MULTILINE,
            )
        ),
        path=str(path),
    )


def describe(values: Iterable[float]) -> dict[str, float | int]:
    vals = list(values)
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals) if vals else math.nan,
        "median": statistics.median(vals) if vals else math.nan,
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "max": max(vals) if vals else math.nan,
    }


def summarize(trajectories: list[Trajectory], expected_group_size: int) -> dict:
    grouped: dict[tuple[int, str], list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        grouped[(trajectory.rollout, trajectory.task_id)].append(trajectory)

    complete_groups = {
        key: vals for key, vals in grouped.items() if len(vals) == expected_group_size
    }
    incomplete_groups = {
        key: len(vals) for key, vals in grouped.items() if len(vals) != expected_group_size
    }

    group_rows = []
    for (rollout, task_id), vals in complete_groups.items():
        rewards = [item.reward for item in vals]
        best = max(rewards)
        worst = min(rewards)
        group_rows.append(
            {
                "rollout": rollout,
                "task_id": task_id,
                "mean": statistics.fmean(rewards),
                "best": best,
                "worst": worst,
                "gap": best - worst,
                "variance": statistics.pvariance(rewards),
                "n_zero": sum(value == 0 for value in rewards),
                "n_below_half": sum(value <= 0.5 for value in rewards),
                "n_below_one": sum(value < 1.0 for value in rewards),
                "n_strictly_below_best": sum(value < best - 1e-12 for value in rewards),
                "steps": sum(item.steps for item in vals),
            }
        )

    mixed = [row for row in group_rows if row["gap"] > 1e-12]
    all_equal = [row for row in group_rows if row["gap"] <= 1e-12]
    all_zero = [row for row in group_rows if row["best"] == 0]
    all_perfect = [row for row in group_rows if row["worst"] == 1]
    partial_only_equal = [
        row
        for row in all_equal
        if row["best"] not in (0.0, 1.0)
    ]

    lower_than_best = sum(row["n_strictly_below_best"] for row in mixed)
    low_reward_with_better_sibling = sum(
        sum(item.reward <= 0.5 and item.reward < row["best"] - 1e-12 for item in complete_groups[(row["rollout"], row["task_id"])])
        for row in mixed
    )
    zero_with_positive_sibling = sum(
        sum(item.reward == 0 and row["best"] > 0 for item in complete_groups[(row["rollout"], row["task_id"])])
        for row in mixed
    )

    reward_buckets = {
        "zero": [item for item in trajectories if item.reward == 0],
        "low_(0,0.5]": [item for item in trajectories if 0 < item.reward <= 0.5],
        "mid_(0.5,1)": [item for item in trajectories if 0.5 < item.reward < 1],
        "perfect": [item for item in trajectories if item.reward == 1],
    }
    bucket_stats = {}
    for name, vals in reward_buckets.items():
        bucket_stats[name] = {
            "count": len(vals),
            "fraction": len(vals) / len(trajectories) if trajectories else math.nan,
            "steps": describe(item.steps for item in vals),
            "actions": describe(item.action_count for item in vals),
            "errors": describe(item.error_observation_count for item in vals),
            "max_steps_fraction": (
                sum(item.steps >= 96 for item in vals) / len(vals) if vals else math.nan
            ),
        }

    task_occurrences = Counter(item.task_id for item in trajectories)
    unique_rollout_tasks = Counter(key[1] for key in grouped)

    examples = sorted(
        (
            item
            for item in trajectories
            if item.reward <= 0.5 and item.steps >= 80
        ),
        key=lambda item: (-item.steps, item.reward, item.task_id),
    )[:20]

    return {
        "coverage": {
            "trajectories": len(trajectories),
            "rollouts": len({item.rollout for item in trajectories}),
            "rollout_min": min((item.rollout for item in trajectories), default=None),
            "rollout_max": max((item.rollout for item in trajectories), default=None),
            "unique_tasks": len(task_occurrences),
            "groups": len(grouped),
            "complete_groups": len(complete_groups),
            "incomplete_groups": len(incomplete_groups),
            "incomplete_group_sizes": Counter(incomplete_groups.values()),
        },
        "trajectory": {
            "reward": describe(item.reward for item in trajectories),
            "steps": describe(item.steps for item in trajectories),
            "actions": describe(item.action_count for item in trajectories),
            "compactions": describe(item.compactions for item in trajectories),
            "pruning_events": describe(item.pruning_events for item in trajectories),
            "terminations": Counter(item.termination for item in trajectories),
            "reward_buckets": bucket_stats,
        },
        "groups": {
            "total_complete": len(group_rows),
            "mixed": len(mixed),
            "mixed_fraction": len(mixed) / len(group_rows) if group_rows else math.nan,
            "all_equal": len(all_equal),
            "all_zero": len(all_zero),
            "all_perfect": len(all_perfect),
            "all_equal_partial": len(partial_only_equal),
            "gap": describe(row["gap"] for row in group_rows),
            "mixed_gap": describe(row["gap"] for row in mixed),
            "trajectories_strictly_below_group_best": lower_than_best,
            "fraction_strictly_below_group_best": lower_than_best
            / (len(complete_groups) * expected_group_size)
            if complete_groups
            else math.nan,
            "low_reward_with_better_sibling": low_reward_with_better_sibling,
            "zero_reward_with_positive_sibling": zero_with_positive_sibling,
        },
        "task_reuse": {
            "trajectory_occurrences": describe(task_occurrences.values()),
            "group_occurrences": describe(unique_rollout_tasks.values()),
        },
        "long_failure_examples": [asdict(item) for item in examples],
    }


def render_markdown(summary: dict) -> str:
    coverage = summary["coverage"]
    trajectory = summary["trajectory"]
    groups = summary["groups"]
    lines = [
        "# ClawGym failure-hindsight opportunity audit",
        "",
        "## Coverage",
        "",
        f"- Parsed trajectories: **{coverage['trajectories']:,}**",
        f"- Rollouts: **{coverage['rollouts']}** ({coverage['rollout_min']}–{coverage['rollout_max']})",
        f"- Unique tasks: **{coverage['unique_tasks']:,}**",
        f"- Complete {8}-trajectory task groups: **{coverage['complete_groups']:,}**",
        f"- Incomplete groups: **{coverage['incomplete_groups']:,}**",
        "",
        "## Group-level reusable signal",
        "",
        f"- Mixed-reward groups: **{groups['mixed']:,}/{groups['total_complete']:,} ({groups['mixed_fraction']:.1%})**",
        f"- All-zero groups: **{groups['all_zero']:,}**",
        f"- All-perfect groups: **{groups['all_perfect']:,}**",
        f"- Equal partial-reward groups: **{groups['all_equal_partial']:,}**",
        f"- Trajectories strictly below a sibling best: **{groups['trajectories_strictly_below_group_best']:,} ({groups['fraction_strictly_below_group_best']:.1%})**",
        f"- Low-reward (<=0.5) trajectories with a better sibling: **{groups['low_reward_with_better_sibling']:,}**",
        f"- Zero-reward trajectories with a positive-reward sibling: **{groups['zero_reward_with_positive_sibling']:,}**",
        f"- Median reward gap within mixed groups: **{groups['mixed_gap']['median']:.4f}**; P90 **{groups['mixed_gap']['p90']:.4f}**",
        "",
        "## Reward and cost buckets",
        "",
        "| Reward bucket | Count | Share | Mean steps | P90 steps | Max-step share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in trajectory["reward_buckets"].items():
        lines.append(
            f"| {name} | {stats['count']:,} | {stats['fraction']:.1%} | "
            f"{stats['steps']['mean']:.2f} | {stats['steps']['p90']:.1f} | "
            f"{stats['max_steps_fraction']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Terminations",
            "",
        ]
    )
    for name, count in trajectory["terminations"].items():
        lines.append(f"- `{name}`: {count:,}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Mixed groups expose an immediate same-task ranking signal, while zero/constant "
            "groups are precisely where scalar GRPO has no within-group advantage signal. "
            "The cost bucket table shows whether failure-conditioned repair can replace full "
            "resampling with short suffix rollouts. This audit does not yet claim causal "
            "recoverability; that requires replaying selected workspace states.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-root",
        type=Path,
        required=True,
        help="Path containing rollout_*/instance-train-*.log",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.logs_root.glob("rollout_*/instance-train-*.log"))
    trajectories = []
    malformed = []
    for path in paths:
        parsed = parse_trajectory(path)
        if parsed is None:
            malformed.append(str(path))
        else:
            trajectories.append(parsed)

    summary = summarize(trajectories, args.group_size)
    summary["coverage"]["files_seen"] = len(paths)
    summary["coverage"]["malformed_files"] = malformed

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
