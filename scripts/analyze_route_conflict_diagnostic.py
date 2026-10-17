#!/usr/bin/env python3
"""Aggregate privacy-safe route-prefix conflict diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _load_audits(path: Path) -> list[dict[str, Any]]:
    audits = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_type") == "trajectory_audit":
                audits.append(row)
    return audits


def _sum_metric(rows: list[dict[str, Any]], key: str) -> int | float:
    return sum(row.get(key, 0) for row in rows)


def _aggregate_logprobs(
    diagnostics: list[dict[str, Any]], category: str
) -> dict[str, Any]:
    rows = [
        item["logprobs"][category]
        for item in diagnostics
        if item.get("logprobs", {}).get("available")
        and category in item["logprobs"]
    ]
    count = int(_sum_metric(rows, "count"))
    abs_sum = float(_sum_metric(rows, "logprob_abs_sum"))
    abs_sq_sum = float(_sum_metric(rows, "logprob_abs_sq_sum"))
    ratio_sum = float(_sum_metric(rows, "ratio_abs_sum"))
    return {
        "count": count,
        "logprob_mae": _ratio(abs_sum, count),
        "logprob_rms": math.sqrt(_ratio(abs_sq_sum, count)),
        "logprob_abs_max": max(
            (float(row.get("logprob_abs_max", 0.0)) for row in rows), default=0.0
        ),
        "logprob_abs_gt_0_01_fraction": _ratio(
            _sum_metric(rows, "logprob_abs_gt_0_01"), count
        ),
        "logprob_abs_gt_0_05_fraction": _ratio(
            _sum_metric(rows, "logprob_abs_gt_0_05"), count
        ),
        "logprob_abs_gt_0_1_fraction": _ratio(
            _sum_metric(rows, "logprob_abs_gt_0_1"), count
        ),
        "ratio_abs_mean": _ratio(ratio_sum, count),
        "ratio_abs_max": max(
            (float(row.get("ratio_abs_max", 0.0)) for row in rows), default=0.0
        ),
        "ratio_abs_gt_0_01_fraction": _ratio(
            _sum_metric(rows, "ratio_abs_gt_0_01"), count
        ),
        "ratio_abs_gt_0_05_fraction": _ratio(
            _sum_metric(rows, "ratio_abs_gt_0_05"), count
        ),
        "ratio_abs_gt_0_1_fraction": _ratio(
            _sum_metric(rows, "ratio_abs_gt_0_1"), count
        ),
    }


def analyze(path: Path) -> dict[str, Any]:
    audits = _load_audits(path)
    all_steps = [step for audit in audits for step in audit.get("steps", [])]
    eligible = [
        step
        for step in all_steps
        if step.get("routing_replay_action") in {"stitched", "split"}
        and step.get("routing_replay_split_reason")
        not in {
            "tool_result_pruned",
            "tool_result_truncated",
            "session_compacted",
            "history_rewritten",
            "template_delta_invalid",
            "token_prefix_rewrite",
        }
    ]
    conflict_steps = [
        step
        for step in all_steps
        if step.get("routing_replay_split_reason") == "route_prefix_conflict"
    ]
    diagnostics = [
        step["routing_replay_conflict_diagnostics"]
        for step in conflict_steps
        if isinstance(step.get("routing_replay_conflict_diagnostics"), dict)
    ]
    route_rows = [item["routes"] for item in diagnostics if "routes" in item]
    membership_boundaries = sum(
        bool(row.get("boundary_has_membership_change")) for row in route_rows
    )
    pure_order_boundaries = sum(
        bool(row.get("boundary_has_order_only_change"))
        and not bool(row.get("boundary_has_membership_change"))
        for row in route_rows
    )
    compared_rows = int(_sum_metric(route_rows, "compared_token_rows"))
    compared_cells = int(_sum_metric(route_rows, "compared_token_layer_cells"))
    compared_ids = int(_sum_metric(route_rows, "compared_route_ids"))
    affected_trajectories = {
        audit.get("session_id")
        for audit in audits
        if any(
            step.get("routing_replay_split_reason") == "route_prefix_conflict"
            for step in audit.get("steps", [])
        )
    }

    per_layer_membership: list[int] = []
    per_layer_order: list[int] = []
    if route_rows:
        layers = len(route_rows[0].get("per_layer_membership_changed_cells", []))
        per_layer_membership = [0] * layers
        per_layer_order = [0] * layers
        for row in route_rows:
            for index, value in enumerate(
                row.get("per_layer_membership_changed_cells", [])
            ):
                per_layer_membership[index] += int(value)
            for index, value in enumerate(row.get("per_layer_order_only_cells", [])):
                per_layer_order[index] += int(value)

    return {
        "trajectory_count": len(audits),
        "affected_trajectory_count": len(affected_trajectories),
        "affected_trajectory_fraction": _ratio(
            len(affected_trajectories), len(audits)
        ),
        "eligible_append_boundaries": len(eligible),
        "strict_conflict_boundaries": len(conflict_steps),
        "strict_conflict_boundary_fraction": _ratio(
            len(conflict_steps), len(eligible)
        ),
        "diagnosed_conflict_boundaries": len(diagnostics),
        "pure_order_only_boundaries": pure_order_boundaries,
        "pure_order_only_boundary_fraction_of_conflicts": _ratio(
            pure_order_boundaries, len(diagnostics)
        ),
        "membership_changed_boundaries": membership_boundaries,
        "membership_changed_boundary_fraction_of_conflicts": _ratio(
            membership_boundaries, len(diagnostics)
        ),
        "compared_token_rows_at_conflicts": compared_rows,
        "any_changed_token_row_fraction": _ratio(
            _sum_metric(route_rows, "any_changed_token_rows"), compared_rows
        ),
        "pure_order_only_token_row_fraction": _ratio(
            _sum_metric(route_rows, "order_only_token_rows"), compared_rows
        ),
        "membership_changed_token_row_fraction": _ratio(
            _sum_metric(route_rows, "membership_changed_token_rows"), compared_rows
        ),
        "compared_token_layer_cells_at_conflicts": compared_cells,
        "order_only_token_layer_fraction": _ratio(
            _sum_metric(route_rows, "order_only_token_layer_cells"), compared_cells
        ),
        "membership_changed_token_layer_fraction": _ratio(
            _sum_metric(route_rows, "membership_changed_token_layer_cells"),
            compared_cells,
        ),
        "compared_route_ids_at_conflicts": compared_ids,
        "ordered_mismatch_route_id_fraction": _ratio(
            _sum_metric(route_rows, "ordered_mismatch_route_ids"), compared_ids
        ),
        "replaced_expert_slot_fraction": _ratio(
            _sum_metric(route_rows, "replaced_expert_slots"), compared_ids
        ),
        "per_layer_membership_changed_cells": per_layer_membership,
        "per_layer_order_only_cells": per_layer_order,
        "logprob_recompute_available_boundaries": sum(
            bool(item.get("logprobs", {}).get("available")) for item in diagnostics
        ),
        "logprob_recompute": {
            category: _aggregate_logprobs(diagnostics, category)
            for category in (
                "all",
                "membership_changed",
                "order_only",
                "route_unchanged",
            )
        },
    }


def _percent(value: float) -> str:
    return f"{100 * value:.4f}%"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    logprob = summary["logprob_recompute"]
    lines = [
        "# Route-prefix conflict diagnostic",
        "",
        f"- 完成轨迹：{summary['trajectory_count']}",
        (
            f"- 冲突边界：{summary['strict_conflict_boundaries']}/"
            f"{summary['eligible_append_boundaries']} "
            f"({_percent(summary['strict_conflict_boundary_fraction'])})"
        ),
        (
            f"- 受影响轨迹：{summary['affected_trajectory_count']} "
            f"({_percent(summary['affected_trajectory_fraction'])})"
        ),
        (
            f"- 纯 Top-k 顺序冲突：{summary['pure_order_only_boundaries']}/"
            f"{summary['diagnosed_conflict_boundaries']} "
            f"({_percent(summary['pure_order_only_boundary_fraction_of_conflicts'])})"
        ),
        (
            f"- 含真正 membership 翻转：{summary['membership_changed_boundaries']}/"
            f"{summary['diagnosed_conflict_boundaries']} "
            f"({_percent(summary['membership_changed_boundary_fraction_of_conflicts'])})"
        ),
        "",
        "## Route 影响比例（仅 conflict 边界的重叠前缀）",
        "",
        f"- token row 任意变化：{_percent(summary['any_changed_token_row_fraction'])}",
        f"- token row 仅顺序变化：{_percent(summary['pure_order_only_token_row_fraction'])}",
        f"- token row 含 membership 变化：{_percent(summary['membership_changed_token_row_fraction'])}",
        f"- token×layer 仅顺序变化：{_percent(summary['order_only_token_layer_fraction'])}",
        f"- token×layer membership 变化：{_percent(summary['membership_changed_token_layer_fraction'])}",
        f"- 有序 route-ID 位置变化：{_percent(summary['ordered_mismatch_route_id_fraction'])}",
        f"- 被替换 expert slot：{_percent(summary['replaced_expert_slot_fraction'])}",
        "",
        "## Prompt 重算 logprob",
        "",
        "| 分类 | token数 | MAE | RMS | |ratio-1|均值 | >1% | >5% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category in ("all", "membership_changed", "order_only", "route_unchanged"):
        row = logprob[category]
        lines.append(
            f"| {category} | {row['count']} | {row['logprob_mae']:.6f} | "
            f"{row['logprob_rms']:.6f} | {row['ratio_abs_mean']:.4%} | "
            f"{row['ratio_abs_gt_0_01_fraction']:.2%} | "
            f"{row['ratio_abs_gt_0_05_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "> 统计单位是相邻 agent 请求边界；同一长前缀在多个边界重复出现时会重复计入。",
            "> logprob 只比较上一轮 assistant response token：原生成值 vs 下一轮 prompt 重算值。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    summary = analyze(args.records)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.markdown_output:
        write_markdown(summary, args.markdown_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
