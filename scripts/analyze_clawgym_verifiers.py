#!/usr/bin/env python3
"""Static audit of ClawGym code verifiers and their criterion-level feedback."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path


def percentile(values: list[int], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return values[lo] if lo == hi else values[lo] * (hi - pos) + values[hi] * (pos - lo)


def criterion_dicts(tree: ast.AST) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            continue
        names: list[str] = []
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
        elif isinstance(node.target, ast.Name):
            names.append(node.target.id)
        keys = [
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        for name in names:
            if name.lower() in {"checks", "criteria", "scores", "components"}:
                found[name] = keys
    return found


def output_paths(text: str) -> list[str]:
    candidates = set()
    for match in re.finditer(
        r"(?:output/|output_dir[^\n]{0,80}?)[\"']([A-Za-z0-9_.\-/]+)[\"']",
        text,
    ):
        candidates.add(match.group(1))
    for match in re.finditer(r"[\"']output/([A-Za-z0-9_.\-/]+)[\"']", text):
        candidates.add(match.group(1))
    return sorted(candidates)


def inspect(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text)
        parse_error = None
    except SyntaxError as exc:
        tree = None
        parse_error = str(exc)
    dicts = criterion_dicts(tree) if tree is not None else {}
    keys = sorted({key for values in dicts.values() for key in values})
    exposes_vector = bool(
        re.search(r"(?:result|output)\.update\(\s*(?:checks|criteria|scores|components)\s*\)", text)
    )
    task_match = re.search(r"task_(\d+)", str(path))
    return {
        "task_id": f"task_{int(task_match.group(1)):04d}" if task_match else path.parent.parent.name,
        "path": str(path),
        "parse_error": parse_error,
        "criterion_dict_names": sorted(dicts),
        "criterion_keys": keys,
        "criterion_count": len(keys),
        "exposes_criterion_vector": exposes_vector,
        "uses_subprocess": bool(re.search(r"subprocess\.(?:run|check_output|Popen)", text)),
        "uses_timeout": "timeout=" in text,
        "mentions_output_dir": "output_dir" in text or "output/" in text,
        "output_path_hints": output_paths(text),
        "prints_json": bool(re.search(r"print\(\s*json\.dumps", text)),
        "has_partial_average": bool(
            re.search(r"passed\s*/\s*total|sum\([^\n]*checks[^\n]*\)\s*/|reward\s*=\s*[^\n]*/[^\n]", text)
        ),
    }


def describe(values: list[int]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else math.nan,
        "median": statistics.median(values) if values else math.nan,
        "p90": percentile(values, 0.9),
        "max": max(values) if values else math.nan,
    }


def summarize(rows: list[dict], used_tasks: set[str]) -> dict:
    def aggregate(selected: list[dict]) -> dict:
        with_criteria = [row for row in selected if row["criterion_count"] > 0]
        exposed = [row for row in selected if row["exposes_criterion_vector"]]
        return {
            "tasks": len(selected),
            "parse_errors": sum(row["parse_error"] is not None for row in selected),
            "with_static_criterion_dict": len(with_criteria),
            "with_static_criterion_dict_fraction": len(with_criteria) / len(selected) if selected else math.nan,
            "exposes_criterion_vector": len(exposed),
            "exposes_criterion_vector_fraction": len(exposed) / len(selected) if selected else math.nan,
            "criterion_count": describe([row["criterion_count"] for row in with_criteria]),
            "uses_subprocess": sum(row["uses_subprocess"] for row in selected),
            "uses_timeout": sum(row["uses_timeout"] for row in selected),
            "prints_json": sum(row["prints_json"] for row in selected),
            "has_partial_average": sum(row["has_partial_average"] for row in selected),
            "output_path_hint_count": describe([len(row["output_path_hints"]) for row in selected]),
        }

    used = [row for row in rows if row["task_id"] in used_tasks]
    return {
        "all_train": aggregate(rows),
        "formal_rl_800_tasks": aggregate(used),
        "formal_rl_task_coverage": len(used),
        "criterion_name_top50": Counter(
            key for row in rows for key in row["criterion_keys"]
        ).most_common(50),
        "rows": rows,
    }


def render(summary: dict) -> str:
    lines = ["# ClawGym verifier structure audit", ""]
    for title, key in (("All 2,000 train tasks", "all_train"), ("Formal-RL 800 tasks", "formal_rl_800_tasks")):
        stats = summary[key]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- Tasks: **{stats['tasks']:,}**",
                f"- Static criterion dictionaries: **{stats['with_static_criterion_dict']:,} ({stats['with_static_criterion_dict_fraction']:.1%})**",
                f"- Criterion vector emitted in verifier JSON: **{stats['exposes_criterion_vector']:,} ({stats['exposes_criterion_vector_fraction']:.1%})**",
                f"- Criteria per structured verifier: mean **{stats['criterion_count']['mean']:.2f}**, median **{stats['criterion_count']['median']:.1f}**, P90 **{stats['criterion_count']['p90']:.1f}**, max **{stats['criterion_count']['max']}**",
                f"- Verifiers invoking subprocesses: **{stats['uses_subprocess']:,}**",
                f"- Verifiers with explicit timeout: **{stats['uses_timeout']:,}**",
                f"- Verifiers printing JSON: **{stats['prints_json']:,}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Implication",
            "",
            "A substantial subset already computes criterion-level booleans and emits them in the "
            "same JSON object as the scalar reward. Capturing this vector adds essentially no "
            "verifier execution cost. It can support criterion coverage, all-failed-group learning, "
            "and artifact-localized credit; unsupported verifiers must fall back to scalar GRPO/OPSD.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.data_root.glob("task_*/reward/check.py"))
    rows = [inspect(path) for path in paths]
    used_tasks: set[str] = set()
    if args.logs_root:
        task_pattern = re.compile(r"^=== TRAIN Task (\S+) ===$", re.MULTILINE)
        for path in args.logs_root.glob("rollout_*/instance-train-*.log"):
            head = path.read_text(encoding="utf-8", errors="replace")[:4096]
            match = task_pattern.search(head)
            if match:
                used_tasks.add(match.group(1))

    summary = summarize(rows, used_tasks)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
