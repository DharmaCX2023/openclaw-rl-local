#!/usr/bin/env python3
import argparse
import ast
import json
import re
import statistics
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def final_eval(log_path: Path):
    result = None
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw_line)
        marker = "eval 0: "
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1]
        try:
            result = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
    return result


def pick(metrics, *keys):
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_root", type=Path)
    args = parser.parse_args()

    runs = []
    for step_dir in sorted(args.batch_root.glob("step*")):
        requested_step = int(step_dir.name.removeprefix("step"))
        for run_dir in sorted(step_dir.glob("run*")):
            logs = sorted((run_dir / "logs").glob("*.log"))
            if not logs:
                continue
            metrics = final_eval(logs[-1])
            if metrics is None:
                continue
            runs.append(
                {
                    "requested_step": requested_step,
                    "iteration": requested_step - 1,
                    "run": int(run_dir.name.split("-", 1)[0].removeprefix("run")),
                    "avg_reward": pick(
                        metrics,
                        "eval/clawgym_eval-avg_reward",
                        "avg_reward",
                    ),
                    "correct_rate": pick(metrics, "correct_rate"),
                    "task_count": pick(metrics, "task_count"),
                    "n_collected": pick(metrics, "n_collected"),
                    "n_skipped": pick(metrics, "n_skipped"),
                    "log": str(logs[-1]),
                }
            )

    aggregates = []
    for step in (20, 40, 60, 80, 100):
        selected = sorted(
            (run for run in runs if run["requested_step"] == step),
            key=lambda run: run["run"],
        )
        rewards = [float(run["avg_reward"]) for run in selected if run["avg_reward"] is not None]
        correct = [float(run["correct_rate"]) for run in selected if run["correct_rate"] is not None]
        aggregates.append(
            {
                "requested_step": step,
                "iteration": step - 1,
                "valid_runs": sum(
                    run["task_count"] == 200
                    and run["n_collected"] == 200
                    and run["n_skipped"] == 0
                    for run in selected
                ),
                "avg_reward_mean": statistics.mean(rewards) if rewards else None,
                "avg_reward_stdev": statistics.stdev(rewards) if len(rewards) > 1 else None,
                "correct_rate_mean": statistics.mean(correct) if correct else None,
            }
        )

    report = {"runs": runs, "aggregates": aggregates}
    json_path = args.batch_root / "aggregate_results.json"
    md_path = args.batch_root / "aggregate_results.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    rows = [
        "| requested step | checkpoint iteration | valid runs | avg_reward mean | stdev | correct_rate mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        fmt = lambda value: "" if value is None else f"{value:.6f}"
        rows.append(
            f"| {item['requested_step']} | {item['iteration']} | {item['valid_runs']}/3 | "
            f"{fmt(item['avg_reward_mean'])} | {fmt(item['avg_reward_stdev'])} | "
            f"{fmt(item['correct_rate_mean'])} |"
        )
    md_path.write_text("\n".join(rows) + "\n")
    print(md_path)


if __name__ == "__main__":
    main()
