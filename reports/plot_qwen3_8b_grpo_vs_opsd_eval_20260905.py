#!/usr/bin/env python3
"""Plot paired Qwen3-8B GRPO/OPSD evaluation curves from TensorBoard."""

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("/root/openclaw-rl-local")
RUNS = {
    "GRPO": ROOT / "outputs/qwen3-8b-grpo-formal-seed0-formal-v3-20260904/tensorboard",
    "OPSD": ROOT / "outputs/qwen3-8b-opsd-formal-seed0-formal-v3-20260904/tensorboard",
}
OUTPUT = ROOT / "reports/qwen3_8b_grpo_vs_opsd_eval_20260905.png"

COLORS = {"GRPO": "#2563EB", "OPSD": "#EA580C"}
TAGS = {
    "Average verifier reward": "eval/clawgym_eval-avg_reward",
    "Correct rate": "eval/clawgym_eval-correct_rate",
}


def load_scalars(directory: Path, tag: str) -> dict[int, float]:
    values: dict[int, float] = {}
    for event_file in sorted(directory.glob("events.out.tfevents*")):
        event = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        event.Reload()
        if tag in event.Tags().get("scalars", []):
            for item in event.Scalars(tag):
                values[int(item.step)] = float(item.value) * 100.0
    return values


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True, constrained_layout=True)

    all_data: dict[str, dict[str, dict[int, float]]] = {}
    for metric_name, tag in TAGS.items():
        all_data[metric_name] = {
            run_name: load_scalars(run_dir, tag) for run_name, run_dir in RUNS.items()
        }

    for axis, (metric_name, run_data) in zip(axes, all_data.items()):
        combined_values: list[float] = []
        for run_name, points in run_data.items():
            steps = sorted(points)
            values = [points[step] for step in steps]
            combined_values.extend(values)
            axis.plot(
                steps,
                values,
                color=COLORS[run_name],
                marker="o",
                markersize=7,
                linewidth=2.5,
                label=run_name,
                zorder=3,
            )
            offset = 0.20 if run_name == "GRPO" else -0.34
            vertical_alignment = "bottom" if offset > 0 else "top"
            for step, value in zip(steps, values):
                axis.text(
                    step,
                    value + offset,
                    f"{value:.2f}%" if metric_name.startswith("Average") else f"{value:.1f}%",
                    color=COLORS[run_name],
                    fontsize=8.8,
                    ha="center",
                    va=vertical_alignment,
                )

        margin = max(1.5, (max(combined_values) - min(combined_values)) * 0.28)
        axis.set_ylim(min(combined_values) - margin, max(combined_values) + margin)
        axis.set_ylabel("Score (%)")
        axis.set_title(metric_name, loc="left")
        axis.grid(axis="both", color="#D1D5DB", linewidth=0.8, alpha=0.75)
        axis.legend(loc="upper left", frameon=False, ncol=2)

    axes[-1].set_xlabel("Training updates (evaluation checkpoint)")
    axes[-1].set_xticks([20, 40, 60, 80, 100])
    fig.suptitle("Qwen3-8B ClawGym Evaluation: GRPO vs. OPSD", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Each point: 200 tasks, n=1, identical deterministic sampling path; no skipped tasks.",
        ha="center",
        color="#4B5563",
        fontsize=9.5,
    )
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
