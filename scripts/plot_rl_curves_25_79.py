#!/usr/bin/env python3
"""Reproduce the rollout 25..79 multi-panel RL curve figure from its CSV."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "outputs/rl-80step-curve-analysis/rl_metrics_rollout_25_to_79_exact.csv"
DEFAULT_PNG = ROOT / "outputs/rl-80step-curve-analysis/rl_curves_rollout_25_to_79_exact.png"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.csv)
    if "step" not in data:
        raise KeyError(f"CSV must contain a 'step' column; got {list(data.columns)}")

    metrics = [column for column in data.columns if column not in {"step", "rollout_id"}]
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
    columns = 4
    rows = int(np.ceil(len(metrics) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(22, 4.25 * rows), sharex=True)
    axes = np.asarray(axes).ravel()
    x = data["step"]
    ticks = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79]

    for index, metric in enumerate(metrics):
        ax = axes[index]
        y = data[metric]
        color = colors[index % len(colors)]
        ax.plot(x, y, color=color, alpha=0.28, linewidth=0.9, label="raw")
        ax.plot(
            x,
            y.rolling(5, center=True, min_periods=1).mean(),
            color=color,
            linewidth=2.2,
            label="centered MA(5)",
        )
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlim(24.5, 79.5)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        if metric == "grad_norm_preclip":
            ax.set_yscale("log")
        elif metric == "ppo_kl":
            ax.set_yscale("symlog", linthresh=1e-5)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    for ax in axes[-columns:]:
        ax.set_xlabel("rollout_id / step (25..79); training interval 25 -> 80")

    fig.suptitle(
        "OpenClaw RL — Exact Rollout 25–79 Curves (Training Interval 25→80)",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
