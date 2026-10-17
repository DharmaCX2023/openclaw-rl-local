#!/usr/bin/env python3
"""Self-contained reproduction of the OpenClaw rollout 25..79 RL curve figure."""

from argparse import ArgumentParser
import base64
import gzip
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_GZIP_BASE64 = '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'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("rl_curves_rollout_25_to_79_exact.png"))
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def main():
    args = parse_args()
    csv_bytes = gzip.decompress(base64.b64decode(DATA_GZIP_BASE64))
    data = pd.read_csv(BytesIO(csv_bytes))

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
