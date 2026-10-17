#!/usr/bin/env python3
"""Create publication-readable plots for the ClawGym OPD precheck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-stats", type=Path, required=True)
    parser.add_argument("--same-policy-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factor = json.loads(args.factor_stats.read_text())
    peer = json.loads(args.same_policy_stats.read_text())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 160,
    })
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)

    eval_steps = factor["h1_verifier_factorization"]["eval_by_checkpoint"]
    steps = sorted(eval_steps, key=int)
    partial = [100 * eval_steps[step]["partial_scalar_reward_rate"] for step in steps]
    mixed_final = [100 * eval_steps[step]["mixed_outcome_rate"] for step in steps]
    axes[0].plot([int(step) for step in steps], partial, "o-", lw=2, label="Partial scalar reward")
    axes[0].plot([int(step) for step in steps], mixed_final, "s--", lw=2, label="Mixed verifier vector")
    axes[0].set(title="Partial-success prevalence", xlabel="RL checkpoint step", ylabel="Trajectory-workspace rate (%)", ylim=(75, 100))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="lower right")

    seed_names = ["seed1", "seed2", "combined"]
    seed_rows = [peer["per_seed"]["seed1"], peer["per_seed"]["seed2"], peer["combined"]]
    rates = np.array([100 * row["rate"] for row in seed_rows])
    lower = rates - np.array([100 * row["wilson_95"][0] for row in seed_rows])
    upper = np.array([100 * row["wilson_95"][1] for row in seed_rows]) - rates
    axes[1].bar(seed_names, rates, color=["#4C78A8", "#72B7B2", "#F58518"], width=0.65)
    axes[1].errorbar(seed_names, rates, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4)
    axes[1].axhline(25, color="#B22222", ls="--", lw=1.5, label="Prespecified gate: 25%")
    for index, value in enumerate(rates):
        axes[1].text(index, value + upper[index] + 1.2, f"{value:.1f}%", ha="center", fontsize=10)
    axes[1].set(title="Same-policy mixed-credit rate", ylabel="Target-action rate (%)", ylim=(0, 45))
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=9, loc="lower right")

    bins = ["<=8", "9-12", "13-16", ">=17"]
    rows = peer["combined"]["by_factor_count"]
    rates = np.array([100 * rows[key]["rate"] for key in bins])
    lower = rates - np.array([100 * rows[key]["wilson_95"][0] for key in bins])
    upper = np.array([100 * rows[key]["wilson_95"][1] for key in bins]) - rates
    axes[2].bar(bins, rates, color="#54A24B", width=0.65)
    axes[2].errorbar(bins, rates, yerr=np.vstack([lower, upper]), fmt="none", ecolor="black", capsize=4)
    axes[2].axhline(25, color="#B22222", ls="--", lw=1.5)
    for index, key in enumerate(bins):
        axes[2].text(index, rates[index] + upper[index] + 1.2, f"{rows[key]['mixed_actions']}/{rows[key]['target_actions']}", ha="center", fontsize=9)
    axes[2].set(title="Mixed credit vs. requirement count", xlabel="Verifier factors per task", ylabel="Target-action rate (%)", ylim=(0, 52))
    axes[2].grid(axis="y", alpha=0.25)

    figure.suptitle("ClawGym Success-Preserving OPD Phase-0 Evidence", fontsize=15, fontweight="bold")
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
