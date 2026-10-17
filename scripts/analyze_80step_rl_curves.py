#!/usr/bin/env python3
"""Extract the 0..79 RL metrics from split/resumed logs and plot diagnostics."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
PATTERNS = {
    "rollout": re.compile(r"rollout\.py:\d+ - perf (\d+): (\{.*\})$"),
    "data": re.compile(r"data\.py:\d+ - rollout (\d+): (\{.*\})$"),
    "train": re.compile(r"model\.py:\d+ - step (\d+): (\{.*\})$"),
    "perf": re.compile(r"train_metric_utils\.py:\d+ - perf (\d+): (\{.*\})$"),
}


def extract(log_dir: Path, max_step: int) -> pd.DataFrame:
    # Resume logs overlap. The chronologically newest complete record wins.
    records: dict[tuple[str, int], tuple[datetime, dict]] = {}
    for path in sorted(log_dir.glob("*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = ANSI_RE.sub("", raw.rstrip())
                timestamp_match = TS_RE.search(line)
                if not timestamp_match:
                    continue
                timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S")
                for kind, pattern in PATTERNS.items():
                    match = pattern.search(line)
                    if not match:
                        continue
                    step = int(match.group(1))
                    if step > max_step:
                        break
                    try:
                        payload = ast.literal_eval(match.group(2))
                    except (SyntaxError, ValueError):
                        break
                    key = (kind, step)
                    if key not in records or timestamp >= records[key][0]:
                        records[key] = (timestamp, payload)
                    break

    rows = []
    for step in range(max_step + 1):
        row: dict[str, float | int] = {"step": step + 1, "rollout_id": step}
        for kind in PATTERNS:
            found = records.get((kind, step))
            if not found:
                continue
            _, payload = found
            for key, value in payload.items():
                if isinstance(value, (int, float)):
                    row[key] = value
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("step")
    if "rollout/truncated_ratio" in frame:
        frame["rollout/natural_termination_ratio"] = 1.0 - frame["rollout/truncated_ratio"]
    if "openclaw/submitted_trajectories" in frame:
        frame["openclaw/submission_ratio"] = frame["openclaw/submitted_trajectories"] / 64.0
    if {"rollout/log_probs", "rollout/ref_log_probs"}.issubset(frame.columns):
        frame["rollout/logprob_shift"] = frame["rollout/log_probs"] - frame["rollout/ref_log_probs"]
    return frame


def series(ax, frame, columns, labels=None, smooth=5, ylim=None):
    labels = labels or columns
    for column, label in zip(columns, labels):
        if column not in frame:
            continue
        ax.plot(frame["step"], frame[column], alpha=0.25, linewidth=1)
        ax.plot(frame["step"], frame[column].rolling(smooth, min_periods=1).mean(), label=label, linewidth=2)
    for checkpoint in range(5, 81, 5):
        ax.axvline(checkpoint, color="0.85", linewidth=0.5, zorder=0)
    ax.grid(alpha=0.2)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8, loc="best")


def plot(frame: pd.DataFrame, output: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(17, 19), sharex=True)
    fig.suptitle("ClawGym GRPO: 80-step RL diagnostics (thin=raw, thick=5-step mean)", fontsize=16)

    series(
        axes[0, 0], frame,
        ["openclaw/avg_reward_submitted", "openclaw/avg_reward_with_missing_as_zero", "rollout/raw_reward"],
        ["reward submitted", "reward missing=0", "train raw reward"], ylim=(0, 1),
    )
    axes[0, 0].set_title("Reward")

    series(
        axes[0, 1], frame,
        ["openclaw/group_all_zero_frac", "openclaw/group_all_one_frac", "openclaw/group_all_same_mid_frac", "openclaw/group_mixed_frac", "openclaw/group_zero_var_frac"],
        ["all-zero groups", "all-one groups", "same-mid groups", "mixed groups", "zero-variance groups"], ylim=(0, 1),
    )
    axes[0, 1].set_title("GRPO group composition")

    series(
        axes[1, 0], frame,
        ["rollout/natural_termination_ratio", "rollout/truncated_ratio", "openclaw/submission_ratio", "rollout/repetition_frac"],
        ["natural termination", "token truncation", "submitted / 64", "repetition"], ylim=(0, 1.05),
    )
    axes[1, 0].set_title("Trajectory completion / failure modes")

    series(
        axes[1, 1], frame,
        ["rollout/response_len/mean", "rollout/response_len/median", "openclaw/avg_steps_per_trajectory"],
        ["response len mean", "response len median", "agent steps / trajectory"],
    )
    axes[1, 1].set_title("Trajectory length (different units share axis)")

    series(
        axes[2, 0], frame,
        ["train/pg_loss", "train/entropy_loss", "train/kl_loss", "train/ppo_kl"],
        ["policy-gradient loss", "entropy metric", "KL metric", "PPO KL"],
    )
    axes[2, 0].axhline(0, color="black", linewidth=0.7)
    axes[2, 0].set_title("Optimization metrics (entropy_coef=0, kl_coef=0)")

    series(
        axes[2, 1], frame,
        ["train/grad_norm", "train/pg_clipfrac", "train/train_rollout_logprob_abs_diff"],
        ["grad norm", "clip fraction", "train-rollout logprob abs diff"],
    )
    axes[2, 1].set_title("Update stability")

    series(
        axes[3, 0], frame,
        ["rollout/rollout_log_probs", "rollout/ref_log_probs", "rollout/log_probs", "rollout/logprob_shift"],
        ["generation logprob", "reference logprob", "post-update logprob", "post-update minus ref"],
    )
    axes[3, 0].set_title("Policy/reference log probabilities")

    series(
        axes[3, 1], frame,
        ["openclaw/avg_task_duration_sec", "perf/rollout_time", "perf/effective_tokens_per_gpu_per_sec"],
        ["avg task duration sec", "rollout wall time sec", "effective tok/GPU/s"],
    )
    axes[3, 1].set_title("System / rollout performance (different units share axis)")

    for ax in axes[-1, :]:
        ax.set_xlabel("training step (1..80); checkpoint every 5 steps")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=170)
    plt.close(fig)


def window_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "openclaw/avg_reward_submitted",
        "openclaw/avg_reward_with_missing_as_zero",
        "rollout/raw_reward",
        "openclaw/group_all_zero_frac",
        "openclaw/group_all_one_frac",
        "openclaw/group_all_same_mid_frac",
        "openclaw/group_mixed_frac",
        "openclaw/group_zero_var_frac",
        "rollout/natural_termination_ratio",
        "rollout/truncated_ratio",
        "openclaw/submission_ratio",
        "rollout/repetition_frac",
        "rollout/response_len/mean",
        "openclaw/avg_steps_per_trajectory",
        "train/entropy_loss",
        "train/kl_loss",
        "train/ppo_kl",
        "train/grad_norm",
        "train/pg_clipfrac",
        "train/train_rollout_logprob_abs_diff",
    ]
    present = [column for column in columns if column in frame]
    labeled = frame.assign(window=pd.cut(frame["step"], [0, 20, 40, 60, 80], labels=["1-20", "21-40", "41-60", "61-80"]))
    return labeled.groupby("window", observed=True)[present].mean().T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-step", type=int, default=79)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = extract(args.log_dir, args.max_step)
    frame.to_csv(args.output_dir / "rl_metrics_80steps.csv", index=False)
    summary = window_summary(frame)
    summary.to_csv(args.output_dir / "rl_metrics_window_summary.csv")
    plot(frame, args.output_dir / "rl_curves_80steps.png")

    completeness = {
        "steps_expected": args.max_step + 1,
        "rollout_metric_steps": int(frame["openclaw/avg_reward_submitted"].notna().sum()),
        "train_metric_steps": int(frame["train/pg_loss"].notna().sum()),
        "data_metric_steps": int(frame["rollout/raw_reward"].notna().sum()),
        "columns": len(frame.columns),
    }
    (args.output_dir / "extraction_summary.json").write_text(json.dumps(completeness, indent=2) + "\n")
    print(json.dumps(completeness, indent=2))
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
