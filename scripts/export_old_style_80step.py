#!/usr/bin/env python3
"""Export the current run in the column order and plot style of the old 80-step figure."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/rl-80step-curve-analysis"
SOURCE = OUT / "rl_metrics_80steps.csv"
CSV = OUT / "rl_80step_original_format.csv"
PNG = OUT / "rl_80step_original_color_style.png"
EXACT_CSV = OUT / "rl_metrics_rollout_25_to_79_exact.csv"
EXACT_PNG = OUT / "rl_curves_rollout_25_to_79_exact.png"
RUN_OUTPUT = ROOT / "outputs/clawgym-official-30b-sft11127-sync2000-rsp8192-ent0-w32mux-20260804-0640"
OPENCLAW_LOGS = RUN_OUTPUT / "openclaw_logs"
OVERLENGTH_STEPS = 12


def col(frame, name):
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def termination_breakdown(source):
    oversized = set()
    oversized_pattern = re.compile(
        r"dropping oversized sample: \d+ tokens > 65536, session=task-(\d+)-(\d+)"
    )
    for path in (RUN_OUTPUT / "logs").glob("*.log"):
        for line in path.open(encoding="utf-8", errors="replace"):
            match = oversized_pattern.search(line)
            if match and int(match.group(1)) < len(source):
                oversized.add((int(match.group(1)), int(match.group(2))))

    reconstruction_path = OUT / "oversized_partial_reconstruction_25_79.json"
    reconstructed = __import__("json").loads(reconstruction_path.read_text())
    partial_64k = {
        (item["rollout_id"], item["task_id"])
        for item in reconstructed
        if item["final_dropped"] and item["partial_credit"]
    }
    full_drop_64k = {
        (item["rollout_id"], item["task_id"])
        for item in reconstructed
        if item["final_dropped"] and item["fully_dropped"]
    }

    rows = []
    for rid in source["rollout_id"].astype(int):
        directory = OPENCLAW_LOGS / f"rollout_{rid}"
        instance_ids = {
            int(path.stem.rsplit("-", 1)[1])
            for path in directory.glob("instance-train-*.log")
        }
        timeout_ids = set()
        step_counts = []
        for path in directory.glob("clawgym-rl-*.log"):
            task_id = int(path.stem.rsplit("-", 1)[1])
            text = path.read_text(encoding="utf-8", errors="replace")
            stderr = text.rsplit("--- STDERR ---", 1)[-1] if "--- STDERR ---" in text else ""
            if "embedded run timeout:" in stderr and "timeoutMs=1000000" in stderr:
                timeout_ids.add(task_id)
        for path in directory.glob("instance-train-*.log"):
            for line in path.open(encoding="utf-8", errors="replace"):
                if line.startswith("Steps: "):
                    step_counts.append(int(line.split(":", 1)[1]))
                    break

        universe = set(range(64))
        oversized_ids = {task_id for rollout_id, task_id in oversized if rollout_id == rid}
        missing_ids = universe - instance_ids
        environment_only = missing_ids - timeout_ids - oversized_ids
        clean_ids = universe - oversized_ids - timeout_ids - environment_only
        partial_ids = {task_id for rollout_id, task_id in partial_64k if rollout_id == rid}
        full_drop_ids = {task_id for rollout_id, task_id in full_drop_64k if rollout_id == rid}
        unknown_split_ids = oversized_ids - partial_ids - full_drop_ids
        rows.append({
            "clean": len(clean_ids) / 64,
            "oversized_64k": len(oversized_ids) / 64,
            "oversized_partial_credit": len(partial_ids) / 64,
            "oversized_fully_dropped": len(full_drop_ids) / 64,
            "oversized_unknown_split": len(unknown_split_ids) / 64,
            "timeout_1000s": len(timeout_ids) / 64,
            "oversized_and_timeout": len(oversized_ids & timeout_ids) / 64,
            "environment_error": len(environment_only) / 64,
        })
    return pd.DataFrame(rows)


def main():
    source = pd.read_csv(SOURCE)
    turns = col(source, "openclaw/avg_steps_per_trajectory").replace(0, np.nan)
    submitted = col(source, "openclaw/submitted_trajectories")
    timed_out = col(source, "openclaw/timed_out_trajectories")
    error_recovered = col(source, "openclaw/skipped_recovered")
    error_lost = col(source, "openclaw/skipped_lost")
    dispatched = (submitted + timed_out + error_recovered + error_lost).replace(0, np.nan)
    termination = termination_breakdown(source)

    # Keep the exact ordering used by the former 20 -> 100 colored figure.
    result = pd.DataFrame({
        "step": np.arange(1, len(source) + 1),
        "episode_reward_01": col(source, "openclaw/avg_reward_submitted"),
        "logged_reward_01_turn_weighted": col(source, "rollout/raw_reward"),
        "logged_reward_m11": 2 * col(source, "rollout/raw_reward") - 1,
        "entropy": col(source, "train/entropy_loss"),
        "pg_clipfrac": col(source, "train/pg_clipfrac"),
        "pg_clipfrac_pct": 100 * col(source, "train/pg_clipfrac"),
        "mean_turns": turns,
        "turn_utilization_12": turns / 12,
        "clean_termination_ratio": termination["clean"],
        "oversized_64k_partial_credit_ratio": termination["oversized_partial_credit"],
        "oversized_64k_fully_dropped_ratio": termination["oversized_fully_dropped"],
        "timeout_1000s_ratio": termination["timeout_1000s"],
        "environment_error_termination_ratio": termination["environment_error"],
        "rollout_truncated_turn_weighted": col(source, "rollout/truncated_ratio"),
        "response_length_per_turn": col(source, "rollout/response_len/mean") / turns,
        "total_length_per_turn": col(source, "rollout/total_lengths") / turns,
        "advantages": col(source, "rollout/advantages"),
        "train_loss": col(source, "train/loss"),
        "pg_loss": col(source, "train/pg_loss"),
        "grad_norm_preclip": col(source, "train/grad_norm"),
        "ppo_kl": col(source, "train/ppo_kl"),
        "kl_loss_to_ref": col(source, "train/kl_loss"),
        "logprob_abs_diff": col(source, "train/train_rollout_logprob_abs_diff"),
        # Requested additions.
        "oversized_64k_total_ratio": termination["oversized_64k"],
        "oversized_64k_unknown_split_ratio": termination["oversized_unknown_split"],
        "all_zero_score_group_ratio": col(source, "openclaw/group_all_zero_frac"),
        "all_one_score_group_ratio": col(source, "openclaw/group_all_one_frac"),
        "mixed_score_group_ratio": col(source, "openclaw/group_mixed_frac"),
        "zero_variance_group_ratio": col(source, "openclaw/group_zero_var_frac"),
        # The old segment starts at 20 and ends at 100: update i spans these boundaries.
        "interval_start_step": np.arange(20, 20 + len(source)),
        "interval_end_step": np.arange(21, 21 + len(source)),
        "source_rollout_id": col(source, "rollout_id"),
    })
    result.to_csv(CSV, index=False, float_format="%.10g")

    # Proxy records needed to distinguish partial retention from a full drop are
    # available only for rollout_id 25..79. Export that verifiable window with
    # a literal x-axis; keep the first column named step for har.py.
    exact = result.loc[result["source_rollout_id"].between(25, 79)].copy()
    exact["step"] = exact["source_rollout_id"].astype(int)
    exact.insert(1, "rollout_id", exact.pop("source_rollout_id").astype(int))
    exact = exact.drop(columns=["interval_start_step", "interval_end_step"])
    exact.to_csv(EXACT_CSV, index=False, float_format="%.10g")

    metrics = list(result.columns[1:30])
    colors = ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
    fig, axes = plt.subplots(8, 4, figsize=(22, 34), sharex=True)
    axes = axes.ravel()
    x = result["step"]
    for index, metric in enumerate(metrics):
        ax = axes[index]
        y = result[metric]
        color = colors[index % len(colors)]
        ax.plot(x, y, color=color, alpha=0.28, linewidth=0.9, label="raw")
        ax.plot(x, y.rolling(5, center=True, min_periods=1).mean(), color=color,
                linewidth=2.2, label="centered MA(5)")
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.22)
        if metric == "grad_norm_preclip":
            ax.set_yscale("log")
        elif metric == "ppo_kl":
            ax.set_yscale("symlog", linthresh=1e-5)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    for ax in axes[-4:]:
        ax.set_xlabel("display step (1..80); interval 20 -> 100")
    fig.suptitle("OpenClaw RL — 80-Step Training Curves (Original Color Style)", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(PNG, dpi=170)
    plt.close(fig)

    exact_metrics = [column for column in exact.columns if column not in {"step", "rollout_id"}]
    fig, axes = plt.subplots(8, 4, figsize=(22, 34), sharex=True)
    axes = axes.ravel()
    x = exact["step"]
    for index, metric in enumerate(exact_metrics):
        ax = axes[index]
        y = exact[metric]
        color = colors[index % len(colors)]
        ax.plot(x, y, color=color, alpha=0.28, linewidth=0.9, label="raw")
        ax.plot(x, y.rolling(5, center=True, min_periods=1).mean(), color=color,
                linewidth=2.2, label="centered MA(5)")
        ax.set_title(metric, fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlim(24.5, 79.5)
        ax.set_xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79])
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        if metric == "grad_norm_preclip":
            ax.set_yscale("log")
        elif metric == "ppo_kl":
            ax.set_yscale("symlog", linthresh=1e-5)
    for ax in axes[len(exact_metrics):]:
        ax.axis("off")
    for ax in axes[-4:]:
        ax.set_xlabel("rollout_id / step (25..79); training interval 25 -> 80")
    fig.suptitle(
        "OpenClaw RL — Exact Rollout 25–79 Curves (Training Interval 25→80)",
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(EXACT_PNG, dpi=170)
    plt.close(fig)

    print(CSV)
    print(PNG)
    print(EXACT_CSV)
    print(EXACT_PNG)
    print(result.shape)
    print(exact.shape)
    print(result.columns.tolist())


if __name__ == "__main__":
    main()
