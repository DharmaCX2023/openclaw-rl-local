"""Merge clean subset reruns into the original 200-task eval results."""

import json
from pathlib import Path

REPO = Path("/root/openclaw-rl-local")
TRAIN = REPO / "outputs/clawgym-official-30b-formal-retrain-routing-tis-det-retry1-20260826-0129"
OUTPUT = REPO / "outputs/clawgym-eval-clean-merged-20260828"
REPLACEMENTS = {
    20: ["task_0190"],
    40: ["task_0104", "task_0165"],
    60: ["task_0045", "task_0119", "task_0138", "task_0140"],
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


summaries = {}
for step, expected_ids in REPLACEMENTS.items():
    original_manifest_path = TRAIN / "eval_artifacts" / f"train_step_{step:06d}" / "manifest.json"
    subset_manifest_path = (
        REPO / f"outputs/clawgym-step{step}-infra-subset-eval-20260828"
        / "eval_artifacts" / f"train_step_{step:06d}" / "manifest.json"
    )
    original_manifest = json.loads(original_manifest_path.read_text())
    subset_manifest = json.loads(subset_manifest_path.read_text())
    original_rows = load_rows(Path(original_manifest["results_jsonl"]))
    subset_rows = load_rows(Path(subset_manifest["results_jsonl"]))
    replacements = {row["task_id"]: row for row in subset_rows}
    if set(replacements) != set(expected_ids):
        raise RuntimeError(f"step {step}: unexpected replacement task ids {sorted(replacements)}")

    merged = []
    for row in original_rows:
        replacement = replacements.get(row["task_id"])
        if replacement is None:
            merged.append(row)
            continue
        clean = dict(replacement)
        clean["flat_index"] = row["flat_index"]
        clean["task_index"] = row["task_index"]
        clean["runtime_task_index"] = row["runtime_task_index"]
        clean["replaces_infrastructure_error"] = row.get("error")
        merged.append(clean)

    merged.sort(key=lambda row: row["flat_index"])
    if len(merged) != 200 or any(row.get("status") != "completed" for row in merged):
        raise RuntimeError(f"step {step}: merged result is not 200 completed tasks")
    if any(row.get("workspace", {}).get("error") for row in merged):
        raise RuntimeError(f"step {step}: merged result retains workspace errors")
    if any(not Path(row.get("workspace", {}).get("archive", "")).is_file() for row in merged):
        raise RuntimeError(f"step {step}: merged result has missing workspace archives")

    out_dir = OUTPUT / f"train_step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged))
    summary = {
        "average_reward": sum(float(row["reward"]) for row in merged) / 200,
        "correct_count": sum(bool(row["correct"]) for row in merged),
        "correct_rate": sum(bool(row["correct"]) for row in merged) / 200,
        "n_collected": 200,
        "n_skipped": 0,
        "task_count": 200,
    }
    manifest = {
        "schema_version": 1,
        "semantic_train_step": step,
        "summary": summary,
        "results_jsonl": str(results_path),
        "original_manifest": str(original_manifest_path),
        "subset_manifest": str(subset_manifest_path),
        "replaced_task_ids": expected_ids,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summaries[step] = summary

clean_sources = {
    0: TRAIN / "eval_artifacts/train_step_000000/manifest.json",
    80: REPO / "outputs/clawgym-step80-supplemental-eval-20260827/eval_artifacts/train_step_000080/manifest.json",
    100: TRAIN / "eval_artifacts/train_step_000100/manifest.json",
}
for step, path in clean_sources.items():
    manifest = json.loads(path.read_text())
    source = manifest["summaries"]["clawgym_eval"]
    summaries[step] = {
        "average_reward": source["average_reward"],
        "correct_count": source["correct_count"],
        "correct_rate": source["correct_rate"],
        "n_collected": source["n_collected"],
        "n_skipped": source["n_skipped"],
        "task_count": source["task_count"],
    }

lines = [
    "# ClawGym RL 0–100 Clean Evaluation Summary",
    "",
    "All rows use the formal deterministic n=1 configuration. Infrastructure-skipped rows at steps 20/40/60 were rerun with their original task indices and request seeds, then merged into the original 200-task results.",
    "",
    "| Step | Average reward | Correct | Collected | Skipped |",
    "|---:|---:|---:|---:|---:|",
]
for step in sorted(summaries):
    s = summaries[step]
    lines.append(
        f"| {step} | {s['average_reward']:.6f} | {s['correct_count']}/200 "
        f"| {s['n_collected']}/200 | {s['n_skipped']} |"
    )
(OUTPUT / "EVAL_0_100_CLEAN_SUMMARY.md").write_text("\n".join(lines) + "\n")
print(OUTPUT / "EVAL_0_100_CLEAN_SUMMARY.md")
