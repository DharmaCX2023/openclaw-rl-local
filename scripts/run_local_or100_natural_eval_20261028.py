#!/usr/bin/env python3
"""Evaluate the local OR step-100 checkpoint (iter_0000099, distcp) with
natural sampling, using the exact same protocol as the released-checkpoint
evaluation: 200 ClawGym-Bench tasks, five runs, temperature 0.7, top-p 1.0,
top-k -1, no fixed sampling seed, deterministic inference disabled.

The only differences from run_released_clawgym_rl_ckpt_eval_20261027.py:
- weights load from the Megatron distcp checkpoint via LOAD_CKPT
  (HF_CKPT stays at its default, the local SFT init);
- ports are offset (serial 31-35) to avoid any clash with earlier runs.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path


REPO = Path("/root/openclaw-rl-local")
PYTHON = REPO / ".venv-openclaw/bin/python"
LAUNCHER = REPO / "clawgym-official/.run_step39_eval_sync_20260812-091700.sh"
EVAL_TASKS = REPO / "clawgym-official/data/clawgym_eval"
OR_ROOT = Path(
    "/mnt/humf/openclaw-rl-checkpoints/clawgym-or100-clean-serper-20261012"
)
ITERATION = 99
OUT = Path(
    "/mnt/humf/openclaw-rl-checkpoints/local-or100-natural-eval-20261028"
)
CHECKPOINTS = {"or100-local": OR_ROOT / f"iter_{ITERATION:07d}"}
REMOTE_HOST = "159.138.50.147"
REMOTE_KEY = "/root/openclaw_cpu_key"


def checkpoint_ready(path: Path) -> tuple[bool, str]:
    if not path.is_dir():
        return False, "missing iteration directory"
    shards = sorted(path.glob("*.distcp"))
    if not shards:
        return False, "no distcp shards"
    empty = [s.name for s in shards if s.stat().st_size == 0]
    if empty:
        return False, f"{len(empty)}/{len(shards)} shards empty"
    return True, f"{len(shards)} distcp shards ready"


def remote_command(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-i",
            REMOTE_KEY,
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=8",
            f"root@{REMOTE_HOST}",
            command,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def clean_stopped_remote_containers() -> None:
    """Remove stopped containers and ClawGym containers owned by dead local PIDs."""
    listed = remote_command("docker ps --format '{{.Names}}'")
    if listed.returncode != 0:
        raise RuntimeError(f"remote Docker listing failed: {listed.stdout.strip()}")
    orphaned: list[str] = []
    for name in listed.stdout.splitlines():
        if not name.startswith("clawgym-rl-"):
            continue
        parts = name.split("-")
        if len(parts) < 4 or not parts[3].isdigit():
            continue
        if not Path(f"/proc/{parts[3]}").exists():
            orphaned.append(name)
    quoted = " ".join(json.dumps(name) for name in orphaned)
    command = (
        "ids=$(docker ps -aq -f status=exited -f status=created -f status=dead); "
        'if [ -n "$ids" ]; then docker rm -f $ids >/dev/null || true; fi; '
        f"if [ -n {json.dumps(quoted)} ]; then docker rm -f {quoted} >/dev/null || true; fi; "
        'printf "removed_orphans=%s running_clawgym=%s\\n" '
        f'"{len(orphaned)}" "$(docker ps --format "{{{{.Names}}}}" | grep -c "^clawgym-rl-" || true)"'
    )
    result = remote_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"remote Docker cleanup failed: {result.stdout.strip()}")
    print(f"DOCKER_CLEANUP {result.stdout.strip()}", flush=True)


def task_ids() -> list[str]:
    ids = sorted(path.name for path in EVAL_TASKS.iterdir() if path.is_dir())
    if len(ids) != 200:
        raise RuntimeError(f"expected 200 eval tasks, found {len(ids)}")
    return ids


def valid_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        metrics = result["metrics"]
        return (
            result["exit_code"] == 0
            and metrics["eval/clawgym_eval-task_count"] == 200
            and metrics["eval/clawgym_eval-n_collected"] == 200
            and metrics["eval/clawgym_eval-n_skipped"] == 0
        )
    except Exception:
        return False


def run_environment(
    model: str,
    repetition: int,
    phase: str,
    run_dir: Path,
    selected_tasks: list[str] | None,
) -> dict[str, str]:
    serial = 30 + repetition
    env = os.environ.copy()
    for key in (
        "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE",
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "NVIDIA_TF32_OVERRIDE",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "NCCL_ALGO",
        "NCCL_PROTO",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "HF_CKPT",
        "REF_LOAD",
    ):
        env.pop(key, None)
    env.update(
        {
            "RUN_STAMP": f"local-or100-{model}-r{repetition}-{phase}",
            "EXP_NAME": f"local-or100-{model}-r{repetition}",
            "SAVE_CKPT": str(run_dir / "save"),
            "LOAD_CKPT": str(run_dir / "checkpoint_view"),
            "OPENCLAW_DISABLE_CHECKPOINT_SAVE": "1",
            "NUM_ROLLOUT": "1",
            "ROLLOUT_BATCH_SIZE": "8",
            "N_SAMPLES_PER_PROMPT": "8",
            "EVAL_INTERVAL": "1000",
            "SKIP_EVAL_BEFORE_TRAIN": "0",
            "OPENCLAW_EVAL_NUM_WORKERS": "8",
            "OPENCLAW_EVAL_N_SAMPLES": "1",
            "OPENCLAW_EVAL_TASKS": str(EVAL_TASKS),
            "OPENCLAW_TRAIN_NUM_WORKERS": "8",
            "OPENCLAW_TRAIN_SHUFFLE_SEED": "0",
            "OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS": "0",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "0",
            "MEGATRON_DETERMINISTIC_MODE": "0",
            "SGLANG_ENABLE_CUDA_GRAPH": "1",
            "EVAL_TEMPERATURE": "0.7",
            "EVAL_TOP_P": "1.0",
            "EVAL_TOP_K": "-1",
            "ROLLOUT_TEMPERATURE": "0.7",
            "ROLLOUT_TOP_P": "1.0",
            "ROLLOUT_TOP_K": "-1",
            "EVAL_MAX_RESPONSE_LEN": "8192",
            "EVAL_MAX_CONTEXT_LEN": "65536",
            "OPENCLAW_MAX_RESPONSE_TOKENS": "8192",
            "OPENCLAW_TRAJECTORY_MAX_TOKENS": "65536",
            "OPENCLAW_CONTEXT_WINDOW": "65536",
            "OPENCLAW_MAX_AGENT_STEPS": "96",
            "OPENCLAW_MAX_COMPACTION_EVENTS": "8",
            "OPENCLAW_REMOTE_BASE": (
                f"/root/clawgym_eval_local_or100_r{repetition}_{phase}"
            ),
            "OPENCLAW_REMOTE_PROXY_PORT": str(60000 + serial),
            "PORT": str(37000 + serial),
            "OPENCLAW_GATEWAY_PORT_BASE": str(20000 + serial * 1000),
            "OPENCLAW_GATEWAY_PORT_MAX_SLOTS": "256",
            "OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY": "8",
            "OPENCLAW_REMOTE_SSH_POOL_SIZE": "8",
            "OPENCLAW_RECORD_FILE": str(run_dir / "official_proxy_records.jsonl"),
            "TENSORBOARD_DIR": str(run_dir / "tensorboard"),
        }
    )
    if selected_tasks is None:
        env["OPENCLAW_EVAL_TASK_IDS"] = ""
        env["OPENCLAW_EVAL_TASK_INDEX_MAP"] = ""
    else:
        mapping = {task_id: int(task_id.rsplit("_", 1)[1]) for task_id in selected_tasks}
        env["OPENCLAW_EVAL_TASK_IDS"] = ",".join(selected_tasks)
        env["OPENCLAW_EVAL_TASK_INDEX_MAP"] = json.dumps(mapping, separators=(",", ":"))
    return env


def launch(
    model: str,
    repetition: int,
    phase: str,
    selected_tasks: list[str] | None,
) -> tuple[int, list[dict]]:
    clean_stopped_remote_containers()
    run_dir = OUT / model / f"run{repetition}" / phase
    (run_dir / "save").mkdir(parents=True, exist_ok=True)
    view = run_dir / "checkpoint_view"
    view.mkdir(parents=True, exist_ok=True)
    link = view / f"iter_{ITERATION:07d}"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(CHECKPOINTS[model])
    (view / "latest_checkpointed_iteration.txt").write_text(f"{ITERATION}\n")
    log_path = run_dir / "run.log"
    env = run_environment(model, repetition, phase, run_dir, selected_tasks)
    print(
        f"EVAL_START model={model} run={repetition} phase={phase} "
        f"tasks={200 if selected_tasks is None else len(selected_tasks)} "
        f"at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    candidates = list((run_dir / "save").glob("eval_artifacts/**/results.jsonl"))
    rows: list[dict] = []
    if len(candidates) == 1:
        rows = [
            json.loads(line)
            for line in candidates[0].read_text().splitlines()
            if line.strip()
        ]
    print(
        f"EVAL_END model={model} run={repetition} phase={phase} "
        f"code={proc.returncode} rows={len(rows)} "
        f"completed={sum(row.get('status') == 'completed' for row in rows)} "
        f"at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )
    return proc.returncode, rows


def write_result(
    model: str, repetition: int, rows_by_task: dict[str, dict]
) -> None:
    ordered_ids = task_ids()
    rows = [rows_by_task[task_id] for task_id in ordered_ids]
    rewards = [float(row["reward"]) for row in rows]
    average = sum(rewards) / len(rewards)
    correct = sum(reward > 0.5 for reward in rewards)
    run_root = OUT / model / f"run{repetition}"
    (run_root / "merged_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    result = {
        "model": model,
        "repetition": repetition,
        "checkpoint": str(CHECKPOINTS[model]),
        "iteration": ITERATION,
        "exit_code": 0,
        "temperature": 0.7,
        "top_p": 1.0,
        "top_k": -1,
        "deterministic_inference": False,
        "fixed_sampling_seed": False,
        "metrics": {
            "eval/clawgym_eval-avg_reward": average,
            "eval/clawgym_eval-correct_count": correct,
            "eval/clawgym_eval-correct_rate": correct / 200,
            "eval/clawgym_eval-task_count": 200,
            "eval/clawgym_eval-n_collected": 200,
            "eval/clawgym_eval-n_skipped": 0,
            "eval/clawgym_eval-avg_reward_collected_only": average,
            "eval/clawgym_eval": average,
        },
    }
    (run_root / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


def aggregate() -> None:
    lines = [
        "# Local OR step-100 (iter_0000099): natural-sampling evaluation",
        "",
        "200 tasks, temperature 0.7, top-p 1.0, top-k -1, "
        "deterministic inference off, no fixed sampling seed.",
        "",
        "| model | run1 | run2 | run3 | run4 | run5 | mean | sample stdev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in CHECKPOINTS:
        values: list[float | None] = []
        for repetition in range(1, 6):
            path = OUT / model / f"run{repetition}/result.json"
            if valid_result(path):
                data = json.loads(path.read_text())
                values.append(data["metrics"]["eval/clawgym_eval-avg_reward"])
            else:
                values.append(None)
        present = [value for value in values if value is not None]
        cells = " | ".join(
            f"{value:.6f}" if value is not None else "" for value in values
        )
        mean = f"{statistics.mean(present):.6f}" if present else ""
        std = f"{statistics.stdev(present):.6f}" if len(present) > 1 else ""
        lines.append(f"| {model} | {cells} | {mean} | {std} |")
    (OUT / "aggregate_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def evaluate_one(model: str, repetition: int) -> None:
    result_path = OUT / model / f"run{repetition}/result.json"
    if valid_result(result_path):
        print(f"EVAL_SKIP model={model} run={repetition} already complete", flush=True)
        return

    expected = set(task_ids())
    completed: dict[str, dict] = {}
    for attempt in range(1, 4):
        phase = f"full-attempt{attempt}"
        _, rows = launch(model, repetition, phase, None)
        for row in rows:
            if (
                row.get("task_id") in expected
                and row.get("status") == "completed"
                and row.get("reward") is not None
            ):
                completed[row["task_id"]] = row
        if completed:
            break
    if not completed:
        raise RuntimeError(f"{model} run {repetition}: no completed full-eval rows")

    for patch_round in range(1, 5):
        missing = sorted(expected - completed.keys())
        if not missing:
            write_result(model, repetition, completed)
            return
        print(
            f"EVAL_PATCH model={model} run={repetition} "
            f"round={patch_round} missing={len(missing)}",
            flush=True,
        )
        _, rows = launch(model, repetition, f"patch{patch_round}", missing)
        for row in rows:
            if (
                row.get("task_id") in expected
                and row.get("status") == "completed"
                and row.get("reward") is not None
            ):
                completed[row["task_id"]] = row
    raise RuntimeError(
        f"{model} run {repetition}: {len(expected - completed.keys())} tasks "
        "still missing after selective retries"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pending = [
        (model, repetition)
        for model in CHECKPOINTS
        for repetition in range(1, 6)
    ]
    while pending:
        ready = [
            item
            for item in pending
            if checkpoint_ready(CHECKPOINTS[item[0]])[0]
        ]
        states = {
            name: checkpoint_ready(path) for name, path in CHECKPOINTS.items()
        }
        print(
            "CKPT_STATUS "
            + " ".join(
                f"{name}={'ready' if ok else detail}"
                for name, (ok, detail) in states.items()
            ),
            flush=True,
        )
        if not ready:
            time.sleep(30)
            continue
        model, repetition = ready[0]
        evaluate_one(model, repetition)
        pending.remove((model, repetition))
        aggregate()
    print(
        f"QUEUE_DONE at={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
