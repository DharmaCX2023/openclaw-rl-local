#!/usr/bin/env python3
"""Measure the real remote OpenClaw Gateway startup path without model calls."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_rollout as rollout  # noqa: E402


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _stratified_tasks(task_root: Path, count: int) -> list[tuple[rollout.TaskSpec, int]]:
    candidates = []
    for entry in sorted(task_root.glob("*/data_entry.json")):
        task = rollout._load_task_spec(entry)
        input_dir = task.task_dir / "input_files"
        candidates.append((task, _tree_bytes(input_dir) if input_dir.is_dir() else 0))
    if not candidates:
        raise RuntimeError(f"no tasks found under {task_root}")
    candidates.sort(key=lambda item: (item[1], item[0].task_id))
    count = min(count, len(candidates))
    if count == 1:
        return [candidates[len(candidates) // 2]]
    return [
        candidates[round(index * (len(candidates) - 1) / (count - 1))]
        for index in range(count)
    ]


def _run_one(task: rollout.TaskSpec, input_bytes: int, task_index: int, rl_port: int) -> dict:
    runtime = None
    attempts = max(1, int(os.environ.get("OPENCLAW_REMOTE_START_ATTEMPTS", "3")))
    started = time.perf_counter()
    try:
        for attempt in range(1, attempts + 1):
            attempt_started = time.perf_counter()
            try:
                semaphore = rollout._remote_docker_start_semaphore()
                queue_started = time.perf_counter()
                with semaphore:
                    rollout._record_train_startup_timing(
                        "queue_wait", time.perf_counter() - queue_started
                    )
                    slot_started = time.perf_counter()
                    try:
                        runtime = rollout._start_container(
                            task,
                            rollout_id=0,
                            task_index=task_index,
                            rl_port=rl_port,
                            session_id=f"startup-measure-{task_index}",
                            model_id=f"main__startup-measure-{task_index}",
                            is_train=True,
                        )
                        health_started = time.perf_counter()
                        try:
                            rollout._wait_for_chat_ready(runtime)
                        finally:
                            rollout._record_train_startup_timing(
                                "health_ready", time.perf_counter() - health_started
                            )
                    finally:
                        rollout._record_train_startup_timing(
                            "slot_held", time.perf_counter() - slot_started
                        )
                rollout._increment_train_startup_metric("successful_attempts")
                break
            except Exception:
                if runtime is not None:
                    rollout._stop_container(runtime, None)
                    runtime = None
                if attempt >= attempts:
                    rollout._increment_train_startup_metric("failed_attempts")
                    raise
                rollout._increment_train_startup_metric("retries")
            finally:
                rollout._increment_train_startup_metric("attempts")
                rollout._record_train_startup_timing(
                    "attempt", time.perf_counter() - attempt_started
                )
        profile_summary = None
        if os.environ.get("OPENCLAW_STARTUP_CPU_PROFILE") == "1":
            assert runtime is not None and runtime.remote_root is not None
            rollout._remote_run(
                f"docker stop -t 20 {shlex.quote(runtime.container_name)} >/dev/null",
                timeout=30,
            )
            profile_path = f"{runtime.remote_root}/workspace/startup.cpuprofile"
            analyzer = b"""import json, sys\nfrom collections import defaultdict\np=json.load(open(sys.argv[1], encoding='utf-8'))\na=defaultdict(int)\nfor n in p.get('nodes', []):\n c=n.get('callFrame', {})\n a[(c.get('functionName') or '(anonymous)', c.get('url') or '')] += int(n.get('hitCount', 0))\nprint(json.dumps([{'samples':v,'function':k[0],'url':k[1]} for k,v in sorted(a.items(), key=lambda x:x[1], reverse=True)[:40]]))\n"""
            analyzed = rollout._remote_run(
                f"python3 - {shlex.quote(profile_path)}",
                input_data=analyzer,
                timeout=30,
            )
            profile_summary = json.loads(analyzed.stdout.decode("utf-8"))
        return {
            "task_id": task.task_id,
            "input_bytes": input_bytes,
            "ready_seconds_including_queue": time.perf_counter() - started,
            "cpu_profile_top": profile_summary,
        }
    finally:
        if runtime is not None:
            rollout._stop_container(runtime, None)


def _summarize(values: list[float]) -> dict[str, float]:
    values = sorted(float(value) for value in values)
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    count = len(values)
    return {
        "avg": sum(values) / count,
        "p50": values[min(count - 1, int(0.50 * count))],
        "p90": values[min(count - 1, int(0.90 * count))],
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rl-port", type=int, default=30699)
    parser.add_argument(
        "--task-root",
        type=Path,
        default=REPO_ROOT / "clawgym-official/data/clawgym_train",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--disable-plugins",
        action="store_true",
        help="Diagnostic ablation: write plugins.enabled=false into the generated config.",
    )
    parser.add_argument(
        "--disable-ui-canvas",
        action="store_true",
        help="Diagnostic ablation: disable the unused Control UI and Canvas host.",
    )
    parser.add_argument(
        "--cpu-profile",
        action="store_true",
        help="Collect a diagnostic Node CPU profile and return its hottest frames.",
    )
    parser.add_argument(
        "--force-built-plugins",
        action="store_true",
        help="Diagnostic ablation: force /app/dist/extensions as bundled plugin root.",
    )
    args = parser.parse_args()

    if args.disable_plugins or args.disable_ui_canvas:
        original_build_gateway_config = rollout._build_gateway_config

        def build_gateway_config_with_ablations(runtime, rl_port):
            original_build_gateway_config(runtime, rl_port)
            config_path = runtime.config_dir / "openclaw.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if args.disable_plugins:
                config["plugins"] = {"enabled": False}
            if args.disable_ui_canvas:
                config.setdefault("gateway", {}).setdefault("controlUi", {})[
                    "enabled"
                ] = False
                config["canvasHost"] = {"enabled": False}
            config_path.write_text(json.dumps(config), encoding="utf-8")

        rollout._build_gateway_config = build_gateway_config_with_ablations

    if args.cpu_profile:
        original_write_secret_env_file = rollout._write_secret_env_file

        def write_secret_env_file_with_cpu_profile(
            container_name, credentials, runtime_env=None
        ):
            path = original_write_secret_env_file(
                container_name, credentials, runtime_env
            )
            with path.open("a", encoding="utf-8") as file_handle:
                file_handle.write(
                    "NODE_OPTIONS=--cpu-prof --cpu-prof-dir=/root/.openclaw/workspace "
                    "--cpu-prof-name=startup.cpuprofile\n"
                )
            return path

        rollout._write_secret_env_file = write_secret_env_file_with_cpu_profile
        os.environ["OPENCLAW_STARTUP_CPU_PROFILE"] = "1"

    if args.force_built_plugins:
        os.environ["OPENCLAW_BUNDLED_PLUGINS_DIR"] = "/app/dist/extensions"

    selected = _stratified_tasks(args.task_root, args.tasks)
    with rollout._TRAIN_METRICS_LOCK:
        rollout._TRAIN_METRICS.clear()
    wall_started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(selected))) as executor:
        futures = {
            executor.submit(_run_one, task, size, index, args.rl_port): task.task_id
            for index, (task, size) in enumerate(selected)
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - wall_started

    with rollout._TRAIN_METRICS_LOCK:
        raw = dict(rollout._TRAIN_METRICS)
    phases = {}
    for phase in (
        "queue_wait",
        "port_allocate",
        "workspace_prepare",
        "archive_build",
        "upload_extract",
        "docker_run",
        "health_ready",
        "slot_held",
        "attempt",
    ):
        phases[phase] = _summarize(raw.get(f"startup_{phase}_seconds", []))
    slot_total = sum(raw.get("startup_slot_held_seconds", []))
    component_totals = {
        phase: sum(raw.get(f"startup_{phase}_seconds", []))
        for phase in (
            "port_allocate",
            "workspace_prepare",
            "archive_build",
            "upload_extract",
            "docker_run",
            "health_ready",
        )
    }
    payload = {
        "task_count": len(selected),
        "worker_count": min(args.workers, len(selected)),
        "start_concurrency": int(
            os.environ.get("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "4")
        ),
        "plugins_disabled": args.disable_plugins,
        "ui_canvas_disabled": args.disable_ui_canvas,
        "cpu_profile_enabled": args.cpu_profile,
        "forced_built_plugins": args.force_built_plugins,
        "wall_seconds": wall_seconds,
        "attempts": raw.get("startup_attempts", 0),
        "successful_attempts": raw.get("startup_successful_attempts", 0),
        "failed_attempts": raw.get("startup_failed_attempts", 0),
        "retries": raw.get("startup_retries", 0),
        "archive_bytes_total": raw.get("startup_archive_bytes", 0),
        "archive_bytes_avg": (
            raw.get("startup_archive_bytes", 0) / raw.get("startup_archive_count", 1)
            if raw.get("startup_archive_count", 0)
            else 0.0
        ),
        "phase_seconds": phases,
        "slot_component_share": {
            phase: total / slot_total if slot_total else 0.0
            for phase, total in component_totals.items()
        },
        "input_bytes": _summarize([size for _, size in selected]),
        "tasks": sorted(results, key=lambda item: item["task_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
