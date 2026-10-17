#!/usr/bin/env python3
"""Run the no-training 8 tasks x 8 samples gate through the production RL path."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = REPO_ROOT / "clawgym-official"
sys.path.insert(0, str(OFFICIAL_ROOT))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_rollout as rollout
from clawgym_rl_api_server import ClawGymRLAPIServer


DEFAULT_MODEL = Path(
    "/mnt/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-sft-iter11127"
)
DEFAULT_PYTHON = REPO_ROOT / ".venv-openclaw/bin/python"
DEFAULT_TASKS = OFFICIAL_ROOT / "data/clawgym_train"
PINNED_IMAGE = (
    "clawgym-rl@sha256:"
    "33a30ab107cb7fef6a84f17b3fb70a0e849654e9201c61dc9f20baeac21385a3"
)


def _wait_http(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last_error = "no response"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = type(exc).__name__
        time.sleep(2)
    raise TimeoutError(f"service did not become ready: {last_error}")


def _start_sglang(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen | None, Any | None]:
    if args.backend_url:
        _wait_http(args.backend_url.rstrip("/") + "/health", 30)
        return None, None

    log_handle = (args.output_dir / "sglang.log").open("wb")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    command = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        args.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.backend_port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--context-length",
        "65536",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--random-seed",
        str(args.engine_seed),
        "--trust-remote-code",
        "--tool-call-parser",
        "qwen25",
    ]
    if not args.enable_cuda_graph:
        command.append("--disable-cuda-graph")
    if getattr(args, "deterministic", False):
        command.append("--enable-deterministic-inference")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    try:
        _wait_http(f"http://127.0.0.1:{args.backend_port}/health", args.serve_timeout)
    # KeyboardInterrupt/SystemExit can arrive while a detached supervisor is
    # shutting down during model startup. They must not orphan the SGLang
    # process group and its GPU scheduler children.
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        log_handle.close()
        raise
    return process, log_handle


def _stop_process(process: subprocess.Popen | None, timeout: int = 30) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _start_reverse_tunnel(args: argparse.Namespace) -> subprocess.Popen:
    command = [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
        "-N",
        "-R",
        f"127.0.0.1:{args.reverse_port}:127.0.0.1:{args.proxy_port}",
        f"{args.remote_user}@{args.remote_host}",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    if process.poll() is not None:
        stderr = process.stderr.read().decode(errors="replace")
        raise RuntimeError(f"reverse tunnel failed: {stderr[:500]}")
    return process


def _configure_environment(args: argparse.Namespace) -> None:
    values = {
        "HOST": "127.0.0.1",
        "PORT": str(args.proxy_port),
        "SGLANG_API_KEY": "",
        "SERVED_MODEL_NAME": args.served_model,
        # FlashInfer's all-reduce fusion currently uses a process-global
        # rank/op-id socket path. Two TP replicas on the same host can then
        # collide even though they are separate SGLang engines. Keep the
        # production-safe setting explicit for TP2 x DP2 gates.
        "SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION": "1",

        "TOOL_CALL_PARSER": "qwen25",
        "OPENCLAW_CONTEXT_WINDOW": "65536",
        "OPENCLAW_TRAJECTORY_MAX_TOKENS": "65536",
        "OPENCLAW_MAX_RESPONSE_TOKENS": "8192",
        "OPENCLAW_TRAIN_MAX_TURNS": "1",
        "OPENCLAW_TRAIN_NUM_WORKERS": str(args.parallel),
        "OPENCLAW_CHAT_TURN_TIMEOUT": str(args.rollout_timeout),
        "OPENCLAW_TRAIN_TASKS": str(args.tasks),
        "OPENCLAW_TRAIN_SHUFFLE_SEED": str(args.task_seed),
        "OPENCLAW_REMOTE_HOST": args.remote_host,
        "OPENCLAW_REMOTE_USER": args.remote_user,
        "OPENCLAW_REMOTE_SSH_KEY": str(args.ssh_key),
        "OPENCLAW_REMOTE_BASE": args.remote_base,
        "OPENCLAW_REMOTE_PROXY_PORT": str(args.reverse_port),
        "OPENCLAW_REMOTE_SSH_POOL_SIZE": str(args.ssh_pool_size),
        "OPENCLAW_GATEWAY_IMAGE": PINNED_IMAGE,
        "OPENCLAW_GATEWAY_PORT_BASE": str(args.gateway_port_base),
        "OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET": str(args.gateway_train_offset),
        "OPENCLAW_GATEWAY_PORT_STRIDE": str(args.gateway_port_stride),
        "OPENCLAW_GATEWAY_PORT_MAX_SLOTS": "1000",
        "CLAWGYM_SERPER_SKILL_DIR": str(
            REPO_ROOT / "sft_contract/recovered_skills/serper"
        ),
        "OPENCLAW_RECORD_ENABLED": "1",
        "OPENCLAW_TOKEN_LEDGER_MODE": "enforce",
        "OPENCLAW_RECORD_FILE": str(args.output_dir / "official_proxy_records.jsonl"),
    }
    if getattr(args, "deterministic", False):
        values["OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE"] = str(
            args.sampling_seed_base
        )
    if getattr(args, "deterministic_tool_call_ids", False):
        values["OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS"] = "1"
    if getattr(args, "eval_mode", False):
        values.update(
            {
                "OPENCLAW_EVAL_TASKS": str(args.tasks),
                "OPENCLAW_EVAL_MAX_TURNS": "1",
                "OPENCLAW_EVAL_NUM_WORKERS": str(args.parallel),
                "OPENCLAW_EVAL_N_SAMPLES": str(args.samples_per_task),
                "OPENCLAW_EVAL_TASK_INDEX_OFFSET": str(getattr(args, "eval_task_index_offset", 0)),
            }
        )
        if getattr(args, "eval_task_ids", ""):
            values["OPENCLAW_EVAL_TASK_IDS"] = args.eval_task_ids
        if getattr(args, "rubric_dataset", None):
            values["OPENCLAW_RUBRIC_DATASET"] = str(args.rubric_dataset)
        if getattr(args, "rubric_archive_dir", None):
            values["OPENCLAW_RUBRIC_ARCHIVE_DIR"] = str(args.rubric_archive_dir)
    os.environ.update(values)
    # clawgym_rl_rollout reads this value at import time. This gate imports the
    # module before parsing its CLI arguments, so updating os.environ alone
    # would silently leave the default 900-second timeout active.
    rollout._CHAT_TURN_TIMEOUT_SECONDS = args.rollout_timeout


def _select_batch(args: argparse.Namespace) -> list[tuple[int, rollout.TaskSpec]]:
    sources = rollout._task_sources_from_env("OPENCLAW_TRAIN_TASKS")
    tasks = [task for source in sources for task in source.tasks]
    batch_args = SimpleNamespace(
        rollout_batch_size=args.task_count,
        n_samples_per_prompt=args.samples_per_task,
    )
    batch = rollout._train_batch(batch_args, tasks, args.rollout_id)
    counts = Counter(task.task_id for _, task in batch)
    if len(counts) != args.task_count:
        raise AssertionError(
            f"expected {args.task_count} unique tasks, got {len(counts)}"
        )
    if set(counts.values()) != {args.samples_per_task}:
        raise AssertionError(f"unexpected samples per task: {dict(counts)}")
    return batch


def _write_selection(
    output_dir: Path, batch: list[tuple[int, rollout.TaskSpec]]
) -> None:
    rows = []
    for task_id in dict.fromkeys(task.task_id for _, task in batch):
        task = next(item for _, item in batch if item.task_id == task_id)
        rows.append(
            {
                "task_id": task.task_id,
                "task_dir": str(task.task_dir),
                "prompt": task.user_query,
            }
        )
    (output_dir / "selected_tasks.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _eval_sources(args: argparse.Namespace) -> list[rollout.TaskSource]:
    sources = rollout._task_sources_from_env("OPENCLAW_EVAL_TASKS")
    task_count = sum(len(source.tasks) for source in sources)
    if task_count != args.task_count:
        raise AssertionError(
            f"expected {args.task_count} eval tasks, discovered {task_count}"
        )
    return sources


def _write_eval_selection(
    output_dir: Path, sources: list[rollout.TaskSource]
) -> None:
    rows = [
        {
            "dataset": source.dataset_name,
            "task_id": task.task_id,
            "task_dir": str(task.task_dir),
            "prompt": task.user_query,
        }
        for source in sources
        for task in source.tasks
    ]
    (output_dir / "selected_tasks.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_eval_report(
    args: argparse.Namespace,
    sources: list[rollout.TaskSource],
    eval_result: dict[str, dict],
    elapsed_seconds: float,
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    datasets = {}
    total_runs = 0
    total_skipped = 0
    all_rewards = []
    for source in sources:
        data = eval_result[source.dataset_name]
        rewards = [float(value) for value in data["rewards"]]
        if len(rewards) != len(source.tasks):
            raise AssertionError(
                f"reward count mismatch for {source.dataset_name}: "
                f"{len(rewards)} != {len(source.tasks)}"
            )
        rows = [
            {"task_id": task.task_id, "reward": reward}
            for task, reward in zip(source.tasks, rewards, strict=True)
        ]
        n_skipped = int(data.get("n_skipped", 0))
        total_runs += len(rows) * args.samples_per_task
        total_skipped += n_skipped
        all_rewards.extend(rewards)
        datasets[source.dataset_name] = {
            "tasks": rows,
            "task_count": len(rows),
            "correct_count": sum(reward > 0.5 for reward in rewards),
            "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "n_collected": int(data.get("n_collected", 0)),
            "n_skipped": n_skipped,
            "avg_reward_collected_only": float(
                data.get("avg_reward_collected_only", 0.0)
            ),
        }

    audit_steps = [step for audit in audits for step in audit.get("steps", [])]
    report = {
        "mode": "full-eval",
        "model": str(args.model),
        "tasks": str(args.tasks),
        "rollout_id": args.rollout_id,
        "n_samples_per_task": args.samples_per_task,
        "datasets": datasets,
        "summary": {
            "task_count": len(all_rewards),
            "total_runs": total_runs,
            "n_skipped": total_skipped,
            "average_reward": (
                sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
            ),
            "correct_count": sum(reward > 0.5 for reward in all_rewards),
            "elapsed_seconds": elapsed_seconds,
            "trajectories_per_minute": (
                total_runs * 60 / elapsed_seconds if elapsed_seconds > 0 else None
            ),
            "trajectory_audits": len(audits),
            "react_steps": len(audit_steps),
            "max_segment_tokens": max(
                (
                    int(step.get("segment_tokens_after_response", 0))
                    for step in audit_steps
                ),
                default=0,
            ),
            "context_splits": sum(
                bool(step.get("split_performed")) for step in audit_steps
            ),
            "compaction_events": sum(
                bool(step.get("compaction_detected")) for step in audit_steps
            ),
        },
        "topology": {
            "gpus": args.gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "cuda_graph_enabled": args.enable_cuda_graph,
            "parallel": args.parallel,
        },
        "passed": total_runs > 0 and total_skipped == 0,
    }
    (args.output_dir / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _recover_failed_session(
    proxy_port: int,
    rollout_id: int,
    task_index: int,
    group_index: int,
) -> str | None:
    session_id = f"task-{rollout_id}-{task_index}"
    try:
        rollout._submit_reward(proxy_port, session_id, 0.0, group_index)
        return None
    except Exception as exc:
        return _safe_exception_text(exc)


def _safe_exception_text(exc: Exception) -> str:
    """Return diagnostics without serializing commands that contain tokens."""
    if isinstance(exc, subprocess.CalledProcessError):
        return (
            "CalledProcessError: command failed with exit status "
            f"{exc.returncode}"
        )
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"TimeoutExpired: command exceeded {exc.timeout} seconds"
    return f"{type(exc).__name__}: {exc}"


def _run_batch(
    args: argparse.Namespace,
    batch: list[tuple[int, rollout.TaskSpec]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runner_args = SimpleNamespace(save=str(args.output_dir))
    results: list[dict[str, Any]] = []
    batch_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_item = {}
        submitted_at = {}
        for task_index, (group_index, task) in enumerate(batch):
            future = executor.submit(
                rollout._run_single_task,
                runner_args,
                task,
                args.rollout_id,
                task_index,
                "main",
                1,
                args.proxy_port,
                group_index,
                "gate",
            )
            future_to_item[future] = (task_index, group_index, task)
            submitted_at[future] = time.monotonic()

        for future in as_completed(future_to_item):
            task_index, group_index, task = future_to_item[future]
            elapsed_seconds = time.monotonic() - submitted_at[future]
            try:
                result = future.result()
                row = {
                    "task_index": task_index,
                    "group_index": group_index,
                    "task_id": task.task_id,
                    "status": "ok",
                    "reward": result.reward,
                    "react_steps": len(result.react_steps),
                    "completion_seconds": round(elapsed_seconds, 3),
                }
            except Exception as exc:
                recovery_error = _recover_failed_session(
                    args.proxy_port,
                    args.rollout_id,
                    task_index,
                    group_index,
                )
                row = {
                    "task_index": task_index,
                    "group_index": group_index,
                    "task_id": task.task_id,
                    "status": "error",
                    "error": _safe_exception_text(exc),
                    "recovery_error": recovery_error,
                    "completion_seconds": round(elapsed_seconds, 3),
                }
            results.append(row)
            print(
                f"[{len(results)}/{len(batch)}] {task.task_id} "
                f"group={group_index} status={row['status']} "
                f"elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )
    batch_elapsed = time.monotonic() - batch_started
    performance = {
        "parallel": args.parallel,
        "batch_elapsed_seconds": round(batch_elapsed, 3),
        "logical_trajectories_per_minute": round(len(batch) * 60 / batch_elapsed, 3),
        "sglang_topology": {
            "gpus": args.gpus,
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "cuda_graph_enabled": args.enable_cuda_graph,
            "mem_fraction_static": args.mem_fraction_static,
        },
    }
    return sorted(results, key=lambda item: item["task_index"]), performance


def _read_audits(record_file: Path) -> list[dict[str, Any]]:
    audits = []
    for line_number, line in enumerate(
        record_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid JSONL at line {line_number}") from exc
        if record.get("record_type") == "trajectory_audit":
            audits.append(record)
    return audits


def _drain_samples(output_queue: queue.Queue) -> list[Any]:
    samples = []
    while not output_queue.empty():
        group_index, batch = output_queue.get_nowait()
        for sample in batch:
            assert sample.group_index == group_index
            samples.append(sample)
    return samples


def _contains_missing_serper_script(output_dir: Path) -> bool:
    canonical_script = "/root/.openclaw/skills/serper/scripts/search.py"
    for path in (output_dir / "openclaw_logs").rglob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        # Only a missing canonical mounted path is a harness failure. Model
        # mistakes such as workspace/~/.openclaw/... must remain learnable.
        for step_block in text.split("--- step"):
            # Keep the path and OS error on the same line. A command may read
            # the canonical source but fail because a different destination
            # directory is missing; the previous block-wide conjunction
            # incorrectly classified that as a missing mounted skill.
            if any(
                canonical_script in line and "no such file" in line
                for line in step_block.splitlines()
            ):
                return True
    return False


def _build_report(
    args: argparse.Namespace,
    batch: list[tuple[int, rollout.TaskSpec]],
    results: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    samples: list[Any],
    performance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = args.task_count * args.samples_per_task
    audit_sessions = {audit["session_id"] for audit in audits}
    sample_sessions = {sample.metadata.get("session_id") for sample in samples}
    selected_counts = Counter(task.task_id for _, task in batch)
    submitted_segments = sum(
        int(audit.get("submitted_segment_count", 0)) for audit in audits
    )
    steps = [step for audit in audits for step in audit.get("steps", [])]
    performance = dict(performance or {})
    generated_tokens = sum(int(step.get("response_tokens", 0)) for step in steps)
    performance["generated_response_tokens"] = generated_tokens
    elapsed_seconds = float(performance.get("batch_elapsed_seconds", 0))
    performance["generated_response_tokens_per_second"] = (
        round(generated_tokens / elapsed_seconds, 3) if elapsed_seconds > 0 else None
    )
    sglang_totals = {
        name: sum(
            audit.get("sglang_stats", {}).get(name, 0) for audit in audits
        )
        for name in (
            "requests",
            "prompt_tokens",
            "cache_hit_tokens",
            "prefill_compute_tokens",
            "request_e2e_seconds",
            "engine_switches",
            "affinity_used",
            "affinity_fallback",
        )
    }
    performance["sglang"] = {
        **sglang_totals,
        "cache_hit_rate": (
            sglang_totals["cache_hit_tokens"] / sglang_totals["prompt_tokens"]
            if sglang_totals["prompt_tokens"]
            else 0.0
        ),
        "avg_request_e2e_seconds": (
            sglang_totals["request_e2e_seconds"] / sglang_totals["requests"]
            if sglang_totals["requests"]
            else 0.0
        ),
    }
    samples_by_session: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        samples_by_session[str(sample.metadata.get("session_id"))].append(sample)
    rewrite_transitions = {
        "compressed_or_rewritten",
        "context_compacted",
        "context_truncated",
        "history_rewritten",
        "session_compacted",
        "template_changed",
        "tool_result_pruned",
        "tool_result_truncated",
        "tool_result_mismatch",
        "template_delta_invalid",
    }
    trajectory_segments = []
    for audit in sorted(audits, key=lambda item: int(item.get("task_index", -1))):
        session_id = str(audit["session_id"])
        audit_steps = audit.get("steps", [])
        session_samples = samples_by_session.get(session_id, [])
        split_reasons = Counter(
            step.get("split_reason")
            for step in audit_steps
            if step.get("split_performed")
        )
        trajectory_segments.append(
            {
                "session_id": session_id,
                "task_id": audit.get("task_id"),
                "task_index": audit.get("task_index"),
                "steps": len(audit_steps),
                "submitted_segments": int(audit.get("submitted_segment_count", 0)),
                "sample_token_lengths": sorted(len(sample.tokens) for sample in session_samples),
                "context_rewrite_splits": sum(
                    int(split_reasons[reason])
                    for reason in rewrite_transitions
                ),
                "trajectory_limit_splits": int(split_reasons["trajectory_limit"]),
                "compaction_events": sum(
                    bool(step.get("compaction_detected")) for step in audit_steps
                ),
                "compacted_tool_outputs": sum(
                    int(step.get("compacted_tool_outputs_added", 0)) for step in audit_steps
                ),
                "exact_context_overflows": int(
                    audit.get("exact_context_overflow_count", 0)
                ),
                "max_consecutive_identical_tool_call_turns": int(
                    audit.get("max_consecutive_identical_tool_call_turns", 0)
                ),
            }
        )
    segment_distribution = Counter(
        row["submitted_segments"] for row in trajectory_segments
    )

    checks = {
        "result_count": len(results) == expected,
        "all_tasks_completed": all(row["status"] == "ok" for row in results),
        "eight_unique_tasks": len(selected_counts) == args.task_count,
        "eight_samples_each": set(selected_counts.values()) == {args.samples_per_task},
        "logical_trajectory_count": len(audit_sessions) == expected
        and sum(int(audit.get("logical_trajectory_count", 0)) for audit in audits)
        == expected,
        "sample_logical_sessions": sample_sessions == audit_sessions,
        "segment_count_matches_samples": submitted_segments == len(samples),
        "sample_lengths_within_65536": all(
            len(sample.tokens) <= 65536 for sample in samples
        ),
        "sample_identity_preserved": all(
            sample.metadata.get("group_index") == sample.group_index
            and sample.metadata.get("traj_index") == sample.index
            for sample in samples
        ),
        "token_logprob_alignment": all(
            step.get("token_alignment_ok") is True for step in steps
        ),
        "request_seed_disabled": (
            args.deterministic
            or all(step.get("sampling_seed") is None for step in steps)
        ),
        "request_seed_enabled_when_deterministic": (
            not args.deterministic
            or all(step.get("sampling_seed") is not None for step in steps)
        ),
        "context_transitions_valid": all(
            step.get("context_transition")
            in ({"initial", "append_only"} | rewrite_transitions)
            for step in steps
        ),
        "token_ledger_mode_enforced": all(
            step.get("ledger_mode") == "enforce" for step in steps
        ),
        "append_turns_use_exact_token_ledger": all(
            step.get("prompt_source") == "exact_token_ledger"
            and int(step.get("assistant_boundary_tokens", 0)) > 0
            and int(step.get("environment_delta_tokens", 0)) > 0
            for step in steps
            if step.get("ledger_transition") == "append_only"
        ),
        "no_token_ledger_invariant_failures": all(
            step.get("context_transition")
            not in {
                "serialization_rewrite_shadow",
                "token_ledger_invariant_failed",
                "template_delta_invalid",
            }
            for step in steps
        ),
        "rewritten_context_always_starts_new_segment": all(
            step.get("split_performed") is True
            and step.get("split_reason") == step.get("context_transition")
            and step.get("prompt_source") == "canonical"
            for step in steps
            if step.get("context_transition") in rewrite_transitions
        ),
        "sample_segment_indices_contiguous": all(
            sorted(
                int(sample.metadata.get("segment_index", -1))
                for sample in samples_by_session.get(str(audit["session_id"]), [])
            )
            == list(range(int(audit.get("submitted_segment_count", 0))))
            for audit in audits
        ),
        "split_samples_share_logical_trajectory": all(
            len({sample.index for sample in session_samples}) <= 1
            and len({sample.group_index for sample in session_samples}) <= 1
            for session_samples in samples_by_session.values()
        ),
        "parsed_tool_calls_all_returned": all(
            step.get("parsed_tool_call_count")
            == step.get("returned_tool_call_count")
            for step in steps
        ),
        "tool_results_linked": all(
            not step.get("has_followup_step")
            or not step.get("returned_tool_call_count")
            or (
                step.get("observed_tool_result_count")
                == step.get("returned_tool_call_count")
                and step.get("tool_result_id_order_matches") is True
            )
            for step in steps
        ),
        "multi_tool_path_exercised": any(
            int(step.get("returned_tool_call_count", 0)) > 1 for step in steps
        ),
        "serper_script_not_missing": not _contains_missing_serper_script(
            args.output_dir
        ),
    }
    report = {
        "gate": "fixed-seed-8x8-production-path",
        "model": str(args.model),
        "rollout_id": args.rollout_id,
        "task_seed": args.task_seed,
        "engine_seed": args.engine_seed,
        "determinism": {
            "sglang_deterministic_inference": args.deterministic,
            "sampling_seed_base": args.sampling_seed_base,
            "deterministic_tool_call_ids": args.deterministic_tool_call_ids,
        },
        "expected_logical_trajectories": expected,
        "selected_task_counts": dict(selected_counts),
        "trajectory_segments": trajectory_segments,
        "segment_distribution": dict(sorted(segment_distribution.items())),
        "results": results,
        "summary": {
            "trajectory_audits": len(audits),
            "logical_sessions": len(audit_sessions),
            "submitted_segments": submitted_segments,
            "samples": len(samples),
            "steps": len(steps),
            "exact_ledger_append_turns": sum(
                step.get("prompt_source") == "exact_token_ledger"
                for step in steps
            ),
            "canonical_prompt_turns": sum(
                step.get("prompt_source") == "canonical" for step in steps
            ),
            "context_rewrite_transitions": sum(
                step.get("context_transition") in rewrite_transitions
                for step in steps
            ),
            "unparsed_raw_tool_call_steps": sum(
                int(step.get("raw_tool_call_count", 0))
                != int(step.get("parsed_tool_call_count", 0))
                for step in steps
            ),
            "compaction_events": sum(
                bool(step.get("compaction_detected")) for step in steps
            ),
            "compacted_tool_outputs": sum(
                int(step.get("compacted_tool_outputs_added", 0)) for step in steps
            ),
            "multi_tool_steps": sum(
                int(step.get("returned_tool_call_count", 0)) > 1 for step in steps
            ),
            "exact_context_overflows": sum(
                int(audit.get("exact_context_overflow_count", 0)) for audit in audits
            ),
            "repeated_tool_call_turns": sum(
                int(step.get("consecutive_identical_tool_call_turns", 0)) > 1
                for step in steps
            ),
            "trajectories_with_10x_tool_call_loop": sum(
                int(audit.get("max_consecutive_identical_tool_call_turns", 0)) >= 10
                for audit in audits
            ),
            "max_consecutive_identical_tool_call_turns": max(
                (
                    int(audit.get("max_consecutive_identical_tool_call_turns", 0))
                    for audit in audits
                ),
                default=0,
            ),
            "trajectories_split_into_multiple_samples": sum(
                row["submitted_segments"] > 1 for row in trajectory_segments
            ),
            "max_samples_per_trajectory": max(segment_distribution, default=0),
        },
        "performance": performance,
        "checks": checks,
        "passed": all(checks.values()),
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--task-count", type=int, default=8)
    parser.add_argument("--samples-per-task", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--engine-seed", type=int, default=12345)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--sampling-seed-base", type=int, default=424242)
    parser.add_argument("--deterministic-tool-call-ids", action="store_true")
    parser.add_argument("--allow-flexible-batch", action="store_true")
    parser.add_argument("--served-model", default="clawgym-sft-fixed-gate")
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--backend-port", type=int, default=30180)
    parser.add_argument("--proxy-port", type=int, default=30181)
    parser.add_argument("--reverse-port", type=int, default=39181)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--serve-timeout", type=int, default=900)
    parser.add_argument("--rollout-timeout", type=int, default=1200)
    parser.add_argument("--remote-host", default="159.138.50.147")
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--ssh-key", type=Path, default=Path("/root/openclaw_cpu_key"))
    parser.add_argument("--remote-base", default="/root/clawgym_rl_fixed_gate")
    parser.add_argument("--ssh-pool-size", type=int, default=8)
    parser.add_argument("--gateway-port-base", type=int, default=30000)
    parser.add_argument("--gateway-train-offset", type=int, default=1000)
    parser.add_argument("--gateway-port-stride", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--eval-task-ids", default="")
    parser.add_argument("--eval-task-index-offset", type=int, default=0)
    parser.add_argument("--rubric-dataset", type=Path)
    parser.add_argument("--rubric-archive-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        args.output_dir = REPO_ROOT / "outputs" / f"fixed-rollout-gate-{stamp}"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.python.is_file():
        raise FileNotFoundError(args.python)
    if not args.tasks.is_dir():
        raise FileNotFoundError(args.tasks)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.eval_mode and (
        not args.allow_flexible_batch
        and (args.task_count != 8 or args.samples_per_task != 8)
    ):
        raise ValueError("the acceptance gate requires exactly 8 tasks x 8 samples")
    if args.eval_mode and args.samples_per_task < 1:
        raise ValueError("eval samples-per-task must be positive")
    if not 1 <= args.parallel <= 64:
        raise ValueError("parallel must be between 1 and 64")
    if not 1 <= args.ssh_pool_size <= 8:
        raise ValueError("ssh-pool-size must be between 1 and 8")
    visible_gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    topology_gpus = args.tensor_parallel_size * args.data_parallel_size
    if topology_gpus != len(visible_gpus):
        raise ValueError(
            "tensor-parallel-size * data-parallel-size must equal the number "
            f"of visible GPUs ({topology_gpus} != {len(visible_gpus)})"
        )
    if min(args.rollout_id, args.task_seed, args.engine_seed) < 0:
        raise ValueError("rollout and seed values must be non-negative")
    if not args.ssh_key.is_file():
        raise FileNotFoundError(args.ssh_key)


def _set_standalone_runtime_contract(args: argparse.Namespace) -> None:
    """Add attributes that Slime normally injects before rollout hooks run."""
    args.save = str(args.output_dir)
    # Eval artifact provenance normally receives this field from Slime's
    # training argument namespace. The standalone gate names the same input
    # ``model``, so expose the compatible attribute before invoking the shared
    # rollout/eval implementation.
    if hasattr(args, "model"):
        args.hf_checkpoint = str(args.model)


def main() -> int:
    args = parse_args()
    _validate_args(args)
    _configure_environment(args)
    batch = None if args.eval_mode else _select_batch(args)
    eval_sources = _eval_sources(args) if args.eval_mode else None

    if args.preflight_only:
        if args.eval_mode:
            assert eval_sources is not None
            selected = {
                source.dataset_name: len(source.tasks) for source in eval_sources
            }
            logical_trajectories = sum(selected.values()) * args.samples_per_task
        else:
            assert batch is not None
            selected = dict(Counter(task.task_id for _, task in batch))
            logical_trajectories = len(batch)
        print(
            json.dumps(
                {
                    "mode": "eval" if args.eval_mode else "gate",
                    "model": str(args.model),
                    "tasks": str(args.tasks),
                    "logical_trajectories": logical_trajectories,
                    "selected_task_counts": selected,
                    "task_seed": args.task_seed,
                    "engine_seed": args.engine_seed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _set_standalone_runtime_contract(args)
    if args.eval_mode:
        assert eval_sources is not None
        _write_eval_selection(args.output_dir, eval_sources)
    else:
        assert batch is not None
        _write_selection(args.output_dir, batch)
    sglang_process = None
    sglang_log = None
    proxy_server = None
    tunnel = None
    output_queue: queue.Queue = queue.Queue(maxsize=100000)
    submission_enabled = threading.Event()
    submission_enabled.set()
    try:
        sglang_process, sglang_log = _start_sglang(args)
        proxy_args = SimpleNamespace(
            hf_checkpoint=str(args.model),
            sglang_router_ip="127.0.0.1",
            sglang_router_port=args.backend_port,
        )
        proxy_server = ClawGymRLAPIServer(
            proxy_args,
            output_queue,
            submission_enabled,
        )
        proxy_server.start()
        proxy_server.wait_until_ready(args.serve_timeout)
        tunnel = _start_reverse_tunnel(args)

        if args.eval_mode:
            assert eval_sources is not None
            started = time.monotonic()
            eval_result = rollout._run_openclaw_eval(args, args.rollout_id)
            elapsed_seconds = time.monotonic() - started
            if eval_result is None:
                raise RuntimeError("OpenClaw eval returned no datasets")
            audits = _read_audits(Path(os.environ["OPENCLAW_RECORD_FILE"]))
            report = _build_eval_report(
                args, eval_sources, eval_result, elapsed_seconds, audits
            )
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 2
        else:
            assert batch is not None
            results, performance = _run_batch(args, batch)
            audits = _read_audits(Path(os.environ["OPENCLAW_RECORD_FILE"]))
            samples = _drain_samples(output_queue)
            report = _build_report(
                args, batch, results, audits, samples, performance
            )
            print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
            failed_checks = [
                name for name, passed in report["checks"].items() if not passed
            ]
            if failed_checks:
                print("failed checks: " + ", ".join(failed_checks), file=sys.stderr)
                return 2
            print("fixed-seed 8x8 rollout gate passed", flush=True)
            return 0
    finally:
        if tunnel is not None:
            _stop_process(tunnel, timeout=10)
        if proxy_server is not None:
            proxy_server.stop()
        _stop_process(sglang_process, timeout=30)
        if sglang_log is not None:
            sglang_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
