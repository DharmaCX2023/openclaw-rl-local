#!/usr/bin/env python3
"""Offline K-way matched-placebo teacher scoring on saved ClawGym rollouts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))

from clawgym_rl_api_server import (  # noqa: E402
    _inject_opsd_feedback,
    _normalize_messages_for_template,
)
from clawgym_rl_rollout import _extract_opsd_teacher_logprobs  # noqa: E402
from opsd_evidence import (  # noqa: E402
    load_evidence_anchor_index,
    load_placebo_mapping,
    render_teacher_feedback,
)


def _extract_tools(prompt_text: str) -> list[dict[str, Any]]:
    starts = [match.end() for match in re.finditer(r"<tools>\n", prompt_text)]
    if not starts:
        raise ValueError("rendered prompt has no tools block")
    start = starts[-1]
    end = prompt_text.find("\n</tools>", start)
    if end < 0:
        raise ValueError("rendered prompt has no closing tools block")
    return [
        json.loads(line)
        for line in prompt_text[start:end].splitlines()
        if line.strip()
    ]


def _active_runs(mask: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(mask):
        while cursor < len(mask) and not mask[cursor]:
            cursor += 1
        start = cursor
        while cursor < len(mask) and mask[cursor]:
            cursor += 1
        if start < cursor:
            runs.append((start, cursor))
    return runs


def _tool_result_failed(row: dict[str, Any]) -> bool:
    state = row.get("next_state") or {}
    text = str(state.get("content") or "")
    return bool(
        re.search(
            r"(?:^|\n)\s*(?:error|exception|failed|failure|timed out)\b"
            r"|command exited with (?:code|status) [1-9]\d*",
            text,
            flags=re.IGNORECASE,
        )
    )


def _action_type(row: dict[str, Any], previous_failed: bool) -> str:
    if row.get("repetition_degenerated") or int(
        row.get("consecutive_identical_tool_call_turns") or 0
    ) > 1:
        return "degenerate_repeat"
    if row.get("hit_response_limit"):
        return "response_truncated"
    failed = _tool_result_failed(row)
    if failed:
        return "tool_error"
    calls = row.get("tool_calls") or []
    names = [
        str((call.get("function") or {}).get("name") or "")
        for call in calls
    ]
    if previous_failed:
        return "recovery_after_error"
    if not names:
        return "final_answer"
    if all(name == "read" for name in names):
        return "successful_read"
    if any(name in {"write", "edit"} for name in names):
        return "modification"
    if "exec" in names:
        command_text = json.dumps(calls, ensure_ascii=False).lower()
        if any(
            marker in command_text
            for marker in ("test", "validat", "check", "pytest", "unittest")
        ):
            return "validation"
        return "exec_other"
    if "process" in names:
        return "process"
    return "other_tool"


def _render_conditioned_prompt(
    *,
    tokenizer,
    row: dict[str, Any],
    anchors: list[dict[str, str]],
) -> list[int]:
    feedback = render_teacher_feedback(
        {
            "rollout_not_fully_successful": True,
            "evidence_anchors": anchors,
        }
    )
    messages = _inject_opsd_feedback(row["messages"], feedback)
    tools = _extract_tools(row["prompt_text"])
    return [
        int(value)
        for value in tokenizer.apply_chat_template(
            _normalize_messages_for_template(messages),
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort=None,
        )
    ]


def _score(
    *,
    generate_url: str,
    routing_key: str,
    prompt_ids: list[int],
    response_ids: list[int],
    timeout: float,
) -> list[float]:
    payload = {
        "input_ids": prompt_ids + response_ids,
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": max(0, len(prompt_ids) - 1),
    }
    response = requests.post(
        generate_url,
        json=payload,
        headers={"X-SMG-Routing-Key": routing_key},
        timeout=timeout,
    )
    response.raise_for_status()
    return _extract_opsd_teacher_logprobs(response.json(), response_ids)


def _weight(delta: float, beta: float, lower: float, upper: float) -> float:
    return min(max(math.exp(-beta * delta), lower), upper)


def _centered_weights(
    rows: list[dict[str, Any]],
    *,
    beta: float,
    lower: float,
    upper: float,
    epsilon: float,
) -> dict[str, float]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
    output: dict[str, float] = {}
    for session_rows in by_session.values():
        deltas = [float(row["primary_delta"]) for row in session_rows]
        if len(deltas) < 2:
            for row in session_rows:
                output[row["turn_key"]] = 1.0
            continue
        median = statistics.median(deltas)
        mad = statistics.median(abs(value - median) for value in deltas)
        if not math.isfinite(mad) or mad <= epsilon:
            for row in session_rows:
                output[row["turn_key"]] = 1.0
            continue
        for row, delta in zip(session_rows, deltas, strict=True):
            z_score = (delta - median) / (mad + epsilon)
            output[row["turn_key"]] = _weight(
                z_score, beta, lower, upper
            )
    return output


def _calibrate_absolute_beta(
    deltas: list[float],
    target_strength: float,
    *,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    candidates = [0.0] + [
        10 ** (-4.0 + 6.0 * index / 4000.0)
        for index in range(4001)
    ]
    best = (float("inf"), 0.0, 0.0)
    for beta in candidates:
        strength = statistics.fmean(
            abs(_weight(delta, beta, lower, upper) - 1.0)
            for delta in deltas
        )
        candidate = (abs(strength - target_strength), beta, strength)
        if candidate < best:
            best = candidate
    return best[1], best[2]


def _bootstrap_category_difference(
    rows: list[dict[str, Any]],
    *,
    good: set[str],
    bad: set[str],
    repeats: int,
    seed: int,
) -> dict[str, float | int | None]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
    sessions = sorted(by_session)
    if not sessions:
        return {"clusters": 0, "estimate": None, "ci_low": None, "ci_high": None}

    def difference(chosen: list[str]) -> float | None:
        selected = [
            row for session in chosen for row in by_session[session]
        ]
        left = [
            float(row["primary_delta"])
            for row in selected
            if row["action_type"] in good
        ]
        right = [
            float(row["primary_delta"])
            for row in selected
            if row["action_type"] in bad
        ]
        if not left or not right:
            return None
        return statistics.fmean(left) - statistics.fmean(right)

    estimate = difference(sessions)
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        chosen = [rng.choice(sessions) for _ in sessions]
        value = difference(chosen)
        if value is not None:
            values.append(value)
    values.sort()
    if not values:
        return {
            "clusters": len(sessions),
            "estimate": estimate,
            "ci_low": None,
            "ci_high": None,
        }
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {
        "clusters": len(sessions),
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in sorted(
            rows, key=lambda item: (item["turn_key"], item["condition"])
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-records", type=Path, required=True)
    parser.add_argument("--step-summary", type=Path, required=True)
    parser.add_argument("--debug-train", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--generate-url",
        default="http://127.0.0.1:30000/generate",
        help=(
            "One URL or a comma-separated list. Sessions are deterministically "
            "pinned so Real and all placebos use the same engine."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=65536)
    parser.add_argument("--centered-beta", type=float, default=0.02)
    parser.add_argument("--clip-lower", type=float, default=0.8)
    parser.add_argument("--clip-upper", type=float, default=1.2)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    if args.k <= 0 or args.concurrency <= 0 or args.retries < 0:
        raise ValueError("k/concurrency must be positive and retries non-negative")

    generate_urls = [
        value.strip()
        for value in args.generate_url.split(",")
        if value.strip()
    ]
    if not generate_urls:
        raise ValueError("at least one generate URL is required")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), local_files_only=True
    )
    anchors = load_evidence_anchor_index(args.anchors)
    mapping = load_placebo_mapping(args.mapping)
    summary_payload = json.loads(args.step_summary.read_text(encoding="utf-8"))
    records = summary_payload["records"]
    debug = torch.load(args.debug_train, weights_only=True)["rollout_data"]
    if len(records) != len(debug["tokens"]):
        raise ValueError("summary/debug sample counts differ")

    proxy: dict[tuple[str, int], dict[str, Any]] = {}
    previous_failed: dict[str, bool] = defaultdict(bool)
    action_types: dict[tuple[str, int], str] = {}
    for line in args.proxy_records.open(encoding="utf-8"):
        row = json.loads(line)
        if "messages" not in row or "step" not in row:
            continue
        key = (str(row["session_id"]), int(row["step"]))
        proxy[key] = row
    for session_id in sorted({key[0] for key in proxy}):
        for key in sorted(
            (key for key in proxy if key[0] == session_id),
            key=lambda value: value[1],
        ):
            row = proxy[key]
            action_types[key] = _action_type(
                row, previous_failed[session_id]
            )
            previous_failed[session_id] = _tool_result_failed(row)

    turns: list[dict[str, Any]] = []
    reconstruction = defaultdict(int)
    for index, record in enumerate(records):
        if not record["opsd_applied"]:
            continue
        if (
            int(record["group_index"]) != int(debug["group_indices"][index])
            or int(record["trajectory_index"])
            != int(debug["sample_indices"][index])
            or int(record["response_token_count"])
            != int(debug["response_lengths"][index])
        ):
            raise ValueError(f"summary/debug row {index} identity mismatch")
        response_length = int(debug["response_lengths"][index])
        tokens = debug["tokens"][index].tolist()
        response_ids = tokens[-response_length:]
        mask = debug["loss_masks"][index].tolist()
        runs = _active_runs(mask)
        step_rows = record["steps"]
        if [end - start for start, end in runs] != [
            int(step["loss_token_count"]) for step in step_rows
        ]:
            raise ValueError(
                f"loss span mismatch for {record['session_id']} segment "
                f"{record['segment_index']}"
            )
        task_id = str(record["task_id"])
        map_row = mapping.get(task_id) or {}
        selected = list(map_row.get("placebos") or [])[: args.k]
        if len(selected) != args.k:
            reconstruction["insufficient_placebos"] += len(step_rows)
            continue
        if task_id not in anchors:
            reconstruction["missing_real_anchor"] += len(step_rows)
            continue

        for step, (start, end) in zip(step_rows, runs, strict=True):
            step_id = int(step["step_id"])
            key = (str(record["session_id"]), step_id)
            proxy_row = proxy.get(key)
            if proxy_row is None:
                reconstruction["missing_proxy_turn"] += 1
                continue
            tools = _extract_tools(proxy_row["prompt_text"])
            base_prompt = [
                int(value)
                for value in tokenizer.apply_chat_template(
                    _normalize_messages_for_template(proxy_row["messages"]),
                    tools=tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    reasoning_effort=None,
                )
            ]
            if len(base_prompt) != int(proxy_row["canonical_prompt_tokens"]):
                raise ValueError(f"canonical prompt mismatch for {key}")
            exact_response = [int(value) for value in response_ids[start:end]]
            conditions = [("real", task_id, anchors[task_id])]
            for placebo_index, placebo in enumerate(selected):
                placebo_task = str(placebo["task_id"])
                if placebo_task == task_id or placebo_task not in anchors:
                    raise ValueError(f"invalid placebo for {task_id}")
                conditions.append(
                    (
                        f"placebo_{placebo_index}",
                        placebo_task,
                        anchors[placebo_task],
                    )
                )
            condition_tasks = {}
            overflow = False
            for condition, condition_task, condition_anchors in conditions:
                prompt_ids = _render_conditioned_prompt(
                    tokenizer=tokenizer,
                    row=proxy_row,
                    anchors=condition_anchors,
                )
                if len(prompt_ids) + len(exact_response) > args.max_context_tokens:
                    overflow = True
                    break
                condition_tasks[condition] = condition_task
            if overflow:
                reconstruction["conditioned_context_overflow"] += 1
                continue
            turns.append(
                {
                    "turn_key": f"{key[0]}:{key[1]}",
                    "session_id": key[0],
                    "step_id": key[1],
                    "task_id": task_id,
                    "group_index": int(record["group_index"]),
                    "trajectory_index": int(record["trajectory_index"]),
                    "segment_index": int(record["segment_index"]),
                    "raw_reward": float(record["raw_reward"]),
                    "advantage": float(record["advantage"]),
                    "response_tokens": exact_response,
                    "response_token_count": len(exact_response),
                    "action_type": action_types.get(key, "unknown"),
                    "proxy_key": [key[0], key[1]],
                    "condition_tasks": condition_tasks,
                }
            )
            reconstruction["ready_turns"] += 1

    if args.prepare_only:
        print(
            json.dumps(
                {
                    "reconstruction": dict(reconstruction),
                    "turns": len(turns),
                    "conditions": len(turns) * (args.k + 1),
                    "trajectories": len(
                        {turn["session_id"] for turn in turns}
                    ),
                    "tasks": len({turn["task_id"] for turn in turns}),
                },
                sort_keys=True,
            )
        )
        return

    prior = {}
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prior[(row["turn_key"], row["condition"])] = row

    jobs = []
    for turn in turns:
        for condition, condition_task in turn["condition_tasks"].items():
            key = (turn["turn_key"], condition)
            if key in prior:
                continue
            jobs.append((turn, condition, condition_task))

    # Long-context scoring can have very few request slots even when SGLang's
    # configured max-running-requests is larger. Preserve parallelism across
    # engines, but serialize requests within each engine so a four-condition
    # turn cannot crash its pinned engine with a request-slot burst.
    engine_semaphores = [threading.Semaphore(1) for _ in generate_urls]

    def run_job(job):
        turn, condition, condition_task = job
        proxy_key = (str(turn["proxy_key"][0]), int(turn["proxy_key"][1]))
        prompt_ids = _render_conditioned_prompt(
            tokenizer=tokenizer,
            row=proxy[proxy_key],
            anchors=anchors[condition_task],
        )
        engine_index = int(
            hashlib.sha256(turn["session_id"].encode()).hexdigest(), 16
        ) % len(generate_urls)
        with engine_semaphores[engine_index]:
            for attempt in range(args.retries + 1):
                try:
                    logprobs = _score(
                        generate_url=generate_urls[engine_index],
                        routing_key=turn["session_id"],
                        prompt_ids=prompt_ids,
                        response_ids=turn["response_tokens"],
                        timeout=args.timeout,
                    )
                    break
                except Exception:
                    if attempt >= args.retries:
                        raise
                    time.sleep(1.0 * (attempt + 1))
        return {
            "turn_key": turn["turn_key"],
            "session_id": turn["session_id"],
            "step_id": turn["step_id"],
            "task_id": turn["task_id"],
            "condition": condition,
            "condition_task_id": condition_task,
            "response_token_count": turn["response_token_count"],
            "mean_logprob": statistics.fmean(logprobs),
            "action_type": turn["action_type"],
            "raw_reward": turn["raw_reward"],
            "advantage": turn["advantage"],
        }

    completed = list(prior.values())
    scoring_failures = []
    if jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = {executor.submit(run_job, job): job for job in jobs}
            for offset, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                job = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    turn, condition, _ = job
                    scoring_failures.append(
                        {
                            "turn_key": turn["turn_key"],
                            "condition": condition,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if len(scoring_failures) <= 10:
                        print(
                            "scoring failure {} {}: {}: {}".format(
                                turn["turn_key"],
                                condition,
                                type(exc).__name__,
                                exc,
                            ),
                            flush=True,
                        )
                if offset % 50 == 0:
                    _write_jsonl(args.output, completed)
                    print(
                        f"scored {offset}/{len(jobs)} new conditions; "
                        f"failures={len(scoring_failures)}",
                        flush=True,
                    )

    _write_jsonl(args.output, completed)

    score_index = {
        (row["turn_key"], row["condition"]): row for row in completed
    }
    analyzed = []
    incomplete = 0
    for turn in turns:
        keys = [
            (turn["turn_key"], "real"),
            *[
                (turn["turn_key"], f"placebo_{index}")
                for index in range(args.k)
            ],
        ]
        if any(key not in score_index for key in keys):
            incomplete += 1
            continue
        real = float(score_index[keys[0]]["mean_logprob"])
        placebo = [
            float(score_index[key]["mean_logprob"]) for key in keys[1:]
        ]
        deltas = [real - value for value in placebo]
        analyzed.append(
            {
                **{
                    key: value
                    for key, value in turn.items()
                    if key
                    not in {
                        "condition_tasks",
                        "proxy_key",
                        "response_tokens",
                    }
                },
                "real_logprob_mean": real,
                "placebo_logprob_means": placebo,
                "placebo_mean": statistics.fmean(placebo),
                "placebo_std": statistics.pstdev(placebo),
                "deltas": deltas,
                "primary_delta": deltas[0],
                "delta_std": statistics.pstdev(deltas),
                "sign_flip": min(deltas) < 0.0 < max(deltas),
            }
        )

    if not analyzed:
        raise RuntimeError("no complete matched-placebo turns were scored")

    centered = _centered_weights(
        analyzed,
        beta=args.centered_beta,
        lower=args.clip_lower,
        upper=args.clip_upper,
        epsilon=args.epsilon,
    )
    for row in analyzed:
        row["centered_weight"] = centered[row["turn_key"]]
    centered_strength = statistics.fmean(
        abs(float(row["centered_weight"]) - 1.0) for row in analyzed
    )
    absolute_beta, absolute_strength = _calibrate_absolute_beta(
        [float(row["primary_delta"]) for row in analyzed],
        centered_strength,
        lower=args.clip_lower,
        upper=args.clip_upper,
    )
    for row in analyzed:
        row["absolute_weight"] = _weight(
            float(row["primary_delta"]),
            absolute_beta,
            args.clip_lower,
            args.clip_upper,
        )

    def summarize_weight(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in analyzed]
        return {
            "mean": statistics.fmean(values),
            "mean_abs_deviation_from_one": statistics.fmean(
                abs(value - 1.0) for value in values
            ),
            "lower_boundary_fraction": sum(
                value <= args.clip_lower + 1e-6 for value in values
            )
            / len(values),
            "upper_boundary_fraction": sum(
                value >= args.clip_upper - 1e-6 for value in values
            )
            / len(values),
        }

    categories: dict[str, dict[str, float | int]] = {}
    for category in sorted({row["action_type"] for row in analyzed}):
        selected = [row for row in analyzed if row["action_type"] == category]
        categories[category] = {
            "turns": len(selected),
            "mean_delta": statistics.fmean(
                float(row["primary_delta"]) for row in selected
            ),
            "mean_centered_weight": statistics.fmean(
                float(row["centered_weight"]) for row in selected
            ),
            "mean_absolute_weight": statistics.fmean(
                float(row["absolute_weight"]) for row in selected
            ),
            "protected_fraction_centered": sum(
                float(row["centered_weight"]) < 1.0 for row in selected
            )
            / len(selected),
        }

    good = {
        "successful_read",
        "modification",
        "validation",
        "recovery_after_error",
    }
    bad = {"tool_error", "degenerate_repeat", "response_truncated"}
    cluster_test = _bootstrap_category_difference(
        analyzed,
        good=good,
        bad=bad,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    placebo_means = [
        statistics.fmean(
            float(row["placebo_logprob_means"][index])
            for row in analyzed
        )
        for index in range(args.k)
    ]
    report = {
        "schema_version": "clawgym_matched_placebo_phase0_v1",
        "inputs": {
            "proxy_records": str(args.proxy_records.resolve()),
            "step_summary": str(args.step_summary.resolve()),
            "debug_train": str(args.debug_train.resolve()),
            "anchors": str(args.anchors.resolve()),
            "mapping": str(args.mapping.resolve()),
            "tokenizer": str(args.tokenizer.resolve()),
            "mapping_sha256": hashlib.sha256(
                args.mapping.read_bytes()
            ).hexdigest(),
        },
        "configuration": {
            "k": args.k,
            "centered_beta": args.centered_beta,
            "absolute_beta_frozen_candidate": absolute_beta,
            "clip_lower": args.clip_lower,
            "clip_upper": args.clip_upper,
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "num_scoring_engines": len(generate_urls),
        },
        "coverage": {
            **dict(reconstruction),
            "analyzed_turns": len(analyzed),
            "incomplete_scoring_turns": incomplete,
            "condition_scoring_failures": len(scoring_failures),
            "logical_trajectories": len(
                {row["session_id"] for row in analyzed}
            ),
            "tasks": len({row["task_id"] for row in analyzed}),
        },
        "stability": {
            "sign_flip_fraction": sum(row["sign_flip"] for row in analyzed)
            / len(analyzed),
            "mean_delta_std_across_placebos": statistics.fmean(
                float(row["delta_std"]) for row in analyzed
            ),
            "median_delta_std_across_placebos": statistics.median(
                float(row["delta_std"]) for row in analyzed
            ),
            "placebo_index_logprob_means": placebo_means,
            "placebo_index_mean_range": max(placebo_means)
            - min(placebo_means),
        },
        "weights": {
            "centered": summarize_weight("centered_weight"),
            "absolute": summarize_weight("absolute_weight"),
            "absolute_strength_match_error": abs(
                absolute_strength - centered_strength
            ),
        },
        "behavior": {
            "categories": categories,
            "good_minus_bad_delta_cluster_bootstrap": cluster_test,
            "good_categories": sorted(good),
            "bad_categories": sorted(bad),
        },
        "scoring_failures": scoring_failures,
        "turns": analyzed,
    }
    report_path = args.output.with_suffix(".report.json")
    _atomic_write(report_path, report)
    beta_path = args.output.with_suffix(".frozen_beta.json")
    _atomic_write(
        beta_path,
        {
            "schema_version": "clawgym_matched_placebo_beta_v1",
            "mapping_sha256": report["inputs"]["mapping_sha256"],
            "centered_beta": args.centered_beta,
            "absolute_beta": absolute_beta,
            "target_mean_abs_weight_deviation": centered_strength,
            "actual_mean_abs_weight_deviation": absolute_strength,
            "must_not_retune_after_eval": True,
        },
    )
    print(
        json.dumps(
            {
                "turns": len(analyzed),
                "trajectories": report["coverage"][
                    "logical_trajectories"
                ],
                "sign_flip_fraction": report["stability"][
                    "sign_flip_fraction"
                ],
                "centered_strength": centered_strength,
                "absolute_beta": absolute_beta,
                "absolute_strength": absolute_strength,
                "report": str(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
