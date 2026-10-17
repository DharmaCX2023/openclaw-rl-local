#!/usr/bin/env python3
"""Extract conservative artifact/action credit from Mosaic Phase-0 results.

The extractor never edits an artifact and never treats a retokenized response
as the original rollout token sequence.  It performs three independent gates:

1. leave-one-file-out verifier credit on the strict Mosaic workspace;
2. exact byte/hash attribution to one observed atomic ``write`` action;
3. exact sampled-token-ID availability and equality for token-loss readiness.

Old audit runs without recorded response IDs may pass gates 1 and 2, but they
are explicitly ineligible for token-level training at gate 3.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable

from transformers import AutoTokenizer

import analyze_mosaic_phase0 as mosaic


WORKSPACE_ROOT = PurePosixPath("/root/.openclaw/workspace")
TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"
PATH_KEYS = ("path", "file_path", "filePath", "file")
MUTATING_TOOLS = {"write", "edit"}
OPAQUE_MUTATING_TOOLS = {"exec", "process"}
RUNTIME_ROOT_FILES = {
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "MEMORY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}


@dataclass(frozen=True)
class RawToolCall:
    call_index: int
    char_start: int
    char_end: int
    payload: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_dump_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _extract_raw_tool_calls(response_text: str) -> tuple[list[RawToolCall], list[str]]:
    """Parse exact delimited blocks while retaining raw character offsets."""

    calls: list[RawToolCall] = []
    errors: list[str] = []
    cursor = 0
    while True:
        start = response_text.find(TOOL_OPEN, cursor)
        if start < 0:
            break
        end_tag = response_text.find(TOOL_CLOSE, start + len(TOOL_OPEN))
        if end_tag < 0:
            errors.append(f"unclosed_tool_call_at_char_{start}")
            break
        end = end_tag + len(TOOL_CLOSE)
        raw_json = response_text[start + len(TOOL_OPEN) : end_tag].strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            errors.append(f"invalid_tool_json_at_char_{start}:{exc.msg}")
        else:
            if isinstance(payload, dict):
                calls.append(RawToolCall(len(calls), start, end, payload))
            else:
                errors.append(f"non_object_tool_json_at_char_{start}")
        cursor = end
    return calls, errors


def _arguments(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("arguments")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _normalize_workspace_path(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value:
        return None, "missing_path"
    path = PurePosixPath(value)
    if path.is_absolute():
        try:
            path = path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            return None, "path_outside_workspace"
    if not path.parts or ".." in path.parts:
        return None, "unsafe_relative_path"
    normalized = path.as_posix()
    if normalized in {".", ""}:
        return None, "empty_relative_path"
    return normalized, None


def _path_from_arguments(arguments: dict[str, Any]) -> tuple[str | None, str | None]:
    for key in PATH_KEYS:
        if key in arguments:
            return _normalize_workspace_path(arguments[key])
    return None, "missing_path"


def _is_runtime_path(relative: str) -> bool:
    if relative in RUNTIME_ROOT_FILES:
        return True
    if relative.startswith((".git/", ".openclaw/", ".venv/", "reward/")):
        return True
    parts = PurePosixPath(relative).parts
    return "__pycache__" in parts or relative.endswith((".pyc", ".pyo"))


def _parse_version_key(key: str) -> tuple[int, int, str] | None:
    if not key.startswith("file:"):
        return None
    try:
        _, mode, size, digest = key.split(":", 3)
        return int(mode, 8), int(size), digest
    except (ValueError, TypeError):
        return None


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("tasks"), list):
        raise ValueError(f"invalid Phase-0 report: {path}")
    return value


def _target_task_rows(
    report: dict[str, Any], excluded: set[str]
) -> list[dict[str, Any]]:
    rows = []
    for row in report["tasks"]:
        if not row.get("supported") or row.get("task_id") in excluded:
            continue
        cross_parent = row.get("cross_parent_complementarity")
        if cross_parent is None:
            cross_parent = bool(
                row.get("strict_super_parent")
                and int(row.get("minimum_source_cover") or 0) >= 2
            )
        if cross_parent:
            rows.append(row)
    return rows


def _load_proxy_records(
    record_path: Path,
    task_rows: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int], str],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, int], int],
]:
    target_ids = {str(row["task_id"]) for row in task_rows}
    session_by_task_index: dict[tuple[str, int], str] = {}
    response_counts: dict[tuple[str, int], int] = {}
    target_sessions: set[str] = set()

    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") != "trajectory_audit":
                continue
            task_id = str(record.get("task_id"))
            if task_id not in target_ids:
                continue
            task_index = int(record["task_index"])
            session_id = str(record["session_id"])
            session_by_task_index[(task_id, task_index)] = session_id
            target_sessions.add(session_id)
            for step in record.get("steps", []):
                response_counts[(session_id, int(step["step"]))] = int(
                    step.get("response_id_count", step.get("response_tokens", -1))
                )

    steps_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            session_id = str(record.get("session_id", ""))
            if session_id in target_sessions and record.get("record_type") is None:
                steps_by_session[session_id].append(record)
    for rows in steps_by_session.values():
        rows.sort(key=lambda row: int(row["step"]))
    return session_by_task_index, dict(steps_by_session), response_counts


def _install_version(target: Path, version: mosaic.FileVersion) -> None:
    if target.exists() or target.is_symlink():
        mosaic._remove_path(target)
    if version.kind == "absent":
        return
    assert version.source_path is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(version.source_path, target)
    target.chmod(version.mode)


def _criterion_delta(
    selected: dict[str, float], alternatives: Iterable[dict[str, float]]
) -> tuple[list[str], list[str]]:
    improved: set[str] = set()
    regressed: set[str] = set()
    for alternative in alternatives:
        keys = set(selected) | set(alternative)
        for key in keys:
            left = float(selected.get(key, 0.0))
            right = float(alternative.get(key, 0.0))
            if left > right + 1e-9:
                improved.add(key)
            elif left < right - 1e-9:
                regressed.add(key)
    return sorted(improved), sorted(regressed)


def _evaluate_artifacts(
    task: dict[str, Any],
    analysis_dir: Path,
    temporary_root: Path,
    checker_timeout: int,
    max_file_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_id = str(task["task_id"])
    extracted = analysis_dir / "extracted" / task_id
    initial_root = extracted / "initial"
    parent_roots = sorted(extracted.glob("parent_*"))
    mosaic_root = Path(str(task["mosaic_workspace"]))
    if not mosaic_root.is_dir():
        raise FileNotFoundError(f"missing strict Mosaic workspace: {mosaic_root}")
    if len(parent_roots) != int(task["parent_count"]):
        raise ValueError(f"parent directory count mismatch for {task_id}")

    initial_manifest = mosaic._manifest(initial_root, max_file_bytes)
    parent_manifests = [mosaic._manifest(root, max_file_bytes) for root in parent_roots]
    selected_manifest = mosaic._manifest(mosaic_root, max_file_bytes)
    checker_mode = str(task.get("checker_workspace_mode") or "absolute")
    selected_reward, selected_vector, selected_seconds = mosaic._run_checker(
        Path(task["task_dir"]), mosaic_root, checker_timeout, checker_mode
    )
    expected_reward = float(task["best_mosaic_reward"])
    if abs(selected_reward - expected_reward) > 1e-6:
        raise ValueError(
            f"strict Mosaic replay mismatch for {task_id}: "
            f"{selected_reward} != {expected_reward}"
        )

    rows = []
    verifier_queries = 1
    verifier_seconds = selected_seconds
    for chosen in task.get("chosen_noninitial_files", []):
        relative = str(chosen["path"])
        selected_key = str(chosen["version"])
        selected_version = selected_manifest.get(relative, mosaic._absent_version())
        row: dict[str, Any] = {
            "task_id": task_id,
            "path": relative,
            "selected_version": selected_key,
            "source_parent_indices": [int(value) for value in chosen.get("sources", [])],
            "selected_reward": selected_reward,
            "artifact_credit_eligible": False,
            "rejection_reasons": [],
        }
        if selected_version.key != selected_key:
            row["rejection_reasons"].append("selected_version_manifest_mismatch")
            rows.append(row)
            continue

        alternatives: dict[str, mosaic.FileVersion] = {}
        for manifest in [initial_manifest, *parent_manifests]:
            version = manifest.get(relative, mosaic._absent_version())
            if version.key != selected_key:
                alternatives.setdefault(version.key, version)
        alternative_rows = []
        for alternative_key, version in sorted(alternatives.items()):
            query_dir = temporary_root / task_id / f"q_{len(rows):04d}_{len(alternative_rows):03d}"
            query_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(mosaic_root, query_dir, copy_function=shutil.copy2)
            _install_version(query_dir / relative, version)
            reward, vector, elapsed = mosaic._run_checker(
                Path(task["task_dir"]), query_dir, checker_timeout, checker_mode
            )
            verifier_queries += 1
            verifier_seconds += elapsed
            alternative_rows.append(
                {
                    "version": alternative_key,
                    "reward": reward,
                    "criterion_vector": vector,
                }
            )
            shutil.rmtree(query_dir)
        row["alternative_count"] = len(alternative_rows)
        if not alternative_rows:
            row["rejection_reasons"].append("no_alternative_version")
            rows.append(row)
            continue
        best_alt_reward = max(float(value["reward"]) for value in alternative_rows)
        best_alternatives = [
            value
            for value in alternative_rows
            if abs(float(value["reward"]) - best_alt_reward) <= 1e-9
        ]
        contribution = selected_reward - best_alt_reward
        improved, regressed = _criterion_delta(
            selected_vector,
            [value["criterion_vector"] for value in best_alternatives],
        )
        row.update(
            {
                "best_alternative_reward": best_alt_reward,
                "reward_contribution": contribution,
                "improved_criteria": improved,
                "regressed_criteria": regressed,
                "best_alternatives": best_alternatives,
            }
        )
        if contribution <= 1e-9:
            row["rejection_reasons"].append("non_positive_leave_one_out_credit")
        if not improved:
            row["rejection_reasons"].append("no_improved_criterion")
        if regressed:
            row["rejection_reasons"].append("criterion_regression")
        if relative == "input" or relative.startswith("input/"):
            row["rejection_reasons"].append("protected_input_path")
        if _is_runtime_path(relative):
            row["rejection_reasons"].append("runtime_or_verifier_path")
        if selected_version.kind != "file":
            row["rejection_reasons"].append("selected_version_is_not_file")
        row["artifact_credit_eligible"] = not row["rejection_reasons"]
        rows.append(row)
    return rows, {
        "task_id": task_id,
        "verifier_queries": verifier_queries,
        "verifier_seconds": verifier_seconds,
    }


def _all_actions(steps: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    actions = []
    errors = []
    sequence = 0
    for record in steps:
        response_text = str(record.get("response_text") or "")
        raw_calls, parse_errors = _extract_raw_tool_calls(response_text)
        errors.extend(f"step_{record['step']}:{value}" for value in parse_errors)
        parsed_calls = record.get("tool_calls") or []
        for raw in raw_calls:
            payload = raw.payload
            name = payload.get("name")
            arguments = _arguments(payload)
            path = None
            path_error = None
            if arguments is not None:
                path, path_error = _path_from_arguments(arguments)
            parsed_id = None
            if raw.call_index < len(parsed_calls):
                parsed_id = parsed_calls[raw.call_index].get("id")
            actions.append(
                {
                    "sequence": sequence,
                    "step": int(record["step"]),
                    "call_index": raw.call_index,
                    "name": str(name) if name is not None else None,
                    "arguments": arguments,
                    "path": path,
                    "path_error": path_error,
                    "char_start": raw.char_start,
                    "char_end": raw.char_end,
                    "response_text": response_text,
                    "response_ids": record.get("response_ids"),
                    "tool_call_id": parsed_id,
                    "observed_tool_call_ids": record.get("observed_tool_call_ids") or [],
                    "tool_result_id_order_matches": record.get(
                        "tool_result_id_order_matches"
                    ),
                    "raw_tool_call_count": int(record.get("raw_tool_call_count", -1)),
                    "parsed_tool_call_count": int(
                        record.get("parsed_tool_call_count", -1)
                    ),
                    "returned_tool_call_count": int(
                        record.get("returned_tool_call_count", -1)
                    ),
                }
            )
            sequence += 1
    return actions, errors


def _token_span(
    tokenizer: Any,
    action: dict[str, Any],
    expected_count: int | None,
) -> dict[str, Any]:
    text = action["response_text"]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(value) for value in encoded["offset_mapping"]]
    stored = action.get("response_ids")
    recorded_span: tuple[int, int] | None = None
    recorded_terminal_token_count = 0
    if stored is not None:
        stored_ids = [int(value) for value in stored]
        content_ids = stored_ids
        decoded = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        while (
            decoded != text
            and eos_token_id is not None
            and content_ids
            and content_ids[-1] == int(eos_token_id)
        ):
            content_ids = content_ids[:-1]
            recorded_terminal_token_count += 1
            decoded = tokenizer.decode(
                content_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        if decoded == text:
            fidelity = "exact_recorded_ids"

            def prefix_length(count: int) -> int:
                return len(
                    tokenizer.decode(
                        content_ids[:count],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )

            def first_boundary(predicate) -> int:
                low, high = 0, len(content_ids)
                while low < high:
                    middle = (low + high) // 2
                    if predicate(prefix_length(middle)):
                        high = middle
                    else:
                        low = middle + 1
                return low

            start = max(
                0,
                first_boundary(lambda length: length > action["char_start"]) - 1,
            )
            end = first_boundary(lambda length: length >= action["char_end"])
            recorded_span = (start, end)
        else:
            fidelity = "recorded_id_decode_mismatch"
    elif expected_count is not None and len(ids) == expected_count:
        fidelity = "retokenized_count_match"
    else:
        fidelity = "retokenized_count_mismatch"
    overlapping = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > action["char_start"] and start < action["char_end"]
    ]
    if recorded_span is not None:
        token_start, token_end = recorded_span
    else:
        token_start = min(overlapping) if overlapping else None
        token_end = max(overlapping) + 1 if overlapping else None
    return {
        "response_token_count_retokenized": len(ids),
        "response_token_count_recorded": len(stored) if stored is not None else None,
        "recorded_terminal_token_count": recorded_terminal_token_count,
        "response_token_count_audit": expected_count,
        "token_fidelity": fidelity,
        "response_token_start": token_start,
        "response_token_end": token_end,
        "tool_call_token_count": (
            token_end - token_start
            if token_start is not None and token_end is not None
            else 0
        ),
    }


def _map_actions(
    artifact: dict[str, Any],
    task: dict[str, Any],
    session_by_task_index: dict[tuple[str, int], str],
    steps_by_session: dict[str, list[dict[str, Any]]],
    response_counts: dict[tuple[str, int], int],
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], list[str]]:
    task_id = str(task["task_id"])
    relative = str(artifact["path"])
    version = _parse_version_key(str(artifact["selected_version"]))
    if version is None:
        return [], ["selected_version_is_not_regular_file"]
    expected_mode, expected_size, expected_digest = version
    mappings = []
    aggregate_rejections: Counter[str] = Counter()
    parent_replay = task["parent_replay"]
    for parent_index in artifact["source_parent_indices"]:
        task_index = int(parent_replay[parent_index]["task_index"])
        session_id = session_by_task_index.get((task_id, task_index))
        if session_id is None:
            aggregate_rejections["source_session_not_found"] += 1
            continue
        actions, parse_errors = _all_actions(steps_by_session.get(session_id, []))
        if parse_errors:
            aggregate_rejections["raw_tool_call_parse_error_in_session"] += 1
        direct_mutations = [
            value
            for value in actions
            if value["path"] == relative and value["name"] in MUTATING_TOOLS
        ]
        for action in actions:
            if action["name"] != "write" or action["path"] != relative:
                continue
            reasons = []
            arguments = action.get("arguments")
            content = arguments.get("content") if isinstance(arguments, dict) else None
            if not isinstance(content, str):
                reasons.append("write_content_is_not_string")
                content_bytes = b""
            else:
                content_bytes = content.encode("utf-8")
                if len(content_bytes) != expected_size:
                    reasons.append("write_size_mismatch")
                if _sha256_bytes(content_bytes) != expected_digest:
                    reasons.append("write_hash_mismatch")
            if len(direct_mutations) != 1:
                reasons.append("path_has_multiple_direct_mutations")
            if not (
                action["raw_tool_call_count"]
                == action["parsed_tool_call_count"]
                == action["returned_tool_call_count"]
                == 1
            ):
                reasons.append("tool_turn_is_not_single_exact_call")
            if (
                not action["tool_call_id"]
                or action["tool_call_id"] not in action["observed_tool_call_ids"]
                or action["tool_result_id_order_matches"] is not True
            ):
                reasons.append("tool_result_not_exactly_observed")
            if any(
                later["sequence"] > action["sequence"]
                and later["name"] in OPAQUE_MUTATING_TOOLS
                for later in actions
            ):
                reasons.append("opaque_mutating_tool_after_write")
            span = _token_span(
                tokenizer,
                action,
                response_counts.get((session_id, int(action["step"]))),
            )
            if span["response_token_start"] is None:
                reasons.append("tool_call_token_span_not_found")
            exact_ids = span["token_fidelity"] == "exact_recorded_ids"
            mapping = {
                "task_id": task_id,
                "session_id": session_id,
                "task_index": task_index,
                "parent_index": parent_index,
                "step": action["step"],
                "tool_call_index": action["call_index"],
                "tool_call_id": action["tool_call_id"],
                "tool_name": "write",
                "path": relative,
                "file_mode": expected_mode,
                "file_size": expected_size,
                "file_sha256": expected_digest,
                "char_start": action["char_start"],
                "char_end": action["char_end"],
                **span,
                "structural_action_eligible": not reasons,
                "token_loss_eligible": bool(not reasons and exact_ids),
                "rejection_reasons": reasons,
            }
            mappings.append(mapping)
            aggregate_rejections.update(reasons)
    if not mappings:
        aggregate_rejections["no_matching_atomic_write"] += 1
    return mappings, sorted(aggregate_rejections)


def _summary(
    artifact_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    run_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible_artifacts = [row for row in artifact_rows if row["artifact_credit_eligible"]]
    structural = [row for row in action_rows if row["structural_action_eligible"]]
    trainable = [row for row in action_rows if row["token_loss_eligible"]]
    rejection_counts: Counter[str] = Counter()
    for row in artifact_rows:
        rejection_counts.update(row.get("rejection_reasons", []))
    for row in action_rows:
        rejection_counts.update(row.get("rejection_reasons", []))
    contributions = [float(row["reward_contribution"]) for row in eligible_artifacts]
    exact_tasks = {row["task_id"] for row in trainable}
    ready = len(trainable) >= 100 and len(exact_tasks) >= 30
    return {
        "cross_parent_tasks_processed": len(task_rows),
        "chosen_artifacts_tested": len(artifact_rows),
        "leave_one_out_credit_eligible_artifacts": len(eligible_artifacts),
        "tasks_with_credit_eligible_artifact": len(
            {row["task_id"] for row in eligible_artifacts}
        ),
        "atomic_write_mappings": len(action_rows),
        "structural_action_eligible": len(structural),
        "tasks_with_structural_action": len({row["task_id"] for row in structural}),
        "exact_token_loss_eligible": len(trainable),
        "tasks_with_exact_token_loss_action": len(exact_tasks),
        "retokenized_count_match_actions": sum(
            row.get("token_fidelity") == "retokenized_count_match"
            for row in action_rows
        ),
        "mean_positive_artifact_credit": (
            statistics.mean(contributions) if contributions else 0.0
        ),
        "median_positive_artifact_credit": (
            statistics.median(contributions) if contributions else 0.0
        ),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "verifier_queries": sum(int(row["verifier_queries"]) for row in run_stats),
        "verifier_seconds": sum(float(row["verifier_seconds"]) for row in run_stats),
        "training_readiness_threshold": {
            "exact_actions": 100,
            "distinct_tasks": 30,
        },
        "ready_for_token_level_ab_test": ready,
        "decision": (
            "ready-for-small-ab"
            if ready
            else "instrument-and-collect-more-exact-token-trajectories"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-report", type=Path, action="append", required=True)
    parser.add_argument("--proxy-records", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--checker-timeout", type=int, default=120)
    parser.add_argument("--max-file-mib", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.phase0_report) != len(args.proxy_records):
        raise ValueError("--phase0-report and --proxy-records counts must match")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    excluded = {str(value) for value in args.exclude_task}
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True, use_fast=True
    )

    artifact_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    run_stats: list[dict[str, Any]] = []
    temporary_root = args.output_dir / ".queries"
    temporary_root.mkdir()
    started = time.monotonic()
    try:
        for run_index, (report_path, record_path) in enumerate(
            zip(args.phase0_report, args.proxy_records, strict=True)
        ):
            report = _load_report(report_path)
            targets = _target_task_rows(report, excluded)
            session_map, steps_by_session, response_counts = _load_proxy_records(
                record_path, targets
            )
            analysis_dir = report_path.parent
            for task in targets:
                per_task, stats = _evaluate_artifacts(
                    task,
                    analysis_dir,
                    temporary_root / f"run_{run_index}",
                    args.checker_timeout,
                    args.max_file_mib * 1024**2,
                )
                for artifact in per_task:
                    artifact["run_index"] = run_index
                    artifact["phase0_report"] = str(report_path.resolve())
                    if artifact["artifact_credit_eligible"]:
                        mappings, map_rejections = _map_actions(
                            artifact,
                            task,
                            session_map,
                            steps_by_session,
                            response_counts,
                            tokenizer,
                        )
                        artifact["action_mapping_rejection_reasons"] = map_rejections
                        for mapping in mappings:
                            mapping["run_index"] = run_index
                            mapping["reward_contribution"] = artifact[
                                "reward_contribution"
                            ]
                            mapping["improved_criteria"] = artifact[
                                "improved_criteria"
                            ]
                            mapping["regressed_criteria"] = artifact[
                                "regressed_criteria"
                            ]
                            action_rows.append(mapping)
                artifact_rows.extend(per_task)
                task_rows.append(
                    {
                        "run_index": run_index,
                        "task_id": task["task_id"],
                        "artifact_count": len(per_task),
                        "credit_eligible_artifacts": sum(
                            row["artifact_credit_eligible"] for row in per_task
                        ),
                    }
                )
                run_stats.append(stats)
                print(
                    f"{task['task_id']}: artifacts={len(per_task)} "
                    "eligible="
                    f"{sum(row['artifact_credit_eligible'] for row in per_task)}",
                    flush=True,
                )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    summary = _summary(artifact_rows, action_rows, task_rows, run_stats)
    summary.update(
        {
            "excluded_tasks": sorted(excluded),
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "tokenizer": str(args.tokenizer.resolve()),
            "scope": (
                "whole-file leave-one-out; positive no-regression artifacts; "
                "single observed write only"
            ),
        }
    )
    (args.output_dir / "artifact_credit_audit.jsonl").write_text(
        "".join(_json_dump_line(row) + "\n" for row in artifact_rows),
        encoding="utf-8",
    )
    (args.output_dir / "action_credit_candidates.jsonl").write_text(
        "".join(_json_dump_line(row) + "\n" for row in action_rows),
        encoding="utf-8",
    )
    (args.output_dir / "task_summary.jsonl").write_text(
        "".join(_json_dump_line(row) + "\n" for row in task_rows),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
