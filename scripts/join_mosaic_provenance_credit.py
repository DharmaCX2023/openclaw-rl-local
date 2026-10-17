#!/usr/bin/env python3
"""Join verifier-positive artifacts to exact provenance action token spans."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import extract_mosaic_artifact_credit as credit


MUTATING_TOOLS = {"write", "edit", "exec", "process"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _load_rollout_records(
    path: Path, target_tasks: set[str]
) -> tuple[
    dict[tuple[str, int], str],
    dict[tuple[str, int], int],
    dict[tuple[str, int], dict[str, Any]],
]:
    session_by_task_index: dict[tuple[str, int], str] = {}
    response_counts: dict[tuple[str, int], int] = {}
    raw_rows = _load_jsonl(path)
    for row in raw_rows:
        if row.get("record_type") != "trajectory_audit":
            continue
        task_id = str(row.get("task_id"))
        if task_id not in target_tasks:
            continue
        session_id = str(row["session_id"])
        session_by_task_index[(task_id, int(row["task_index"]))] = session_id
        for step in row.get("steps", []):
            response_counts[(session_id, int(step["step"]))] = int(
                step.get("response_id_count", step.get("response_tokens", -1))
            )
    target_sessions = set(session_by_task_index.values())
    steps = {
        (str(row["session_id"]), int(row["step"])): row
        for row in raw_rows
        if row.get("record_type") is None
        and str(row.get("session_id")) in target_sessions
    }
    return session_by_task_index, response_counts, steps


def _selected_digest(version_key: str) -> str | None:
    parsed = credit._parse_version_key(version_key)
    return parsed[2] if parsed is not None else None


def _write_matches_digest(action: dict[str, Any], digest: str) -> bool:
    arguments = action.get("arguments")
    content = arguments.get("content") if isinstance(arguments, dict) else None
    return isinstance(content, str) and hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest() == digest


def _choose_action(
    actions: list[dict[str, Any]], relative: str, digest: str
) -> tuple[dict[str, Any] | None, list[str]]:
    direct = []
    for action in actions:
        if action.get("path") != relative or action.get("name") not in {"write", "edit"}:
            continue
        if action["name"] == "write" and not _write_matches_digest(action, digest):
            continue
        direct.append(action)
    if len(direct) == 1:
        return direct[0], []
    if len(direct) > 1:
        return None, ["multiple_direct_actions_match_artifact"]
    mutating = [value for value in actions if value.get("name") in MUTATING_TOOLS]
    if len(mutating) == 1:
        return mutating[0], []
    if not mutating:
        return None, ["no_mutating_action_in_provenance_event"]
    return None, ["ambiguous_multi_mutating_action_group"]


def _later_path_mutation(
    events: list[dict[str, Any]], event_index: int, relative: str
) -> bool:
    return any(
        mutation.get("path") == relative
        for event in events
        if int(event.get("event_index", -1)) > event_index
        for mutation in event.get("mutations", [])
    )


def _main(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    report = json.loads(args.phase0_report.read_text(encoding="utf-8"))
    report_tasks = {str(row["task_id"]): row for row in report["tasks"]}
    artifacts = [
        row
        for row in _load_jsonl(args.artifact_credit_audit)
        if row.get("artifact_credit_eligible")
    ]
    target_tasks = {str(row["task_id"]) for row in artifacts}
    session_map, response_counts, step_records = _load_rollout_records(
        args.proxy_records, target_tasks
    )
    events_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance_errors = 0
    for row in _load_jsonl(args.provenance_records):
        if row.get("record_type") == "workspace_provenance_error":
            provenance_errors += 1
        elif row.get("record_type") == "workspace_provenance_event":
            events_by_session[str(row["session_id"])].append(row)
    for rows in events_by_session.values():
        rows.sort(key=lambda value: int(value["event_index"]))
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer), trust_remote_code=True, use_fast=True
    )

    audit_rows = []
    raw_candidates = []
    for artifact in artifacts:
        task_id = str(artifact["task_id"])
        relative = str(artifact["path"])
        digest = _selected_digest(str(artifact["selected_version"]))
        row = {
            "task_id": task_id,
            "path": relative,
            "reward_contribution": artifact["reward_contribution"],
            "improved_criteria": artifact["improved_criteria"],
            "matches": [],
            "exact_action_matches": 0,
        }
        if digest is None:
            row["rejection_reasons"] = ["artifact_version_is_not_regular_file"]
            audit_rows.append(row)
            continue
        task = report_tasks[task_id]
        for parent_index in artifact["source_parent_indices"]:
            task_index = int(task["parent_replay"][parent_index]["task_index"])
            session_id = session_map.get((task_id, task_index))
            if session_id is None:
                row["matches"].append({"rejection_reasons": ["source_session_missing"]})
                continue
            session_events = events_by_session.get(session_id, [])
            for event in session_events:
                matching_mutations = [
                    mutation
                    for mutation in event.get("mutations", [])
                    if mutation.get("path") == relative
                    and (mutation.get("after") or {}).get("sha256") == digest
                ]
                if not matching_mutations:
                    continue
                match = {
                    "session_id": session_id,
                    "task_index": task_index,
                    "parent_index": parent_index,
                    "event_index": event["event_index"],
                    "source_step": event["source_step"],
                    "tool_names": event["tool_names"],
                    "attribution_unit": event["attribution_unit"],
                    "rejection_reasons": [],
                }
                if not event.get("provenance_eligible"):
                    match["rejection_reasons"].append("provenance_event_ineligible")
                    match["rejection_reasons"].extend(event.get("rejection_reasons", []))
                if _later_path_mutation(
                    session_events, int(event["event_index"]), relative
                ):
                    match["rejection_reasons"].append("later_mutation_same_path")
                step_key = (session_id, int(event["source_step"]))
                step_record = step_records.get(step_key)
                if step_record is None:
                    match["rejection_reasons"].append("source_step_record_missing")
                    row["matches"].append(match)
                    continue
                actions, parse_errors = credit._all_actions([step_record])
                if parse_errors:
                    match["rejection_reasons"].append("raw_tool_call_parse_error")
                action, choose_reasons = _choose_action(actions, relative, digest)
                match["rejection_reasons"].extend(choose_reasons)
                if action is None:
                    row["matches"].append(match)
                    continue
                if action.get("tool_call_id") not in event.get("tool_call_ids", []):
                    match["rejection_reasons"].append("chosen_action_not_in_event")
                span = credit._token_span(
                    tokenizer,
                    action,
                    response_counts.get(step_key),
                )
                if span["token_fidelity"] != "exact_recorded_ids":
                    match["rejection_reasons"].append("response_ids_not_exact")
                if span["response_token_start"] is None:
                    match["rejection_reasons"].append("action_token_span_missing")
                match.update(
                    {
                        "tool_call_id": action.get("tool_call_id"),
                        "tool_name": action.get("name"),
                        "tool_call_index": action.get("call_index"),
                        **span,
                    }
                )
                match["exact_action_eligible"] = not match["rejection_reasons"]
                if match["exact_action_eligible"]:
                    row["exact_action_matches"] += 1
                    raw_candidates.append(
                        {
                            "task_id": task_id,
                            "session_id": session_id,
                            "task_index": task_index,
                            "step": int(event["source_step"]),
                            "tool_call_index": int(action["call_index"]),
                            "tool_call_id": action["tool_call_id"],
                            "tool_name": action["name"],
                            "response_token_start": span["response_token_start"],
                            "response_token_end": span["response_token_end"],
                            "tool_call_token_count": span["tool_call_token_count"],
                            "token_fidelity": span["token_fidelity"],
                            "artifact_path": relative,
                            "artifact_sha256": digest,
                            "reward_contribution": float(
                                artifact["reward_contribution"]
                            ),
                            "improved_criteria": artifact["improved_criteria"],
                            "attribution_unit": event["attribution_unit"],
                        }
                    )
                row["matches"].append(match)
        if not row["matches"]:
            row["rejection_reasons"] = ["no_final_hash_provenance_match"]
        audit_rows.append(row)

    aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in raw_candidates:
        key = (
            candidate["session_id"],
            candidate["step"],
            candidate["tool_call_index"],
        )
        if key not in aggregated:
            aggregated[key] = {
                **candidate,
                "artifact_paths": [candidate.pop("artifact_path")],
                "artifact_sha256s": [candidate.pop("artifact_sha256")],
                "artifact_credits": [candidate["reward_contribution"]],
            }
        else:
            value = aggregated[key]
            value["artifact_paths"].append(candidate["artifact_path"])
            value["artifact_sha256s"].append(candidate["artifact_sha256"])
            value["artifact_credits"].append(candidate["reward_contribution"])
            value["reward_contribution"] = max(
                value["reward_contribution"], candidate["reward_contribution"]
            )
            value["improved_criteria"] = sorted(
                set(value["improved_criteria"]) | set(candidate["improved_criteria"])
            )
    candidates = list(aggregated.values())
    for value in candidates:
        value["artifact_paths"] = sorted(value["artifact_paths"])
        value["artifact_sha256s"] = sorted(value["artifact_sha256s"])
        value["artifact_credits"] = sorted(value["artifact_credits"], reverse=True)
        value["credit_aggregation"] = "max_not_sum"
    candidates.sort(key=lambda row: (row["task_id"], row["session_id"], row["step"]))
    summary = {
        "verifier_positive_artifacts": len(artifacts),
        "artifacts_with_any_provenance_match": sum(bool(row["matches"]) for row in audit_rows),
        "artifacts_with_exact_action": sum(row["exact_action_matches"] > 0 for row in audit_rows),
        "exact_action_candidates": len(candidates),
        "tasks_with_exact_action": len({row["task_id"] for row in candidates}),
        "single_exec_actions": sum(row["tool_name"] == "exec" for row in candidates),
        "multi_tool_turn_actions": sum(
            row["attribution_unit"] == "action_group" for row in candidates
        ),
        "provenance_errors": provenance_errors,
        "credit_aggregation": "max artifact credit per action; never sum",
    }
    for name, rows in (
        ("provenance_artifact_audit.jsonl", audit_rows),
        ("exact_action_candidates.jsonl", candidates),
    ):
        (args.output_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-report", type=Path, required=True)
    parser.add_argument("--artifact-credit-audit", type=Path, required=True)
    parser.add_argument("--proxy-records", type=Path, required=True)
    parser.add_argument("--provenance-records", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(_main(parse_args()), ensure_ascii=False, indent=2))
