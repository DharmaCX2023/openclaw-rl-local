#!/usr/bin/env python3
"""Map verified mixed-effect artifact swaps back to logged policy tool calls.

Only an artifact produced by exactly one full-file ``write`` whose content is
byte-identical to the archived artifact is called ``exact_single_write``.
For such cases, the artifact transplant is equivalent to replaying that
logged write call on the earlier workspace.  Edits and exec-generated files
remain provenance-only evidence and are not promoted to action-level causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ACTION_RE = re.compile(r"^ACTION:\s+([^\s]+)\s+(\{.*\})\s*$")
WORKSPACE_PREFIXES = (
    "/root/.openclaw/workspace/",
    "/workspace/",
)


def _normalize_path(value: str) -> str:
    value = value.strip()
    for prefix in WORKSPACE_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    return str(PurePosixPath(value))


def _actions(log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(log_path.read_text(errors="replace").splitlines(), 1):
        match = ACTION_RE.match(line)
        if not match:
            continue
        tool, raw = match.groups()
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            continue
        path = arguments.get("path") or arguments.get("file_path")
        if tool not in {"write", "edit"} or not isinstance(path, str):
            continue
        record: dict[str, Any] = {
            "line": line_number,
            "tool": tool,
            "path": _normalize_path(path),
        }
        if tool == "write" and isinstance(arguments.get("content"), str):
            content = arguments["content"].encode("utf-8")
            record["content_sha256"] = hashlib.sha256(content).hexdigest()
            record["content_bytes"] = len(content)
        records.append(record)
    return records


def _archive_file_digest(archive: Path, relative: str) -> tuple[str, int] | None:
    names = {relative, f"./{relative}"}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if member.name not in names or not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is None:
                return None
            data = extracted.read()
            return hashlib.sha256(data).hexdigest(), len(data)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-audit", type=Path, required=True)
    parser.add_argument("--swap-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    factor = json.loads(args.factor_audit.resolve(strict=True).read_text())
    swap = json.loads(args.swap_audit.resolve(strict=True).read_text())
    metadata = {
        (step, record["task_id"]): {
            "instance_log": record.get("instance_log"),
            "agent_steps": record.get("agent_steps"),
            "factor_count": record.get("factor_count"),
        }
        for step, step_payload in factor["eval_steps"].items()
        for record in step_payload["records"]
    }
    output_records: list[dict[str, Any]] = []
    for pair in swap["records"]:
        right_metadata = metadata.get((pair["right_step"], pair["task_id"]), {})
        log_value = right_metadata.get("instance_log")
        parsed_actions = _actions(Path(log_value)) if log_value and Path(log_value).is_file() else []
        for intervention in pair["interventions"]:
            # H2's denominator is failure-relevant interventions: an artifact
            # must actually correct at least one pre-existing failed factor.
            # Skipping non-corrections also avoids repeatedly decompressing a
            # workspace archive for causally inert files.
            if not intervention.get("checker_ok") or not intervention.get("corrections"):
                continue
            artifact = intervention["artifact"]
            matching = [action for action in parsed_actions if action["path"] == artifact]
            archive_digest = None
            if len(matching) == 1 and matching[0]["tool"] == "write":
                archive_digest = _archive_file_digest(Path(pair["right_archive"]), artifact)
            exact_single_write = False
            if len(matching) == 1 and matching[0]["tool"] == "write" and archive_digest:
                exact_single_write = (
                    matching[0].get("content_sha256") == archive_digest[0]
                    and matching[0].get("content_bytes") == archive_digest[1]
                )
            output_records.append({
                "task_id": pair["task_id"],
                "transition": f"{pair['left_step']}->{pair['right_step']}",
                "artifact": artifact,
                "corrections": intervention["corrections"],
                "regressions": intervention["regressions"],
                "strict_mixed_effect": bool(
                    intervention["corrections"] and intervention["regressions"]
                ),
                "repeat_stable_if_mixed": intervention.get("repeat_stable"),
                "right_instance_log": log_value,
                "right_agent_steps": right_metadata.get("agent_steps"),
                "factor_count": right_metadata.get("factor_count"),
                "matching_mutating_actions": matching,
                "matching_mutating_action_count": len(matching),
                "archive_file_sha256": archive_digest[0] if archive_digest else None,
                "archive_file_bytes": archive_digest[1] if archive_digest else None,
                "exact_single_write": exact_single_write,
                "claim_scope": (
                    "action-replay-equivalent on the earlier workspace"
                    if exact_single_write
                    else "artifact-level causality with action provenance only"
                ),
            })
    mixed = [record for record in output_records if record["strict_mixed_effect"]]
    mapped_mixed = [record for record in mixed if record["matching_mutating_action_count"]]
    exact = [record for record in output_records if record["exact_single_write"]]
    exact_correction = [record for record in exact if record["corrections"]]
    exact_mixed = [record for record in exact_correction if record["regressions"]]
    exact_mixed_stable = [
        record for record in exact_mixed if record["repeat_stable_if_mixed"]
    ]
    exact_tasks = {(record["transition"], record["task_id"]) for record in exact_mixed_stable}
    payload = {
        "schema_version": 1,
        "summary": {
            "artifact_interventions_ok": len(output_records),
            "strict_mixed_artifacts": len(mixed),
            "mixed_artifacts_with_direct_write_or_edit_provenance": len(mapped_mixed),
            "mixed_direct_provenance_rate": len(mapped_mixed) / len(mixed) if mixed else 0.0,
            "exact_single_write_artifacts": len(exact),
            "exact_single_write_rate": len(exact) / len(output_records) if output_records else 0.0,
            "failure_relevant_exact_single_writes": len(exact_correction),
            "mixed_credit_exact_single_writes": len(exact_mixed_stable),
            "rho_mixed_action_replay": (
                len(exact_mixed_stable) / len(exact_correction)
                if exact_correction else 0.0
            ),
            "task_pairs_with_mixed_exact_single_write": len(exact_tasks),
            "mixed_exact_single_write_by_transition": dict(sorted(Counter(
                record["transition"] for record in exact_mixed_stable
            ).items())),
            "interpretation": (
                "exact_single_write means the logged policy write content is byte-identical "
                "to the transplanted artifact, so the intervention replays that action's state "
                "effect on the earlier workspace. This still does not estimate the regression "
                "caused by an OPD gradient update."
            ),
        },
        "records": output_records,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
