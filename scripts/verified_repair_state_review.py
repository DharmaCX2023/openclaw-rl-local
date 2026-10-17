"""Evidence-backed review of known OpenClaw initialization-only file drift.

No workspace is changed. Never erase dates from arbitrary task files or tool
observations. Terminal bytes may stand in for pre-action bytes only when their
SHA256 exactly equals the recorded pre-action fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile

STATE_PATH = ".openclaw/workspace-state.json"
TIMESTAMPS = {"bootstrapSeededAt", "setupCompletedAt"}
ISO_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z")


def read_proven_bytes(archive, fingerprint):
    if fingerprint.get("type") != "file":
        raise ValueError("state metadata is not a regular file")
    with tarfile.open(Path(archive), "r:gz") as handle:
        members = [member for member in handle.getmembers()
                   if member.name.removeprefix("./") == STATE_PATH]
        if len(members) != 1 or not members[0].isfile() or members[0].size > 4096:
            raise ValueError("missing, duplicate or oversized runtime metadata")
        data = handle.extractfile(members[0]).read()
    if hashlib.sha256(data).hexdigest() != fingerprint["sha256"]:
        raise ValueError("terminal metadata is not proven equal to pre-action metadata")
    return data


def canonical_runtime_metadata(data):
    value = json.loads(data)
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("unsupported runtime metadata version")
    if not set(value) <= {"version"} | TIMESTAMPS:
        raise ValueError("unknown field: this is not an initialization-only exception")
    if not set(value) & TIMESTAMPS:
        raise ValueError("no known initialization timestamp")
    for key in set(value) & TIMESTAMPS:
        if not isinstance(value[key], str) or not ISO_UTC.fullmatch(value[key]):
            raise ValueError("invalid initialization timestamp")
        # Presence remains semantic: setup completion is not the same as merely
        # seeding bootstrap files. Only the timestamp value is a nuisance.
        value[key] = "<validated-initialization-timestamp>"
    return value


def review_runtime_drift(source, replay):
    branch = replay.get("replay", {})
    comparison = branch.get("pre_action_file_comparison", {})
    report = {"eligible": False, "policy": "openclaw-initialization-metadata-only-v1",
              "observations_normalized": False, "workspace_modified": False,
              "process_and_external_state_proven": False}
    if replay.get("status") != "completed" or not branch.get("replacement_done"):
        return report | {"reason": "replay_not_completed"}
    if branch.get("observation_differences"):
        return report | {"reason": "tool_observation_difference_requires_review"}
    if comparison.get("status") == "match":
        return report | {"eligible": True, "reason": "exact_files_and_observations"}
    if comparison.get("changed_paths") != [STATE_PATH]:
        return report | {"reason": "other_workspace_difference"}
    try:
        step = source["steps"][branch["target_step"] - 1]
        original = step["file_state"]["files"][STATE_PATH]
        current = branch["actual_pre_action_file_state"]["files"][STATE_PATH]
        if original["mode"] != current["mode"]:
            raise ValueError("permissions changed")
        old_bytes = read_proven_bytes(source["terminal_snapshot"]["archive"], original)
        new_bytes = read_proven_bytes(replay["terminal_snapshot"]["archive"], current)
        old_state = canonical_runtime_metadata(old_bytes)
        new_state = canonical_runtime_metadata(new_bytes)
        if old_state != new_state:
            raise ValueError("bootstrap/setup state changed, not just its timestamp")
        report.update(eligible=True, reason="proven_initialization_timestamp_only",
                      changed_path=STATE_PATH, source_pre_action_sha256=original["sha256"],
                      replay_pre_action_sha256=current["sha256"],
                      pre_action_bytes_proven_by_terminal_hash=True,
                      normalized_fields=sorted(set(old_state) & TIMESTAMPS),
                      upstream_writer="openclaw/src/agents/workspace.ts:400")
    except (ValueError, KeyError, IndexError, OSError, tarfile.TarError) as exc:
        report.update(reason="metadata_exception_not_proven", error_type=type(exc).__name__,
                      detail=str(exc)[:180])
    return report
