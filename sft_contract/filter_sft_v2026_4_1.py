#!/usr/bin/env python3
"""Build the OpenClaw v2026.4.1 six-tool, fixed-skill SFT dataset.

This script does not train.  It only validates, normalizes, length-filters, and
emits an auditable rejection file whose records retain the source line number.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ALLOWED_TOOLS = ("read", "edit", "write", "exec", "process", "web_fetch")
DISALLOWED_EXEC_ARGS = {"ask", "elevated", "host", "node", "security"}

# Canonical recovered Serper skill and the whitespace-only raw variant.
SERPER_CANONICAL_SHA256 = "a24e8ad9ba871137952ac143d8a2db867694bd24616d01939e0a4ae958b6fe67"
SERPER_WHITESPACE_SHA256 = "74a3d07db423f37c03363fa0935dd2332538aebd2bc1897532cd6d0c10081672"
SERPER_OLD_SHA256 = "9b161a88e59fb749880d0908d491627dead7a3f06fc03a28227d2af5802937b0"
SERPER_TRUNCATED_SHA256 = "eb786855125db9e7d462105eccd83e2c8e6af4605aad00207947af2488e6aa0d"

# These limited reads came from a different bundled-skill revision than the
# pinned v2026.4.1 tree.  All three occur in source line 21425.
MISMATCHED_BUNDLED_SKILL_SHA256 = {
    "1821215fccb10184ad5074a07fcbb39dcb68f9d3ac29113514221dbc7c92039e": "gh-issues",
    "59a412d34ad5b8d50a54192aaf58cdb528b744fd850d94e449101f39b68fd407": "github",
    "7bbce1d7819d9ba9222b802bcac94c59dcaafcaa3129512d0445ab77a4d7d6f7": "weather",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(message.get("content") or ""):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            calls.append({"name": "__parse_error__", "_error": str(exc)})
            continue
        calls.append(payload if isinstance(payload, dict) else {"name": "__parse_error__"})
    # The published rows primarily use XML calls, but support native calls too.
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                calls.append({"name": "__parse_error__", "_error": str(exc)})
                continue
        calls.append({"name": function.get("name"), "arguments": arguments})
    return calls


def iter_calls(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for message in row.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            yield from parse_calls(message)


def validate_calls(
    row: dict[str, Any],
    validators: dict[str, Draft7Validator],
) -> list[str]:
    reasons: list[str] = []
    for call in iter_calls(row):
        name = call.get("name")
        if name == "__parse_error__":
            reasons.append("tool_call_parse_error")
            continue
        if name not in validators:
            reasons.append(f"tool_outside_six:{name}")
            continue
        args = call.get("arguments")
        if not isinstance(args, dict):
            reasons.append(f"schema_incompatible:{name}:arguments_not_object")
            continue
        if name == "exec":
            blocked = sorted(set(args) & DISALLOWED_EXEC_ARGS)
            if blocked:
                reasons.append("exec_policy_override:" + ",".join(blocked))
        errors = sorted(validators[name].iter_errors(args), key=lambda e: list(e.path))
        if errors:
            detail = errors[0].message.replace("\n", " ")[:180]
            reasons.append(f"schema_incompatible:{name}:{detail}")
    return sorted(set(reasons))


def is_managed_serper_path(path: str) -> bool:
    path = path.replace("\\", "/")
    return (
        path in {
            "/root/.openclaw/skills/serper/SKILL.md",
            "~/.openclaw/skills/serper/SKILL.md",
        }
        or path.endswith("/.openclaw/skills/serper/SKILL.md")
    )


def normalize_and_audit_skill_observations(
    row: dict[str, Any],
    canonical_serper: str,
) -> tuple[list[str], Counter[str]]:
    """Pair calls with tool results in execution order and audit runtime skills."""
    reasons: list[str] = []
    stats: Counter[str] = Counter()
    pending: list[dict[str, Any]] = []

    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            pending.extend(parse_calls(message))
            continue
        if message.get("role") != "tool" or not pending:
            continue

        call = pending.pop(0)
        if call.get("name") != "read":
            continue
        args = call.get("arguments") or {}
        path = str(args.get("path") or args.get("file_path") or args.get("filePath") or args.get("file") or "")
        if "SKILL.md" not in path:
            continue

        content = message.get("content")
        if not isinstance(content, str):
            continue
        digest = sha256_text(content)

        if is_managed_serper_path(path):
            if digest == SERPER_CANONICAL_SHA256:
                stats["serper_canonical"] += 1
            elif digest == SERPER_WHITESPACE_SHA256:
                message["content"] = canonical_serper
                stats["serper_whitespace_normalized"] += 1
            elif digest == SERPER_OLD_SHA256:
                reasons.append("runtime_skill_revision_mismatch:serper_old")
            elif digest == SERPER_TRUNCATED_SHA256:
                stats["serper_limited_read"] += 1
            # Other observations are not skill contents: paths can be shadowed
            # by task files, and failed reads are valid/reproducible outcomes.

        mismatched_name = MISMATCHED_BUNDLED_SKILL_SHA256.get(digest)
        if mismatched_name:
            reasons.append(f"runtime_skill_revision_mismatch:{mismatched_name}")

    return sorted(set(reasons)), stats


def normalize_messages(messages: list[dict[str, Any]], system_prompt: str) -> tuple[list[dict[str, Any]], int]:
    result = []
    removed = 0
    for message in copy.deepcopy(messages):
        if message.get("role") == "assistant" and not (message.get("content") or "").strip() and not message.get("tool_calls"):
            removed += 1
            continue
        result.append(message)
    if result and result[0].get("role") == "system":
        result[0]["content"] = system_prompt
    else:
        result.insert(0, {"role": "system", "content": system_prompt})
    return result, removed


def reject_record(
    source_line: int,
    reasons: list[str],
    row: dict[str, Any],
    token_length: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "source_line": source_line,
        "reject_reasons": reasons,
        "trajectory": row,
    }
    if token_length is not None:
        record["token_length"] = token_length
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejects", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--contract-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    system_prompt = (contract_dir / "system_prompt.txt").read_text(encoding="utf-8")
    schemas = json.loads((contract_dir / "six_tool_schemas.json").read_text(encoding="utf-8"))
    canonical_serper = (contract_dir / "recovered_skills/serper/SKILL.md").read_text(encoding="utf-8")
    validators = {
        item["function"]["name"]: Draft7Validator(item["function"]["parameters"])
        for item in schemas
    }
    assert tuple(validators) == ALLOWED_TOOLS
    assert sha256_text(canonical_serper) == SERPER_CANONICAL_SHA256

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    input_path = Path(args.input)
    output_path = Path(args.output)
    rejects_path = Path(args.rejects)
    report_path = Path(args.report)
    for path in (output_path, rejects_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    skill_stats: Counter[str] = Counter()
    schema_audit_counts: Counter[str] = Counter()
    schema_audit_source_lines: dict[str, list[int]] = {}
    rejected_lines: dict[str, list[int]] = {}
    kept_lengths: list[int] = []
    all_candidate_lengths: list[int] = []
    batch: list[tuple[int, dict[str, Any]]] = []

    def write_reject(dst: Any, line_no: int, reasons: list[str], row: dict[str, Any], length: int | None = None) -> None:
        counts["rejected"] += 1
        for reason in reasons:
            reason_counts[reason] += 1
            rejected_lines.setdefault(reason, []).append(line_no)
        dst.write(json.dumps(reject_record(line_no, reasons, row, length), ensure_ascii=False) + "\n")

    def flush_batch(output_dst: Any, reject_dst: Any) -> None:
        if not batch:
            return
        rendered = [
            tokenizer.apply_chat_template(
                row["messages"],
                tools=row["tools"],
                tokenize=False,
                add_generation_prompt=False,
            )
            for _, row in batch
        ]
        encoded = tokenizer(rendered, add_special_tokens=False, return_length=True, verbose=False)
        lengths = encoded["length"]
        for (line_no, row), length in zip(batch, lengths, strict=True):
            length = int(length)
            all_candidate_lengths.append(length)
            if length > args.max_tokens:
                write_reject(reject_dst, line_no, ["length_over_64k"], row, length)
            else:
                output_dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["kept"] += 1
                kept_lengths.append(length)
        batch.clear()

    with (
        input_path.open(encoding="utf-8") as source,
        output_path.open("w", encoding="utf-8") as output_dst,
        rejects_path.open("w", encoding="utf-8") as reject_dst,
    ):
        for line_no, line in enumerate(source, 1):
            counts["seen"] += 1
            raw_row = json.loads(line)
            row = copy.deepcopy(raw_row)
            call_findings = validate_calls(row, validators)
            # Invalid six-tool arguments are authentic recovery steps: the
            # runtime returns a validation observation and the model can fix
            # the call later. Audit them without rejecting the full trajectory.
            schema_findings = [
                reason for reason in call_findings if reason.startswith("schema_incompatible:")
            ]
            for finding in schema_findings:
                schema_audit_counts[finding] += 1
                schema_audit_source_lines.setdefault(finding, []).append(line_no)
            reasons = [
                reason for reason in call_findings if not reason.startswith("schema_incompatible:")
            ]
            skill_reasons, per_row_skill_stats = normalize_and_audit_skill_observations(row, canonical_serper)
            reasons = sorted(set(reasons + skill_reasons))
            skill_stats.update(per_row_skill_stats)

            if reasons:
                write_reject(reject_dst, line_no, reasons, raw_row)
            else:
                row["messages"], removed = normalize_messages(row.get("messages") or [], system_prompt)
                counts["empty_assistant_messages_removed"] += removed
                row["tools"] = copy.deepcopy(schemas)
                batch.append((line_no, row))
                if len(batch) >= args.batch_size:
                    flush_batch(output_dst, reject_dst)

            if line_no % 500 == 0:
                print(
                    json.dumps(
                        {
                            "progress": line_no,
                            "kept": counts["kept"],
                            "rejected": counts["rejected"],
                        }
                    ),
                    flush=True,
                )
        flush_batch(output_dst, reject_dst)

    def length_summary(values: list[int]) -> dict[str, int] | None:
        if not values:
            return None
        values = sorted(values)
        return {
            "min": values[0],
            "p50": values[len(values) // 2],
            "p90": values[int(len(values) * 0.90) - 1],
            "p95": values[int(len(values) * 0.95) - 1],
            "max": values[-1],
        }

    report = {
        "contract": {
            "openclaw_version": "2026.4.1",
            "openclaw_commit": "da64a978e5814567f7797cc34fbe29b61b7eae7a",
            "allowed_tools": list(ALLOWED_TOOLS),
            "system_prompt_sha256": sha256_text(system_prompt),
            "tool_schemas_sha256": sha256_text((contract_dir / "six_tool_schemas.json").read_text(encoding="utf-8")),
            "max_tokens": args.max_tokens,
            "tokenizer": args.tokenizer,
        },
        "paths": {
            "input": str(input_path),
            "output": str(output_path),
            "rejects": str(rejects_path),
        },
        "counts": dict(counts),
        "reject_reason_counts": dict(reason_counts),
        "reject_source_lines": rejected_lines,
        "non_rejecting_schema_audit_counts": dict(schema_audit_counts),
        "non_rejecting_schema_audit_source_lines": schema_audit_source_lines,
        "skill_normalization": dict(skill_stats),
        "candidate_length_summary": length_summary(all_candidate_lengths),
        "kept_length_summary": length_summary(kept_lengths),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
