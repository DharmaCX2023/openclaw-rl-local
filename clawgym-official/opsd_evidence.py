"""Evidence Anchor parsing, validation, storage loading, and prompt rendering."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any


_ANCHOR_FIELDS = ("condition", "artifact", "verified_fact", "explanation")


def _clean_text(value: Any, field: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"anchor {field} must be a string")
    text = value.strip()
    if required and not text:
        raise ValueError(f"anchor {field} must not be empty")
    if "\x00" in text:
        raise ValueError(f"anchor {field} contains NUL")
    return text


def normalize_evidence_anchor(value: Any) -> dict[str, str]:
    """Normalize one structured anchor while retaining legacy string input."""

    if isinstance(value, str):
        return {
            "condition": _clean_text(value, "condition"),
            "artifact": "",
            "verified_fact": "",
            "explanation": "",
        }
    if not isinstance(value, dict):
        raise ValueError("anchor must be a string or object")
    unknown = set(value) - set(_ANCHOR_FIELDS) - {"anchor_id", "source"}
    if unknown:
        raise ValueError(f"anchor has unsupported fields: {sorted(unknown)}")
    return {
        "condition": _clean_text(value.get("condition"), "condition"),
        "artifact": _clean_text(
            value.get("artifact", ""), "artifact", required=False
        ),
        "verified_fact": _clean_text(
            value.get("verified_fact", ""), "verified_fact", required=False
        ),
        "explanation": _clean_text(
            value.get("explanation", ""), "explanation", required=False
        ),
    }


def parse_evidence_anchor_xml(text: str, *, task_id: str) -> dict[str, Any]:
    """Parse the strict XML emitted by the offline Anchor generator."""

    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("DTD and entities are not allowed in anchor XML")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid anchor XML: {exc}") from exc
    if root.tag != "evidences" or root.attrib:
        raise ValueError("anchor XML root must be an attribute-free <evidences>")

    anchors: list[dict[str, str]] = []
    for node in root:
        if node.tag != "evidence" or node.attrib:
            raise ValueError(
                "<evidences> may contain only attribute-free <evidence> nodes"
            )
        children: dict[str, str] = {}
        for child in node:
            if (
                child.tag not in _ANCHOR_FIELDS
                or child.tag in children
                or child.attrib
                or list(child)
            ):
                raise ValueError(
                    "each evidence must contain one flat copy of every anchor field"
                )
            children[child.tag] = "".join(child.itertext())
        if set(children) != set(_ANCHOR_FIELDS):
            raise ValueError(f"incomplete evidence fields: {sorted(children)}")
        anchors.append(normalize_evidence_anchor(children))
    if not anchors:
        raise ValueError("anchor XML contains no evidence")
    return {
        "schema_version": "clawgym_evidence_anchor_v1",
        "task_id": _clean_text(task_id, "task_id"),
        "anchors": anchors,
    }


def _records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return payload["tasks"]
    if isinstance(payload, dict) and "task_id" in payload:
        return [payload]
    if isinstance(payload, dict):
        return [
            {"task_id": str(task_id), "anchors": anchors}
            for task_id, anchors in payload.items()
        ]
    if isinstance(payload, list):
        return payload
    raise ValueError("unsupported Evidence Anchor file shape")


def load_evidence_anchors(
    path: str | Path, task_id: str
) -> list[dict[str, str]]:
    """Load normalized anchors from JSON or JSONL internal storage."""

    anchor_path = Path(path)
    raw = anchor_path.read_text(encoding="utf-8")
    if anchor_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        records = _records_from_payload(json.loads(raw))
    matches = [
        row
        for row in records
        if isinstance(row, dict) and str(row.get("task_id")) == task_id
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate Evidence Anchor record for {task_id}")
    if not matches:
        return []
    anchors = matches[0].get("anchors", [])
    if isinstance(anchors, str):
        anchors = [anchors]
    if not isinstance(anchors, list):
        raise ValueError("record anchors must be a list")
    return [normalize_evidence_anchor(anchor) for anchor in anchors]


@lru_cache(maxsize=8)
def load_evidence_anchor_index(path: str | Path) -> dict[str, list[dict[str, str]]]:
    """Load and normalize an Evidence Anchor file once per process."""

    anchor_path = Path(path)
    raw = anchor_path.read_text(encoding="utf-8")
    if anchor_path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        records = _records_from_payload(json.loads(raw))
    result: dict[str, list[dict[str, str]]] = {}
    for row in records:
        if not isinstance(row, dict) or "task_id" not in row:
            raise ValueError("anchor record must contain task_id")
        task_id = _clean_text(str(row["task_id"]), "task_id")
        if task_id in result:
            raise ValueError(f"duplicate Evidence Anchor record for {task_id}")
        anchors = row.get("anchors", [])
        if isinstance(anchors, str):
            anchors = [anchors]
        if not isinstance(anchors, list):
            raise ValueError("record anchors must be a list")
        result[task_id] = [normalize_evidence_anchor(anchor) for anchor in anchors]
    return result


@lru_cache(maxsize=8)
def load_placebo_mapping(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a deterministic task-to-placebo mapping produced offline."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "clawgym_matched_placebo_v1":
        raise ValueError("unsupported matched-placebo mapping schema")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict):
        raise ValueError("matched-placebo mapping tasks must be an object")
    return tasks


def render_teacher_feedback(feedback: dict[str, Any]) -> str:
    """Render privileged feedback as plain text, never XML/HTML markup."""

    guidance_mode = str(feedback.get("teacher_guidance_mode", "opsd"))
    if guidance_mode not in {"opsd", "sdar"}:
        raise ValueError("unsupported_teacher_guidance_mode")
    if guidance_mode == "opsd" and feedback.get("rollout_not_fully_successful") is not True:
        raise ValueError("rollout_not_fully_successful_required")
    anchors = [
        normalize_evidence_anchor(value)
        for value in feedback.get("evidence_anchors") or []
    ]
    if not anchors:
        raise ValueError("missing_evidence_anchors")

    lines = [
        "Teacher-only auxiliary context",
        "==============================",
        "",
        (
            "The following evidence is provided only as auxiliary context "
            "for scoring the recorded assistant actions."
        ),
    ]
    for index, anchor in enumerate(anchors, start=1):
        lines.extend(
            ["", f"Evidence {index}", f"Condition: {anchor['condition']}"]
        )
        if anchor["artifact"]:
            lines.append(f"Relevant artifact: {anchor['artifact']}")
        if anchor["verified_fact"]:
            lines.append(f"Verified fact: {anchor['verified_fact']}")
        if anchor["explanation"]:
            lines.append(f"Why it matters: {anchor['explanation']}")
    lines.extend(
        [
            "",
            "Do not reproduce or explicitly discuss this auxiliary context.",
        ]
    )
    return "\n".join(lines)
