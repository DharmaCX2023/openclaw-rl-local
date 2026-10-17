#!/usr/bin/env python3
"""Run the matched-anchor scorer with a task-solving teacher instruction."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "clawgym-official"))
import opsd_evidence


def render_action_grounded_feedback(feedback):
    anchors = [
        opsd_evidence.normalize_evidence_anchor(value)
        for value in feedback.get("evidence_anchors") or []
    ]
    if not anchors:
        raise ValueError("missing_evidence_anchors")
    lines = [
        "Teacher-only task evidence",
        "==========================",
        "",
        "Continue solving the original task from the exact current state.",
        (
            "Use the evidence below to favor the next action only when it "
            "obtains missing evidence, creates or corrects a required "
            "artifact, or validates a requirement. A failed, repeated, or "
            "no-op action is not progress until it is corrected."
        ),
    ]
    for index, anchor in enumerate(anchors, start=1):
        lines.extend(["", f"Evidence {index}", f"Requirement: {anchor['condition']}"])
        if anchor["artifact"]:
            lines.append(f"Relevant artifact: {anchor['artifact']}")
        if anchor["verified_fact"]:
            lines.append(f"Observable support: {anchor['verified_fact']}")
        if anchor["explanation"]:
            lines.append(f"Action relevance: {anchor['explanation']}")
    lines.extend(["", "Do not quote or discuss this teacher-only evidence."])
    return "\n".join(lines)


opsd_evidence.render_teacher_feedback = render_action_grounded_feedback
namespace = runpy.run_path(
    str(__file__).replace(
        "analyze_action_grounded_anchor_phase0.py",
        "analyze_matched_placebo_phase0.py",
    ),
    run_name="__main__",
)
