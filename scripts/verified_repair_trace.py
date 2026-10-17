"""Mechanical, paginated access to one trajectory; no verifier or file browser."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def content_hash(value: Any) -> str:
    return hashlib.sha256(text_content(value).encode()).hexdigest()


class TraceReader:
    def __init__(self, steps: list[dict]):
        self.steps = {int(step["agent_step"]): step for step in steps}
        if len(self.steps) != len(steps) or sorted(self.steps) != list(range(1, len(steps) + 1)):
            raise ValueError("agent step IDs must be unique and contiguous")
        self.access_log = []

    def read(self, step: int, part: str, offset: int = 0, max_chars: int = 12000) -> dict:
        if type(step) is not int or step not in self.steps:
            raise ValueError("unknown agent step")
        if part not in {"assistant", "observations", "context"}:
            raise ValueError("part must be assistant, observations, or context")
        if type(offset) is not int or offset < 0:
            raise ValueError("invalid character offset")
        if type(max_chars) is not int or not 1 <= max_chars <= 12000:
            raise ValueError("max_chars must be in [1,12000]")
        row = self.steps[step]
        field = {"assistant": "response_text", "observations": "observations", "context": "messages"}[part]
        if field not in row:
            raise ValueError("requested original record is missing")
        value = text_content(row[field])
        if offset > len(value):
            raise ValueError("offset beyond record")
        end = min(len(value), offset + max_chars)
        result = {
            "agent_step": step, "part": part, "offset": offset, "end_offset": end,
            "total_chars": len(value), "text": value[offset:end],
            "next_offset": end if end < len(value) else None,
            "has_more": end < len(value), "sha256": content_hash(value),
        }
        self.access_log.append({k: v for k, v in result.items() if k != "text"})
        return result

    def fully_read(self, step: int, part: str) -> bool:
        pages = sorted((item for item in self.access_log
                        if item["agent_step"] == step and item["part"] == part),
                       key=lambda item: item["offset"])
        cursor = 0
        for page in pages:
            if page["offset"] > cursor:
                return False
            cursor = max(cursor, page["end_offset"])
        return bool(pages) and cursor == pages[-1]["total_chars"]

    def directory(self) -> str:
        lines = ["Mechanical agent-step index. Excerpts are literal and incomplete.",
                 "A tool reporting success is not proof that the task requirement was met.",
                 "Use read_trace to inspect full assistant, observations, or pre-action context."]
        for number, row in self.steps.items():
            attempted = max(len(row.get("tool_calls", [])), row["response_text"].count("<tool_call>"))
            lines.append(f"\nAgent step {number}: assistant_chars={len(row['response_text'])}, "
                         f"attempted_tool_blocks={attempted}, parsed_tool_calls={len(row.get('tool_calls', []))}")
            for index, call in enumerate(row.get("tool_calls", []), 1):
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"unparsed_arguments": args}
                details = {}
                for key, value in args.items():
                    raw = text_content(value)
                    details[key] = raw if len(raw) <= 180 else {
                        "chars": len(raw), "literal_excerpt": raw[:180], "has_more": True,
                    }
                lines.append(f"  call {index}: {fn['name']} {json.dumps(details, ensure_ascii=False)}")
            observations = row.get("observations")
            if observations is None:
                lines.append("  observations: MISSING (not synthesized)")
            else:
                for index, observation in enumerate(observations, 1):
                    raw = text_content(observation)
                    lines.append(f"  result {index}: chars={len(raw)}, literal_excerpt="
                                 + json.dumps(raw[:180], ensure_ascii=False)
                                 + (", has_more=true" if len(raw) > 180 else ""))
        return "\n".join(lines)

    def full_history(self) -> str:
        # Never concatenate rendered segment prefixes or repeat the whole system prompt.
        return "\n\n".join(
            f"Agent step {number}\nAssistant:\n{row['response_text']}\n"
            f"Actual tool observations:\n{text_content(row.get('observations'))}"
            for number, row in self.steps.items()
        )


READ_TRACE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_trace",
        "description": "Read an exact original field of this trajectory by agent step. "
                       "Offsets are characters. Follow next_offset to read remaining content. "
                       "This cannot read arbitrary files or verifier information.",
        "parameters": {
            "type": "object",
            "properties": {
                "step": {"type": "integer", "minimum": 1},
                "part": {"type": "string", "enum": ["assistant", "observations", "context"]},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 12000},
            },
            "required": ["step", "part"],
            "additionalProperties": False,
        },
    },
}

SELECT_REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "select_repair",
        "description": "Select one counterfactual repair boundary and diagnose it. "
                       "call_index is zero-based for an attempted tool-call block; use -1 "
                       "only for an agent step with no attempted tool calls. The executable "
                       "replacement is generated in a separate native action pass.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "agent_step": {"type": "integer", "minimum": 1},
                "call_index": {"type": "integer", "minimum": -1},
                "diagnosis": {"type": "string", "minLength": 1},
            },
            "required": ["agent_step", "call_index", "diagnosis"],
            "additionalProperties": False,
        },
    },
}



SUBMIT_REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_repair",
        "description": "Submit one complete counterfactual replacement action at a chosen pre-action agent step.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "agent_step": {"type": "integer", "minimum": 1},
                "call_index": {"type": "integer", "minimum": 0},
                "diagnosis": {"type": "string", "minLength": 1},
                "replacement_action": {
                    "type": "string", "minLength": 1,
                    "description": "Complete native assistant action. Include exact <tool_call> blocks when tools are needed.",
                },
            },
            "required": ["agent_step", "call_index", "diagnosis", "replacement_action"],
            "additionalProperties": False,
        },
    },
}

DECLINE_REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "decline_repair",
        "description": "Decline when the record does not support one useful executable repair.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}
