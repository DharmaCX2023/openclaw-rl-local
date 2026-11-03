import asyncio
import base64
import copy
import hashlib
import json
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from itertools import count
from typing import Any

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from slime.utils.processing_utils import load_tokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from slime.utils.types import Sample

from opsd_evidence import render_teacher_feedback

logger = logging.getLogger(__name__)

_NON_STANDARD_BODY_KEYS: set[str] = set()
_COMPACTED_TOOL_OUTPUT_MARKER = "[compacted: tool output removed to free context]"
_TRUNCATED_TOOL_OUTPUT_MARKER = "[truncated: output exceeded context limit]"


@dataclass
class SessionTokenLedger:
    """Exact policy context plus the structural state returned to OpenClaw."""

    messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float]
    template_tools: list[dict[str, Any]] | None
    template_kwargs: dict[str, Any]
    reasoning_effort: Any
    transition: str = "initial"
    transition_detail: str = ""
    environment_delta_ids: list[int] = field(default_factory=list)
    assistant_boundary_ids: list[int] = field(default_factory=list)




def _require_dict(value, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    return value


def _require_list(value, field_name: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _require_bool(value, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a bool")
    return value


def _openai_error_response(
    *,
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    """Return an error envelope understood by OpenAI-compatible clients.

    FastAPI's HTTPException(detail=...) serializes errors under detail. The
    OpenAI adapter used by OpenClaw only preserves provider errors from the
    standard error.message envelope; otherwise it reduces the failure to
    <status> status code (no body), preventing overflow classification.
    """

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


def _parse_request_model(value) -> tuple[str, str]:
    model_value = _require_string(value, "model")
    parts = model_value.split("__", 1)
    if len(parts) != 2:
        raise TypeError("model must be '<turn_type>__<session_id>'")
    turn_type = _require_string(parts[0], "turn_type").strip().lower()
    session_id = _require_string(parts[1], "session_id")
    return turn_type, session_id


def _flatten_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if (
                isinstance(item, dict)
                and "type" in item
                and item["type"] == "text"
                and "text" in item
            ):
                parts.append(str(item["text"]))
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


def _count_compacted_tool_outputs(messages: list[dict]) -> int:
    return sum(
        _COMPACTED_TOOL_OUTPUT_MARKER
        in _flatten_message_content(message.get("content"))
        for message in messages
    )


def _count_new_tool_result_markers(
    previous_messages: list[dict[str, Any]],
    current_messages: list[dict[str, Any]],
    marker: str,
) -> int:
    """Count tool results rewritten to a marker since the prior request.

    OpenClaw keeps pruning placeholders in later prompts, so counting marker
    occurrences in the whole history repeatedly over-counts one rewrite. Match
    stable tool-call IDs and only count a transition into the marker instead.
    """

    previous_by_id = {
        str(message.get("tool_call_id")): _flatten_message_content(
            message.get("content")
        )
        for message in previous_messages
        if message.get("role") == "tool" and message.get("tool_call_id")
    }
    count_new = 0
    for message in current_messages:
        if message.get("role") != "tool" or not message.get("tool_call_id"):
            continue
        tool_call_id = str(message["tool_call_id"])
        previous_content = previous_by_id.get(tool_call_id)
        current_content = _flatten_message_content(message.get("content"))
        if (
            previous_content is not None
            and marker not in previous_content
            and marker in current_content
        ):
            count_new += 1
    return count_new


def _json_semantic_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return _json_semantic_value(json.loads(value))
        except (json.JSONDecodeError, ValueError, TypeError):
            return value
    if isinstance(value, list):
        return [_json_semantic_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _json_semantic_value(item)
            for key, item in value.items()
            if key != "index"
        }
    return value


def _tool_call_signature(tool_calls: list[dict[str, Any]]) -> str | None:
    """Return a stable action signature without tool-call IDs or formatting noise."""
    if not tool_calls:
        return None
    normalized = []
    for tool_call in tool_calls:
        function = tool_call.get("function") or {}
        normalized.append(
            {
                "name": str(function.get("name") or ""),
                "arguments": _json_semantic_value(function.get("arguments")),
            }
        )
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _advance_tool_call_repeat_streak(
    state: dict[str, tuple[str, int]],
    session_id: str,
    tool_calls: list[dict[str, Any]],
) -> int:
    """Audit consecutive identical action turns without changing model behavior."""
    signature = _tool_call_signature(tool_calls)
    if signature is None:
        state.pop(session_id, None)
        return 0
    previous_signature, previous_streak = state.get(session_id, ("", 0))
    streak = previous_streak + 1 if previous_signature == signature else 1
    state[session_id] = (signature, streak)
    return streak


def _semantic_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep fields that affect the ClawGym chat template or tool linkage."""
    role = "system" if message.get("role") == "developer" else message.get("role")
    normalized: dict[str, Any] = {
        "role": role,
        "content": _flatten_message_content(message.get("content")),
    }
    if role == "assistant":
        normalized["tool_calls"] = []
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            normalized["tool_calls"].append(
                {
                    "id": str(tool_call.get("id") or ""),
                    "type": str(tool_call.get("type") or "function"),
                    "name": str(function.get("name") or ""),
                    "arguments": _json_semantic_value(function.get("arguments")),
                }
            )
    elif role == "tool":
        normalized["tool_call_id"] = str(message.get("tool_call_id") or "")
        if "name" in message:
            normalized["name"] = str(message.get("name") or "")
    return normalized


def _semantic_messages_equal(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> bool:
    return [_semantic_message(message) for message in left] == [
        _semantic_message(message) for message in right
    ]


def _semantic_tools_equal(left: Any, right: Any) -> bool:
    return _json_semantic_value(left) == _json_semantic_value(right)


def _classify_structural_append(
    state: SessionTokenLedger,
    messages: list[dict[str, Any]],
    template_tools: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], str]:
    """Classify a new OpenClaw request without comparing re-rendered policy text."""
    if not _semantic_tools_equal(state.template_tools, template_tools):
        return "template_changed", [], "tool schema changed"
    if len(messages) <= len(state.messages):
        return "history_rewritten", [], "message history did not grow"
    if not _semantic_messages_equal(
        state.messages, messages[: len(state.messages)]
    ):
        return "history_rewritten", [], "previous message prefix changed"

    appended = messages[len(state.messages) :]
    if not appended or appended[0].get("role") != "assistant":
        return "history_rewritten", appended, "previous assistant message missing"
    if not _semantic_messages_equal([state.assistant_message], [appended[0]]):
        return "history_rewritten", appended, "previous assistant message changed"

    expected_ids = [
        str(tool_call.get("id") or "")
        for tool_call in state.assistant_message.get("tool_calls") or []
    ]
    observed_ids = [
        str(message.get("tool_call_id") or "")
        for message in appended[1:]
        if message.get("role") == "tool"
    ]
    unexpected_roles = [
        str(message.get("role"))
        for message in appended[1:]
        if message.get("role") != "tool"
    ]
    if unexpected_roles:
        return (
            "history_rewritten",
            appended,
            "unexpected appended roles: " + ",".join(unexpected_roles),
        )
    if expected_ids != observed_ids:
        return (
            "tool_result_mismatch",
            appended,
            f"expected tool results {expected_ids!r}, observed {observed_ids!r}",
        )
    return "append_only", appended, ""

def _normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for message in messages:
        item = dict(message)
        if "role" in item and item["role"] == "developer":
            item["role"] = "system"
        if "content" in item and not isinstance(item["content"], str):
            item["content"] = _flatten_message_content(item["content"])
        # Parse tool_calls arguments from JSON string to dict to match SGLang behavior.
        # SGLang does json.loads on arguments before passing to apply_chat_template,
        # which changes formatting (e.g. adds spaces after colons). We must do the same
        # to keep training tokens consistent with inference tokens.
        if "tool_calls" in item and isinstance(item["tool_calls"], list):
            new_tool_calls = []
            for tc in item["tool_calls"]:
                tc = dict(tc)
                if "function" in tc and isinstance(tc["function"], dict):
                    tc["function"] = dict(tc["function"])
                    args = tc["function"].get("arguments")
                    if isinstance(args, str):
                        try:
                            tc["function"]["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            pass
                new_tool_calls.append(tc)
            item["tool_calls"] = new_tool_calls
        normalized.append(item)
    return normalized
def _apply_chat_template_ids(
    tokenizer,
    messages: list[dict[str, Any]],
    *,
    template_tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any],
    reasoning_effort: Any,
    add_generation_prompt: bool,
) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        _normalize_messages_for_template(messages),
        tools=template_tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        reasoning_effort=reasoning_effort,
        **template_kwargs,
    )
    return [int(token_id) for token_id in token_ids]


def _opsd_feedback_text(feedback: dict[str, Any]) -> str:
    return render_teacher_feedback(feedback)


def _inject_opsd_feedback(
    messages: list[dict[str, Any]], feedback_text: str
) -> list[dict[str, Any]]:
    hinted = copy.deepcopy(messages)
    for message in hinted:
        if message.get("role") != "user":
            continue
        suffix = "\n\n" + feedback_text
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = content + suffix
        elif isinstance(content, list):
            message["content"] = list(content) + [
                {"type": "text", "text": suffix}
            ]
        else:
            raise ValueError("unsupported_user_content")
        return hinted
    raise ValueError("missing_user_task_message")


def _prepare_opsd_segment(
    tokenizer,
    segment: dict[str, Any],
    pending: dict[int, dict[str, Any]],
    feedback: dict[str, Any] | None,
    *,
    max_context_tokens: int,
    infrastructure_failure: bool,
    require_incomplete: bool = True,
    evidence_key: str = "evidence_anchors",
) -> tuple[list[int], bool, str]:
    if infrastructure_failure:
        return [], False, "infrastructure_failure"
    if not feedback:
        return [], False, "missing_feedback"
    if require_incomplete and feedback.get("rollout_not_fully_successful") is not True:
        # Absolute/centered OPSD only redistributes negative-advantage credit.
        return [], False, ""
    first_step = pending[segment["agent_steps"][0]]
    try:
        render_kwargs = {
            "template_tools": first_step.get("template_tools"),
            "template_kwargs": first_step.get("template_kwargs") or {},
            "reasoning_effort": first_step.get("reasoning_effort"),
            "add_generation_prompt": True,
        }
        original_prompt = _apply_chat_template_ids(
            tokenizer, first_step["messages"], **render_kwargs
        )
        if original_prompt != list(first_step["prompt_ids"]):
            return [], False, "render_mismatch"
        rendered_feedback = dict(feedback)
        if evidence_key != "evidence_anchors":
            rendered_feedback["evidence_anchors"] = feedback.get(evidence_key)
        if not require_incomplete:
            rendered_feedback["teacher_guidance_mode"] = "sdar"
        feedback_text = _opsd_feedback_text(rendered_feedback)
        hinted_messages = _inject_opsd_feedback(
            first_step["messages"], feedback_text
        )
        hint_prompt = _apply_chat_template_ids(
            tokenizer, hinted_messages, **render_kwargs
        )
        response_tokens = list(
            segment["tokens"][-len(segment["loss_mask"]) :]
        )
        teacher_tokens = hint_prompt + response_tokens
        if len(teacher_tokens) > max_context_tokens:
            return [], False, "hint_overflow"
        return teacher_tokens, True, ""
    except (TypeError, ValueError) as exc:
        return [], False, str(exc)


def _prepare_opsd_pair(
    tokenizer,
    segment: dict[str, Any],
    pending: dict[int, dict[str, Any]],
    feedback: dict[str, Any] | None,
    *,
    max_context_tokens: int,
    infrastructure_failure: bool,
    require_incomplete: bool,
    matched_placebo: bool,
) -> tuple[list[int], list[int], bool, str]:
    real_tokens, real_eligible, real_reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        feedback,
        max_context_tokens=max_context_tokens,
        infrastructure_failure=infrastructure_failure,
        require_incomplete=require_incomplete,
    )
    if not matched_placebo:
        return real_tokens, [], real_eligible, real_reason
    if not real_eligible:
        return [], [], False, real_reason

    placebo_tokens, placebo_eligible, placebo_reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        feedback,
        max_context_tokens=max_context_tokens,
        infrastructure_failure=infrastructure_failure,
        require_incomplete=require_incomplete,
        evidence_key="placebo_evidence_anchors",
    )
    if not placebo_eligible:
        reason = placebo_reason or "missing_placebo_evidence_anchors"
        return [], [], False, f"placebo_{reason}"
    return real_tokens, placebo_tokens, True, ""


def _missing_assistant_boundary_ids(
    response_ids: list[int], boundary_ids: list[int]
) -> list[int]:
    """Avoid duplicating a stop token when a backend includes it in output ids."""
    matched = 0
    for size in range(min(len(response_ids), len(boundary_ids)), 0, -1):
        if response_ids[-size:] == boundary_ids[:size]:
            matched = size
            break
    return list(boundary_ids[matched:])


def _build_incremental_prompt_ids(
    tokenizer,
    state: SessionTokenLedger,
    messages: list[dict[str, Any]],
    appended: list[dict[str, Any]],
    *,
    template_tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any],
    reasoning_effort: Any,
) -> tuple[list[int], list[int], list[int]]:
    """Append environment tokens to exact policy tokens without re-rendering actions."""
    if not appended or appended[0].get("role") != "assistant":
        raise ValueError("incremental prompt requires the prior assistant message")

    canonical_generation_prefix = _apply_chat_template_ids(
        tokenizer,
        state.messages,
        template_tools=template_tools,
        template_kwargs=template_kwargs,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )
    canonical_empty_assistant = _apply_chat_template_ids(
        tokenizer,
        state.messages + [{"role": "assistant", "content": "", "tool_calls": []}],
        template_tools=template_tools,
        template_kwargs=template_kwargs,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=False,
    )
    boundary_ids = _strict_prompt_append_delta(
        canonical_generation_prefix, canonical_empty_assistant
    )
    if boundary_ids is None or not boundary_ids:
        raise ValueError("chat template did not expose an assistant closing boundary")

    canonical_through_assistant = _apply_chat_template_ids(
        tokenizer,
        state.messages + [appended[0]],
        template_tools=template_tools,
        template_kwargs=template_kwargs,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=False,
    )
    canonical_next_prompt = _apply_chat_template_ids(
        tokenizer,
        messages,
        template_tools=template_tools,
        template_kwargs=template_kwargs,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True,
    )
    environment_delta_ids = _strict_prompt_append_delta(
        canonical_through_assistant, canonical_next_prompt
    )
    if environment_delta_ids is None:
        # Some chat templates render an assistant differently depending on
        # whether it is the last message. Qwen3, for example, inserts an empty
        # <think> block for a terminal assistant but removes it once a tool
        # result follows. That decoration is not part of the policy response
        # and must not force one training segment per agent turn.
        #
        # Recover only the stable assistant-closing boundary and the exact
        # canonical environment suffix. Deliberately avoid a longest-common-
        # prefix splice: policy response tokens remain exactly those returned
        # by SGLang; only tool/environment tokens come from the template.
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        special_positions = [
            index
            for index, token_id in enumerate(boundary_ids)
            if token_id in special_ids
        ]
        if not special_positions:
            raise ValueError(
                "new environment messages are not a canonical template suffix"
            )
        stable_boundary_ids = boundary_ids[special_positions[-1] :]
        boundary_offset = None
        search_start = len(canonical_generation_prefix)
        for offset in range(
            search_start,
            len(canonical_next_prompt) - len(stable_boundary_ids) + 1,
        ):
            if (
                canonical_next_prompt[
                    offset : offset + len(stable_boundary_ids)
                ]
                == stable_boundary_ids
            ):
                boundary_offset = offset
                break
        if boundary_offset is None:
            raise ValueError(
                "new environment messages are not a canonical template suffix"
            )
        boundary_ids = stable_boundary_ids
        environment_delta_ids = canonical_next_prompt[
            boundary_offset + len(boundary_ids) :
        ]

    missing_boundary_ids = _missing_assistant_boundary_ids(
        state.response_ids, boundary_ids
    )
    prompt_ids = (
        list(state.prompt_ids)
        + list(state.response_ids)
        + missing_boundary_ids
        + environment_delta_ids
    )
    return prompt_ids, missing_boundary_ids, environment_delta_ids




def _extract_generate_output_logprobs(
    output: dict[str, Any],
) -> tuple[list[int], list[float]]:
    meta_info = _require_dict(output["meta_info"], "meta_info")
    pairs = _require_list(
        meta_info["output_token_logprobs"], "meta_info.output_token_logprobs"
    )
    token_ids: list[int] = []
    logprobs: list[float] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            raise TypeError("output_token_logprobs items must be [logprob, token_id]")
        logprobs.append(float(pair[0]))
        token_ids.append(int(pair[1]))
    return token_ids, logprobs


def _build_generate_sampling_params(
    body: dict[str, Any],
    generation_defaults: dict[str, Any],
    tool_call_parser: str,
) -> tuple[ChatCompletionRequest, FunctionCallParser | None, dict[str, Any]]:
    request = ChatCompletionRequest.model_validate(body)
    if request.n != 1:
        raise ValueError("ClawGym proxy requires exactly one completion")

    parser = None
    tool_call_constraint = None
    if request.tools and request.tool_choice != "none":
        request.skip_special_tokens = False
        parser = FunctionCallParser(request.tools, tool_call_parser)
        tool_call_constraint = parser.get_structure_constraint(request.tool_choice)

    sampling_params = request.to_sampling_params(
        stop=request.stop,
        model_generation_config=generation_defaults,
        tool_call_constraint=tool_call_constraint,
    )
    response_limit = int(os.environ.get("OPENCLAW_MAX_RESPONSE_TOKENS", "8192"))
    requested_limit = sampling_params.get("max_new_tokens")
    if requested_limit is None:
        sampling_params["max_new_tokens"] = response_limit
    else:
        sampling_params["max_new_tokens"] = min(int(requested_limit), response_limit)
    return request, parser, sampling_params


def _parse_generated_assistant(
    raw_text: str,
    parser: FunctionCallParser | None,
    tool_call_id_namespace: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    if parser is None or not parser.has_tool_call(raw_text):
        return raw_text, []

    content, parsed_calls = parser.parse_non_stream(raw_text)
    tool_calls = []
    for index, call in enumerate(parsed_calls):
        arguments = call.parameters
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        if tool_call_id_namespace is None:
            tool_call_id = f"call{uuid.uuid4().hex[:24]}"
        else:
            digest = hashlib.sha256(
                (
                    f"{tool_call_id_namespace}:{index}:{call.name}:"
                    f"{arguments}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            tool_call_id = f"call{digest}"
        tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "index": index,
                "function": {
                    "name": call.name,
                    "arguments": arguments,
                },
            }
        )
    return content, tool_calls


def _count_raw_qwen_tool_calls(raw_text: str) -> int:
    return raw_text.count("<tool_call>")


def _optional_recorded_response_ids(response_ids: list[int]) -> dict[str, Any]:
    """Persist exact sampled IDs only for explicitly opted-in audit runs."""

    if os.environ.get("OPENCLAW_RECORD_RESPONSE_IDS", "0") != "1":
        return {}
    return {"response_ids": [int(token_id) for token_id in response_ids]}


def _has_incomplete_qwen_tool_call(raw_text: str) -> bool:
    """Detect a tool-call block cut off before its closing delimiter.

    This is diagnostic only. It must not repair, drop, or mask sampled tokens.
    """

    return raw_text.count("<tool_call>") > raw_text.count("</tool_call>")


def _token_repetition_diagnostics(
    token_ids: list[int], ngram_size: int = 32, repeat_threshold: int = 8
) -> dict[str, float | int | bool]:
    """Return inexpensive, token-native repetition diagnostics for one turn."""

    total_ngrams = len(token_ids) - ngram_size + 1
    if total_ngrams <= 0:
        return {
            "repeated_ngram_ratio": 0.0,
            "max_ngram_repeat": 0,
            "repetition_degenerated": False,
        }
    counts = Counter(
        tuple(token_ids[index : index + ngram_size])
        for index in range(total_ngrams)
    )
    repeated_occurrences = sum(count - 1 for count in counts.values() if count > 1)
    max_repeat = max(counts.values(), default=0)
    return {
        "repeated_ngram_ratio": repeated_occurrences / total_ngrams,
        "max_ngram_repeat": max_repeat,
        "repetition_degenerated": max_repeat >= repeat_threshold,
    }


def _collect_new_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for message in reversed(messages):
        role = message.get("role")
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id") or "")
            if tool_call_id in seen_ids:
                continue
            seen_ids.add(tool_call_id)
            results.append(
                {
                    "tool_call_id": tool_call_id,
                    "content": _flatten_message_content(message.get("content")),
                }
            )
        elif role == "assistant":
            break
    results.reverse()
    return results


def _extract_aligned_generate_response(
    output: dict[str, Any],
) -> tuple[str, list[int], list[float]]:
    raw_text = _require_string(output["text"], "text")
    response_ids, response_logprobs = _extract_generate_output_logprobs(output)
    if len(response_ids) != len(response_logprobs):
        raise ValueError(
            "SGLang response token/logprob mismatch: "
            f"tokens={len(response_ids)}, logprobs={len(response_logprobs)}"
        )

    meta_info = _require_dict(output["meta_info"], "meta_info")
    completion_tokens = meta_info.get("completion_tokens")
    if completion_tokens is not None and int(completion_tokens) != len(response_ids):
        raise ValueError(
            "SGLang completion count does not match raw generated token ids: "
            f"completion_tokens={completion_tokens}, token_ids={len(response_ids)}"
        )
    return raw_text, response_ids, response_logprobs


def _strict_prompt_append_delta(
    accumulated_ids: list[int], prompt_ids: list[int]
) -> list[int] | None:
    """Return the exact appended suffix, or None when the prompt rewrites prior context."""
    accumulated_len = len(accumulated_ids)
    if len(prompt_ids) < accumulated_len:
        return None
    if prompt_ids[:accumulated_len] != accumulated_ids:
        return None
    return list(prompt_ids[accumulated_len:])


def _shared_prefix_length(left: list[int], right: list[int]) -> int:
    shared = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        shared += 1
    return shared


def _routing_prefix_matches(
    route_chunks: list[np.ndarray], candidate_routes: np.ndarray
) -> bool:
    """Return whether candidate routes preserve every accumulated route row.

    Chunks avoid repeatedly concatenating a growing multi-turn route tensor.
    Each row is aligned to one input token; a sequence with N tokens therefore
    carries N-1 route rows.
    """

    offset = 0
    for chunk in route_chunks:
        end = offset + int(chunk.shape[0])
        if end > int(candidate_routes.shape[0]):
            return False
        if not np.array_equal(chunk, candidate_routes[offset:end]):
            return False
        offset = end
    return True


def _routing_prefix_diagnostics(
    route_chunks: list[np.ndarray], candidate_routes: np.ndarray
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Aggregate ordered and set-wise route-prefix differences."""

    compared_rows = sum(int(chunk.shape[0]) for chunk in route_chunks)
    if compared_rows > int(candidate_routes.shape[0]):
        raise ValueError("candidate route prefix is shorter than accumulated routes")
    membership_row_mask = np.zeros(compared_rows, dtype=np.bool_)
    order_only_row_mask = np.zeros(compared_rows, dtype=np.bool_)
    num_layers = int(candidate_routes.shape[1])
    topk = int(candidate_routes.shape[2])
    per_layer_order_only = np.zeros(num_layers, dtype=np.int64)
    per_layer_membership = np.zeros(num_layers, dtype=np.int64)
    ordered_mismatch_route_ids = 0
    replaced_expert_slots = 0
    order_only_cells = 0
    membership_cells = 0
    offset = 0

    for chunk in route_chunks:
        end = offset + int(chunk.shape[0])
        candidate = candidate_routes[offset:end]
        ordered_equal = np.all(chunk == candidate, axis=-1)
        set_equal = np.all(
            np.sort(chunk, axis=-1) == np.sort(candidate, axis=-1), axis=-1
        )
        order_only = np.logical_and(~ordered_equal, set_equal)
        membership = ~set_equal
        order_only_row_mask[offset:end] = np.any(order_only, axis=1)
        membership_row_mask[offset:end] = np.any(membership, axis=1)
        per_layer_order_only += np.sum(order_only, axis=0, dtype=np.int64)
        per_layer_membership += np.sum(membership, axis=0, dtype=np.int64)
        order_only_cells += int(np.count_nonzero(order_only))
        membership_cells += int(np.count_nonzero(membership))
        ordered_mismatch_route_ids += int(np.count_nonzero(chunk != candidate))

        # Top-k IDs are unique. Count old expert slots absent from the new set.
        intersection_size = np.zeros(chunk.shape[:2], dtype=np.int16)
        for topk_index in range(topk):
            intersection_size += np.any(
                chunk[:, :, topk_index, None] == candidate, axis=-1
            )
        replaced_expert_slots += int(
            np.sum(topk - intersection_size, dtype=np.int64)
        )
        offset = end

    compared_cells = compared_rows * num_layers
    compared_route_ids = compared_cells * topk
    membership_rows = int(np.count_nonzero(membership_row_mask))
    order_only_rows = int(
        np.count_nonzero(np.logical_and(order_only_row_mask, ~membership_row_mask))
    )
    any_changed_rows = int(
        np.count_nonzero(np.logical_or(order_only_row_mask, membership_row_mask))
    )
    diagnostics = {
        "compared_token_rows": compared_rows,
        "compared_token_layer_cells": compared_cells,
        "compared_route_ids": compared_route_ids,
        "ordered_mismatch_route_ids": ordered_mismatch_route_ids,
        "ordered_mismatch_route_id_fraction": (
            ordered_mismatch_route_ids / compared_route_ids
            if compared_route_ids
            else 0.0
        ),
        "order_only_token_layer_cells": order_only_cells,
        "order_only_token_layer_fraction": (
            order_only_cells / compared_cells if compared_cells else 0.0
        ),
        "membership_changed_token_layer_cells": membership_cells,
        "membership_changed_token_layer_fraction": (
            membership_cells / compared_cells if compared_cells else 0.0
        ),
        "replaced_expert_slots": replaced_expert_slots,
        "replaced_expert_slot_fraction": (
            replaced_expert_slots / compared_route_ids if compared_route_ids else 0.0
        ),
        "any_changed_token_rows": any_changed_rows,
        "any_changed_token_row_fraction": (
            any_changed_rows / compared_rows if compared_rows else 0.0
        ),
        "order_only_token_rows": order_only_rows,
        "order_only_token_row_fraction": (
            order_only_rows / compared_rows if compared_rows else 0.0
        ),
        "membership_changed_token_rows": membership_rows,
        "membership_changed_token_row_fraction": (
            membership_rows / compared_rows if compared_rows else 0.0
        ),
        "boundary_has_order_only_change": bool(order_only_cells),
        "boundary_has_membership_change": bool(membership_cells),
        "per_layer_order_only_cells": per_layer_order_only.tolist(),
        "per_layer_membership_changed_cells": per_layer_membership.tolist(),
    }
    return diagnostics, membership_row_mask, order_only_row_mask


def _aligned_prompt_logprobs(
    meta_info: dict[str, Any], prompt_ids: list[int]
) -> list[float | None] | None:
    """Align opt-in SGLang input logprobs to prompt token positions."""

    pairs = meta_info.get("input_token_logprobs")
    if not isinstance(pairs, list) or not pairs:
        return None
    parsed_ids: list[int] = []
    parsed_logprobs: list[float | None] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            return None
        try:
            parsed_ids.append(int(pair[1]))
            parsed_logprobs.append(None if pair[0] is None else float(pair[0]))
        except (TypeError, ValueError):
            return None
    for start in range(0, len(prompt_ids) - len(parsed_ids) + 1):
        if parsed_ids == prompt_ids[start : start + len(parsed_ids)]:
            aligned: list[float | None] = [None] * len(prompt_ids)
            aligned[start : start + len(parsed_ids)] = parsed_logprobs
            return aligned
    return None


def _logprob_recompute_diagnostics(
    current: dict[str, Any],
    prompt_logprobs: list[float | None] | None,
    membership_row_mask: np.ndarray,
    order_only_row_mask: np.ndarray,
) -> dict[str, Any]:
    """Compare original rollout logprobs with next-request prompt recomputation."""

    categories: dict[str, list[tuple[float, float]]] = {
        "all": [],
        "membership_changed": [],
        "order_only": [],
        "route_unchanged": [],
    }
    if prompt_logprobs is None:
        return {"available": False, "reason": "missing_or_unaligned_input_logprobs"}
    token_offset = int(current["loss_token_offset"])
    for local_index, (mask, old_logprob) in enumerate(
        zip(current["loss_mask"], current["rollout_log_probs"])
    ):
        if not mask:
            continue
        token_position = token_offset + local_index
        if token_position <= 0 or token_position >= len(prompt_logprobs):
            continue
        new_logprob = prompt_logprobs[token_position]
        if new_logprob is None:
            continue
        old_value = float(old_logprob)
        new_value = float(new_logprob)
        abs_diff = abs(new_value - old_value)
        ratio_abs_diff = abs(
            float(np.expm1(np.clip(new_value - old_value, -50, 50)))
        )
        item = (abs_diff, ratio_abs_diff)
        categories["all"].append(item)
        route_row = token_position - 1
        if route_row < len(membership_row_mask) and membership_row_mask[route_row]:
            categories["membership_changed"].append(item)
        elif route_row < len(order_only_row_mask) and order_only_row_mask[route_row]:
            categories["order_only"].append(item)
        else:
            categories["route_unchanged"].append(item)

    result: dict[str, Any] = {"available": bool(categories["all"])}
    for name, values in categories.items():
        abs_values = np.asarray([value[0] for value in values], dtype=np.float64)
        ratio_values = np.asarray([value[1] for value in values], dtype=np.float64)
        result[name] = {
            "count": int(abs_values.size),
            "logprob_abs_sum": float(abs_values.sum()),
            "logprob_abs_sq_sum": float(np.square(abs_values).sum()),
            "logprob_abs_max": float(abs_values.max()) if abs_values.size else 0.0,
            "logprob_abs_gt_0_01": int(np.count_nonzero(abs_values > 0.01)),
            "logprob_abs_gt_0_05": int(np.count_nonzero(abs_values > 0.05)),
            "logprob_abs_gt_0_1": int(np.count_nonzero(abs_values > 0.1)),
            "ratio_abs_sum": float(ratio_values.sum()),
            "ratio_abs_max": float(ratio_values.max()) if ratio_values.size else 0.0,
            "ratio_abs_gt_0_01": int(np.count_nonzero(ratio_values > 0.01)),
            "ratio_abs_gt_0_05": int(np.count_nonzero(ratio_values > 0.05)),
            "ratio_abs_gt_0_1": int(np.count_nonzero(ratio_values > 0.1)),
        }
    return result


def _concatenate_route_chunks(route_chunks: list[np.ndarray]) -> np.ndarray:
    if not route_chunks:
        raise ValueError("routing replay segment has no route chunks")
    if len(route_chunks) == 1:
        return route_chunks[0]
    return np.concatenate(route_chunks, axis=0)


def _build_stitched_routing_replay_segments(
    pending: dict[int, dict[str, Any]],
    sorted_steps: list[int],
    *,
    max_context_tokens: int,
    num_layers: int,
    router_topk: int,
    session_id: str,
    step_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build exact replay segments, stitching only stable token/route prefixes."""

    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    segment_index = 0

    def flush() -> None:
        nonlocal current
        if current is None or not current["loss_mask"]:
            current = None
            return
        routes = _concatenate_route_chunks(current.pop("route_chunks"))
        expected_rows = len(current["tokens"]) - 1
        if int(routes.shape[0]) != expected_rows:
            raise ValueError(
                "stitched routing replay no longer aligns with tokens: "
                f"got {tuple(routes.shape)}, expected first dimension "
                f"{expected_rows}, session={session_id}, "
                f"segment={current['segment_index']}"
            )
        current["routes"] = routes
        if not (
            len(current["response_step_ids"])
            == len(current["loss_mask"])
            == len(current["rollout_log_probs"])
        ):
            raise ValueError(
                "routing replay response-space metadata is misaligned: "
                f"steps={len(current['response_step_ids'])}, "
                f"mask={len(current['loss_mask'])}, "
                f"logprobs={len(current['rollout_log_probs'])}, "
                f"session={session_id}, segment={current['segment_index']}"
            )
        current["turn_count"] = len(current["agent_steps"])
        current["stitched_turns"] = max(0, current["turn_count"] - 1)
        segments.append(current)
        current = None

    for request_index, step_num in enumerate(sorted_steps):
        step_data = pending[step_num]
        prompt_ids = list(step_data["prompt_ids"])
        response_ids = list(step_data["response_ids"])
        response_logprobs = list(step_data["response_logprobs"])
        routes = step_data.get("rollout_routed_experts")
        if routes is None:
            raise ValueError(
                "missing rollout_routed_experts for agent request "
                f"session={session_id}, step={step_num}"
            )
        request_tokens = prompt_ids + response_ids
        expected_shape = (
            len(request_tokens) - 1,
            int(num_layers),
            int(router_topk),
        )
        if tuple(routes.shape) != expected_shape:
            raise ValueError(
                "rollout_routed_experts no longer aligns with request "
                f"tokens: got {tuple(routes.shape)}, expected "
                f"{expected_shape}, session={session_id}, step={step_num}"
            )
        if len(response_ids) != len(response_logprobs):
            raise ValueError(
                "routing replay response/logprob length mismatch: "
                f"response={len(response_ids)}, "
                f"logprobs={len(response_logprobs)}, "
                f"session={session_id}, step={step_num}"
            )
        if len(request_tokens) > max_context_tokens:
            flush()
            step_details[request_index].update(
                {
                    "routing_replay_action": "dropped",
                    "routing_replay_prefix_route_rows": 0,
                    "routing_replay_prefix_match": False,
                    "routing_replay_split_reason": "trajectory_limit",
                    "routing_replay_segment_index": segment_index,
                }
            )
            segment_index += 1
            continue

        if current is None:
            current = {
                "tokens": request_tokens,
                "loss_mask": [1] * len(response_ids),
                "response_step_ids": [step_num] * len(response_ids),
                "rollout_log_probs": response_logprobs,
                "route_chunks": [routes],
                "prompt_text": step_data["prompt_text"],
                "response_text": step_data["response_text"],
                "agent_steps": [step_num],
                "segment_index": segment_index,
                "split_reason": "initial" if not segments else "after_drop",
                "hit_response_limit": bool(
                    step_data.get("hit_response_limit", False)
                ),
                "loss_token_offset": len(prompt_ids),
            }
            step_details[request_index].update(
                {
                    "routing_replay_action": "initial",
                    "routing_replay_prefix_route_rows": 0,
                    "routing_replay_prefix_match": True,
                    "routing_replay_split_reason": None,
                    "routing_replay_segment_index": segment_index,
                }
            )
            continue

        accumulated_tokens = current["tokens"]
        delta_ids = _strict_prompt_append_delta(
            accumulated_tokens, prompt_ids
        )
        declared_transition = step_data.get("ledger_transition")
        declared_rewrite = declared_transition not in {
            None,
            "initial",
            "append_only",
        }
        accumulated_route_rows = len(accumulated_tokens) - 1
        route_prefix_matches = (
            not declared_rewrite
            and delta_ids is not None
            and _routing_prefix_matches(current["route_chunks"], routes)
        )
        conflict_diagnostics = None
        if (
            os.environ.get("OPENCLAW_ROUTING_CONFLICT_DIAGNOSTICS", "0") == "1"
            and not declared_rewrite
            and delta_ids is not None
            and not route_prefix_matches
        ):
            (
                route_diagnostics,
                membership_row_mask,
                order_only_row_mask,
            ) = _routing_prefix_diagnostics(current["route_chunks"], routes)
            conflict_diagnostics = {
                "routes": route_diagnostics,
                "logprobs": _logprob_recompute_diagnostics(
                    current,
                    step_data.get("prompt_input_logprobs"),
                    membership_row_mask,
                    order_only_row_mask,
                ),
            }

        split_reason = None
        if declared_rewrite:
            split_reason = str(declared_transition)
        elif delta_ids is None:
            split_reason = "token_prefix_rewrite"
        elif not route_prefix_matches:
            split_reason = "route_prefix_conflict"

        if split_reason is None:
            accumulated_tokens.extend(delta_ids)
            current["loss_mask"].extend([0] * len(delta_ids))
            current["response_step_ids"].extend([-1] * len(delta_ids))
            current["rollout_log_probs"].extend([0.0] * len(delta_ids))
            accumulated_tokens.extend(response_ids)
            current["loss_mask"].extend([1] * len(response_ids))
            current["response_step_ids"].extend([step_num] * len(response_ids))
            current["rollout_log_probs"].extend(response_logprobs)
            current["route_chunks"].append(
                routes[accumulated_route_rows:]
            )
            if accumulated_tokens != request_tokens:
                raise ValueError(
                    "stitched routing replay tokens differ from exact "
                    f"request, session={session_id}, step={step_num}"
                )
            current["response_text"] += step_data["response_text"]
            current["agent_steps"].append(step_num)
            current["hit_response_limit"] = (
                current["hit_response_limit"]
                or bool(step_data.get("hit_response_limit", False))
            )
            action = "stitched"
        else:
            flush()
            segment_index += 1
            current = {
                "tokens": request_tokens,
                "loss_mask": [1] * len(response_ids),
                "response_step_ids": [step_num] * len(response_ids),
                "rollout_log_probs": response_logprobs,
                "route_chunks": [routes],
                "prompt_text": step_data["prompt_text"],
                "response_text": step_data["response_text"],
                "agent_steps": [step_num],
                "segment_index": segment_index,
                "split_reason": split_reason,
                "hit_response_limit": bool(
                    step_data.get("hit_response_limit", False)
                ),
                "loss_token_offset": len(prompt_ids),
            }
            action = "split"

        step_details[request_index].update(
            {
                "routing_replay_action": action,
                "routing_replay_prefix_route_rows": accumulated_route_rows,
                "routing_replay_prefix_match": route_prefix_matches,
                "routing_replay_split_reason": split_reason,
                "routing_replay_segment_index": segment_index,
                "routing_replay_conflict_diagnostics": conflict_diagnostics,
            }
        )

    flush()
    return segments


def _reward_score(sample: Sample) -> float:
    reward = _require_dict(sample.reward, "sample.reward")
    return float(reward["score"])


async def reward_func(args, sample_or_samples, **kwargs):
    if isinstance(sample_or_samples, list):
        rewards = []
        for sample in sample_or_samples:
            rewards.append({"score": _reward_score(sample)})
        return rewards

    sample = sample_or_samples
    return {"score": _reward_score(sample)}


async def generate(
    args, sample: Sample, sampling_params, evaluation: bool = False
) -> Sample:
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    if isinstance(sample.prompt, list):
        messages = sample.prompt
    else:
        messages = [{"role": "user", "content": str(sample.prompt)}]

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    payload = {
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        output = response.json()

    text, token_ids, logprobs = _extract_aligned_generate_response(output)

    sample.tokens = input_ids + token_ids
    sample.response = text
    sample.response_length = len(token_ids)
    sample.rollout_log_probs = logprobs
    sample.loss_mask = [1] * len(token_ids)
    sample.status = Sample.Status.COMPLETED
    return sample


class ClawGymRLAPIServer:
    def __init__(
        self, args, output_queue: queue.Queue, submission_enabled: threading.Event
    ):
        self.args = args
        self.output_queue = output_queue
        self.submission_enabled = submission_enabled
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.sglang_generate_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        )
        self.sglang_health_url = (
            f"http://{args.sglang_router_ip}:{args.sglang_router_port}/health"
        )
        self.tool_call_parser = os.getenv("TOOL_CALL_PARSER", "qwen25")
        generation_config_path = os.path.join(
            args.hf_checkpoint, "generation_config.json"
        )
        self.generation_defaults: dict[str, Any] = {}
        if os.path.isfile(generation_config_path):
            with open(
                generation_config_path, encoding="utf-8"
            ) as generation_config_file:
                generation_config = json.load(generation_config_file)
            for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
                if key in generation_config:
                    self.generation_defaults[key] = generation_config[key]
        # The policy that Megatron uses to recompute rollout logprobs must be
        # exactly the policy executed by SGLang. OpenClaw usually omits these
        # fields, so make the explicit Slime CLI values authoritative instead
        # of silently falling back to generation_config.json.
        for key, arg_name in (
            ("temperature", "rollout_temperature"),
            ("top_p", "rollout_top_p"),
            ("top_k", "rollout_top_k"),
        ):
            value = getattr(args, arg_name, None)
            if value is not None:
                self.generation_defaults[key] = value
        self.expected_api_key = os.getenv("SGLANG_API_KEY", "")
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "30000"))
        self.served_model_name = os.getenv("SERVED_MODEL_NAME", "qwen3-4b")
        self._record_file = ""
        if os.getenv("OPENCLAW_RECORD_ENABLED", "0") == "1":
            self._record_file = os.environ["OPENCLAW_RECORD_FILE"]

        self._index_counter = count(0)
        self._group_counter = count(0)
        self._step_counts: dict[str, int] = {}
        self._finalized_sessions: set[str] = set()
        self._pending_step_data: dict[str, dict[int, dict[str, Any]]] = {}
        # Compact idempotency receipts make reward submission safe to retry.
        # Long, heavily-pruned trajectories can take longer than the client
        # timeout to materialize all segments after consuming the session.
        self._completed_reward_submissions: dict[str, dict[str, Any]] = {}
        self._pending_records: dict[str, dict[str, Any]] = {}
        self._conversation_logs: dict[str, list[dict[str, Any]]] = {}
        self._session_raw_prompts: dict[str, dict[str, str]] = {}
        self._session_metadata: dict[str, dict[str, Any]] = {}
        self._session_context_overflows: dict[str, list[dict[str, int]]] = {}
        self._session_sglang_stats: dict[str, dict[str, Any]] = {}
        # Successful LLM summary requests. Tool-result pruning is a separate,
        # deterministic OpenClaw context rewrite and must never consume this budget.
        self._session_compaction_counts: dict[str, int] = {}
        self._session_compaction_request_counts: dict[str, int] = {}
        self._session_pending_compaction_rewrites: dict[str, int] = {}
        self._session_tool_result_pruning_events: dict[str, int] = {}
        self._session_tool_results_pruned: dict[str, int] = {}
        self._session_tool_result_truncation_events: dict[str, int] = {}
        self._session_tool_results_truncated: dict[str, int] = {}
        self._session_unclassified_history_rewrites: dict[str, int] = {}
        self._session_budget_terminations: dict[str, dict[str, Any]] = {}
        self._tool_call_repeat_state: dict[str, tuple[str, int]] = {}
        self._token_ledger_mode = os.getenv(
            "OPENCLAW_TOKEN_LEDGER_MODE", "enforce"
        ).strip().lower()
        if self._token_ledger_mode not in {"legacy", "audit", "enforce"}:
            raise ValueError(
                "OPENCLAW_TOKEN_LEDGER_MODE must be legacy, audit, or enforce"
            )
        self._session_token_ledgers: dict[str, SessionTokenLedger] = {}
        self._eval_scores: list[float] = []
        self._eval_scores_lock = threading.Lock()
        self._sglang_ready = threading.Event()
        self._full_prompt_logged = False

        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

        if self._record_file:
            os.makedirs(os.path.dirname(self._record_file), exist_ok=True)
            open(self._record_file, "w").close()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="ClawGym-RL Proxy")
        app.state.owner = self

        @app.get("/healthz")
        async def healthz():
            return {"ok": True}

        @app.post("/reset_sessions")
        async def reset_sessions(request: Request):
            owner: ClawGymRLAPIServer = request.app.state.owner
            owner._step_counts.clear()
            owner._finalized_sessions.clear()
            owner._pending_step_data.clear()
            owner._completed_reward_submissions.clear()
            owner._pending_records.clear()
            owner._conversation_logs.clear()
            owner._session_raw_prompts.clear()
            owner._session_metadata.clear()
            owner._session_token_ledgers.clear()
            owner._session_context_overflows.clear()
            owner._session_sglang_stats.clear()
            owner._session_compaction_counts.clear()
            owner._session_compaction_request_counts.clear()
            owner._session_pending_compaction_rewrites.clear()
            owner._session_tool_result_pruning_events.clear()
            owner._session_tool_results_pruned.clear()
            owner._session_tool_result_truncation_events.clear()
            owner._session_tool_results_truncated.clear()
            owner._session_unclassified_history_rewrites.clear()
            owner._session_budget_terminations.clear()
            owner._tool_call_repeat_state.clear()
            return {"ok": True}

        @app.post("/register_session")
        async def register_session(request: Request):
            owner: ClawGymRLAPIServer = request.app.state.owner
            body = await request.json()
            target_sid = _require_string(body["session_id"], "session_id")
            metadata = _require_dict(body["metadata"], "metadata")
            allowed_keys = {
                "task_id",
                "phase",
                "rollout_id",
                "task_index",
                "group_index",
                "task_selection_seed",
                "fixed_rollout_seed",
                "sampling_identity",
            }
            unexpected = sorted(set(metadata) - allowed_keys)
            if unexpected:
                raise HTTPException(
                    status_code=400,
                    detail="unsupported session metadata keys: "
                    + ", ".join(unexpected),
                )
            owner._session_metadata[target_sid] = {
                key: metadata.get(key) for key in sorted(allowed_keys)
            }
            return {"ok": True, "session_id": target_sid}

        @app.post("/get_conversation_log")
        async def get_conversation_log(request: Request):
            owner: ClawGymRLAPIServer = request.app.state.owner
            body = await request.json()
            target_sid = _require_string(body["session_id"], "session_id")
            if target_sid not in owner._conversation_logs:
                raise HTTPException(
                    status_code=404, detail=f"conversation log not found: {target_sid}"
                )
            # Seal before returning the trajectory. An upstream generation can
            # still be in flight when the remote agent times out; it must not
            # append another training turn after this snapshot is consumed.
            owner._finalized_sessions.add(target_sid)
            steps = owner._conversation_logs.pop(target_sid)
            raw_prompts = owner._session_raw_prompts.pop(target_sid, {})
            context_overflows = list(
                owner._session_context_overflows.get(target_sid, [])
            )
            sglang_stats = dict(
                owner._session_sglang_stats.get(target_sid, {})
            )
            compaction_count = owner._session_compaction_counts.get(target_sid, 0)
            compaction_model_call_count = owner._session_compaction_request_counts.get(
                target_sid, 0
            )
            tool_result_pruning_events = owner._session_tool_result_pruning_events.get(
                target_sid, 0
            )
            tool_results_pruned = owner._session_tool_results_pruned.get(target_sid, 0)
            tool_result_truncation_events = (
                owner._session_tool_result_truncation_events.get(target_sid, 0)
            )
            tool_results_truncated = owner._session_tool_results_truncated.get(
                target_sid, 0
            )
            unclassified_history_rewrites = (
                owner._session_unclassified_history_rewrites.get(target_sid, 0)
            )
            budget_termination = owner._session_budget_terminations.get(target_sid)
            owner._tool_call_repeat_state.pop(target_sid, None)
            session_metadata = dict(owner._session_metadata.get(target_sid, {}))
            if session_metadata.get("group_index") is None:
                owner._flush_pending_record(target_sid, None)
                owner._pending_step_data.pop(target_sid, None)
                owner._step_counts.pop(target_sid, None)
                owner._session_metadata.pop(target_sid, None)
                owner._session_token_ledgers.pop(target_sid, None)
                owner._session_context_overflows.pop(target_sid, None)
                owner._session_sglang_stats.pop(target_sid, None)
                owner._session_compaction_counts.pop(target_sid, None)
                owner._session_compaction_request_counts.pop(target_sid, None)
                owner._session_pending_compaction_rewrites.pop(target_sid, None)
                owner._session_tool_result_pruning_events.pop(target_sid, None)
                owner._session_tool_results_pruned.pop(target_sid, None)
                owner._session_tool_result_truncation_events.pop(target_sid, None)
                owner._session_tool_results_truncated.pop(target_sid, None)
                owner._session_unclassified_history_rewrites.pop(target_sid, None)
                owner._session_budget_terminations.pop(target_sid, None)

            return {
                "ok": True,
                "session_id": target_sid,
                "steps": steps,
                "raw_prompts": raw_prompts,
                "session_metadata": session_metadata,
                "exact_context_overflows": context_overflows,
                "sglang_stats": sglang_stats,
                "compaction_count": compaction_count,
                "compaction_model_call_count": compaction_model_call_count,
                "tool_result_pruning_events": tool_result_pruning_events,
                "tool_results_pruned": tool_results_pruned,
                "tool_result_truncation_events": tool_result_truncation_events,
                "tool_results_truncated": tool_results_truncated,
                "unclassified_history_rewrites": unclassified_history_rewrites,
                "budget_termination": budget_termination,
            }

        @app.post("/set_reward")
        async def set_reward(request: Request):
            owner: ClawGymRLAPIServer = request.app.state.owner
            body = await request.json()
            reward_value = float(body["reward"])
            # Sanitize NaN/Inf reward (e.g. reward.sh divide-by-zero, math-verify
            # exception in __mul__) so the whole GRPO group's mean/std doesn't
            # become NaN and silently kill that group's gradient.
            if not math.isfinite(reward_value):
                logger.warning(
                    f"non-finite reward {reward_value!r} for session {body.get('session_id')!r}, clamping to 0.0"
                )
                reward_value = 0.0
            target_sid = _require_string(body["session_id"], "session_id")
            group_index_override = int(body["group_index"])
            opsd_feedback = body.get("opsd_feedback")
            infrastructure_failure = bool(body.get("infrastructure_failure", False))

            if target_sid not in owner._pending_step_data:
                completed = owner._completed_reward_submissions.get(target_sid)
                if completed is not None:
                    return {**completed, "idempotent_replay": True}
                raise HTTPException(
                    status_code=404, detail=f"pending session not found: {target_sid}"
                )

            # Clean up step tracking for this session
            if target_sid in owner._step_counts:
                del owner._step_counts[target_sid]
            owner._flush_pending_record(target_sid, None)

            step_details: list[dict[str, Any]] = []
            pending = owner._pending_step_data.pop(target_sid)
            sorted_steps = sorted(pending)
            session_metadata = dict(owner._session_metadata.get(target_sid, {}))
            context_overflows = owner._session_context_overflows.pop(target_sid, [])
            sglang_stats = getattr(owner, "_session_sglang_stats", {}).pop(
                target_sid, {}
            )
            compaction_count = owner._session_compaction_counts.pop(target_sid, 0)
            compaction_model_call_count = owner._session_compaction_request_counts.pop(
                target_sid, 0
            )
            owner._session_pending_compaction_rewrites.pop(target_sid, None)
            tool_result_pruning_events = owner._session_tool_result_pruning_events.pop(
                target_sid, 0
            )
            tool_results_pruned = owner._session_tool_results_pruned.pop(target_sid, 0)
            tool_result_truncation_events = (
                owner._session_tool_result_truncation_events.pop(target_sid, 0)
            )
            tool_results_truncated = owner._session_tool_results_truncated.pop(
                target_sid, 0
            )
            unclassified_history_rewrites = (
                owner._session_unclassified_history_rewrites.pop(target_sid, 0)
            )
            budget_termination = owner._session_budget_terminations.pop(
                target_sid, None
            )
            sample_metadata = {
                **session_metadata,
                "sglang_stats": sglang_stats,
                "budget_termination": budget_termination,
                "compaction_count": compaction_count,
                "compaction_model_call_count": compaction_model_call_count,
                "tool_result_pruning_events": tool_result_pruning_events,
                "tool_results_pruned": tool_results_pruned,
                "tool_result_truncation_events": tool_result_truncation_events,
                "tool_results_truncated": tool_results_truncated,
                "unclassified_history_rewrites": unclassified_history_rewrites,
            }

            for position, step_num in enumerate(sorted_steps):
                step_data = pending[step_num]
                user_messages = []
                for message in step_data["messages"]:
                    if message["role"] in {"user", "system", "tool"}:
                        user_messages.append(
                            {
                                "role": message["role"],
                                "content": _flatten_message_content(message["content"])[
                                    :500
                                ],
                            }
                        )

                next_tool_results: list[dict[str, Any]] = []
                if position + 1 < len(sorted_steps):
                    next_step_data = pending[sorted_steps[position + 1]]
                    next_tool_results = _collect_new_tool_results(
                        next_step_data["messages"]
                    )
                expected_tool_call_ids = [
                    tool_call["id"] for tool_call in step_data["tool_calls"]
                ]
                observed_tool_call_ids = [
                    result["tool_call_id"] for result in next_tool_results
                ]
                step_details.append(
                    {
                        "step": step_num,
                        "input": user_messages,
                        "response": step_data["response_text"],
                        "tool_calls": step_data["tool_calls"],
                        "prompt_tokens": len(step_data["prompt_ids"]),
                        "response_tokens": len(step_data["response_ids"]),
                        "response_id_count": len(step_data["response_ids"]),
                        "response_logprob_count": len(step_data["response_logprobs"]),
                        "finish_reason": step_data.get("finish_reason", "stop"),
                        "response_limit_tokens": step_data.get("response_limit_tokens", 0),
                        "configured_response_limit_tokens": step_data.get(
                            "configured_response_limit_tokens", 0
                        ),
                        "hit_response_limit": bool(
                            step_data.get("hit_response_limit", False)
                        ),
                        "incomplete_tool_call_at_limit": bool(
                            step_data.get("incomplete_tool_call_at_limit", False)
                        ),
                        "repeated_ngram_ratio": float(
                            step_data.get("repeated_ngram_ratio", 0.0)
                        ),
                        "max_ngram_repeat": int(
                            step_data.get("max_ngram_repeat", 0)
                        ),
                        "repetition_degenerated": bool(
                            step_data.get("repetition_degenerated", False)
                        ),
                        "token_alignment_ok": (
                            len(step_data["response_ids"])
                            == len(step_data["response_logprobs"])
                        ),
                        "sampling_seed": step_data["sampling_seed"],
                        "has_followup_step": position + 1 < len(sorted_steps),
                        "ledger_mode": step_data.get("ledger_mode", "legacy"),
                        "ledger_transition": step_data.get("ledger_transition"),
                        "ledger_transition_detail": step_data.get("ledger_transition_detail", ""),
                        "prompt_source": step_data.get("prompt_source", "canonical"),
                        "canonical_prompt_tokens": step_data.get("canonical_prompt_tokens"),
                        "candidate_prompt_tokens": step_data.get("candidate_prompt_tokens"),
                        "assistant_boundary_tokens": step_data.get("assistant_boundary_tokens", 0),
                        "environment_delta_tokens": step_data.get("environment_delta_tokens", 0),
                        "compacted_tool_output_count": _count_compacted_tool_outputs(
                            step_data["messages"]
                        ),
                        "tool_results_pruned": step_data.get("tool_results_pruned", 0),
                        "tool_results_truncated": step_data.get(
                            "tool_results_truncated", 0
                        ),
                        "raw_tool_call_count": step_data["raw_tool_call_count"],
                        "parsed_tool_call_count": step_data["parsed_tool_call_count"],
                        "returned_tool_call_count": step_data[
                            "returned_tool_call_count"
                        ],
                        "consecutive_identical_tool_call_turns": step_data.get(
                            "consecutive_identical_tool_call_turns", 0
                        ),
                        "observed_tool_result_count": len(next_tool_results),
                        "tool_call_ids": expected_tool_call_ids,
                        "observed_tool_call_ids": observed_tool_call_ids,
                        "tool_result_id_order_matches": (
                            observed_tool_call_ids == expected_tool_call_ids
                        ),
                    }
                )

            all_tokens: list[int] = []
            all_loss_mask: list[int] = []
            all_logprobs: list[float] = []
            all_step_ids: list[int] = []
            segment_agent_steps: list[int] = []
            prompt_text = ""
            response_text = ""
            last_response_text: str | None = None

            def prepare_dense_opsd(sample: Sample) -> None:
                owner_args = getattr(owner, "args", None)
                guidance_enabled = bool(
                    getattr(owner_args, "opsd_enable", False)
                    or getattr(owner_args, "sdar_enable", False)
                )
                if not guidance_enabled or getattr(owner_args, "use_rollout_routing_replay", False):
                    return
                sample.opsd_step_ids = list(all_step_ids)
                segment = {
                    "tokens": list(sample.tokens),
                    "loss_mask": list(sample.loss_mask),
                    "agent_steps": list(segment_agent_steps),
                }
                (
                    sample.opsd_teacher_tokens,
                    sample.opsd_placebo_teacher_tokens,
                    sample.opsd_eligible,
                    sample.opsd_fallback_reason,
                ) = _prepare_opsd_pair(
                    owner.tokenizer,
                    segment,
                    pending,
                    opsd_feedback,
                    max_context_tokens=int(
                        os.environ.get("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536")
                    ),
                    infrastructure_failure=infrastructure_failure,
                    require_incomplete=not bool(
                        getattr(owner_args, "sdar_enable", False)
                        or str(getattr(owner_args, "opsd_mode", "absolute"))
                        in {"agentopsd", "bb_agentopsd"}
                    ),
                    matched_placebo=bool(
                        getattr(owner_args, "opsd_matched_placebo", False)
                    ),
                )
                sample.opsd_placebo_task_id = str(
                    (opsd_feedback or {}).get("placebo_task_id") or ""
                )

            # A prompt may only extend the exact token sequence accumulated so far.
            # OpenClaw compression or any other rewrite flushes the current segment
            # and starts from the exact new prompt.  Never splice at a longest common
            # prefix: rollout logprobs were sampled under the new prompt, not a hybrid.
            #
            # All split samples from the SAME rollout share one traj_index so
            # slime's --dynamic-history GRPO normalization treats them as one
            # trajectory (one vote per traj in group baseline) instead of N
            # independent samples (which would over-weight long trajs that
            # split into many pieces).
            traj_index = next(owner._index_counter)
            samples_to_submit: list[Sample] = []
            segment_index = 0
            segment_hit_response_limit = False

            for index, step_num in enumerate(sorted_steps):
                step_data = pending[step_num]
                prompt_ids = step_data["prompt_ids"]
                response_ids = step_data["response_ids"]
                response_logprobs = step_data["response_logprobs"]
                tool_results_pruned_at_step = int(
                    step_data.get("tool_results_pruned", 0)
                )
                tool_results_truncated_at_step = int(
                    step_data.get("tool_results_truncated", 0)
                )
                session_compacted_at_step = (
                    step_data.get("ledger_transition") == "session_compacted"
                )

                if index == 0:
                    # First step: start new trajectory
                    all_tokens.extend(prompt_ids)
                    prompt_text = step_data["prompt_text"]
                    step_details[index].update(
                        {
                            "context_transition": "initial",
                            "shared_prefix_tokens": 0,
                            "previous_segment_tokens": 0,
                            "current_prompt_tokens": len(prompt_ids),
                            "split_performed": False,
                            "split_reason": None,
                            "segment_index": segment_index,
                            "compaction_detected": session_compacted_at_step,
                            "tool_result_pruning_detected": (
                                tool_results_pruned_at_step > 0
                            ),
                            "tool_result_truncation_detected": (
                                tool_results_truncated_at_step > 0
                            ),
                        }
                    )
                elif os.environ.get("OPENCLAW_ASSEMBLE_OFFICIAL_SPLICE", "0") == "1":
                    # Public ClawGym assembly: keep the old suffix and append the
                    # new prompt from the first mismatched token. Flush only when
                    # the previous assistant text is gone, or the window is full.
                    current_prompt = step_data.get("prompt_text") or ""
                    context_changed = not (
                        last_response_text
                        and last_response_text.strip()
                        and last_response_text.strip() in current_prompt
                    )
                    max_tokens = int(
                        os.environ.get("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536")
                    )
                    exceeds_limit = len(all_tokens) + len(response_ids) > max_tokens
                    previous_segment_tokens = len(all_tokens)
                    shared_prefix_tokens = _shared_prefix_length(all_tokens, prompt_ids)
                    split_performed = context_changed or exceeds_limit
                    if context_changed:
                        context_transition = "official_context_changed"
                        split_reason = "official_context_changed"
                    elif exceeds_limit:
                        context_transition = "trajectory_limit"
                        split_reason = "trajectory_limit"
                    else:
                        context_transition = "official_splice"
                        split_reason = None
                    if split_performed:
                        if all_loss_mask:
                            sample = Sample()
                            sample.prompt = prompt_text
                            sample.response = response_text
                            sample.tokens = list(all_tokens)
                            sample.response_length = len(all_loss_mask)
                            sample.loss_mask = list(all_loss_mask)
                            sample.rollout_log_probs = list(all_logprobs)
                            sample.status = (
                                Sample.Status.TRUNCATED
                                if segment_hit_response_limit
                                else Sample.Status.COMPLETED
                            )
                            sample.index = traj_index
                            sample.group_index = group_index_override
                            sample.reward = {"score": reward_value}
                            sample.metadata = {
                                **sample_metadata,
                                "session_id": target_sid,
                                "traj_index": traj_index,
                                "group_index": group_index_override,
                                "segment_index": segment_index,
                            }
                            prepare_dense_opsd(sample)
                            samples_to_submit.append(sample)
                        all_tokens = list(prompt_ids)
                        all_loss_mask = []
                        all_logprobs = []
                        all_step_ids = []
                        segment_agent_steps = []
                        prompt_text = step_data["prompt_text"]
                        response_text = ""
                        segment_index += 1
                        segment_hit_response_limit = False
                    else:
                        delta_ids = list(prompt_ids[shared_prefix_tokens:])
                        if delta_ids:
                            all_tokens.extend(delta_ids)
                            all_loss_mask.extend([0] * len(delta_ids))
                            all_logprobs.extend([0.0] * len(delta_ids))
                            all_step_ids.extend([-1] * len(delta_ids))
                    step_details[index].update(
                        {
                            "context_transition": context_transition,
                            "shared_prefix_tokens": shared_prefix_tokens,
                            "previous_segment_tokens": previous_segment_tokens,
                            "current_prompt_tokens": len(prompt_ids),
                            "split_performed": split_performed,
                            "split_reason": split_reason,
                            "segment_index": segment_index,
                            "compaction_detected": session_compacted_at_step,
                            "tool_result_pruning_detected": (
                                tool_results_pruned_at_step > 0
                            ),
                            "tool_result_truncation_detected": (
                                tool_results_truncated_at_step > 0
                            ),
                        }
                    )
                else:
                    previous_segment_tokens = len(all_tokens)
                    shared_prefix_tokens = _shared_prefix_length(all_tokens, prompt_ids)
                    delta_ids = _strict_prompt_append_delta(all_tokens, prompt_ids)
                    max_tokens = int(
                        os.environ.get("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536")
                    )
                    prompt_was_rewritten = delta_ids is None
                    declared_transition = step_data.get("ledger_transition")
                    declared_rewrite = declared_transition not in {
                        None,
                        "initial",
                        "append_only",
                    }
                    next_step_exceeds_limit = (
                        len(prompt_ids) + len(response_ids) > max_tokens
                    )
                    if declared_rewrite:
                        context_transition = str(declared_transition)
                    elif prompt_was_rewritten:
                        if (
                            declared_transition == "append_only"
                            and step_data.get("prompt_source") != "exact_token_ledger"
                        ):
                            context_transition = "serialization_rewrite_shadow"
                        elif declared_transition == "append_only":
                            context_transition = "token_ledger_invariant_failed"
                        else:
                            context_transition = "compressed_or_rewritten"
                    else:
                        context_transition = "append_only"

                    context_requires_split = declared_rewrite or prompt_was_rewritten
                    split_performed = (
                        context_requires_split or next_step_exceeds_limit
                    )
                    split_reason = None
                    if context_requires_split:
                        split_reason = context_transition
                    elif next_step_exceeds_limit:
                        split_reason = "trajectory_limit"

                    if context_requires_split or next_step_exceeds_limit:
                        # Flush the valid prefix, then train this step only under its
                        # exact prompt.  Split segments keep the same trajectory and
                        # group identity, and therefore the same terminal reward.
                        if all_loss_mask:
                            sample = Sample()
                            sample.prompt = prompt_text
                            sample.response = response_text
                            sample.tokens = list(all_tokens)
                            sample.response_length = len(all_loss_mask)
                            sample.loss_mask = list(all_loss_mask)
                            sample.rollout_log_probs = list(all_logprobs)
                            sample.status = (
                                Sample.Status.TRUNCATED
                                if segment_hit_response_limit
                                else Sample.Status.COMPLETED
                            )
                            sample.index = traj_index
                            sample.group_index = group_index_override
                            sample.reward = {"score": reward_value}
                            sample.metadata = {
                                **sample_metadata,
                                "session_id": target_sid,
                                "traj_index": traj_index,
                                "group_index": group_index_override,
                                "segment_index": segment_index,
                            }
                            prepare_dense_opsd(sample)
                            samples_to_submit.append(sample)

                        all_tokens = list(prompt_ids)
                        all_loss_mask = []
                        all_logprobs = []
                        all_step_ids = []
                        segment_agent_steps = []
                        prompt_text = step_data["prompt_text"]
                        response_text = ""
                        segment_index += 1
                        segment_hit_response_limit = False
                    elif delta_ids:
                        all_tokens.extend(delta_ids)
                        all_loss_mask.extend([0] * len(delta_ids))
                        all_logprobs.extend([0.0] * len(delta_ids))
                        all_step_ids.extend([-1] * len(delta_ids))
                    step_details[index].update(
                        {
                            "context_transition": context_transition,
                            "shared_prefix_tokens": shared_prefix_tokens,
                            "previous_segment_tokens": previous_segment_tokens,
                            "current_prompt_tokens": len(prompt_ids),
                            "split_performed": split_performed,
                            "split_reason": split_reason,
                            "segment_index": segment_index,
                            "compaction_detected": session_compacted_at_step,
                            "tool_result_pruning_detected": (
                                tool_results_pruned_at_step > 0
                            ),
                            "tool_result_truncation_detected": (
                                tool_results_truncated_at_step > 0
                            ),
                        }
                    )

                segment_hit_response_limit = (
                    segment_hit_response_limit
                    or bool(step_data.get("hit_response_limit", False))
                )
                all_tokens.extend(response_ids)
                all_loss_mask.extend([1] * len(response_ids))
                all_logprobs.extend(response_logprobs)
                all_step_ids.extend([step_num] * len(response_ids))
                segment_agent_steps.append(step_num)
                response_text += step_data["response_text"]
                last_response_text = step_data["response_text"]
                step_details[index]["segment_tokens_after_response"] = len(all_tokens)

            # Flush the last trajectory
            max_ctx = int(os.environ.get("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536"))
            if all_loss_mask:
                if len(all_tokens) <= max_ctx:
                    sample = Sample()
                    sample.prompt = prompt_text
                    sample.response = response_text
                    sample.tokens = list(all_tokens)
                    sample.response_length = len(all_loss_mask)
                    sample.loss_mask = list(all_loss_mask)
                    sample.rollout_log_probs = list(all_logprobs)
                    sample.status = (
                        Sample.Status.TRUNCATED
                        if segment_hit_response_limit
                        else Sample.Status.COMPLETED
                    )
                    sample.index = traj_index
                    sample.group_index = group_index_override
                    sample.reward = {"score": reward_value}
                    sample.metadata = {
                        **sample_metadata,
                        "session_id": target_sid,
                        "traj_index": traj_index,
                        "group_index": group_index_override,
                        "segment_index": segment_index,
                    }
                    prepare_dense_opsd(sample)
                    samples_to_submit.append(sample)
                else:
                    logger.warning(
                        f"[ClawGym-RL] dropping oversized sample: {len(all_tokens)} tokens > {max_ctx}, "
                        f"session={target_sid}, steps={len(sorted_steps)}"
                    )

            if getattr(
                getattr(owner, "args", None),
                "use_rollout_routing_replay",
                False,
            ):
                # Preserve exact routes while avoiding repeated full-prefix
                # training.  Rewrites or route conflicts start a new exact
                # segment; all segments retain one trajectory/group identity.
                routing_segments = _build_stitched_routing_replay_segments(
                    pending,
                    sorted_steps,
                    max_context_tokens=max_ctx,
                    num_layers=int(owner.args.num_layers),
                    router_topk=int(owner.args.moe_router_topk),
                    session_id=target_sid,
                    step_details=step_details,
                )
                samples_to_submit = []
                for segment in routing_segments:
                    sample = Sample()
                    sample.prompt = segment["prompt_text"]
                    sample.response = segment["response_text"]
                    sample.tokens = list(segment["tokens"])
                    sample.response_length = len(segment["loss_mask"])
                    sample.loss_mask = list(segment["loss_mask"])
                    sample.rollout_log_probs = list(
                        segment["rollout_log_probs"]
                    )
                    sample.rollout_routed_experts = segment["routes"]
                    sample.opsd_step_ids = list(segment["response_step_ids"])
                    if (
                        getattr(owner.args, "opsd_enable", False)
                        or getattr(owner.args, "sdar_enable", False)
                    ):
                        (
                            sample.opsd_teacher_tokens,
                            sample.opsd_placebo_teacher_tokens,
                            sample.opsd_eligible,
                            sample.opsd_fallback_reason,
                        ) = _prepare_opsd_pair(
                            owner.tokenizer,
                            segment,
                            pending,
                            opsd_feedback,
                            max_context_tokens=max_ctx,
                            infrastructure_failure=infrastructure_failure,
                            require_incomplete=not bool(
                                getattr(owner.args, "sdar_enable", False)
                                or str(getattr(owner.args, "opsd_mode", "absolute"))
                                in {"agentopsd", "bb_agentopsd"}
                            ),
                            matched_placebo=bool(
                                getattr(
                                    owner.args,
                                    "opsd_matched_placebo",
                                    False,
                                )
                            ),
                        )
                        sample.opsd_placebo_task_id = str(
                            (opsd_feedback or {}).get("placebo_task_id") or ""
                        )

                    sample.status = (
                        Sample.Status.TRUNCATED
                        if segment["hit_response_limit"]
                        else Sample.Status.COMPLETED
                    )
                    sample.index = traj_index
                    sample.group_index = group_index_override
                    sample.reward = {"score": reward_value}
                    sample.metadata = {
                        **sample_metadata,
                        "session_id": target_sid,
                        "traj_index": traj_index,
                        "group_index": group_index_override,
                        "segment_index": segment["segment_index"],
                        "routing_replay_scope": (
                            "stitched_agent_requests"
                            if segment["turn_count"] > 1
                            else "single_agent_request"
                        ),
                        "routing_replay_agent_steps": segment[
                            "agent_steps"
                        ],
                        "routing_replay_turn_count": segment["turn_count"],
                        "routing_replay_stitched_turns": segment[
                            "stitched_turns"
                        ],
                        "routing_replay_split_reason": segment[
                            "split_reason"
                        ],
                    }
                    samples_to_submit.append(sample)

            # Also filter any mid-trajectory samples that exceeded the limit
            samples_to_submit = [
                s for s in samples_to_submit if len(s.tokens) <= max_ctx
            ]

            audit_keys = (
                "step",
                "prompt_tokens",
                "response_tokens",
                "response_id_count",
                "response_logprob_count",
                "finish_reason",
                "response_limit_tokens",
                "configured_response_limit_tokens",
                "hit_response_limit",
                "incomplete_tool_call_at_limit",
                "repeated_ngram_ratio",
                "max_ngram_repeat",
                "repetition_degenerated",
                "token_alignment_ok",
                "sampling_seed",
                "has_followup_step",
                "ledger_mode",
                "ledger_transition",
                "ledger_transition_detail",
                "prompt_source",
                "canonical_prompt_tokens",
                "candidate_prompt_tokens",
                "assistant_boundary_tokens",
                "environment_delta_tokens",
                "raw_tool_call_count",
                "parsed_tool_call_count",
                "returned_tool_call_count",
                "consecutive_identical_tool_call_turns",
                "observed_tool_result_count",
                "tool_call_ids",
                "observed_tool_call_ids",
                "tool_result_id_order_matches",
                "compacted_tool_output_count",
                "tool_results_pruned",
                "tool_results_truncated",
                "compaction_detected",
                "tool_result_pruning_detected",
                "tool_result_truncation_detected",
                "context_transition",
                "shared_prefix_tokens",
                "previous_segment_tokens",
                "current_prompt_tokens",
                "split_performed",
                "split_reason",
                "segment_index",
                "segment_tokens_after_response",
                "routing_replay_action",
                "routing_replay_prefix_route_rows",
                "routing_replay_prefix_match",
                "routing_replay_split_reason",
                "routing_replay_segment_index",
                "routing_replay_conflict_diagnostics",
            )
            routing_replay_samples = [
                sample
                for sample in samples_to_submit
                if str(sample.metadata.get("routing_replay_scope", "")).endswith(
                    "agent_requests"
                )
                or sample.metadata.get("routing_replay_scope")
                == "single_agent_request"
            ]
            owner._append_audit_record(
                {
                    **session_metadata,
                    "record_type": "trajectory_audit",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": target_sid,
                    "group_index": group_index_override,
                    "traj_index": traj_index,
                    "logical_trajectory_count": 1,
                    "submitted_segment_count": len(samples_to_submit),
                    "routing_replay_segment_count": len(
                        routing_replay_samples
                    ),
                    "routing_replay_turn_count": sum(
                        int(
                            sample.metadata.get(
                                "routing_replay_turn_count", 0
                            )
                        )
                        for sample in routing_replay_samples
                    ),
                    "routing_replay_stitched_turns": sum(
                        int(
                            sample.metadata.get(
                                "routing_replay_stitched_turns", 0
                            )
                        )
                        for sample in routing_replay_samples
                    ),
                    "routing_replay_route_conflict_splits": sum(
                        sample.metadata.get("routing_replay_split_reason")
                        == "route_prefix_conflict"
                        for sample in routing_replay_samples
                    ),
                    "exact_context_overflow_count": len(context_overflows),
                    "exact_context_overflows": context_overflows,
                    "sglang_stats": sglang_stats,
                    "compaction_count": compaction_count,
                    "compaction_model_call_count": compaction_model_call_count,
                    "tool_result_pruning_events": tool_result_pruning_events,
                    "tool_results_pruned": tool_results_pruned,
                    "tool_result_truncation_events": tool_result_truncation_events,
                    "tool_results_truncated": tool_results_truncated,
                    "unclassified_history_rewrites": unclassified_history_rewrites,
                    "budget_termination": budget_termination,
                    "max_consecutive_identical_tool_call_turns": max(
                        (
                            int(
                                detail.get(
                                    "consecutive_identical_tool_call_turns", 0
                                )
                            )
                            for detail in step_details
                        ),
                        default=0,
                    ),
                    "steps": [
                        {key: detail.get(key) for key in audit_keys}
                        for detail in step_details
                    ],
                }
            )
            owner._session_metadata.pop(target_sid, None)
            getattr(owner, "_session_token_ledgers", {}).pop(target_sid, None)
            owner._tool_call_repeat_state.pop(target_sid, None)

            for s in samples_to_submit:
                await asyncio.to_thread(owner.output_queue.put, (s.group_index, [s]))

            response_payload = {
                "ok": True,
                "reward": reward_value,
                "submitted": len(samples_to_submit),
                "steps": step_details,
            }
            owner._completed_reward_submissions[target_sid] = {
                "ok": True,
                "reward": reward_value,
                "submitted": len(samples_to_submit),
                "session_id": target_sid,
            }
            return response_payload

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: str | None = Header(default=None),
        ):
            owner: ClawGymRLAPIServer = request.app.state.owner
            await owner._check_auth(authorization)
            if not owner.submission_enabled.is_set():
                raise HTTPException(
                    status_code=503, detail="submission paused for weight update"
                )

            body = await request.json()
            turn_type, session_id = _parse_request_model(body["model"])
            stream = _require_bool(body["stream"], "stream")
            result = await owner._handle_request(
                body, session_id=session_id, turn_type=turn_type
            )
            if isinstance(result, Response):
                return result
            if stream:
                return StreamingResponse(
                    owner._stream_response(result), media_type="text/event-stream"
                )
            return JSONResponse(content=result["response"])

        return app

    async def _check_auth(self, authorization: str | None):
        if not self.expected_api_key:
            return
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization.split(" ", 1)[1].strip()
        if token != self.expected_api_key:
            raise HTTPException(status_code=401, detail="invalid api key")

    def _buffer_record(
        self,
        session_id: str,
        step_num: int,
        messages: list[dict[str, Any]],
        prompt_text: str,
        response_text: str,
        tool_calls: list[dict[str, Any]],
        tool_audit: dict[str, Any],
    ):
        if not self._record_file:
            return
        self._pending_records[session_id] = {
            "session_id": session_id,
            "step": step_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "prompt_text": prompt_text,
            "response_text": response_text,
            "tool_calls": tool_calls,
            **tool_audit,
        }

    def _flush_pending_record(
        self,
        session_id: str,
        next_state: dict[str, Any] | None,
        observed_tool_results: list[dict[str, Any]] | None = None,
    ):
        if session_id not in self._pending_records:
            return
        record = self._pending_records.pop(session_id)
        observed_tool_results = observed_tool_results or []
        expected_tool_call_ids = [
            str(tool_call.get("id") or "") for tool_call in record.get("tool_calls", [])
        ]
        observed_tool_call_ids = [
            str(result.get("tool_call_id") or "") for result in observed_tool_results
        ]
        record["next_state"] = next_state
        record["observed_tool_result_count"] = len(observed_tool_results)
        record["observed_tool_call_ids"] = observed_tool_call_ids
        record["tool_result_id_order_matches"] = (
            observed_tool_call_ids == expected_tool_call_ids
        )
        if not self._record_file:
            return
        with open(self._record_file, "a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_audit_record(self, record: dict[str, Any]):
        if not self._record_file:
            return
        with open(self._record_file, "a", encoding="utf-8") as file_handle:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_previous_step_observations(
        self, session_id: str, messages: list[dict[str, Any]], turn_type: str
    ) -> None:
        tool_results = _collect_new_tool_results(messages)
        observations = [result["content"] for result in tool_results]
        observed_tool_call_ids = [result["tool_call_id"] for result in tool_results]
        conversation = self._conversation_logs.get(session_id, [])
        if conversation:
            previous_step = conversation[-1]
            expected_tool_call_ids = [
                tool_call["id"] for tool_call in previous_step.get("tool_calls", [])
            ]
            previous_step["observations"] = observations
            previous_step["observed_tool_call_ids"] = observed_tool_call_ids
            previous_step["observed_tool_result_count"] = len(tool_results)
            previous_step["tool_result_id_order_matches"] = (
                observed_tool_call_ids == expected_tool_call_ids
            )
        if turn_type == "main" and self._step_counts.get(session_id, 0) > 0:
            self._flush_pending_record(
                session_id, messages[-1] if messages else None, tool_results
            )

    def _budget_terminal_response(
        self,
        session_id: str,
        turn_type: str,
        messages: list[dict[str, Any]],
        model: str,
        termination: dict[str, Any],
    ) -> dict[str, Any]:
        self._session_budget_terminations[session_id] = termination
        # A compact-model request carries summary instructions, not new tool
        # observations for the previous policy action.
        if turn_type != "compact":
            self._update_previous_step_observations(session_id, messages, turn_type)
        logger.warning(
            "[ClawGym-RL] terminating session=%s at execution budget: %s",
            session_id,
            json.dumps(termination, ensure_ascii=False, sort_keys=True),
        )
        output = {
            "id": f"chatcmpl-budget-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "",
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "metadata": {"clawgym_budget_termination": termination},
            "session_id": session_id,
        }
        return {"response": output}

    def _assert_session_open(self, session_id: str) -> None:
        if session_id in self._finalized_sessions:
            raise HTTPException(status_code=409, detail="session trajectory already finalized")

    async def _handle_request(
        self, body: dict[str, Any], session_id: str, turn_type: str
    ):
        self._assert_session_open(session_id)
        messages = _require_list(body["messages"], "messages")
        tools = body["tools"] if "tools" in body else None

        # Store raw system/user prompts on the first request per session
        if session_id not in self._session_raw_prompts:
            raw = {}
            for msg in messages:
                role = msg.get("role", "")
                if role == "system":
                    raw["system_prompt"] = _flatten_message_content(
                        msg.get("content", "")
                    )
                elif role == "user" and "user_prompt" not in raw:
                    raw["user_prompt"] = _flatten_message_content(
                        msg.get("content", "")
                    )
            if tools:
                raw["tools"] = json.dumps(tools, ensure_ascii=False)
            self._session_raw_prompts[session_id] = raw

        # Dump the raw prompt on the first session only (full), subsequent sessions only log user prompt
        if session_id not in self._step_counts:
            if not self._full_prompt_logged:
                self._full_prompt_logged = True
                logger.info("[ClawGym-RL] === FULL SYSTEM PROMPT (logged once) ===")
                for i, msg in enumerate(messages):
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    preview = content if isinstance(content, str) else str(content)
                    logger.info(
                        "[ClawGym-RL]   message[%d] role=%s content=%s",
                        i,
                        role,
                        preview,
                    )
                if tools:
                    logger.info(
                        "[ClawGym-RL]   tools=%s", json.dumps(tools, ensure_ascii=False)
                    )
                logger.info("[ClawGym-RL] === END FULL SYSTEM PROMPT ===")
            else:
                user_content = ""
                for msg in messages:
                    if msg.get("role") == "user":
                        user_content = _flatten_message_content(msg.get("content", ""))[
                            :500
                        ]
                        break
                logger.info(
                    "[ClawGym-RL] session=%s user_prompt=%s", session_id, user_content
                )

        model_name = _require_string(body["model"], "model")
        existing_termination = getattr(
            self, "_session_budget_terminations", {}
        ).get(session_id)
        if existing_termination is not None:
            return self._budget_terminal_response(
                session_id, turn_type, messages, model_name, existing_termination
            )

        completed_agent_steps = len(self._conversation_logs.get(session_id, []))
        max_agent_steps = int(os.environ.get("OPENCLAW_MAX_AGENT_STEPS", "96"))
        if turn_type != "compact" and completed_agent_steps >= max_agent_steps:
            termination = {
                "reason": "max_agent_steps",
                "limit": max_agent_steps,
                "completed_agent_steps": completed_agent_steps,
                "compaction_count": getattr(
                    self, "_session_compaction_counts", {}
                ).get(session_id, 0),
            }
            return self._budget_terminal_response(
                session_id, turn_type, messages, model_name, termination
            )
        if turn_type == "compact":
            compaction_count = getattr(
                self, "_session_compaction_counts", {}
            ).get(session_id, 0)
            max_compactions = int(
                os.environ.get("OPENCLAW_MAX_COMPACTION_EVENTS", "8")
            )
            if compaction_count >= max_compactions:
                termination = {
                    "reason": "max_compaction_events",
                    "limit": max_compactions,
                    "completed_agent_steps": completed_agent_steps,
                    "compaction_count": compaction_count,
                    "attempted_compaction_count": compaction_count + 1,
                }
                return self._budget_terminal_response(
                    session_id, turn_type, messages, model_name, termination
                )

        forward_body = {
            key: value
            for key, value in body.items()
            if key not in _NON_STANDARD_BODY_KEYS
        }
        forward_body["stream"] = False
        forward_body.pop("stream_options", None)
        _require_string(forward_body["model"], "model")

        chat_request, parser, sampling_params = _build_generate_sampling_params(
            forward_body,
            self.generation_defaults,
            self.tool_call_parser,
        )
        sampling_seed = None
        sampling_seed_base = os.environ.get(
            "OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE", ""
        ).strip()
        session_metadata = self._session_metadata.get(session_id, {})
        sampling_identity = session_metadata.get("sampling_identity") or session_id
        if not isinstance(sampling_identity, str):
            raise TypeError("session sampling_identity must be a string")
        if sampling_seed_base:
            compact_ordinal = getattr(
                self, "_session_compaction_request_counts", {}
            ).get(session_id, 0)
            ordinal = completed_agent_steps if turn_type == "main" else compact_ordinal
            seed_material = (
                f"{int(sampling_seed_base)}:{sampling_identity}:{turn_type}:{ordinal}"
            ).encode("utf-8")
            sampling_seed = int.from_bytes(
                hashlib.sha256(seed_material).digest()[:4], "little"
            ) & 0x7FFFFFFF
            sampling_params["sampling_seed"] = sampling_seed
        template_tools = None
        if chat_request.tools and chat_request.tool_choice != "none":
            template_tools = [tool.model_dump() for tool in chat_request.tools]
        template_kwargs = dict(chat_request.chat_template_kwargs or {})
        canonical_prompt_ids = _apply_chat_template_ids(
            self.tokenizer,
            messages,
            template_tools=template_tools,
            template_kwargs=template_kwargs,
            reasoning_effort=chat_request.reasoning_effort,
            add_generation_prompt=True,
        )

        ledger_transition = "initial"
        ledger_transition_detail = ""
        assistant_boundary_ids: list[int] = []
        environment_delta_ids: list[int] = []
        candidate_prompt_ids: list[int] | None = None
        ledgers = getattr(self, "_session_token_ledgers", {})
        tool_results_pruned = 0
        tool_results_truncated = 0
        # Compaction summaries are context-management calls, not policy actions.
        # Keep their token state separate from the main agent ledger so the next
        # real request detects rewritten history and starts a new RL segment.
        previous_ledger = None if turn_type == "compact" else ledgers.get(session_id)
        if previous_ledger is not None:
            ledger_transition, appended, ledger_transition_detail = (
                _classify_structural_append(
                    previous_ledger,
                    messages,
                    template_tools,
                )
            )
            if ledger_transition == "history_rewritten":
                tool_results_pruned = _count_new_tool_result_markers(
                    previous_ledger.messages,
                    messages,
                    _COMPACTED_TOOL_OUTPUT_MARKER,
                )
                tool_results_truncated = _count_new_tool_result_markers(
                    previous_ledger.messages,
                    messages,
                    _TRUNCATED_TOOL_OUTPUT_MARKER,
                )
                pending_compaction = getattr(
                    self, "_session_pending_compaction_rewrites", {}
                ).get(session_id, 0)
                if pending_compaction:
                    ledger_transition = "session_compacted"
                elif tool_results_pruned:
                    ledger_transition = "tool_result_pruned"
                elif tool_results_truncated:
                    ledger_transition = "tool_result_truncated"
            if turn_type == "main":
                # A successful compact call owns only the immediately following
                # main-context rebuild. Do not leak it into a later rewrite.
                getattr(
                    self, "_session_pending_compaction_rewrites", {}
                ).pop(session_id, None)
                if ledger_transition == "session_compacted":
                    compactions = self._session_compaction_counts
                    compactions[session_id] = compactions.get(session_id, 0) + 1
                if tool_results_pruned:
                    pruning_events = self._session_tool_result_pruning_events
                    pruning_events[session_id] = pruning_events.get(session_id, 0) + 1
                    pruned = self._session_tool_results_pruned
                    pruned[session_id] = pruned.get(session_id, 0) + tool_results_pruned
                if tool_results_truncated:
                    truncation_events = self._session_tool_result_truncation_events
                    truncation_events[session_id] = (
                        truncation_events.get(session_id, 0) + 1
                    )
                    truncated = self._session_tool_results_truncated
                    truncated[session_id] = (
                        truncated.get(session_id, 0) + tool_results_truncated
                    )
                if ledger_transition == "history_rewritten":
                    rewrites = self._session_unclassified_history_rewrites
                    rewrites[session_id] = rewrites.get(session_id, 0) + 1
            if ledger_transition == "append_only":
                try:
                    (
                        candidate_prompt_ids,
                        assistant_boundary_ids,
                        environment_delta_ids,
                    ) = _build_incremental_prompt_ids(
                        self.tokenizer,
                        previous_ledger,
                        messages,
                        appended,
                        template_tools=template_tools,
                        template_kwargs=template_kwargs,
                        reasoning_effort=chat_request.reasoning_effort,
                    )
                except ValueError as exc:
                    ledger_transition = "template_delta_invalid"
                    ledger_transition_detail = str(exc)
                    logger.warning(
                        "[ClawGym-RL] token ledger fallback for session=%s: %s",
                        session_id,
                        exc,
                    )

        ledger_mode = getattr(self, "_token_ledger_mode", "enforce")
        use_exact_ledger = (
            ledger_mode == "enforce"
            and ledger_transition == "append_only"
            and candidate_prompt_ids is not None
        )
        prompt_ids = (
            list(candidate_prompt_ids)
            if use_exact_ledger
            else list(canonical_prompt_ids)
        )
        prompt_source = "exact_token_ledger" if use_exact_ledger else "canonical"
        prompt_text = self.tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generate_body = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        if (
            os.environ.get("OPENCLAW_ROUTING_CONFLICT_DIAGNOSTICS", "0") == "1"
            and previous_ledger is not None
            and ledger_transition == "append_only"
        ):
            # Only rescore the immediately preceding assistant response and
            # the appended environment suffix. Full-prefix prompt logprobs can
            # require an unnecessarily large vocabulary projection for 20-60K
            # contexts and are not needed to diagnose this boundary.
            generate_body["logprob_start_len"] = len(previous_ledger.prompt_ids)
        if getattr(self.args, "use_rollout_routing_replay", False):
            generate_body["return_routed_experts"] = True

        context_window = int(os.environ.get("OPENCLAW_CONTEXT_WINDOW", "65536"))
        max_new_tokens = int(sampling_params.get("max_new_tokens") or 0)
        # SGLang rejects requests whose prompt + completion budget is exactly
        # equal to the configured context length, so keep one token of headroom.
        available_response_tokens = context_window - len(prompt_ids) - 1
        requested_total_tokens = len(prompt_ids) + max_new_tokens
        if available_response_tokens <= 0:
            event = {
                "prompt_tokens": len(prompt_ids),
                "max_new_tokens": max_new_tokens,
                "requested_total_tokens": requested_total_tokens,
                "context_window": context_window,
                "overflow_tokens": requested_total_tokens - context_window,
            }
            if not hasattr(self, "_session_context_overflows"):
                self._session_context_overflows = {}
            self._session_context_overflows.setdefault(session_id, []).append(event)
            logger.warning(
                "[ClawGym-RL] exact context guard for session=%s: "
                "prompt=%d + response=%d > context=%d (overflow=%d); "
                "returning 400 for OpenClaw overflow recovery",
                session_id,
                len(prompt_ids),
                max_new_tokens,
                context_window,
                requested_total_tokens - context_window,
            )
            return _openai_error_response(
                status_code=400,
                message=(
                    "Prompt is too long: requested "
                    f"{requested_total_tokens} tokens ({len(prompt_ids)} prompt + "
                    f"{max_new_tokens} completion), which exceeds the model's "
                    f"maximum context length of {context_window} tokens."
                ),
                code="context_length_exceeded",
            )
        if max_new_tokens > available_response_tokens:
            logger.info(
                "[ClawGym-RL] fitting response budget for session=%s turn_type=%s: "
                "prompt=%d requested_response=%d fitted_response=%d context=%d",
                session_id,
                turn_type,
                len(prompt_ids),
                max_new_tokens,
                available_response_tokens,
                context_window,
            )
            sampling_params["max_new_tokens"] = available_response_tokens

        sglang_request_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            # Retry on 503 (KV cache full) — usually transient, clears in a few seconds
            max_retries = 5
            retry_delay = 2.0
            for attempt in range(max_retries + 1):
                sglang_response = await client.post(
                    self.sglang_generate_url,
                    json=generate_body,
                    headers={"X-SMG-Routing-Key": session_id},
                )
                if sglang_response.status_code != 503 or attempt == max_retries:
                    break
                logger.info(
                    "[ClawGym-RL] SGLang 503 for session=%s, retry %d/%d",
                    session_id,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
            if sglang_response.status_code != 200:
                logger.warning(
                    "[ClawGym-RL] SGLang returned %s for session=%s",
                    sglang_response.status_code,
                    session_id,
                )
                return _openai_error_response(
                    status_code=sglang_response.status_code,
                    message=f"upstream SGLang error: {sglang_response.text[:500]}",
                    error_type="upstream_error",
                )
            generate_output = _require_dict(sglang_response.json(), "generate response")
        # No await occurs between this check and recording the response.
        # Reject late output, rather than assigning it to an archived turn.
        self._assert_session_open(session_id)
        sglang_request_e2e_seconds = time.perf_counter() - sglang_request_started

        raw_response_text, response_ids, response_logprobs = (
            _extract_aligned_generate_response(generate_output)
        )
        tool_call_id_namespace = None
        if os.environ.get("OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS", "0") == "1":
            tool_call_id_namespace = (
                f"{sampling_identity}:{turn_type}:{completed_agent_steps}"
            )
        content, tool_calls = _parse_generated_assistant(
            raw_response_text,
            parser,
            tool_call_id_namespace=tool_call_id_namespace,
        )
        raw_tool_call_count = _count_raw_qwen_tool_calls(raw_response_text)
        parsed_tool_call_count = len(tool_calls)
        # Match the official ClawGym-Agents behaviour of relaying only the
        # first tool call per assistant turn. Off by default; the single-tool
        # eval queues set OPENCLAW_MAX_TOOL_CALLS_PER_TURN=1.
        max_tool_calls_per_turn = int(
            os.environ.get("OPENCLAW_MAX_TOOL_CALLS_PER_TURN", "0") or 0
        )
        if (
            max_tool_calls_per_turn > 0
            and len(tool_calls) > max_tool_calls_per_turn
        ):
            tool_calls = tool_calls[:max_tool_calls_per_turn]
        if not hasattr(self, "_tool_call_repeat_state"):
            self._tool_call_repeat_state = {}
        consecutive_identical_tool_call_turns = _advance_tool_call_repeat_streak(
            self._tool_call_repeat_state,
            session_id,
            tool_calls,
        )

        meta_info = _require_dict(generate_output["meta_info"], "meta_info")
        prompt_input_logprobs = None
        if os.environ.get("OPENCLAW_ROUTING_CONFLICT_DIAGNOSTICS", "0") == "1":
            prompt_input_logprobs = _aligned_prompt_logprobs(meta_info, prompt_ids)
        rollout_routed_experts = None
        if getattr(self.args, "use_rollout_routing_replay", False):
            encoded_routes = meta_info.get("routed_experts")
            if not isinstance(encoded_routes, str) or not encoded_routes:
                raise ValueError(
                    "SGLang response is missing routed_experts while "
                    "use_rollout_routing_replay is enabled"
                )
            route_rows = len(prompt_ids) + len(response_ids) - 1
            expected_shape = (
                route_rows,
                int(self.args.num_layers),
                int(self.args.moe_router_topk),
            )
            decoded_routes = np.frombuffer(
                base64.b64decode(encoded_routes.encode("ascii")),
                dtype=np.int32,
            )
            expected_values = math.prod(expected_shape)
            if decoded_routes.size != expected_values:
                raise ValueError(
                    "SGLang routed_experts shape mismatch: "
                    f"got {decoded_routes.size} int32 values, expected "
                    f"{expected_values} for shape {expected_shape}"
                )
            # ``np.frombuffer`` inherits the immutable bytes backing store.
            # Megatron converts this array with ``torch.from_numpy``; keep the
            # replay tensor explicitly writable to avoid undefined behavior if
            # a downstream transform ever mutates it.
            rollout_routed_experts = decoded_routes.reshape(
                expected_shape
            ).copy()
        prompt_token_count = int(meta_info.get("prompt_tokens") or len(prompt_ids))
        cache_hit_tokens = min(
            prompt_token_count,
            max(0, int(meta_info.get("cached_tokens") or 0)),
        )
        prefill_compute_tokens = max(0, prompt_token_count - cache_hit_tokens)
        worker_id = sglang_response.headers.get("x-slime-worker-id", "")
        affinity_used = sglang_response.headers.get("x-slime-affinity-used") == "1"
        affinity_fallback = (
            sglang_response.headers.get("x-slime-affinity-fallback") == "1"
        )
        session_stats = self._session_sglang_stats.setdefault(
            session_id,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "cache_hit_tokens": 0,
                "prefill_compute_tokens": 0,
                "request_e2e_seconds": 0.0,
                "engine_switches": 0,
                "affinity_used": 0,
                "affinity_fallback": 0,
                "last_worker_id": "",
            },
        )
        previous_worker_id = session_stats["last_worker_id"]
        if worker_id and previous_worker_id and worker_id != previous_worker_id:
            session_stats["engine_switches"] += 1
        session_stats["last_worker_id"] = worker_id or previous_worker_id
        session_stats["requests"] += 1
        session_stats["prompt_tokens"] += prompt_token_count
        session_stats["cache_hit_tokens"] += cache_hit_tokens
        session_stats["prefill_compute_tokens"] += prefill_compute_tokens
        session_stats["request_e2e_seconds"] += sglang_request_e2e_seconds
        session_stats["affinity_used"] += int(affinity_used)
        session_stats["affinity_fallback"] += int(affinity_fallback)
        finish_info = meta_info.get("finish_reason") or {}
        finish_reason = str(finish_info.get("type") or "stop")
        if tool_calls and finish_reason == "stop":
            finish_reason = "tool_calls"
        response_limit_tokens = int(sampling_params.get("max_new_tokens") or 0)
        configured_response_limit_tokens = int(
            os.environ.get("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
        )
        hit_response_limit = (
            finish_reason == "length"
            or (response_limit_tokens > 0 and len(response_ids) >= response_limit_tokens)
        )
        incomplete_tool_call_at_limit = (
            hit_response_limit and _has_incomplete_qwen_tool_call(raw_response_text)
        )
        repetition_diagnostics = _token_repetition_diagnostics(response_ids)
        reasoning_content = ""
        assistant_msg = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
        }
        output = {
            "id": str(meta_info.get("id") or f"chatcmpl-{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _require_string(forward_body["model"], "model"),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_msg,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": int(meta_info.get("prompt_tokens", len(prompt_ids))),
                "completion_tokens": len(response_ids),
                "total_tokens": int(meta_info.get("prompt_tokens", len(prompt_ids)))
                + len(response_ids),
            },
            "metadata": {"weight_version": meta_info.get("weight_version")},
        }

        if turn_type == "compact":
            if not hasattr(self, "_session_compaction_request_counts"):
                self._session_compaction_request_counts = {}
            request_counts = self._session_compaction_request_counts
            request_count = request_counts.get(session_id, 0) + 1
            request_counts[session_id] = request_count
            if not hasattr(self, "_session_pending_compaction_rewrites"):
                self._session_pending_compaction_rewrites = {}
            pending = self._session_pending_compaction_rewrites
            pending[session_id] = pending.get(session_id, 0) + 1
            logger.info(
                "[ClawGym-RL] successful compaction model call for session=%s call_count=%d",
                session_id,
                request_count,
            )
            output["session_id"] = session_id
            return {"response": output}

        # OpenClaw sends full history each time. Keep only the ordered tool
        # results after the most recent assistant message and preserve their IDs.
        tool_results = _collect_new_tool_results(messages)
        observations = [result["content"] for result in tool_results]
        observed_tool_call_ids = [result["tool_call_id"] for result in tool_results]
        if session_id not in self._conversation_logs:
            self._conversation_logs[session_id] = []

        if self._conversation_logs[session_id]:
            prev_step = self._conversation_logs[session_id][-1]
            expected_tool_call_ids = [
                tool_call["id"] for tool_call in prev_step.get("tool_calls", [])
            ]
            prev_step["observations"] = observations
            prev_step["observed_tool_call_ids"] = observed_tool_call_ids
            prev_step["observed_tool_result_count"] = len(tool_results)
            prev_step["tool_result_id_order_matches"] = (
                observed_tool_call_ids == expected_tool_call_ids
            )

        step_record = {
            "step": len(self._conversation_logs[session_id]) + 1,
            "content": content.strip(),
            "reasoning_content": reasoning_content.strip() if reasoning_content else "",
            "tool_calls": [
                {
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"],
                }
                for tool_call in tool_calls
            ],
            "observations": [],
            "raw_tool_call_count": raw_tool_call_count,
            "parsed_tool_call_count": parsed_tool_call_count,
            "returned_tool_call_count": len(tool_calls),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "finish_reason": finish_reason,
            "response_limit_tokens": response_limit_tokens,
            "configured_response_limit_tokens": configured_response_limit_tokens,
            "hit_response_limit": hit_response_limit,
            "incomplete_tool_call_at_limit": incomplete_tool_call_at_limit,
            **repetition_diagnostics,
            "consecutive_identical_tool_call_turns": (
                consecutive_identical_tool_call_turns
            ),
            "sampling_seed": sampling_seed,
            "sampling_identity": sampling_identity,
            "sampling_params": {
                key: sampling_params.get(key)
                for key in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                    "max_new_tokens",
                )
                if key in sampling_params
            },
        }
        self._conversation_logs[session_id].append(step_record)

        if turn_type == "main":
            prev_step_num = 0
            if session_id in self._step_counts:
                prev_step_num = self._step_counts[session_id]
            if prev_step_num > 0 and messages:
                self._flush_pending_record(session_id, messages[-1], tool_results)

            response_text = raw_response_text
            if not response_ids and not response_text.strip() and not tool_calls:
                output["session_id"] = session_id
                return {"response": output}

            step_num = prev_step_num + 1
            self._step_counts[session_id] = step_num
            step_data = {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response_logprobs": response_logprobs,
                "prompt_input_logprobs": prompt_input_logprobs,
                "rollout_routed_experts": rollout_routed_experts,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "messages": messages,
                "template_tools": copy.deepcopy(template_tools),
                "template_kwargs": copy.deepcopy(template_kwargs),
                "reasoning_effort": chat_request.reasoning_effort,
                "tool_calls": tool_calls,
                "raw_tool_call_count": raw_tool_call_count,
                "parsed_tool_call_count": parsed_tool_call_count,
                "returned_tool_call_count": len(tool_calls),
                "finish_reason": finish_reason,
                "response_limit_tokens": response_limit_tokens,
                "configured_response_limit_tokens": configured_response_limit_tokens,
                "hit_response_limit": hit_response_limit,
                "incomplete_tool_call_at_limit": incomplete_tool_call_at_limit,
                **repetition_diagnostics,
                "consecutive_identical_tool_call_turns": (
                    consecutive_identical_tool_call_turns
                ),
                "sampling_seed": sampling_seed,
                "ledger_mode": ledger_mode,
                "ledger_transition": ledger_transition,
                "ledger_transition_detail": ledger_transition_detail,
                "prompt_source": prompt_source,
                "canonical_prompt_tokens": len(canonical_prompt_ids),
                "candidate_prompt_tokens": (
                    len(candidate_prompt_ids) if candidate_prompt_ids is not None else None
                ),
                "assistant_boundary_tokens": len(assistant_boundary_ids),
                "environment_delta_tokens": len(environment_delta_ids),
                "tool_results_pruned": tool_results_pruned,
                "tool_results_truncated": tool_results_truncated,
            }
            if not hasattr(self, "_session_token_ledgers"):
                self._session_token_ledgers = {}
            self._session_token_ledgers[session_id] = SessionTokenLedger(
                messages=copy.deepcopy(messages),
                assistant_message=copy.deepcopy(assistant_msg),
                prompt_ids=list(prompt_ids),
                response_ids=list(response_ids),
                response_logprobs=list(response_logprobs),
                template_tools=copy.deepcopy(template_tools),
                template_kwargs=copy.deepcopy(template_kwargs),
                reasoning_effort=chat_request.reasoning_effort,
                transition=ledger_transition,
                transition_detail=ledger_transition_detail,
                environment_delta_ids=list(environment_delta_ids),
                assistant_boundary_ids=list(assistant_boundary_ids),
            )
            if session_id not in self._pending_step_data:
                self._pending_step_data[session_id] = {}
            self._pending_step_data[session_id][step_num] = step_data
            self._buffer_record(
                session_id,
                step_num,
                messages,
                prompt_text,
                response_text,
                tool_calls,
                {
                    "raw_tool_call_count": raw_tool_call_count,
                    "parsed_tool_call_count": parsed_tool_call_count,
                    "returned_tool_call_count": len(tool_calls),
                    "finish_reason": finish_reason,
                    "response_limit_tokens": response_limit_tokens,
                    "configured_response_limit_tokens": configured_response_limit_tokens,
                    "hit_response_limit": hit_response_limit,
                    "incomplete_tool_call_at_limit": incomplete_tool_call_at_limit,
                    **repetition_diagnostics,
                    "consecutive_identical_tool_call_turns": (
                        consecutive_identical_tool_call_turns
                    ),
                    "sampling_seed": sampling_seed,
                    "sampling_params": {
                        key: sampling_params.get(key)
                        for key in (
                            "temperature",
                            "top_p",
                            "top_k",
                            "min_p",
                            "repetition_penalty",
                            "max_new_tokens",
                        )
                        if key in sampling_params
                    },
                    "tool_call_ids": [tool_call["id"] for tool_call in tool_calls],
                    "ledger_mode": ledger_mode,
                    "ledger_transition": ledger_transition,
                    "ledger_transition_detail": ledger_transition_detail,
                    "prompt_source": prompt_source,
                    "canonical_prompt_tokens": len(canonical_prompt_ids),
                    "candidate_prompt_tokens": (
                        len(candidate_prompt_ids) if candidate_prompt_ids is not None else None
                    ),
                    "assistant_boundary_tokens": len(assistant_boundary_ids),
                    "environment_delta_tokens": len(environment_delta_ids),
                    "tool_results_pruned": tool_results_pruned,
                    "tool_results_truncated": tool_results_truncated,
                    **_optional_recorded_response_ids(response_ids),
                },
            )

        if turn_type != "main":
            if not hasattr(self, "_session_token_ledgers"):
                self._session_token_ledgers = {}
            self._session_token_ledgers[session_id] = SessionTokenLedger(
                messages=copy.deepcopy(messages),
                assistant_message=copy.deepcopy(assistant_msg),
                prompt_ids=list(prompt_ids),
                response_ids=list(response_ids),
                response_logprobs=list(response_logprobs),
                template_tools=copy.deepcopy(template_tools),
                template_kwargs=copy.deepcopy(template_kwargs),
                reasoning_effort=chat_request.reasoning_effort,
                transition=ledger_transition,
                transition_detail=ledger_transition_detail,
                environment_delta_ids=list(environment_delta_ids),
                assistant_boundary_ids=list(assistant_boundary_ids),
            )

        output["session_id"] = session_id
        return {"response": output}

    async def _stream_response(self, result: dict[str, Any]):
        payload = result["response"]
        choice = payload["choices"][0]
        message = choice["message"]
        delta = {"role": "assistant", "content": message["content"]}
        if message["tool_calls"]:
            delta["tool_calls"] = message["tool_calls"]

        chunk_base = {
            "id": payload["id"],
            "object": "chat.completion.chunk",
            "created": payload["created"],
            "model": payload["model"],
            "session_id": payload["session_id"],
        }
        first = {
            **chunk_base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        final = {
            **chunk_base,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
            ],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    def drain_eval_scores(self) -> list[float]:
        with self._eval_scores_lock:
            scores = list(self._eval_scores)
            self._eval_scores.clear()
            return scores

    def reset_eval_scores(self):
        with self._eval_scores_lock:
            self._eval_scores.clear()

    def purge_record_files(self):
        if self._record_file:
            open(self._record_file, "w").close()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._sglang_ready.clear()
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="info"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        readiness_thread = threading.Thread(
            target=self._wait_for_sglang_ready, daemon=True
        )
        readiness_thread.start()

    def wait_until_ready(self, timeout: float):
        if not self._sglang_ready.wait(timeout=timeout):
            raise TimeoutError("SGLang chat endpoint did not become ready in time")

    def _wait_for_sglang_ready(self):
        while True:
            try:
                response = httpx.get(
                    self.sglang_health_url,
                    timeout=5,
                )
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        self._sglang_ready.set()
        logger.info("[ClawGym-RL] policy server ready")

    def stop(self):
        self._sglang_ready.clear()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
