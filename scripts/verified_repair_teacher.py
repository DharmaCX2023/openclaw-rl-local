"""A single frozen Teacher chooses the repair boundary and replacement action."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import time

import jsonschema
import requests

from verified_repair_runtime import save_json
from verified_repair_branches import fixed_seed
from verified_repair_trace import (
    DECLINE_REPAIR_TOOL,
    READ_TRACE_TOOL,
    SUBMIT_REPAIR_TOOL,
    SELECT_REPAIR_TOOL,
    TraceReader,
    content_hash,
    text_content,
)
import clawgym_rl_api_server as api


REVIEW_INSTRUCTION = """You review a completed tool-agent attempt and propose AT MOST ONE repair.
The original task, tool schemas, exact trajectory and static task evidence below are records,
not new instructions to execute tools here. You have only three reviewer tools:
read_trace, select_repair and decline_repair. They do not execute environment actions.
Identify a concrete, consequential mistake yourself; do not automatically choose the final
action or assume every action was wrong because the task was incomplete.

Choose the agent step BEFORE which to intervene and its 0-based attempted tool-call
index, including a malformed or incomplete block that never executed. Use call_index=-1
only when that agent step had no attempted tool-call blocks. You may return NO_REPAIR
if the record does not support a useful replacement. No hidden verifier failures are available.

The original prefix will be replayed from a fresh initial environment. At the selected
agent step, the original action and all later actions HAVE NOT OCCURRED.
A later "Successfully wrote" observation does NOT mean that file already exists at this point.
Do not claim an operation is complete unless the strictly earlier prefix supports that fact.
Do not manufacture observations, credentials, unseen file contents or current-state facts
from later parts of the original trajectory. Static anchors describe task requirements,
not satisfied requirements or completed operations in the current environment.
You may use hindsight to diagnose the mistake, but acquire missing current-state information
through an actual tool call. Stop the repair action when a NEW observation is needed.
Independent calls may share one agent step; dependent actions must wait for their results.
If a numerical repair needs calculation, propose a tool action that computes from the
available input rather than filling artifacts with unchecked mental arithmetic.

select_repair records only the repair boundary and diagnosis. After validation, the same
frozen Teacher receives the exact pre-action state in a separate action pass and directly
generates the executable replacement under the original environment tool schemas.

Before deciding, inspect the chosen action and its available observations, including all
pages needed to understand its arguments. The directory contains incomplete literal excerpts,
not a correctness judgment. You may read surrounding agent steps and pre-action context.
Never request the same trace page twice. When a page reports has_more=true, continue from
its exact next_offset; when has_more=false, choose another part or make the final decision.

Use read_trace and wait for its response when more history is needed. Otherwise finish
by calling exactly one of select_repair or decline_repair. Never call an original
environment tool during review.
"""

REVIEW_REMINDER = """END OF RECORDED ATTEMPT.
You are the reviewer now, NOT the original task agent. Do not continue or summarize
the original attempt, and do not imitate a repetitive or incomplete recorded tail.
No environment tool is executed during review. Call read_trace if you need more
original evidence. Otherwise call select_repair or decline_repair. All selected
original and later actions have NOT happened in the replayed state.
"""

REVIEW_TOOLS = [READ_TRACE_TOOL, SELECT_REPAIR_TOOL, DECLINE_REPAIR_TOOL]
DIRECT_HISTORY_MAX_INPUT_TOKENS = 32768


ACTION_ABANDON_TOOL = {
    "type": "function",
    "function": {
        "name": "abandon_repair_action",
        "description": (
            "Abandon this selected boundary when the exact pre-action state does not "
            "support a useful executable change. This is a review control decision, "
            "not an environment action and never a training target."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "minLength": 1}},
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
}


class SemanticNoopRepair(ValueError):
    """The proposed environment action is equivalent to the recorded action."""


def _one_tag(raw: str, name: str) -> str:
    matches = re.findall(rf"<{name}>(.*?)</{name}>", raw, re.S)
    if len(matches) != 1:
        raise ValueError("expected exactly one " + name)
    return matches[0].strip()


def canonical_tool_call_signature(calls):
    signature = []
    for call in calls:
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        signature.append((
            function["name"],
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ))
    return signature


def parse_proposal(raw, reader, tools, parser, tokenizer, identity):
    if "<no_repair>" in raw:
        if "<replacement_action>" in raw:
            raise ValueError("ambiguous repair and no_repair")
        return {"status": "no_repair", "reason": _one_tag(raw, "no_repair")}
    if raw.count("<replacement_action>") != 1:
        raise ValueError("expected one replacement_action start marker")
    metadata, action = raw.split("<replacement_action>", 1)
    step = int(_one_tag(metadata, "repair_step"))
    call = int(_one_tag(metadata, "repair_call"))
    diagnosis = _one_tag(metadata, "diagnosis")
    if "</replacement_action>" in action:
        action, remainder = action.split("</replacement_action>", 1)
        if remainder.strip():
            raise ValueError("review text follows the final action boundary")
    action = action.strip()
    if step not in reader.steps:
        raise ValueError("Teacher selected an unknown agent step")
    original = reader.steps[step]
    original_calls = original.get("tool_calls", [])
    attempted_count = max(len(original_calls), api._count_raw_qwen_tool_calls(original["response_text"]))
    if (attempted_count and not 1 <= call <= attempted_count) or (not attempted_count and call != 0):
        raise ValueError("selected call is outside the chosen agent step")
    if not action or not diagnosis:
        raise ValueError("empty replacement or diagnosis")
    if any(marker in action for marker in (
        "<tool_response>", "<|im_start|>", "<|im_end|>", "<diagnosis>", "read_trace",
    )):
        raise ValueError("replacement includes diagnostic tool, role or observation markup")
    content, calls = api._parse_generated_assistant(action, parser, identity + ":repair")
    if api._has_incomplete_qwen_tool_call(action) or api._count_raw_qwen_tool_calls(action) != len(calls):
        raise ValueError("incomplete or unparsed replacement tool call")
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for item in calls:
        function = item["function"]
        if function["name"] not in schemas:
            raise ValueError("replacement tool is not in the original environment")
        jsonschema.validate(json.loads(function["arguments"]), schemas[function["name"]])
    # IDs and JSON whitespace are transport metadata. When every recorded block
    # parsed successfully, an identical canonical call sequence cannot be a
    # counterfactual repair and must not be executed or used as a preference.
    if (attempted_count == len(original_calls)
            and canonical_tool_call_signature(calls) == canonical_tool_call_signature(original_calls)):
        raise SemanticNoopRepair("replacement_semantically_identical_to_original_action")
    ids = tokenizer.encode(action, add_special_tokens=False)
    if not ids or len(ids) > 8192:
        raise ValueError("replacement exceeds the original per-step action budget")
    if len(original.get("prompt_ids", [])) + len(ids) >= 65536:
        raise ValueError("complete replacement does not fit the chosen live context")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != action:
        raise ValueError("replacement tokenization is not text-exact")
    return {
        "status": "proposed", "agent_step": step, "call_index": call,
        "original_attempted_tool_blocks": attempted_count,
        "original_parsed_tool_calls": len(original_calls),
        "diagnosis": diagnosis, "text": action, "ids": ids, "tool_calls": calls,
        "content": content, "is_policy_rollout": False,
        "target_tokenization": "replacement-only, excludes all review metadata",
        "accepted_preference": False,
    }


def parse_structured_proposal(arguments, reader, tools, parser, tokenizer, identity):
    expected = SUBMIT_REPAIR_TOOL["function"]["parameters"]
    jsonschema.validate(arguments, expected)
    # Reuse exactly one validation path for step/call range, native tool
    # parsing, schema, context budget and text-exact tokenization.
    escaped = {
        key: str(arguments[key]).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for key in ("diagnosis",)
    }
    raw = (
        f"<repair_step>{arguments['agent_step']}</repair_step>"
        f"<repair_call>{arguments['call_index']}</repair_call>"
        f"<diagnosis>{escaped['diagnosis']}</diagnosis>"
        f"<replacement_action>{arguments['replacement_action']}"
    )
    value = parse_proposal(raw, reader, tools, parser, tokenizer, identity)
    value["diagnosis"] = arguments["diagnosis"]
    value["review_submission_protocol"] = "native_submit_repair_tool_v1"
    return value


def parse_repair_selection(arguments, reader):
    jsonschema.validate(arguments, SELECT_REPAIR_TOOL["function"]["parameters"])
    step = arguments["agent_step"]
    teacher_call_index = arguments["call_index"]
    if step not in reader.steps:
        raise ValueError("Teacher selected an unknown agent step")
    original = reader.steps[step]
    original_calls = original.get("tool_calls", [])
    attempted_count = max(
        len(original_calls), api._count_raw_qwen_tool_calls(original["response_text"])
    )
    if ((attempted_count and not 0 <= teacher_call_index < attempted_count)
            or (not attempted_count and teacher_call_index != -1)):
        raise ValueError("selected call is outside the chosen agent step")
    internal_call_number = teacher_call_index + 1
    return {
        "agent_step": step,
        "call_index": internal_call_number,
        "teacher_call_index": teacher_call_index,
        "diagnosis": arguments["diagnosis"],
        "original_attempted_tool_blocks": attempted_count,
        "original_parsed_tool_calls": len(original_calls),
    }


def native_action_from_required_json(raw, schema):
    """Convert constrained original-tool JSON into exact Qwen tool-call text."""
    value = json.loads(raw)
    jsonschema.validate(value, schema)
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("replacement action must contain one to eight tool calls")
    return "\n".join(
        "<tool_call>\n"
        + json.dumps(
            {"name": item["name"], "arguments": item["parameters"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</tool_call>"
        for item in value
    )


def required_action_decision_from_json(raw, schema):
    """Separate the action-pass control decision from executable native tools."""
    value = json.loads(raw)
    jsonschema.validate(value, schema)
    controls = [item for item in value if item["name"] == "abandon_repair_action"]
    if controls:
        if len(value) != 1:
            raise ValueError("abandon_repair_action must be the only action-pass call")
        return {
            "status": "no_repair",
            "reason": controls[0]["parameters"]["reason"],
        }
    return {
        "status": "action",
        "text": native_action_from_required_json(raw, schema),
    }


def proposal_arguments_from_selection(selection, action):
    """Keep internal selection audit fields outside the strict submission schema."""
    return {
        "agent_step": selection["agent_step"],
        "call_index": selection["call_index"],
        "diagnosis": selection["diagnosis"],
        "replacement_action": action,
    }


def expose_selected_step(reader, selection):
    """Return every exact page of a selected step before final confirmation."""
    pages = []
    for part in ("assistant", "observations"):
        if reader.fully_read(selection["agent_step"], part):
            continue
        offset = 0
        while True:
            page = reader.read(selection["agent_step"], part, offset=offset)
            pages.append(page)
            if not page["has_more"]:
                break
            offset = page["next_offset"]
    return {
        "status": "selection_requires_confirmation",
        "selected_agent_step": selection["agent_step"],
        "teacher_call_index": selection["teacher_call_index"],
        "exact_pages": pages,
        "instruction": "Inspect these exact pages, then select_repair again to confirm or decline_repair.",
    }

def generate_replacement_action(selection, reader, tools, action_parser, proxy, g, request_dir, identity):
    step = reader.steps[selection["agent_step"]]
    action_prompt = (
        "You are producing the executable counterfactual replacement for one recorded "
        "tool-agent step. The messages above are the exact state BEFORE that step; the "
        "original selected action and every later action have not happened.\n\n"
        "Reviewer diagnosis (hindsight only; it is not an environment observation):\n"
        + selection["diagnosis"]
        + "\n\nOriginal action being replaced (quoted for comparison only):\n"
        + step["response_text"]
        + "\n\nGenerate exactly one complete replacement assistant step using the "
        "available environment tools. Preserve any correct independent calls from the "
        "original step. Do not claim success or invent observations. If a new observation "
        "is needed, stop after the tool call that obtains it. If the exact pre-action "
        "state does not support a useful executable change, call abandon_repair_action "
        "instead. Return either environment tool calls or that single control call."
    )
    messages = copy.deepcopy(step["messages"])
    messages.append({"role": "user", "content": action_prompt})
    if any(tool["function"]["name"] == "abandon_repair_action" for tool in tools):
        raise ValueError("environment unexpectedly defines the reserved repair control tool")
    action_tools = list(tools) + [ACTION_ABANDON_TOOL]
    body = {
        "model": g.served_model,
        "messages": messages,
        "tools": action_tools,
        "tool_choice": "required",
    }
    chat, _, sampling = api._build_generate_sampling_params(
        body, proxy.generation_defaults, proxy.tool_call_parser
    )
    schema = json.loads(sampling["json_schema"])
    schema["minItems"] = 1
    schema["maxItems"] = 8
    required_schema = json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )
    sampling["json_schema"] = required_schema
    template_tools = [tool.model_dump() for tool in chat.tools]
    input_ids = api._apply_chat_template_ids(
        proxy.tokenizer,
        messages,
        template_tools=template_tools,
        template_kwargs=dict(chat.chat_template_kwargs or {}),
        reasoning_effort=chat.reasoning_effort,
        add_generation_prompt=True,
    )
    max_new_tokens = min(8192, 65535 - len(input_ids))
    if max_new_tokens <= 0:
        raise ValueError("replacement action Teacher context budget exhausted")
    params = dict(sampling)
    params.update(
        temperature=.7,
        top_p=.8,
        top_k=20,
        max_new_tokens=max_new_tokens,
        sampling_seed=fixed_seed(161803, identity + ":replacement-action"),
    )
    request_path = request_dir / "replacement-action-request.json"
    save_json(request_path, {
        "messages": messages,
        "input_ids": input_ids,
        "sampling_params": params,
        "tools": template_tools,
    })
    started = time.monotonic()
    response = requests.post(
        f"http://127.0.0.1:{g.backend_port}/generate",
        json={"input_ids": input_ids, "sampling_params": params, "return_logprob": True},
        timeout=900,
    )
    response.raise_for_status()
    output = response.json()
    save_json(request_path.with_suffix(".response.json"), output)
    raw, response_ids, _ = api._extract_aligned_generate_response(output)
    generation_audit = {
        "request_file": str(request_path),
        "input_tokens": len(input_ids),
        "response_tokens": len(response_ids),
        "seconds": time.monotonic() - started,
        "control_encoding": "sglang_required_original_tool_or_abandon_json_array",
        "native_action_synthesized_mechanically": True,
    }
    decision = required_action_decision_from_json(raw, schema)
    if decision["status"] == "no_repair":
        return {
            "status": "no_repair",
            "reason": decision["reason"],
            "agent_step": selection["agent_step"],
            "call_index": selection["call_index"],
            "teacher_call_index": selection["teacher_call_index"],
            "diagnosis": selection["diagnosis"],
            "original_attempted_tool_blocks": selection["original_attempted_tool_blocks"],
            "original_parsed_tool_calls": selection["original_parsed_tool_calls"],
            "review_submission_protocol": "two_stage_action_abandon_v1",
            "action_generation": {
                **generation_audit,
                "native_action_synthesized_mechanically": False,
            },
        }
    action = decision["text"]
    proposal = parse_structured_proposal(
        proposal_arguments_from_selection(selection, action),
        reader,
        tools,
        action_parser,
        proxy.tokenizer,
        identity,
    )
    for key in ("teacher_call_index", "original_attempted_tool_blocks",
                "original_parsed_tool_calls"):
        proposal[key] = selection[key]
    proposal["review_submission_protocol"] = "two_stage_select_then_native_action_v1"
    proposal["action_generation"] = generation_audit
    return proposal




def parse_reviewer_calls(raw, parser, identity, required_json_schema):
    """Parse either native Qwen tags or SGLang required-tool JSON.

    SGLang's required tool constraint intentionally emits a JSON array rather
    than model-native tags. This is reviewer control-plane data only; it is
    never used as the replacement action target.
    """
    content, calls = api._parse_generated_assistant(raw, parser, identity)
    if calls:
        return content, calls, "native_qwen_tool_tags"
    value = json.loads(raw)
    schema = json.loads(required_json_schema)
    jsonschema.validate(value, schema)
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError("required-tool output must contain exactly one reviewer call")
    calls = []
    allowed = {tool["function"]["name"] for tool in REVIEW_TOOLS}
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "parameters"}:
            raise ValueError("unexpected required-tool object fields")
        name, arguments = item["name"], item["parameters"]
        if name not in allowed or not isinstance(arguments, dict):
            raise ValueError("unknown reviewer tool or non-object parameters")
        serialized = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(
            f"{identity}:{index}:{name}:{serialized}".encode()).hexdigest()[:24]
        calls.append({"id": "call" + digest, "type": "function", "index": index,
                      "function": {"name": name, "arguments": serialized}})
    return "", calls, "sglang_required_tool_json_array"
def read_trace_page(reader, arguments):
    """Return one mechanical page without reinjecting an already-read page."""
    offset = arguments.get("offset", 0)
    for previous in reversed(reader.access_log):
        if (previous["agent_step"] == arguments["step"]
                and previous["part"] == arguments["part"]
                and previous["offset"] == offset):
            return {
                "error": "trace_page_already_read",
                "agent_step": arguments["step"], "part": arguments["part"],
                "offset": offset, "end_offset": previous["end_offset"],
                "has_more": previous["has_more"],
                "next_offset": previous["next_offset"],
                "instruction": ("continue from next_offset" if previous["has_more"]
                                else "choose another part or submit the final decision"),
                "no_text_reinjected": True,
            }, True
    return reader.read(**arguments), False




def review_trajectory(source, anchors, proxy, g, destination):
    destination = Path(destination)
    reader = TraceReader(source["steps"])
    initial = reader.steps[1]
    original_tools = initial["tools"]
    identity = source["session_id"]
    fingerprint = content_hash({
        "steps": source["steps"], "anchors": anchors,
        "instruction": REVIEW_INSTRUCTION, "reminder": REVIEW_REMINDER,
        "action_protocol": "two-stage-original-tools-v2-action-decline-noop",
        "action_control_tool": ACTION_ABANDON_TOOL,
        "review_tools": REVIEW_TOOLS, "model": str(g.model),
    })
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_sha256"] != fingerprint:
            raise ValueError("cached Teacher proposal belongs to a different trace/protocol")
        return previous
    body = {"model": g.served_model, "messages": [], "tools": REVIEW_TOOLS,
            "tool_choice": "required"}
    chat, parser, sampling = api._build_generate_sampling_params(
        body, proxy.generation_defaults, proxy.tool_call_parser,
    )
    required_json_schema = sampling.get("json_schema")
    if not required_json_schema:
        raise RuntimeError("required reviewer tools did not produce a JSON schema constraint")
    # SGLang represents a required tool call as a JSON array. Its generated
    # schema has no maxItems constraint, which permits a model to emit hundreds
    # of repeated read_trace calls in one turn. Reviewer interaction is
    # intentionally sequential: one read, observe its result, then decide what
    # to inspect next. Constrain both decoding and the defensive parser to one
    # call per request.
    required_schema_object = json.loads(required_json_schema)
    required_schema_object["minItems"] = 1
    required_schema_object["maxItems"] = 1
    required_json_schema = json.dumps(
        required_schema_object, ensure_ascii=False, separators=(",", ":"))
    sampling["json_schema"] = required_json_schema
    _, _, decision_sampling = api._build_generate_sampling_params(
        {"model": g.served_model, "messages": [],
         "tools": [SELECT_REPAIR_TOOL, DECLINE_REPAIR_TOOL], "tool_choice": "required"},
        proxy.generation_defaults, proxy.tool_call_parser,
    )
    decision_schema_object = json.loads(decision_sampling["json_schema"])
    decision_schema_object["minItems"] = 1
    decision_schema_object["maxItems"] = 1
    decision_json_schema = json.dumps(
        decision_schema_object, ensure_ascii=False, separators=(",", ":"))


    _, action_parser, _ = api._build_generate_sampling_params(
        {"model": g.served_model, "messages": initial["messages"], "tools": original_tools},
        proxy.generation_defaults, proxy.tool_call_parser,
    )
    template_tools = [tool.model_dump() for tool in chat.tools]
    template_kwargs = dict(chat.chat_template_kwargs or {})
    record = {
        "protocol": "teacher-two-stage-action-repair-v5", "session_id": identity,
        "input_sha256": fingerprint, "policy": str(g.model), "started_at": time.time(),
        "requests": [], "status": "reviewing", "verifier_feedback_provided": False,
        "task_reward_provided": False, "trajectory_local_read_only": True,
    }
    header = (
        "Original agent initial context (quoted record):\n" + text_content(initial["messages"])
        + "\n\nNative environment tool schemas (not callable during this review):\n"
        + text_content(original_tools)
        + "\n\nStatic task evidence anchors:\n" + text_content(anchors) + "\n\n"
    )

    def render(messages):
        return api._apply_chat_template_ids(
            proxy.tokenizer, messages, template_tools=template_tools,
            template_kwargs=template_kwargs, reasoning_effort=chat.reasoning_effort,
            add_generation_prompt=True,
        )

    messages = [
        {"role": "system", "content": REVIEW_INSTRUCTION},
        {"role": "user", "content": header + reader.full_history() + "\n\n" + REVIEW_REMINDER},
    ]
    if (len(render(messages)) > DIRECT_HISTORY_MAX_INPUT_TOKENS
            or any(len(step["response_text"]) > 12000 for step in reader.steps.values())):
        messages[1]["content"] = header + reader.directory() + "\n\n" + REVIEW_REMINDER
        record["history_mode"] = "directory_with_read_trace"
    else:
        record["history_mode"] = "full_history"
    started = time.monotonic()
    record["format_errors"] = []
    consecutive_duplicate_reads = 0
    force_final_decision = False
    try:
        # This is a bounded review budget, never a Student continuation limit.
        for request_index in range(25):
            ids = render(messages)
            if len(ids) + 8192 >= 65536:
                record.update(status="no_proposal", reason="teacher_context_budget_exhausted")
                break
            params = dict(sampling)
            active_json_schema = decision_json_schema if force_final_decision else required_json_schema
            params["json_schema"] = active_json_schema
            params.update(temperature=.7, top_p=.8, top_k=20, max_new_tokens=8192,
                          sampling_seed=fixed_seed(161803, f"{identity}:review:{request_index}"))
            request_path = destination.parent / identity / f"request-{request_index:02d}.json"
            save_json(request_path, {
                "messages": copy.deepcopy(messages), "input_ids": ids,
                "sampling_params": params, "tools": template_tools,
            })
            call_started = time.monotonic()
            response = requests.post(
                f"http://127.0.0.1:{g.backend_port}/generate",
                json={"input_ids": ids, "sampling_params": params, "return_logprob": True},
                timeout=900,
            )
            response.raise_for_status()
            output = response.json()
            save_json(request_path.with_suffix(".response.json"), output)
            raw, response_ids, _ = api._extract_aligned_generate_response(output)
            record["requests"].append({
                "index": request_index, "request_file": str(request_path),
                "input_tokens": len(ids), "response_tokens": len(response_ids),
                "seconds": time.monotonic() - call_started,
            })
            try:
                content, calls, call_encoding = parse_reviewer_calls(
                    raw, parser, identity + f":review:{request_index}", active_json_schema)
                finish_type = output["meta_info"].get("finish_reason", {}).get("type")
                if finish_type == "length":
                    if call_encoding != "sglang_required_tool_json_array":
                        raise ValueError("length-limited Teacher review call")
                    # Parsing plus schema validation proves the complete one-call
                    # object ended before the limit; any tail is JSON whitespace.
                    record["requests"][-1]["accepted_complete_json_at_length"] = True
                if not calls or api._has_incomplete_qwen_tool_call(raw):
                    raise ValueError("Teacher returned no complete reviewer tool call")
                names = [call["function"]["name"] for call in calls]
                decisions = [name for name in names if name in {"select_repair", "decline_repair"}]
                if decisions:
                    if len(calls) != 1:
                        raise ValueError("a final reviewer decision must be the only tool call")
                    arguments = json.loads(calls[0]["function"]["arguments"])
                    if names[0] == "decline_repair":
                        jsonschema.validate(arguments, DECLINE_REPAIR_TOOL["function"]["parameters"])
                        proposal = {"status": "no_repair", "reason": arguments["reason"],
                                    "review_submission_protocol": "native_decline_repair_tool_v1"}
                    else:
                        selection = parse_repair_selection(arguments, reader)
                        if record["history_mode"] == "directory_with_read_trace":
                            missing = [part for part in ("assistant", "observations")
                                       if not reader.fully_read(selection["agent_step"], part)]
                            if missing:
                                messages.append({
                                    "role": "assistant", "content": content, "tool_calls": calls,
                                })
                                exposure = expose_selected_step(reader, selection)
                                messages.append({
                                    "role": "tool", "tool_call_id": calls[0]["id"],
                                    "name": "select_repair", "content": text_content(exposure),
                                })
                                record.setdefault("selection_evidence_exposures", []).append({
                                    "request": request_index, "agent_step": selection["agent_step"],
                                    "parts": missing, "page_count": len(exposure["exact_pages"]),
                                })
                                record["requests"][-1]["selection_evidence_exposed"] = True
                                consecutive_duplicate_reads = 0
                                # The selecting response preceded this exact evidence.
                                # Require the same Teacher to confirm or decline on a
                                # subsequent decision-only request before action generation.
                                force_final_decision = True
                                continue
                        proposal = generate_replacement_action(
                            selection, reader, original_tools, action_parser, proxy, g,
                            destination.parent / identity, identity,
                        )
                    record.update(proposal, raw_teacher_response=raw)
                    record["review_call_encoding"] = call_encoding
                    break
                if any(name != "read_trace" for name in names):
                    raise ValueError("unknown reviewer tool")
                if request_index == 24:
                    record.update(status="no_proposal", reason="teacher_read_request_budget_exhausted")
                    break
                # Validate every call before appending a tool-call turn or executing
                # any read; malformed review output is never an environment action.
                arguments_list = []
                for call in calls:
                    fn = call["function"]
                    if fn["name"] != "read_trace":
                        raise ValueError("Teacher attempted an environment tool during review")
                    arguments = json.loads(fn["arguments"])
                    jsonschema.validate(arguments, READ_TRACE_TOOL["function"]["parameters"])
                    arguments_list.append(arguments)
                messages.append({"role": "assistant", "content": content, "tool_calls": calls})
                record["requests"][-1]["review_call_encoding"] = call_encoding
                duplicate_in_request = False
                for call, arguments in zip(calls, arguments_list):
                    try:
                        value, duplicate = read_trace_page(reader, arguments)
                        duplicate_in_request = duplicate_in_request or duplicate
                    except ValueError as exc:
                        value = {"error": str(exc), "no_file_read_performed": True}
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "name": "read_trace", "content": text_content(value)})
                consecutive_duplicate_reads = consecutive_duplicate_reads + 1 if duplicate_in_request else 0
                if duplicate_in_request:
                    record["duplicate_trace_reads"] = record.get("duplicate_trace_reads", 0) + 1
                    record["requests"][-1]["duplicate_trace_read"] = True
                if consecutive_duplicate_reads >= 3 and not force_final_decision:
                    force_final_decision = True
                    record["forced_final_decision_after_duplicate_reads"] = request_index
                    messages.append({
                        "role": "user", "content":
                        "The read-only review budget is now closed. Call exactly one of "
                        "select_repair or decline_repair using the evidence already inspected."
                    })
            except SemanticNoopRepair as exc:
                record.update(
                    status="no_proposal",
                    reason=str(exc),
                    raw_teacher_response=raw,
                    review_call_encoding=call_encoding,
                )
                break
            except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
                reason = type(exc).__name__ + ":" + str(exc)[:400]
                record["format_errors"].append({"request": request_index, "reason": reason})
                if len(record["format_errors"]) > 2:
                    record.update(status="invalid_proposal", reason=reason, raw_teacher_response=raw)
                    break
                # Same Teacher, no new task feedback, at most two format retries.
                # Preserve raw output; do not supply a chosen step or repair.
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Review protocol validation failed: " + reason
                     + "\nNothing in the previous reply was executed.\n" + REVIEW_REMINDER},
                ])
        else:
            record.update(status="no_proposal", reason="teacher_request_budget_exhausted")
    except requests.RequestException as exc:
        record.update(status="infrastructure_error", reason=type(exc).__name__)
    except (ValueError, KeyError, TypeError, jsonschema.ValidationError) as exc:
        record.update(status="invalid_proposal", reason=type(exc).__name__ + ":" + str(exc)[:600])
    record.update(seconds=time.monotonic() - started, finished_at=time.time(),
                  trace_reads=reader.access_log)
    save_json(destination, record)
    return record
