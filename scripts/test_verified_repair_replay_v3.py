from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from verified_repair_sampler_patch import OLD, NEW, SOURCE_SHA256, patched_source
from verified_repair_trace import TraceReader


def test_overlay_is_pinned_and_leaves_shared_source_unchanged():
    import hashlib
    source = Path(__file__).resolve().parents[1] / (
        ".venv-openclaw/lib/python3.12/site-packages/sglang/srt/layers/sampler.py"
    )
    before = source.read_bytes()
    patched = patched_source(before)
    assert hashlib.sha256(before).hexdigest() == SOURCE_SHA256
    assert NEW in patched and OLD not in patched
    assert source.read_bytes() == before
    with pytest.raises(RuntimeError, match="re-audit"):
        patched_source(before + b"\n")


def test_midpoint_uniform_excludes_both_uint32_endpoints():
    import math
    for hashed in (0, 1, 2**32 - 2, 2**32 - 1):
        uniform = (hashed + 0.5) / float(2**32)
        assert 0 < uniform < 1
        assert math.isfinite(-math.log(-math.log(uniform)))


def _reader():
    return TraceReader([{
        "agent_step": 1, "response_text": "read first",
        "messages": [{"role": "user", "content": "original query"}],
        "tool_calls": [{"id": "a", "function": {"name": "read", "arguments": '{"path":"input/a"}'}}],
        "observations": ["abc" * 10000],
    }])


def test_trace_pages_preserve_exact_original_and_declare_missing_tail():
    reader = _reader()
    result = reader.read(1, "observations", max_chars=12000)
    assert result["has_more"] and result["next_offset"] == 12000
    text = result["text"]
    while result["has_more"]:
        result = reader.read(1, "observations", offset=result["next_offset"])
        text += result["text"]
    import json
    assert json.loads(text) == ["abc" * 10000]
    assert "has_more=true" in reader.directory()


@pytest.mark.parametrize("kwargs", [
    {"step": 0, "part": "assistant"},
    {"step": 1, "part": "../../.env"},
    {"step": 1, "part": "verifier"},
    {"step": 1, "part": "assistant", "offset": -1},
    {"step": 1, "part": "assistant", "max_chars": 12001},
    {"step": True, "part": "assistant"},
])
def test_trace_reader_has_no_arbitrary_filesystem_or_verifier_access(kwargs):
    with pytest.raises(ValueError):
        _reader().read(**kwargs)


def test_trace_index_does_not_infer_success_from_tool_message():
    reader = _reader()
    reader.steps[1]["observations"] = ["Successfully wrote 10 bytes"]
    assert "not proof" in reader.directory()
    assert "criterion_pass" not in reader.directory()


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def _native_tools():
    return [{"type": "function", "function": {
        "name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                     "required": ["path"], "additionalProperties": False},
    }}]


def _proposal(raw):
    from verified_repair_teacher import parse_proposal
    from verified_repair_replay_proxy import api
    tools = _native_tools()
    _, parser, _ = api._build_generate_sampling_params(
        {"model": "test", "messages": [{"role": "user", "content": "query"}], "tools": tools},
        {}, "qwen25",
    )
    return parse_proposal(raw, _reader(), tools, parser, CharacterTokenizer(), "unit")


def test_teacher_metadata_is_separated_from_native_action_and_selects_explicit_step():
    raw = ('<repair_step>1</repair_step><repair_call>1</repair_call>'
           '<diagnosis>Read the required input, not an invented result.</diagnosis>'
           '<replacement_action><tool_call>\n{"name":"read","arguments":{"path":"input/b"}}'
           '\n</tool_call></replacement_action>')
    value = _proposal(raw)
    assert value["agent_step"] == 1 and value["call_index"] == 1
    assert value["tool_calls"][0]["function"]["name"] == "read"
    assert "diagnosis" not in CharacterTokenizer().decode(value["ids"])
    assert not value["is_policy_rollout"]


def test_final_native_action_can_end_at_eos_without_an_extra_wrapper():
    raw = ('<repair_step>1</repair_step><repair_call>1</repair_call>'
           '<diagnosis>Read the correct path.</diagnosis><replacement_action>\n'
           '<tool_call>\n{"name":"read","arguments":{"path":"input/b"}}\n</tool_call>')
    result = _proposal(raw)
    assert result["status"] == "proposed"
    assert result["text"].endswith("</tool_call>")
    assert "<replacement_action>" not in CharacterTokenizer().decode(result["ids"])
    with pytest.raises(ValueError):
        _proposal(raw + "</replacement_action>Additional diagnosis must not be action text.")
    with pytest.raises(ValueError):
        _proposal(raw[:-len("</tool_call>")])


def test_teacher_can_decline_without_manufacturing_preference():
    assert _proposal("<no_repair>No supported mistake found.</no_repair>")["status"] == "no_repair"


def test_semantically_identical_replacement_is_rejected_before_replay():
    from verified_repair_teacher import SemanticNoopRepair
    raw = (
        "<repair_step>1</repair_step><repair_call>1</repair_call>"
        "<diagnosis>The original call allegedly needs repair.</diagnosis>"
        "<replacement_action><tool_call>\n"
        '{"name":"read","arguments":{"path": "input/a"}}'
        "\n</tool_call></replacement_action>"
    )
    with pytest.raises(SemanticNoopRepair, match="semantically_identical"):
        _proposal(raw)


def test_teacher_can_select_an_incomplete_attempt_that_never_executed():
    from verified_repair_teacher import parse_proposal
    from verified_repair_replay_proxy import api
    reader = _reader()
    reader.steps[1].update(
        response_text='<tool_call>\n{"name":"read","arguments":{"path":"input/',
        tool_calls=[], observations=[])
    tools = _native_tools()
    _, parser, _ = api._build_generate_sampling_params(_body(), {}, "qwen25")
    raw = ('<repair_step>1</repair_step><repair_call>1</repair_call>'
           '<diagnosis>The attempted read was incomplete and did not execute.</diagnosis>'
           '<replacement_action><tool_call>\n{"name":"read","arguments":{"path":"input/b"}}'
           '\n</tool_call></replacement_action>')
    value = parse_proposal(raw, reader, tools, parser, CharacterTokenizer(), "unit")
    assert value["call_index"] == 1
    assert value["original_attempted_tool_blocks"] == 1
    assert value["original_parsed_tool_calls"] == 0
    assert "attempted_tool_blocks=1, parsed_tool_calls=0" in reader.directory()


@pytest.mark.parametrize("step,call", [(2,1), (1,2), (1,0)])
def test_teacher_cannot_select_nonexistent_step_or_call(step, call):
    with pytest.raises(ValueError):
        _proposal(f"<repair_step>{step}</repair_step><repair_call>{call}</repair_call>"
                  "<diagnosis>test</diagnosis><replacement_action>done</replacement_action>")


def test_tool_metadata_changes_do_not_trigger_automatic_replay_failure():
    from verified_repair_replay_proxy import compare_observations, compare_file_state
    assert compare_observations(["pid 123"], ["pid 456"])["status"] == "changed_requires_review"
    state = {"status": "ok", "files": {"output/a": {"sha256": "a"}}}
    assert compare_file_state(state, state)["status"] == "match"
    assert not compare_file_state(state, state)["process_and_external_state_proven"]
    other = {"status": "ok", "files": {"output/a": {"sha256": "b"}}}
    assert compare_file_state(state, other)["status"] == "material_file_divergence"


def test_paged_selection_requires_complete_read_not_an_excerpt():
    reader = _reader()
    reader.read(1, "observations", offset=10)
    assert not reader.fully_read(1, "observations")
    page = reader.read(1, "observations")
    while page["has_more"]:
        page = reader.read(1, "observations", page["next_offset"])
    assert reader.fully_read(1, "observations")
    assert not reader.fully_read(1, "assistant")

def test_repeated_trace_page_is_not_reinjected():
    from verified_repair_teacher import read_trace_page
    reader = _reader()
    arguments = {"step": 1, "part": "observations", "offset": 0, "max_chars": 4}
    first, duplicate = read_trace_page(reader, arguments)
    assert not duplicate and len(first["text"]) == 4
    second, duplicate = read_trace_page(reader, arguments)
    assert duplicate and second["error"] == "trace_page_already_read"
    assert second["no_text_reinjected"]
    assert len(reader.access_log) == 1



@pytest.fixture
def replay_proxy(tmp_path, monkeypatch):
    import queue
    import threading
    from types import SimpleNamespace
    from verified_repair_replay_proxy import ReplayProxy
    monkeypatch.setenv("OPENCLAW_RECORD_ENABLED", "0")
    state = SimpleNamespace(output=tmp_path, fingerprint=lambda runtime: {"status": "ok", "files": {}})
    model = Path(__file__).resolve().parents[1] / "outputs/clawgym-8b-sft-formal-20260902-165551/final_model"
    return ReplayProxy(SimpleNamespace(hf_checkpoint=str(model), sglang_router_ip="127.0.0.1",
                       sglang_router_port=9), queue.Queue(), threading.Event(), remote_state=state)


def _wire_action():
    text = '<tool_call>\n{"name":"read","arguments":{"path":"input/a"}}\n</tool_call>'
    call = {"id": "source-call-1", "type": "function",
            "function": {"name": "read", "arguments": '{"path":"input/a"}'}}
    wire = {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]},
                          "finish_reason": "tool_calls"}], "usage": {}}
    return text, wire


def _body():
    return {"model": "main__unit", "messages": [{"role": "user", "content": "Inspect input/a"}],
            "tools": _native_tools()}


def test_forced_prefix_preserves_raw_tokens_without_trainable_fake_logprobs(replay_proxy):
    import copy
    proxy = replay_proxy
    sid = "unit"
    body = _body()
    raw, wire = _wire_action()
    ids = proxy.tokenizer.encode(raw, add_special_tokens=False)
    event = {"kind": "main"}
    proxy._forced_response(sid, body, event, {"wire_response": wire, "ids": ids, "text": raw})
    ledger = proxy._session_token_ledgers[sid]
    assert not proxy._pending_step_data.get(sid)
    assert not ledger.response_logprobs
    assert ledger.response_ids == ids
    next_body = copy.deepcopy(body)
    next_body["messages"] += [wire["choices"][0]["message"],
                             {"role": "tool", "tool_call_id": "source-call-1", "content": "Actual text"}]
    next_state = proxy._prompt_state(sid, next_body, "main")
    assert next_state["transition"] == "append_only"
    prefix = ledger.prompt_ids + ids
    assert next_state["prompt_ids"][:len(prefix)] == prefix


def test_compaction_replay_is_auxiliary_and_not_an_agent_step(replay_proxy):
    proxy = replay_proxy
    raw, wire = _wire_action()
    event = {"kind": "compact"}
    proxy._forced_response("unit", _body(), event, {"wire_response": wire, "ids": [], "text": "summary"})
    assert proxy._step_counts.get("unit", 0) == 0
    assert proxy._session_compaction_request_counts["unit"] == 1
    assert not proxy._pending_step_data.get("unit")
    assert "unit" not in proxy._session_token_ledgers


def test_replay_target_never_calls_student_and_charges_original_prefix_time(replay_proxy, monkeypatch):
    import asyncio
    import copy
    import time
    from types import SimpleNamespace
    from verified_repair_replay_proxy import api
    proxy = replay_proxy
    raw, wire = _wire_action()
    body = _body()
    ids = proxy.tokenizer.encode(raw, add_special_tokens=False)
    source_event = {"kind": "main", "agent_step": 1, "body": body,
                    "file_state": {"status": "ok", "files": {}},
                    "original_elapsed_before_request": 210}
    branch = {"source_events": [source_event], "source_steps": [], "target_step": 1,
              "original_budget_seconds": 1000,
              "replacement": {"wire_response": wire, "ids": ids, "text": raw}}
    proxy.register_runtime(SimpleNamespace(session_id="unit"), branch=branch)
    proxy.begin_execution("unit")

    async def forbidden(*args, **kwargs):
        pytest.fail("replay prefix unexpectedly invoked model inference")

    monkeypatch.setattr(api.ClawGymRLAPIServer, "_handle_request", forbidden)
    output = asyncio.run(proxy._handle_request(copy.deepcopy(body), "unit", "main"))
    assert output["response"]["choices"][0]["message"]["tool_calls"]
    assert branch["remaining_policy_seconds"] == 790
    assert 788 < proxy.deadlines["unit"]-time.monotonic() <= 790
    assert branch["replacement_done"]
    assert not proxy._pending_step_data.get("unit")
    assert proxy._step_counts["unit"] == 1


def test_expired_original_budget_blocks_new_inference(replay_proxy, monkeypatch):
    import asyncio
    from verified_repair_replay_proxy import api
    proxy = replay_proxy
    proxy.deadlines["unit"] = 0

    async def forbidden(*args, **kwargs):
        pytest.fail("expired task invoked inference")

    monkeypatch.setattr(api.ClawGymRLAPIServer, "_handle_request", forbidden)
    output = asyncio.run(proxy._handle_request(_body(), "unit", "main"))
    assert output["response"]["metadata"]["clawgym_budget_termination"]["reason"] == "original_task_wall_budget"


def test_gateway_budget_extension_applies_only_to_replay_transport(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import run_verified_repair_replay_pilot as runner
    config = {"agents": {"defaults": {"timeoutSeconds": 1000}}}
    path = tmp_path / "openclaw.json"

    def original(runtime, port):
        path.write_text(json.dumps(config))

    monkeypatch.setattr(runner, "_ORIGINAL_GATEWAY_CONFIG", original)
    runtime = SimpleNamespace(config_dir=tmp_path)
    runner._RUNTIME_OPTIONS.replay = False
    runner.gateway_config(runtime, 1)
    assert json.loads(path.read_text())["agents"]["defaults"]["timeoutSeconds"] == 1000
    try:
        runner._RUNTIME_OPTIONS.replay = True
        runner.gateway_config(runtime, 1)
        assert json.loads(path.read_text())["agents"]["defaults"]["timeoutSeconds"] == 2400
    finally:
        runner._RUNTIME_OPTIONS.replay = False


def test_runtime_metadata_exception_does_not_erase_setup_state_or_unknown_fields():
    import json
    from verified_repair_state_review import canonical_runtime_metadata
    old = {"version": 1, "bootstrapSeededAt": "2026-09-07T11:00:00.001Z"}
    new = dict(old, bootstrapSeededAt="2026-09-07T12:00:00.002Z")
    assert canonical_runtime_metadata(json.dumps(old)) == canonical_runtime_metadata(json.dumps(new))
    setup = dict(new, setupCompletedAt="2026-09-07T12:00:01.002Z")
    assert canonical_runtime_metadata(json.dumps(old)) != canonical_runtime_metadata(json.dumps(setup))
    for invalid in (dict(new, task_result=12), dict(new, version=2),
                    dict(new, bootstrapSeededAt="arbitrary task text")):
        with pytest.raises(ValueError):
            canonical_runtime_metadata(json.dumps(invalid))


def test_runtime_review_requires_pre_action_hash_and_exact_observations(tmp_path):
    import hashlib
    import io
    import json
    import tarfile
    from verified_repair_state_review import STATE_PATH, review_runtime_drift
    sides = []
    for index in range(2):
        value = json.dumps({"version": 1, "bootstrapSeededAt": f"2026-09-07T11:00:0{index}.001Z"}).encode()
        path = tmp_path / f"{index}.tar.gz"
        with tarfile.open(path, "w:gz") as handle:
            info = tarfile.TarInfo("./" + STATE_PATH)
            info.size = len(value)
            handle.addfile(info, io.BytesIO(value))
        sides.append({"archive": str(path), "file": {
            "type": "file", "mode": 420, "sha256": hashlib.sha256(value).hexdigest()}})
    source = {"steps": [{"file_state": {"files": {STATE_PATH: sides[0]["file"]}}}],
              "terminal_snapshot": {"archive": sides[0]["archive"]}}
    branch = {"target_step": 1, "replacement_done": True, "observation_differences": [],
              "pre_action_file_comparison": {"status": "material_file_divergence", "changed_paths": [STATE_PATH]},
              "actual_pre_action_file_state": {"files": {STATE_PATH: sides[1]["file"]}}}
    replay = {"status": "completed", "replay": branch,
              "terminal_snapshot": {"archive": sides[1]["archive"]}}
    assert review_runtime_drift(source, replay)["eligible"]
    branch["observation_differences"] = [{"agent_step": 1}]
    assert not review_runtime_drift(source, replay)["eligible"]
    branch["observation_differences"] = []
    branch["actual_pre_action_file_state"]["files"][STATE_PATH]["sha256"] = "not-proven"
    assert not review_runtime_drift(source, replay)["eligible"]


@pytest.mark.parametrize("invalid_attempts,expected_calls,expected_status", [
    (1, 2, "no_repair"), (99, 3, "invalid_proposal"),
])
def test_teacher_format_feedback_is_bounded_and_has_no_hidden_verifier_feedback(
    replay_proxy, tmp_path, monkeypatch, invalid_attempts, expected_calls, expected_status,
):
    from types import SimpleNamespace
    import verified_repair_teacher as teacher
    row = _reader().steps[1]
    row.update(tools=_native_tools(), observations=["real input text"])
    source = {"session_id": "review-unit", "steps": [row],
              "verifier": {"reward": 0, "private": "HIDDEN_VERIFIER_PRIVATE_SENTINEL"}}
    sent = []

    def post(url, *, json, timeout):
        sent.append(json)
        raw = ("The original task is done." if len(sent) <= invalid_attempts
               else '<tool_call>\n{"name":"decline_repair","arguments":'
                    '{"reason":"No justified correction found."}}\n</tool_call>')
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "diagnostic_raw": raw, "meta_info": {"finish_reason": {"type": "stop"}}})

    monkeypatch.setattr(teacher.requests, "post", post)
    monkeypatch.setattr(teacher.api, "_extract_aligned_generate_response",
                        lambda out: (out["diagnostic_raw"],
                                     replay_proxy.tokenizer.encode(out["diagnostic_raw"]), []))
    g = SimpleNamespace(model="frozen-test-model", served_model="test", backend_port=9)
    record = teacher.review_trajectory(source, ["Static task requirement"], replay_proxy, g,
                                       tmp_path/"teacher.json")
    assert record["status"] == expected_status
    assert len(sent) == expected_calls
    assert len(record["format_errors"]) == min(invalid_attempts, 3)
    assert record["trace_reads"] == []
    for payload in sent:
        text = replay_proxy.tokenizer.decode(payload["input_ids"])
        assert "HIDDEN_VERIFIER_PRIVATE_SENTINEL" not in text
        assert teacher.REVIEW_REMINDER in text
        assert payload["sampling_params"]["temperature"] == .7
        assert payload["sampling_params"]["json_schema"] is not None
    assert "Nothing in the previous reply was executed." in replay_proxy.tokenizer.decode(sent[1]["input_ids"])


def test_native_submit_repair_tool_keeps_only_embedded_action_as_target():
    from verified_repair_teacher import parse_structured_proposal
    from verified_repair_replay_proxy import api
    tools = _native_tools()
    _, parser, _ = api._build_generate_sampling_params(_body(), {}, "qwen25")
    action = '<tool_call>\n{"name":"read","arguments":{"path":"input/b"}}\n</tool_call>'
    value = parse_structured_proposal({
        "agent_step": 1, "call_index": 1,
        "diagnosis": "The original path was unsupported <and> must be observed first.",
        "replacement_action": action,
    }, _reader(), tools, parser, CharacterTokenizer(), "unit")
    assert value["text"] == action
    assert CharacterTokenizer().decode(value["ids"]) == action
    assert "diagnosis" not in value["text"]
    assert "submit_repair" not in value["text"]
    assert value["review_submission_protocol"] == "native_submit_repair_tool_v1"



def test_two_stage_repair_selection_validates_exact_step_and_call_range():
    from verified_repair_teacher import parse_repair_selection

    selection = parse_repair_selection({
        "agent_step": 1,
        "call_index": 0,
        "diagnosis": "The recorded action read the wrong input path.",
    }, _reader())
    assert selection["agent_step"] == 1
    assert selection["call_index"] == 1
    assert selection["teacher_call_index"] == 0
    assert selection["original_attempted_tool_blocks"] == 1
    for invalid in (
        {"agent_step": 2, "call_index": 0, "diagnosis": "bad step"},
        {"agent_step": 1, "call_index": -1, "diagnosis": "bad call"},
        {"agent_step": 1, "call_index": 1, "diagnosis": "bad call"},
    ):
        with pytest.raises(ValueError):
            parse_repair_selection(invalid, _reader())


def test_internal_selection_audit_stays_outside_strict_submission_schema():
    import jsonschema
    from verified_repair_teacher import proposal_arguments_from_selection
    from verified_repair_trace import SUBMIT_REPAIR_TOOL

    selection = parse_selection = {
        "agent_step": 1, "call_index": 1, "teacher_call_index": 0,
        "diagnosis": "Use the observed input.",
        "original_attempted_tool_blocks": 1, "original_parsed_tool_calls": 1,
    }
    action = '<tool_call>\n{"name":"read","arguments":{"path":"input/b"}}\n</tool_call>'
    arguments = proposal_arguments_from_selection(selection, action)
    assert set(arguments) == {"agent_step", "call_index", "diagnosis", "replacement_action"}
    assert parse_selection["teacher_call_index"] == 0
    jsonschema.validate(arguments, SUBMIT_REPAIR_TOOL["function"]["parameters"])


def test_selected_step_exposure_returns_complete_literal_pages():
    from verified_repair_teacher import expose_selected_step
    from verified_repair_trace import text_content

    reader = _reader()
    selection = {
        "agent_step": 1, "call_index": 1, "teacher_call_index": 0,
        "diagnosis": "Inspect the exact selected step.",
    }
    exposure = expose_selected_step(reader, selection)
    assert exposure["status"] == "selection_requires_confirmation"
    assert reader.fully_read(1, "assistant")
    assert reader.fully_read(1, "observations")
    by_part = {}
    for page in exposure["exact_pages"]:
        by_part.setdefault(page["part"], "")
        by_part[page["part"]] += page["text"]
    assert by_part["assistant"] == reader.steps[1]["response_text"]


def test_directory_selection_is_confirmed_after_exact_step_exposure(
    replay_proxy, tmp_path, monkeypatch,
):
    from types import SimpleNamespace
    import verified_repair_teacher as teacher

    row = _reader().steps[1]
    row["tools"] = _native_tools()
    source = {"session_id": "selection-confirm-unit", "steps": [row]}
    sent = []
    replies = [
        (
            '<tool_call>\n{"name":"select_repair","arguments":'
            '{"agent_step":1,"call_index":0,"diagnosis":"Inspect this exact step."}}'
            "\n</tool_call>"
        ),
        (
            '<tool_call>\n{"name":"decline_repair","arguments":'
            '{"reason":"Exact evidence does not support an executable change."}}'
            "\n</tool_call>"
        ),
    ]

    def post(url, *, json, timeout):
        sent.append(json)
        raw = replies[len(sent) - 1]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "diagnostic_raw": raw,
            "meta_info": {"finish_reason": {"type": "stop"}},
        })

    monkeypatch.setattr(teacher, "DIRECT_HISTORY_MAX_INPUT_TOKENS", 1)
    monkeypatch.setattr(teacher.requests, "post", post)
    monkeypatch.setattr(
        teacher.api,
        "_extract_aligned_generate_response",
        lambda out: (
            out["diagnostic_raw"],
            replay_proxy.tokenizer.encode(out["diagnostic_raw"]),
            [],
        ),
    )
    monkeypatch.setattr(
        teacher,
        "generate_replacement_action",
        lambda *args, **kwargs: pytest.fail("unconfirmed selection generated an action"),
    )
    g = SimpleNamespace(model="frozen-test-model", served_model="test", backend_port=9)
    record = teacher.review_trajectory(
        source, ["Static task requirement"], replay_proxy, g, tmp_path / "teacher.json"
    )
    assert record["status"] == "no_repair"
    assert len(sent) == 2
    assert len(record["selection_evidence_exposures"]) == 1
    second_prompt = replay_proxy.tokenizer.decode(sent[1]["input_ids"])
    assert "selection_requires_confirmation" in second_prompt
    assert "read_trace" not in sent[1]["sampling_params"]["json_schema"]


def test_required_original_tool_json_becomes_exact_native_action():
    import json
    import jsonschema
    from verified_repair_teacher import native_action_from_required_json
    from verified_repair_replay_proxy import api

    body = {"model": "test", "messages": [], "tools": _native_tools(),
            "tool_choice": "required"}
    _, _, sampling = api._build_generate_sampling_params(body, {}, "qwen25")
    schema = json.loads(sampling["json_schema"])
    schema["minItems"] = 1
    schema["maxItems"] = 8
    raw = '[{"name":"read","parameters":{"path":"input/b"}}]'
    assert native_action_from_required_json(raw, schema) == (
        '<tool_call>\n{"name":"read","arguments":{"path":"input/b"}}\n</tool_call>'
    )
    with pytest.raises(jsonschema.ValidationError):
        native_action_from_required_json(
            '[{"name":"read","parameters":{"unknown":"input/b"}}]', schema)


def test_action_pass_can_abandon_without_creating_a_training_target():
    import json
    from verified_repair_teacher import (
        ACTION_ABANDON_TOOL,
        required_action_decision_from_json,
    )
    from verified_repair_replay_proxy import api

    tools = _native_tools() + [ACTION_ABANDON_TOOL]
    body = {"model": "test", "messages": [], "tools": tools, "tool_choice": "required"}
    _, _, sampling = api._build_generate_sampling_params(body, {}, "qwen25")
    schema = json.loads(sampling["json_schema"])
    schema["minItems"] = 1
    schema["maxItems"] = 8
    decision = required_action_decision_from_json(
        '[{"name":"abandon_repair_action","parameters":'
        '{"reason":"The exact state supports no executable change."}}]',
        schema,
    )
    assert decision == {
        "status": "no_repair",
        "reason": "The exact state supports no executable change.",
    }
    with pytest.raises(ValueError, match="must be the only"):
        required_action_decision_from_json(
            '[{"name":"read","parameters":{"path":"input/a"}},'
            '{"name":"abandon_repair_action","parameters":{"reason":"uncertain"}}]',
            schema,
        )


def test_sglang_required_tool_json_array_is_strictly_parsed(replay_proxy):
    import json
    from verified_repair_teacher import REVIEW_TOOLS, parse_reviewer_calls
    from verified_repair_replay_proxy import api
    body = {"model": "test", "messages": [], "tools": REVIEW_TOOLS, "tool_choice": "required"}
    _, parser, sampling = api._build_generate_sampling_params(body, {}, "qwen25")
    raw = '[{"name":"read_trace","parameters":{"step":2,"part":"assistant"}}]'
    content, calls, encoding = parse_reviewer_calls(raw, parser, "unit", sampling["json_schema"])
    assert content == ""
    assert encoding == "sglang_required_tool_json_array"
    assert calls[0]["function"]["name"] == "read_trace"
    assert json.loads(calls[0]["function"]["arguments"]) == {"step": 2, "part": "assistant"}
    for invalid in (
        '[{"name":"write","parameters":{"path":"output/a"}}]',
        '[{"name":"read_trace","parameters":{"step":0,"part":"assistant"}}]',
        '[{"name":"read_trace","parameters":{"step":2,"part":"assistant"},"extra":1}]',
        ('[{"name":"read_trace","parameters":{"step":2,"part":"assistant"}},'
         '{"name":"read_trace","parameters":{"step":3,"part":"assistant"}}]'),
    ):
        with pytest.raises((ValueError, json.JSONDecodeError, __import__("jsonschema").ValidationError)):
            parse_reviewer_calls(invalid, parser, "unit", sampling["json_schema"])
