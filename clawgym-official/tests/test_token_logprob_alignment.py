import asyncio
import base64
import json
from itertools import count
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_api_server as proxy


class _FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        return [10, 11]

    def decode(self, token_ids, **kwargs):
        assert token_ids == [10, 11]
        return "rendered prompt"


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, output):
        self._output = output
        self.headers = {}

    def json(self):
        return self._output


class _FakeAsyncClient:
    calls = []
    output = None
    status_code = 200
    response_text = ""

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json, headers=None):
        self.__class__.calls.append((url, json))
        response = _FakeResponse(self.__class__.output)
        response.status_code = self.__class__.status_code
        response.text = self.__class__.response_text
        return response


def _generate_output(*, completion_tokens=2, text="raw generated text", pairs=None):
    if pairs is None:
        pairs = [[-0.125, 20, "raw"], [-0.25, 21, " generated text"]]
    return {
        "text": text,
        "meta_info": {
            "id": "request-id",
            "prompt_tokens": 2,
            "completion_tokens": completion_tokens,
            "finish_reason": {"type": "stop", "matched": None},
            "weight_version": "weight-1",
            "output_token_logprobs": pairs,
        },
    }


def _owner():
    owner = object.__new__(proxy.ClawGymRLAPIServer)
    owner.args = SimpleNamespace(
        use_rollout_routing_replay=False,
        num_layers=2,
        moe_router_topk=2,
    )
    owner.tokenizer = _FakeTokenizer()
    owner.sglang_generate_url = "http://sglang/generate"
    owner.tool_call_parser = "qwen25"
    owner.generation_defaults = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }
    owner._session_sglang_stats = {}
    owner._step_counts = {}
    owner._pending_step_data = {}
    owner._completed_reward_submissions = {}
    owner._pending_records = {}
    owner._conversation_logs = {}
    owner._session_raw_prompts = {}
    owner._session_metadata = {}
    owner._session_context_overflows = {}
    owner._session_compaction_counts = {}
    owner._session_compaction_request_counts = {}
    owner._session_pending_compaction_rewrites = {}
    owner._session_tool_result_pruning_events = {}
    owner._session_tool_results_pruned = {}
    owner._session_tool_result_truncation_events = {}
    owner._session_tool_results_truncated = {}
    owner._session_unclassified_history_rewrites = {}
    owner._session_budget_terminations = {}
    owner._session_token_ledgers = {}
    owner._tool_call_repeat_state = {}
    owner._token_ledger_mode = "enforce"
    owner._full_prompt_logged = True
    owner._record_file = ""
    owner._index_counter = count(0)
    owner.output_queue = queue.Queue()
    return owner


def test_extracts_exact_generated_token_ids_and_logprobs():
    text, token_ids, logprobs = proxy._extract_aligned_generate_response(
        _generate_output()
    )
    assert text == "raw generated text"
    assert token_ids == [20, 21]
    assert logprobs == [-0.125, -0.25]


def test_length_limit_diagnostics_preserve_incomplete_sample(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "2")
    raw_text = '<tool_call>\n{"name":"read","arguments":'
    output = _generate_output(
        completion_tokens=2,
        text=raw_text,
        pairs=[[-0.1, 20, "a"], [-0.2, 21, "b"]],
    )
    output["meta_info"]["finish_reason"]["type"] = "length"
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = output
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__limit-session",
        "messages": [{"role": "user", "content": "read"}],
        "tools": [_read_tool()],
        "stream": False,
        "max_tokens": 2,
    }

    result = asyncio.run(
        owner._handle_request(body, session_id="limit-session", turn_type="main")
    )

    step = owner._pending_step_data["limit-session"][1]
    assert result["response"]["choices"][0]["finish_reason"] == "length"
    assert step["response_limit_tokens"] == 2
    assert step["configured_response_limit_tokens"] == 2
    assert step["hit_response_limit"] is True
    assert step["incomplete_tool_call_at_limit"] is True
    assert step["response_ids"] == [20, 21]
    assert step["response_logprobs"] == [-0.1, -0.2]


def test_token_repetition_diagnostics_detects_eight_repeated_blocks():
    diagnostics = proxy._token_repetition_diagnostics(list(range(32)) * 8)

    assert diagnostics["max_ngram_repeat"] == 8
    assert diagnostics["repeated_ngram_ratio"] > 0
    assert diagnostics["repetition_degenerated"] is True


def test_rejects_completion_count_mismatch():
    with pytest.raises(ValueError, match="completion count"):
        proxy._extract_aligned_generate_response(_generate_output(completion_tokens=3))


def test_rejects_malformed_logprob_pair():
    output = _generate_output()
    output["meta_info"]["output_token_logprobs"] = [[-0.125]]
    with pytest.raises(TypeError, match="items must be"):
        proxy._extract_aligned_generate_response(output)


def test_sampling_params_use_model_defaults_and_response_cap(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    request, parser, sampling = proxy._build_generate_sampling_params(
        {
            "model": "main__session",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
            "max_tokens": 99,
        },
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "qwen25",
    )
    assert request.model == "main__session"
    assert parser is None
    assert sampling["temperature"] == 0.7
    assert sampling["top_p"] == 0.8
    assert sampling["top_k"] == 20
    assert sampling["max_new_tokens"] == 8


def test_exact_context_guard_rejects_before_sglang(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "2")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "side__overflow-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8,
    }

    response = asyncio.run(
        owner._handle_request(
            body,
            session_id="overflow-session",
            turn_type="side",
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["code"] == "context_length_exceeded"
    assert "prompt is too long" in payload["error"]["message"].lower()
    assert "maximum context length" in payload["error"]["message"].lower()
    assert _FakeAsyncClient.calls == []
    assert owner._session_context_overflows["overflow-session"] == [
        {
            "prompt_tokens": 2,
            "max_new_tokens": 8,
            "requested_total_tokens": 10,
            "context_window": 2,
            "overflow_tokens": 8,
        }
    ]



def test_response_budget_is_fitted_to_remaining_context(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "9")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "compact__fit-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8,
    }

    result = asyncio.run(
        owner._handle_request(body, session_id="fit-session", turn_type="compact")
    )

    assert result["response"]["usage"]["completion_tokens"] == 2
    assert len(_FakeAsyncClient.calls) == 1
    assert _FakeAsyncClient.calls[0][1]["sampling_params"]["max_new_tokens"] == 6
    assert owner._session_context_overflows.get("fit-session", []) == []
    assert owner._conversation_logs.get("fit-session", []) == []
    assert owner._pending_step_data.get("fit-session", {}) == {}


def test_streaming_exact_context_guard_returns_json_error_envelope(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "2")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    owner.expected_api_key = ""
    owner.submission_enabled = __import__("threading").Event()
    owner.submission_enabled.set()
    client = TestClient(owner._build_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "side__stream-overflow-session",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [],
            "stream": True,
            "max_tokens": 8,
        },
    )

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    payload = response.json()
    assert payload["error"]["code"] == "context_length_exceeded"
    assert "maximum context length" in payload["error"]["message"].lower()
    assert _FakeAsyncClient.calls == []


def test_upstream_error_uses_openai_compatible_envelope(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "64")
    monkeypatch.setattr(_FakeAsyncClient, "status_code", 400)
    monkeypatch.setattr(
        _FakeAsyncClient,
        "response_text",
        (
            "Requested token count exceeds the model's maximum context length "
            "of 64 tokens."
        ),
    )
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = None
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "side__upstream-overflow-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8,
    }

    response = asyncio.run(
        owner._handle_request(
            body,
            session_id="upstream-overflow-session",
            turn_type="side",
        )
    )

    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "upstream_error"
    assert "maximum context length" in payload["error"]["message"].lower()
    assert len(_FakeAsyncClient.calls) == 1


def test_identical_tool_call_streak_uses_semantic_arguments():
    state = {}
    first = [
        {
            "id": "call-a",
            "function": {
                "name": "exec",
                "arguments": '{"command":"echo ok","timeout":10}',
            },
        }
    ]
    same_semantics = [
        {
            "id": "call-b",
            "function": {
                "name": "exec",
                "arguments": '{ "timeout": 10, "command": "echo ok" }',
            },
        }
    ]
    changed = [
        {
            "id": "call-c",
            "function": {
                "name": "exec",
                "arguments": '{"command":"echo changed","timeout":10}',
            },
        }
    ]

    assert proxy._advance_tool_call_repeat_streak(state, "session", first) == 1
    assert (
        proxy._advance_tool_call_repeat_streak(state, "session", same_semantics) == 2
    )
    assert proxy._advance_tool_call_repeat_streak(state, "session", changed) == 1
    assert proxy._advance_tool_call_repeat_streak(state, "session", []) == 0
    assert "session" not in state


def test_request_seed_is_not_sent_to_sglang_and_is_audited_as_disabled(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": True,
        "max_tokens": 8192,
    }

    asyncio.run(owner._handle_request(body, session_id="session", turn_type="main"))

    _, generate_body = _FakeAsyncClient.calls[0]
    assert "seed" not in generate_body["sampling_params"]
    assert generate_body["sampling_params"]["sampling_seed"] is None
    assert owner._pending_step_data["session"][1]["sampling_seed"] is None


def test_empty_request_seed_base_is_treated_as_disabled(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE", "   ")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": True,
        "max_tokens": 8192,
    }

    asyncio.run(owner._handle_request(body, session_id="session", turn_type="main"))

    _, generate_body = _FakeAsyncClient.calls[0]
    assert generate_body["sampling_params"]["sampling_seed"] is None
    assert owner._pending_step_data["session"][1]["sampling_seed"] is None


def test_eval_sampling_seed_is_stable_across_runtime_sessions(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE", "20260826")
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)

    seeds = []
    for runtime_session in ("eval-task-0-42", "eval-task-99-42"):
        _FakeAsyncClient.calls = []
        owner = _owner()
        owner._session_metadata[runtime_session] = {
            "sampling_identity": "eval:v1:clawgym_eval:task_0042:sample:0"
        }
        body = {
            "model": f"main__{runtime_session}",
            "messages": [{"role": "user", "content": "test"}],
            "tools": [],
            "stream": True,
            "max_tokens": 8192,
        }
        asyncio.run(
            owner._handle_request(
                body, session_id=runtime_session, turn_type="main"
            )
        )
        seeds.append(
            _FakeAsyncClient.calls[0][1]["sampling_params"]["sampling_seed"]
        )
        assert owner._conversation_logs[runtime_session][0][
            "sampling_identity"
        ] == "eval:v1:clawgym_eval:task_0042:sample:0"

    assert seeds[0] == seeds[1]


def test_main_turn_stores_raw_tokens_without_retokenizing(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": True,
        "max_tokens": 8192,
    }

    result = asyncio.run(
        owner._handle_request(body, session_id="session", turn_type="main")
    )

    assert len(_FakeAsyncClient.calls) == 1
    url, generate_body = _FakeAsyncClient.calls[0]
    assert url == "http://sglang/generate"
    assert generate_body["input_ids"] == [10, 11]
    assert generate_body["return_logprob"] is True

    step = owner._pending_step_data["session"][1]
    assert step["prompt_ids"] == [10, 11]
    assert step["response_ids"] == [20, 21]
    assert step["response_logprobs"] == [-0.125, -0.25]
    assert step["response_text"] == "raw generated text"
    assert (
        result["response"]["choices"][0]["message"]["content"] == "raw generated text"
    )


def test_full_routing_replay_falls_back_on_token_prefix_rewrite(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "65536")
    monkeypatch.setenv("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536")
    routes = np.arange(3 * 2 * 2, dtype=np.int32).reshape(3, 2, 2)
    output = _generate_output()
    output["meta_info"]["routed_experts"] = base64.b64encode(
        routes.tobytes()
    ).decode("ascii")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = output
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)

    owner = _owner()
    owner.args.use_rollout_routing_replay = True
    owner._session_metadata["routing-session"] = {
        "task_id": "task-routing",
        "rollout_id": 0,
        "task_index": 0,
        "group_index": 7,
    }
    body = {
        "model": "main__routing-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8192,
    }
    asyncio.run(
        owner._handle_request(
            body, session_id="routing-session", turn_type="main"
        )
    )
    asyncio.run(
        owner._handle_request(
            body, session_id="routing-session", turn_type="main"
        )
    )

    assert len(_FakeAsyncClient.calls) == 2
    assert all(call[1]["return_routed_experts"] is True for call in _FakeAsyncClient.calls)
    client = TestClient(owner._build_app())
    reward_response = client.post(
        "/set_reward",
        json={
            "session_id": "routing-session",
            "reward": 0.5,
            "group_index": 7,
        },
    )
    assert reward_response.status_code == 200
    assert reward_response.json()["submitted"] == 2

    samples = [
        owner.output_queue.get_nowait()[1][0],
        owner.output_queue.get_nowait()[1][0],
    ]
    assert samples[0].index == samples[1].index
    assert [sample.group_index for sample in samples] == [7, 7]
    for sample in samples:
        assert sample.tokens == [10, 11, 20, 21]
        assert sample.response_length == 2
        assert sample.loss_mask == [1, 1]
        assert sample.rollout_log_probs == [-0.125, -0.25]
        assert np.array_equal(sample.rollout_routed_experts, routes)
        assert sample.rollout_routed_experts.flags.writeable is True
        assert sample.metadata["routing_replay_scope"] == "single_agent_request"
        assert sample.metadata["routing_replay_turn_count"] == 1
    assert samples[0].metadata["routing_replay_split_reason"] == "initial"
    assert (
        samples[1].metadata["routing_replay_split_reason"]
        == "history_rewritten"
    )


def _read_tool():
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def _two_read_calls_text():
    return (
        '<tool_call>\n{"name":"read","arguments":{"path":"a.txt"}}\n</tool_call>\n'
        '<tool_call>\n{"name":"read","arguments":{"path":"b.txt"}}\n</tool_call>'
    )


def test_qwen_parser_preserves_all_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    _, parser, _ = proxy._build_generate_sampling_params(
        {
            "model": "main__session",
            "messages": [{"role": "user", "content": "read both"}],
            "tools": [_read_tool()],
            "stream": False,
            "max_tokens": 8192,
        },
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "qwen25",
    )
    content, tool_calls = proxy._parse_generated_assistant(
        _two_read_calls_text(), parser
    )

    assert content == ""
    assert len(tool_calls) == 2
    assert [call["index"] for call in tool_calls] == [0, 1]
    assert [call["function"]["name"] for call in tool_calls] == ["read", "read"]
    assert [
        json.loads(call["function"]["arguments"])["path"] for call in tool_calls
    ] == [
        "a.txt",
        "b.txt",
    ]
    assert len({call["id"] for call in tool_calls}) == 2
    assert all(call["id"].isalnum() for call in tool_calls)


def test_stable_sampling_identity_also_stabilizes_tool_call_ids():
    _, parser, _ = proxy._build_generate_sampling_params(
        {
            "model": "main__session",
            "messages": [{"role": "user", "content": "read both"}],
            "tools": [_read_tool()],
            "stream": False,
            "max_tokens": 8192,
        },
        {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "qwen25",
    )
    namespace = "eval:v1:clawgym_eval:task_0042:sample:0:main:0"
    _, first = proxy._parse_generated_assistant(
        _two_read_calls_text(), parser, tool_call_id_namespace=namespace
    )
    _, repeated = proxy._parse_generated_assistant(
        _two_read_calls_text(), parser, tool_call_id_namespace=namespace
    )

    assert [call["id"] for call in first] == [
        call["id"] for call in repeated
    ]
    assert len({call["id"] for call in first}) == 2


def test_main_turn_returns_and_stores_every_tool_call(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    raw_text = _two_read_calls_text()
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output(
        completion_tokens=1,
        text=raw_text,
        pairs=[[-0.125, 20, raw_text]],
    )
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__session",
        "messages": [{"role": "user", "content": "read both"}],
        "tools": [_read_tool()],
        "stream": True,
        "max_tokens": 8192,
    }

    result = asyncio.run(
        owner._handle_request(body, session_id="session", turn_type="main")
    )

    returned_calls = result["response"]["choices"][0]["message"]["tool_calls"]
    stored_step = owner._pending_step_data["session"][1]
    assert len(returned_calls) == 2
    assert stored_step["tool_calls"] == returned_calls
    assert stored_step["raw_tool_call_count"] == 2
    assert stored_step["parsed_tool_call_count"] == 2
    assert stored_step["returned_tool_call_count"] == 2
    assert result["response"]["choices"][0]["finish_reason"] == "tool_calls"


def test_collects_success_and_failure_results_in_order():
    messages = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "call_a", "content": "first"},
        {"role": "tool", "tool_call_id": "call_b", "content": "ERROR: missing"},
        {"role": "tool", "tool_call_id": "call_c", "content": "third"},
    ]
    results = proxy._collect_new_tool_results(messages)
    assert [result["tool_call_id"] for result in results] == [
        "call_a",
        "call_b",
        "call_c",
    ]
    assert len(results) == 3
    assert results[1]["content"] == "ERROR: missing"


def test_persistent_record_links_every_result_id(tmp_path):
    owner = _owner()
    owner._record_file = str(tmp_path / "records.jsonl")
    tool_calls = [
        {
            "id": "call_a",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path":"a"}'},
        },
        {
            "id": "call_b",
            "type": "function",
            "function": {"name": "read", "arguments": '{"path":"b"}'},
        },
    ]
    owner._buffer_record(
        "session",
        1,
        [{"role": "user", "content": "read both"}],
        "prompt",
        _two_read_calls_text(),
        tool_calls,
        {
            "raw_tool_call_count": 2,
            "parsed_tool_call_count": 2,
            "returned_tool_call_count": 2,
            "tool_call_ids": ["call_a", "call_b"],
        },
    )
    owner._flush_pending_record(
        "session",
        {"role": "tool", "tool_call_id": "call_b", "content": "second"},
        [
            {"tool_call_id": "call_a", "content": "first"},
            {"tool_call_id": "call_b", "content": "second"},
        ],
    )

    record = json.loads((tmp_path / "records.jsonl").read_text())
    assert record["raw_tool_call_count"] == 2
    assert record["parsed_tool_call_count"] == 2
    assert record["returned_tool_call_count"] == 2
    assert record["observed_tool_result_count"] == 2
    assert record["tool_call_ids"] == ["call_a", "call_b"]
    assert record["observed_tool_call_ids"] == ["call_a", "call_b"]
    assert record["tool_result_id_order_matches"] is True


def test_response_ids_are_recorded_only_when_explicitly_enabled(monkeypatch):
    monkeypatch.delenv("OPENCLAW_RECORD_RESPONSE_IDS", raising=False)
    assert proxy._optional_recorded_response_ids([10, 20]) == {}

    monkeypatch.setenv("OPENCLAW_RECORD_RESPONSE_IDS", "1")
    assert proxy._optional_recorded_response_ids([10, 20]) == {
        "response_ids": [10, 20]
    }



class _LedgerTokenizer:
    @staticmethod
    def _encode(text):
        return [ord(character) + 1000 for character in text]

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        rendered = ""
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role in {"system", "user"}:
                rendered += f"<{role}>{content}</{role}>"
            elif role == "assistant":
                rendered += f"<assistant>{content}"
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    rendered += (
                        f"<tool_call>{{\"name\":\"{function['name']}\","
                        f"\"arguments\":{arguments}}}</tool_call>"
                    )
                rendered += "</assistant>"
            elif role == "tool":
                rendered += (
                    f"<tool:{message.get('tool_call_id', '')}>"
                    f"{content}</tool>"
                )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self._encode(rendered)

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id - 1000) for token_id in token_ids)


class _ConditionalReasoningLedgerTokenizer(_LedgerTokenizer):
    """Mimic Qwen3 last-assistant-only empty reasoning decoration."""

    all_special_ids = [ord("<") + 1000]

    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        rendered = ""
        for index, message in enumerate(messages):
            role = message["role"]
            content = message.get("content") or ""
            if role in {"system", "user"}:
                rendered += f"<{role}>{content}</{role}>"
            elif role == "assistant":
                rendered += "<assistant>"
                if index == len(messages) - 1:
                    rendered += "<think></think>"
                rendered += content
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call["function"]
                    arguments = function.get("arguments")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    rendered += (
                        f"<tool_call>{{\"name\":\"{function['name']}\","
                        f"\"arguments\":{arguments}}}</tool_call>"
                    )
                rendered += "</assistant>"
            elif role == "tool":
                rendered += (
                    f"<tool:{message.get('tool_call_id', '')}>"
                    f"{content}</tool>"
                )
        if add_generation_prompt:
            rendered += "<assistant>"
        return self._encode(rendered)


def _ledger_tool_call(call_id, path):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "read",
            "arguments": json.dumps(
                {"path": path}, ensure_ascii=False, separators=(",", ":")
            ),
        },
    }


def test_structural_append_ignores_json_whitespace_but_links_every_tool_result():
    tokenizer = _LedgerTokenizer()
    initial_messages = [{"role": "user", "content": "read both"}]
    calls = [
        _ledger_tool_call("call_a", "a.txt"),
        _ledger_tool_call("call_b", "b.txt"),
    ]
    assistant = {"role": "assistant", "content": "", "tool_calls": calls}
    raw_action = (
        '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>'
        '<tool_call>{"name":"read","arguments":{"path":"b.txt"}}</tool_call>'
    )
    initial_prompt_ids = proxy._apply_chat_template_ids(
        tokenizer,
        initial_messages,
        template_tools=[_read_tool()],
        template_kwargs={},
        reasoning_effort=None,
        add_generation_prompt=True,
    )
    state = proxy.SessionTokenLedger(
        messages=initial_messages,
        assistant_message=assistant,
        prompt_ids=initial_prompt_ids,
        response_ids=tokenizer._encode(raw_action),
        response_logprobs=[-0.1] * len(raw_action),
        template_tools=[_read_tool()],
        template_kwargs={},
        reasoning_effort=None,
    )
    next_messages = [
        *initial_messages,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    **calls[0],
                    "function": {
                        **calls[0]["function"],
                        "arguments": '{"path": "a.txt"}',
                    },
                },
                {
                    **calls[1],
                    "function": {
                        **calls[1]["function"],
                        "arguments": '{"path": "b.txt"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "first"},
        {"role": "tool", "tool_call_id": "call_b", "content": "second"},
    ]

    transition, appended, detail = proxy._classify_structural_append(
        state, next_messages, [_read_tool()]
    )
    assert transition == "append_only"
    assert detail == ""

    prompt_ids, boundary_ids, environment_ids = (
        proxy._build_incremental_prompt_ids(
            tokenizer,
            state,
            next_messages,
            appended,
            template_tools=[_read_tool()],
            template_kwargs={},
            reasoning_effort=None,
        )
    )
    rendered = tokenizer.decode(prompt_ids)
    assert rendered == (
        "<user>read both</user><assistant>"
        + raw_action
        + "</assistant>"
        + "<tool:call_a>first</tool>"
        + "<tool:call_b>second</tool>"
        + "<assistant>"
    )
    assert boundary_ids == tokenizer._encode("</assistant>")
    assert environment_ids == tokenizer._encode(
        "<tool:call_a>first</tool><tool:call_b>second</tool><assistant>"
    )
    assert prompt_ids[: len(initial_prompt_ids)] == initial_prompt_ids
    assert prompt_ids[
        len(initial_prompt_ids) : len(initial_prompt_ids) + len(raw_action)
    ] == tokenizer._encode(raw_action)


def test_incremental_prompt_ignores_last_assistant_only_reasoning_decoration():
    tokenizer = _ConditionalReasoningLedgerTokenizer()
    initial_messages = [{"role": "user", "content": "read it"}]
    call = _ledger_tool_call("call_a", "a.txt")
    assistant = {"role": "assistant", "content": "", "tool_calls": [call]}
    raw_action = (
        '<tool_call>{"name":"read","arguments":{"path":"a.txt"}}</tool_call>'
    )
    initial_prompt_ids = proxy._apply_chat_template_ids(
        tokenizer,
        initial_messages,
        template_tools=[_read_tool()],
        template_kwargs={},
        reasoning_effort=None,
        add_generation_prompt=True,
    )
    state = proxy.SessionTokenLedger(
        messages=initial_messages,
        assistant_message=assistant,
        prompt_ids=initial_prompt_ids,
        response_ids=tokenizer._encode(raw_action),
        response_logprobs=[-0.1] * len(raw_action),
        template_tools=[_read_tool()],
        template_kwargs={},
        reasoning_effort=None,
    )
    next_messages = [
        *initial_messages,
        assistant,
        {"role": "tool", "tool_call_id": "call_a", "content": "result"},
    ]

    transition, appended, _ = proxy._classify_structural_append(
        state, next_messages, [_read_tool()]
    )
    assert transition == "append_only"
    prompt_ids, boundary_ids, environment_ids = (
        proxy._build_incremental_prompt_ids(
            tokenizer,
            state,
            next_messages,
            appended,
            template_tools=[_read_tool()],
            template_kwargs={},
            reasoning_effort=None,
        )
    )

    assert tokenizer.decode(boundary_ids) == "</assistant>"
    assert tokenizer.decode(environment_ids) == (
        "<tool:call_a>result</tool><assistant>"
    )
    assert tokenizer.decode(prompt_ids) == (
        "<user>read it</user><assistant>"
        + raw_action
        + "</assistant><tool:call_a>result</tool><assistant>"
    )


def test_structural_append_rejects_missing_multi_tool_result_and_history_rewrite():
    messages = [{"role": "user", "content": "read both"}]
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _ledger_tool_call("call_a", "a.txt"),
            _ledger_tool_call("call_b", "b.txt"),
        ],
    }
    state = proxy.SessionTokenLedger(
        messages=messages,
        assistant_message=assistant,
        prompt_ids=[1],
        response_ids=[2],
        response_logprobs=[-0.1],
        template_tools=[_read_tool()],
        template_kwargs={},
        reasoning_effort=None,
    )
    missing_result_messages = [
        *messages,
        assistant,
        {"role": "tool", "tool_call_id": "call_a", "content": "first"},
    ]
    transition, _, detail = proxy._classify_structural_append(
        state, missing_result_messages, [_read_tool()]
    )
    assert transition == "tool_result_mismatch"
    assert "call_b" in detail

    rewritten_messages = [
        {"role": "user", "content": "changed"},
        assistant,
        {"role": "tool", "tool_call_id": "call_a", "content": "first"},
        {"role": "tool", "tool_call_id": "call_b", "content": "second"},
    ]
    transition, _, detail = proxy._classify_structural_append(
        state, rewritten_messages, [_read_tool()]
    )
    assert transition == "history_rewritten"
    assert "prefix changed" in detail


def test_agent_step_budget_stops_without_sampling_or_dropping_prior_step(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_MAX_AGENT_STEPS", "1")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    body = {
        "model": "main__budget-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8192,
    }

    asyncio.run(
        owner._handle_request(body, session_id="budget-session", turn_type="main")
    )
    result = asyncio.run(
        owner._handle_request(body, session_id="budget-session", turn_type="main")
    )

    assert len(_FakeAsyncClient.calls) == 1
    assert len(owner._conversation_logs["budget-session"]) == 1
    assert sorted(owner._pending_step_data["budget-session"]) == [1]
    termination = owner._session_budget_terminations["budget-session"]
    assert termination == {
        "reason": "max_agent_steps",
        "limit": 1,
        "completed_agent_steps": 1,
        "compaction_count": 0,
    }
    response = result["response"]
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["tool_calls"] == []
    assert response["usage"]["completion_tokens"] == 0


def test_compaction_route_does_not_consume_agent_budget_or_create_rl_sample(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "64")
    monkeypatch.setenv("OPENCLAW_MAX_AGENT_STEPS", "2")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    main_body = {
        "model": "main__route-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8,
    }
    compact_body = {**main_body, "model": "compact__route-session"}

    asyncio.run(
        owner._handle_request(main_body, session_id="route-session", turn_type="main")
    )
    asyncio.run(
        owner._handle_request(compact_body, session_id="route-session", turn_type="compact")
    )
    asyncio.run(
        owner._handle_request(compact_body, session_id="route-session", turn_type="compact")
    )
    asyncio.run(
        owner._handle_request(main_body, session_id="route-session", turn_type="main")
    )

    assert len(_FakeAsyncClient.calls) == 4
    assert len(owner._conversation_logs["route-session"]) == 2
    assert sorted(owner._pending_step_data["route-session"]) == [1, 2]
    assert owner._step_counts["route-session"] == 2
    assert owner._session_compaction_counts["route-session"] == 1
    assert owner._session_compaction_request_counts["route-session"] == 2
    assert owner._pending_step_data["route-session"][2]["ledger_transition"] == "session_compacted"
    assert "route-session" not in owner._session_budget_terminations


def test_compaction_budget_stops_ninth_compaction_without_sampling(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_MAX_AGENT_STEPS", "96")
    monkeypatch.setenv("OPENCLAW_MAX_COMPACTION_EVENTS", "8")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    first_body = {
        "model": "main__compact-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8192,
    }
    asyncio.run(
        owner._handle_request(
            first_body, session_id="compact-session", turn_type="main"
        )
    )
    owner._session_compaction_counts["compact-session"] = 8
    compacted_body = {**first_body, "model": "compact__compact-session"}

    result = asyncio.run(
        owner._handle_request(
            compacted_body, session_id="compact-session", turn_type="compact"
        )
    )

    assert len(_FakeAsyncClient.calls) == 1
    assert len(owner._conversation_logs["compact-session"]) == 1

    termination = owner._session_budget_terminations["compact-session"]
    assert termination["reason"] == "max_compaction_events"
    assert termination["compaction_count"] == 8
    assert termination["attempted_compaction_count"] == 9
    assert result["response"]["usage"]["completion_tokens"] == 0


def test_tool_result_pruning_rewrites_context_without_consuming_compaction_budget(
    monkeypatch,
):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8")
    monkeypatch.setenv("OPENCLAW_CONTEXT_WINDOW", "256")
    monkeypatch.setenv("OPENCLAW_MAX_COMPACTION_EVENTS", "1")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    first_body = {
        "model": "main__prune-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8,
    }
    asyncio.run(
        owner._handle_request(first_body, session_id="prune-session", turn_type="main")
    )
    ledger = owner._session_token_ledgers["prune-session"]
    ledger.messages = [
        {"role": "user", "content": "test"},
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": "a long tool result",
        },
    ]
    pruned_body = {
        **first_body,
        "messages": [
            {"role": "user", "content": "test"},
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": proxy._COMPACTED_TOOL_OUTPUT_MARKER,
            },
        ],
    }
    asyncio.run(
        owner._handle_request(pruned_body, session_id="prune-session", turn_type="main")
    )

    assert owner._session_compaction_counts == {}
    assert owner._session_tool_result_pruning_events["prune-session"] == 1
    assert owner._session_tool_results_pruned["prune-session"] == 1
    step = owner._pending_step_data["prune-session"][2]
    assert step["ledger_transition"] == "tool_result_pruned"
    assert step["prompt_source"] == "canonical"
    assert step["tool_results_pruned"] == 1


def test_budget_terminated_training_trajectory_is_verified_and_submitted(monkeypatch):
    monkeypatch.setenv("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    monkeypatch.setenv("OPENCLAW_MAX_AGENT_STEPS", "1")
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.output = _generate_output()
    monkeypatch.setattr(proxy.httpx, "AsyncClient", _FakeAsyncClient)
    owner = _owner()
    owner._session_metadata["reward-session"] = {
        "task_id": "task-test",
        "rollout_id": 0,
        "task_index": 0,
        "group_index": 0,
    }
    body = {
        "model": "main__reward-session",
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "stream": False,
        "max_tokens": 8192,
    }
    asyncio.run(
        owner._handle_request(body, session_id="reward-session", turn_type="main")
    )
    asyncio.run(
        owner._handle_request(body, session_id="reward-session", turn_type="main")
    )
    client = TestClient(owner._build_app())

    log_response = client.post(
        "/get_conversation_log", json={"session_id": "reward-session"}
    )
    assert log_response.status_code == 200
    assert log_response.json()["budget_termination"]["reason"] == "max_agent_steps"
    reward_response = client.post(
        "/set_reward",
        json={"session_id": "reward-session", "reward": 0.467, "group_index": 0},
    )

    assert reward_response.status_code == 200
    assert reward_response.json()["submitted"] == 1
    group_index, samples = owner.output_queue.get_nowait()
    assert group_index == 0
    assert len(samples) == 1
    retry_response = client.post(
        "/set_reward",
        json={"session_id": "reward-session", "reward": 0.467, "group_index": 0},
    )
    assert retry_response.status_code == 200
    assert retry_response.json()["idempotent_replay"] is True
    assert retry_response.json()["submitted"] == 1
    assert owner.output_queue.empty()
    sample = samples[0]
    assert sample.reward == {"score": 0.467}
    assert sample.response_length == 2
    assert sample.metadata["budget_termination"]["reason"] == "max_agent_steps"
