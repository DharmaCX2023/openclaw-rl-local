from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("analyze_mosaic_phase0", REPO_ROOT / "scripts/analyze_mosaic_phase0.py")
credit = _load(
    "extract_mosaic_artifact_credit",
    REPO_ROOT / "scripts/extract_mosaic_artifact_credit.py",
)


class _CharacterTokenizer:
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(value) for value in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(value) for value in ids)


def _record(response_text: str, response_ids=None):
    call_id = "call_demo"
    result = {
        "step": 3,
        "response_text": response_text,
        "tool_calls": [{"id": call_id}],
        "observed_tool_call_ids": [call_id],
        "tool_result_id_order_matches": True,
        "raw_tool_call_count": 1,
        "parsed_tool_call_count": 1,
        "returned_tool_call_count": 1,
    }
    if response_ids is not None:
        result["response_ids"] = response_ids
    return result


def test_raw_tool_call_parser_preserves_exact_span_and_escaped_content():
    payload = {
        "name": "write",
        "arguments": {"path": "output/a.json", "content": "{\n  \"x\": 1\n}"},
    }
    block = "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"
    text = "analysis first\n" + block + "\nafter"
    calls, errors = credit._extract_raw_tool_calls(text)

    assert errors == []
    assert len(calls) == 1
    assert text[calls[0].char_start : calls[0].char_end] == block
    assert calls[0].payload == payload


def test_workspace_path_normalization_rejects_escape_and_external_absolute_path():
    assert credit._normalize_workspace_path("output/a.txt") == ("output/a.txt", None)
    assert credit._normalize_workspace_path(
        "/root/.openclaw/workspace/output/a.txt"
    ) == ("output/a.txt", None)
    assert credit._normalize_workspace_path("../secret") == (
        None,
        "unsafe_relative_path",
    )
    assert credit._normalize_workspace_path("/etc/passwd") == (
        None,
        "path_outside_workspace",
    )


def test_retokenized_count_match_is_not_exact_rollout_token_fidelity():
    payload = {
        "name": "write",
        "arguments": {"path": "out.txt", "content": "OK"},
    }
    text = "<tool_call>" + json.dumps(payload) + "</tool_call>"
    action = credit._all_actions([_record(text)])[0][0]
    span = credit._token_span(_CharacterTokenizer(), action, len(text))

    assert span["token_fidelity"] == "retokenized_count_match"
    assert span["response_token_start"] == 0
    assert span["response_token_end"] == len(text)


def test_atomic_write_requires_recorded_exact_ids_for_token_loss():
    content = "OK"
    payload = {
        "name": "write",
        "arguments": {"path": "out.txt", "content": content},
    }
    text = "<tool_call>" + json.dumps(payload) + "</tool_call>"
    artifact = {
        "task_id": "task_demo",
        "path": "out.txt",
        "selected_version": (
            f"file:0644:{len(content.encode())}:"
            f"{hashlib.sha256(content.encode()).hexdigest()}"
        ),
        "source_parent_indices": [0],
    }
    task = {"task_id": "task_demo", "parent_replay": [{"task_index": 7}]}
    session_map = {("task_demo", 7): "session_demo"}
    counts = {("session_demo", 3): len(text)}

    rows, _ = credit._map_actions(
        artifact,
        task,
        session_map,
        {"session_demo": [_record(text)]},
        counts,
        _CharacterTokenizer(),
    )
    assert len(rows) == 1
    assert rows[0]["structural_action_eligible"] is True
    assert rows[0]["token_fidelity"] == "retokenized_count_match"
    assert rows[0]["token_loss_eligible"] is False

    exact_ids = [ord(value) for value in text]
    exact_rows, _ = credit._map_actions(
        artifact,
        task,
        session_map,
        {"session_demo": [_record(text, exact_ids)]},
        counts,
        _CharacterTokenizer(),
    )
    assert exact_rows[0]["token_fidelity"] == "exact_recorded_ids"
    assert exact_rows[0]["token_loss_eligible"] is True


def test_multi_tool_turn_is_not_structurally_eligible():
    first = {
        "name": "write",
        "arguments": {"path": "out.txt", "content": "OK"},
    }
    second = {"name": "read", "arguments": {"path": "input.txt"}}
    text = "".join(
        f"<tool_call>{json.dumps(payload)}</tool_call>" for payload in (first, second)
    )
    record = _record(text, [ord(value) for value in text])
    record.update(
        {
            "tool_calls": [{"id": "call_demo"}, {"id": "call_other"}],
            "observed_tool_call_ids": ["call_demo", "call_other"],
            "raw_tool_call_count": 2,
            "parsed_tool_call_count": 2,
            "returned_tool_call_count": 2,
        }
    )
    digest = hashlib.sha256(b"OK").hexdigest()
    artifact = {
        "task_id": "task_demo",
        "path": "out.txt",
        "selected_version": f"file:0644:2:{digest}",
        "source_parent_indices": [0],
    }
    task = {"task_id": "task_demo", "parent_replay": [{"task_index": 7}]}
    rows, _ = credit._map_actions(
        artifact,
        task,
        {("task_demo", 7): "session_demo"},
        {"session_demo": [record]},
        {("session_demo", 3): len(text)},
        _CharacterTokenizer(),
    )

    assert rows[0]["structural_action_eligible"] is False
    assert "tool_turn_is_not_single_exact_call" in rows[0]["rejection_reasons"]


def test_recorded_ids_remain_exact_when_retokenization_uses_different_ids():
    class SpecialTokenizer(_CharacterTokenizer):
        SPECIAL = 999_001

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            # Deliberately emulate a round-trip tokenizer that does not recover
            # the one sampled special token.
            return super().__call__(text, add_special_tokens, return_offsets_mapping)

        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            pieces = []
            for value in ids:
                pieces.append("<tool_call>" if value == self.SPECIAL else chr(value))
            return "".join(pieces)

    text = '<tool_call>{"name":"write"}</tool_call>'
    stored = [SpecialTokenizer.SPECIAL, *map(ord, '{"name":"write"}</tool_call>')]
    action = {
        "response_text": text,
        "response_ids": stored,
        "char_start": 0,
        "char_end": len(text),
    }
    span = credit._token_span(SpecialTokenizer(), action, len(stored))

    assert span["token_fidelity"] == "exact_recorded_ids"
    assert span["response_token_start"] == 0
    assert span["response_token_end"] == len(stored)
    assert span["response_token_count_retokenized"] != len(stored)


def test_declared_eos_suffix_is_allowed_but_excluded_from_action_span():
    class EosTokenizer(_CharacterTokenizer):
        eos_token_id = 999_002

        def decode(
            self,
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            return "".join(
                "<|im_end|>" if value == self.eos_token_id else chr(value)
                for value in ids
            )

    text = '<tool_call>{"name":"write"}</tool_call>'
    stored = [*map(ord, text), EosTokenizer.eos_token_id]
    action = {
        "response_text": text,
        "response_ids": stored,
        "char_start": 0,
        "char_end": len(text),
    }
    span = credit._token_span(EosTokenizer(), action, len(stored))

    assert span["token_fidelity"] == "exact_recorded_ids"
    assert span["recorded_terminal_token_count"] == 1
    assert span["response_token_end"] == len(stored) - 1
