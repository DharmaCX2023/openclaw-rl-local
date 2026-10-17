from itertools import count
import json
import numpy as np
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

from clawgym_rl_api_server import (
    ClawGymRLAPIServer,
    _prepare_opsd_pair,
    _prepare_opsd_segment,
    _build_stitched_routing_replay_segments,
    _routing_prefix_diagnostics,
    _strict_prompt_append_delta,
)


def _step(prompt_ids, response_ids, prompt_text, response_text, logprob):
    return {
        "messages": [{"role": "user", "content": prompt_text}],
        "prompt_ids": list(prompt_ids),
        "response_ids": list(response_ids),
        "response_logprobs": [logprob] * len(response_ids),
        "prompt_text": prompt_text,
        "response_text": response_text,
        "tool_calls": [],
        "raw_tool_call_count": 0,
        "parsed_tool_call_count": 0,
        "returned_tool_call_count": 0,
        "sampling_seed": None,
    }


def _submit(
    monkeypatch,
    steps,
    max_tokens=65536,
    record_file="",
    session_metadata=None,
    owner_args=None,
    tokenizer=None,
    opsd_feedback=None,
):
    monkeypatch.setenv("OPENCLAW_TRAJECTORY_MAX_TOKENS", str(max_tokens))
    owner = object.__new__(ClawGymRLAPIServer)
    if owner_args is not None:
        owner.args = owner_args
    if tokenizer is not None:
        owner.tokenizer = tokenizer
    owner._pending_step_data = {"session": steps}
    owner._completed_reward_submissions = {}
    owner._step_counts = {"session": len(steps)}
    owner._pending_records = {}
    owner._record_file = str(record_file)
    owner._session_metadata = {"session": dict(session_metadata or {})}
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
    owner._index_counter = count(100)
    owner.output_queue = queue.Queue()

    with TestClient(owner._build_app()) as client:
        body = {"session_id": "session", "group_index": 7, "reward": 0.75}
        if opsd_feedback is not None:
            body["opsd_feedback"] = opsd_feedback
        response = client.post("/set_reward", json=body)
    assert response.status_code == 200, response.text

    samples = []
    while not owner.output_queue.empty():
        group_index, batch = owner.output_queue.get_nowait()
        assert group_index == 7
        assert len(batch) == 1
        samples.append(batch[0])
    assert response.json()["submitted"] == len(samples)
    return samples


def test_register_session_metadata_accepts_only_audit_fields():
    owner = object.__new__(ClawGymRLAPIServer)
    owner._session_metadata = {}
    with TestClient(owner._build_app()) as client:
        response = client.post(
            "/register_session",
            json={
                "session_id": "session",
                "metadata": {
                    "task_id": "task_0042",
                    "rollout_id": 12,
                    "task_index": 3,
                    "group_index": 7,
                    "task_selection_seed": 0,
                    "fixed_rollout_seed": 123,
                    "sampling_identity": "eval:v1:clawgym_eval:task_0042:sample:0",
                },
            },
        )
        rejected = client.post(
            "/register_session",
            json={
                "session_id": "session",
                "metadata": {"secret": "must-not-be-recorded"},
            },
        )

    assert response.status_code == 200
    assert owner._session_metadata["session"]["task_id"] == "task_0042"
    assert owner._session_metadata["session"]["fixed_rollout_seed"] == 123
    assert owner._session_metadata["session"]["sampling_identity"] == (
        "eval:v1:clawgym_eval:task_0042:sample:0"
    )
    assert rejected.status_code == 400
    assert "secret" not in owner._session_metadata["session"]


def test_strict_append_delta():
    assert _strict_prompt_append_delta([1, 2, 3], [1, 2, 3, 4, 5]) == [4, 5]
    assert _strict_prompt_append_delta([1, 2, 3], [1, 2, 3]) == []
    assert _strict_prompt_append_delta([1, 2, 3], [1, 2]) is None
    assert _strict_prompt_append_delta([1, 2, 3], [1, 2, 9, 4]) is None


def test_routing_prefix_diagnostics_separates_order_and_membership():
    old = np.asarray(
        [
            [[3, 8], [12, 5]],
            [[7, 3], [5, 12]],
        ],
        dtype=np.int32,
    )
    new = np.asarray(
        [
            [[8, 3], [12, 5]],
            [[7, 4], [5, 12]],
        ],
        dtype=np.int32,
    )

    stats, membership_rows, order_only_rows = _routing_prefix_diagnostics(
        [old], new
    )

    assert stats["compared_token_rows"] == 2
    assert stats["compared_token_layer_cells"] == 4
    assert stats["compared_route_ids"] == 8
    assert stats["ordered_mismatch_route_ids"] == 3
    assert stats["order_only_token_layer_cells"] == 1
    assert stats["membership_changed_token_layer_cells"] == 1
    assert stats["replaced_expert_slots"] == 1
    assert membership_rows.tolist() == [False, True]
    assert order_only_rows.tolist() == [True, False]


def _routing_step(prompt_ids, response_ids, routes, *, transition=None):
    step = _step(prompt_ids, response_ids, "prompt", "response", -0.1)
    step["rollout_routed_experts"] = np.asarray(
        routes, dtype=np.int32
    ).reshape(-1, 1, 1)
    step["ledger_transition"] = transition
    return step


def _routing_segments(steps):
    details = [{} for _ in steps]
    segments = _build_stitched_routing_replay_segments(
        steps,
        sorted(steps),
        max_context_tokens=65536,
        num_layers=1,
        router_topk=1,
        session_id="routing-test",
        step_details=details,
    )
    return segments, details


def test_routing_replay_stitches_stable_token_and_route_prefix():
    steps = {
        0: _routing_step([1, 2], [3], [10, 11], transition="initial"),
        1: _routing_step(
            [1, 2, 3, 4], [5], [10, 11, 12, 13],
            transition="append_only",
        ),
    }

    segments, details = _routing_segments(steps)

    assert len(segments) == 1
    assert segments[0]["tokens"] == [1, 2, 3, 4, 5]
    assert segments[0]["loss_mask"] == [1, 0, 1]
    assert segments[0]["response_step_ids"] == [0, -1, 1]
    assert segments[0]["rollout_log_probs"] == [-0.1, 0.0, -0.1]
    assert segments[0]["routes"][:, 0, 0].tolist() == [10, 11, 12, 13]
    assert segments[0]["turn_count"] == 2
    assert segments[0]["stitched_turns"] == 1
    assert details[1]["routing_replay_action"] == "stitched"


def test_routing_replay_splits_on_route_prefix_conflict():
    steps = {
        0: _routing_step([1, 2], [3], [10, 11], transition="initial"),
        1: _routing_step(
            [1, 2, 3, 4], [5], [99, 11, 12, 13],
            transition="append_only",
        ),
    }

    segments, details = _routing_segments(steps)

    assert [segment["tokens"] for segment in segments] == [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
    ]
    assert segments[1]["split_reason"] == "route_prefix_conflict"
    assert segments[0]["response_step_ids"] == [0]
    assert segments[1]["response_step_ids"] == [1]
    assert details[1]["routing_replay_prefix_match"] is False


def test_routing_replay_splits_on_declared_history_rewrite():
    steps = {
        0: _routing_step([1, 2], [3], [10, 11], transition="initial"),
        1: _routing_step(
            [1, 2, 3, 4], [5], [10, 11, 12, 13],
            transition="tool_result_pruned",
        ),
    }

    segments, _ = _routing_segments(steps)

    assert len(segments) == 2
    assert segments[1]["split_reason"] == "tool_result_pruned"
    assert segments[0]["response_step_ids"] == [0]
    assert segments[1]["response_step_ids"] == [1]


def test_dense_nonrouting_path_prepares_opsd_teacher_tokens(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            return [1, 2] if content == "task" else [10, 11]

    samples = _submit(
        monkeypatch,
        {0: _step([1, 2], [3, 4], "task", "answer", -0.2)},
        owner_args=SimpleNamespace(
            opsd_enable=True, use_rollout_routing_replay=False
        ),
        tokenizer=Tokenizer(),
        opsd_feedback={
            "rollout_not_fully_successful": True,
            "evidence_anchors": [
                {
                    "condition": "result must be valid",
                    "artifact": "output/result.json",
                    "verified_fact": "the verifier parsed it",
                    "explanation": "the format is required",
                }
            ],
        },
    )

    assert len(samples) == 1
    assert samples[0].opsd_eligible is True
    assert samples[0].opsd_step_ids == [0, 0]
    assert samples[0].opsd_teacher_tokens == [10, 11, 3, 4]


def test_agentopsd_prepares_teacher_tokens_for_complete_trajectory(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            return [1, 2] if content == "task" else [10, 11]

    samples = _submit(
        monkeypatch,
        {0: _step([1, 2], [3, 4], "task", "answer", -0.2)},
        owner_args=SimpleNamespace(
            opsd_enable=True,
            opsd_mode="agentopsd",
            sdar_enable=False,
            opsd_matched_placebo=False,
            use_rollout_routing_replay=False,
        ),
        tokenizer=Tokenizer(),
        opsd_feedback={
            "rollout_not_fully_successful": False,
            "evidence_anchors": [{"condition": "result must be valid"}],
        },
    )

    assert len(samples) == 1
    assert samples[0].opsd_eligible is True
    assert samples[0].opsd_teacher_tokens == [10, 11, 3, 4]


def test_exact_append_stays_in_one_segment(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 3, 4], [5], "p1 r1 tool", "r2", -0.2),
        },
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.tokens == [1, 2, 3, 4, 5]
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert sample.index == 100
    assert sample.group_index == 7
    assert sample.reward == {"score": 0.75}


def test_compressed_prompt_flushes_instead_of_common_prefix_splice(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            # Contains the old response text, so the removed text heuristic would
            # have treated it as unchanged. Tokens share a prefix but rewrite 3.
            1: _step([1, 2, 9, 4], [5], "p1 r1 compressed", "r2", -0.2),
        },
    )

    assert len(samples) == 2
    assert samples[0].tokens == [1, 2, 3]
    assert samples[1].tokens == [1, 2, 9, 4, 5]
    assert [sample.index for sample in samples] == [100, 100]
    assert [sample.group_index for sample in samples] == [7, 7]
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert [sample.reward for sample in samples] == [{"score": 0.75}, {"score": 0.75}]


def test_compression_audit_is_persisted(monkeypatch, tmp_path):
    record_file = tmp_path / "rollout.jsonl"
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 9, 4], [5], "compressed", "r2", -0.2),
        },
        record_file=record_file,
        session_metadata={
            "task_id": "task_0042",
            "rollout_id": 12,
            "task_index": 3,
            "group_index": 7,
            "task_selection_seed": 0,
            "fixed_rollout_seed": 123,
        },
    )

    records = [
        json.loads(line)
        for line in record_file.read_text(encoding="utf-8").splitlines()
    ]
    audit = next(
        record for record in records if record["record_type"] == "trajectory_audit"
    )
    assert audit["logical_trajectory_count"] == 1
    assert audit["task_id"] == "task_0042"
    assert audit["rollout_id"] == 12
    assert audit["task_index"] == 3
    assert audit["task_selection_seed"] == 0
    assert audit["fixed_rollout_seed"] == 123
    assert all(sample.metadata["task_id"] == "task_0042" for sample in samples)
    assert all(sample.metadata["rollout_id"] == 12 for sample in samples)
    assert all(sample.metadata["group_index"] == 7 for sample in samples)
    assert audit["submitted_segment_count"] == 2
    assert [step["context_transition"] for step in audit["steps"]] == [
        "initial",
        "compressed_or_rewritten",
    ]
    assert audit["steps"][1]["shared_prefix_tokens"] == 2
    assert audit["steps"][1]["split_performed"] is True
    assert audit["steps"][1]["split_reason"] == "compressed_or_rewritten"


def test_shorter_prompt_flushes(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2], [4], "compressed", "r2", -0.2),
        },
    )
    assert [sample.tokens for sample in samples] == [[1, 2, 3], [1, 2, 4]]


def test_system_prompt_rewrite_starts_exact_new_segment(monkeypatch):
    steps = {
        0: _step([10, 20], [30], "system-v1", "r1", -0.1),
        1: _step([11, 20, 30, 40], [50], "system-v2", "r2", -0.2),
    }
    samples = _submit(monkeypatch, steps)

    assert [sample.tokens for sample in samples] == [
        [10, 20, 30],
        [11, 20, 30, 40, 50],
    ]
    real_step_sequences = {
        tuple(step["prompt_ids"] + step["response_ids"]) for step in steps.values()
    }
    assert all(tuple(sample.tokens) in real_step_sequences for sample in samples)


def test_append_after_compression_stays_in_new_segment(monkeypatch):
    steps = {
        0: _step([1, 2], [3], "initial", "r1", -0.1),
        1: _step([9, 10], [11], "compressed", "r2", -0.2),
        2: _step([9, 10, 11, 12], [13], "compressed plus tool", "r3", -0.3),
    }
    samples = _submit(monkeypatch, steps)

    assert [sample.tokens for sample in samples] == [
        [1, 2, 3],
        [9, 10, 11, 12, 13],
    ]
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1]
    assert [sample.index for sample in samples] == [100, 100]


def test_multiple_compressions_create_replayable_segments(monkeypatch):
    steps = {
        0: _step([1], [2], "p0", "r0", -0.1),
        1: _step([3, 4], [5], "p1", "r1", -0.2),
        2: _step([6, 7, 8], [9], "p2", "r2", -0.3),
    }
    samples = _submit(monkeypatch, steps)

    assert [sample.tokens for sample in samples] == [
        [1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
    ]
    assert [sample.metadata["segment_index"] for sample in samples] == [0, 1, 2]
    assert len({sample.index for sample in samples}) == 1
    assert len({sample.group_index for sample in samples}) == 1


def test_real_65536_token_boundary_is_kept(monkeypatch):
    prompt_ids = list(range(65534))
    response_ids = [70000, 70001]
    samples = _submit(
        monkeypatch,
        {0: _step(prompt_ids, response_ids, "near limit", "r", -0.1)},
        max_tokens=65536,
    )

    assert len(samples) == 1
    assert len(samples[0].tokens) == 65536
    assert samples[0].tokens == prompt_ids + response_ids


def test_response_limited_segment_is_truncated_but_retained(monkeypatch):
    step = _step([1, 2], [3, 4], "prompt", "sampled", -0.2)
    step["hit_response_limit"] = True

    samples = _submit(monkeypatch, {0: step})

    assert len(samples) == 1
    assert samples[0].status.value == "truncated"
    assert samples[0].tokens == [1, 2, 3, 4]
    assert samples[0].loss_mask == [1, 1]
    assert samples[0].rollout_log_probs == [-0.2, -0.2]



def test_sdar_hint_accepts_fully_successful_rollout():
    tokenizer = _FakeTemplateTokenizer()
    messages = [{"role": "user", "content": "task"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, tools=None, tokenize=True, add_generation_prompt=True, reasoning_effort=None
    )
    pending = {
        0: {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "template_tools": None,
            "template_kwargs": {},
            "reasoning_effort": None,
        }
    }
    segment = {
        "agent_steps": [0],
        "tokens": prompt_ids + [7],
        "loss_mask": [1],
    }
    teacher_tokens, eligible, reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        {
            "rollout_not_fully_successful": False,
            "evidence_anchors": ["result must be valid"],
        },
        max_context_tokens=4096,
        infrastructure_failure=False,
        require_incomplete=False,
    )
    assert eligible is True
    assert reason == ""
    assert teacher_tokens[-1:] == [7]
    assert teacher_tokens != segment["tokens"]

def test_exact_trajectory_limit_is_kept(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 3, 4], [5], "p1 r1 tool", "r2", -0.2),
        },
        max_tokens=5,
    )
    assert len(samples) == 1
    assert len(samples[0].tokens) == 5


def test_oversized_new_segment_is_not_submitted(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 3, 4], [5], "p1 r1 tool", "r2", -0.2),
        },
        max_tokens=4,
    )
    assert [sample.tokens for sample in samples] == [[1, 2, 3]]
    assert all(len(sample.tokens) <= 4 for sample in samples)
class _FakeTemplateTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tools,
        tokenize,
        add_generation_prompt,
        reasoning_effort,
        **kwargs,
    ):
        text = "|".join(str(message.get("content", "")) for message in messages)
        return [ord(character) for character in text] + ([999] if add_generation_prompt else [])


def test_opsd_hint_rendering_preserves_segment_response_and_step_ids():
    tokenizer = _FakeTemplateTokenizer()
    messages = [{"role": "user", "content": "task"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort=None,
    )
    pending = {
        3: {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "template_tools": None,
            "template_kwargs": {},
            "reasoning_effort": None,
        }
    }
    segment = {
        "agent_steps": [3],
        "tokens": prompt_ids + [41, 42],
        "loss_mask": [1, 1],
    }
    feedback = {
        "rollout_not_fully_successful": True,
        "evidence_anchors": ["JSON must parse"],
    }

    teacher_tokens, eligible, reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        feedback,
        max_context_tokens=4096,
        infrastructure_failure=False,
    )

    assert eligible is True
    assert reason == ""
    assert teacher_tokens[-2:] == [41, 42]
    assert teacher_tokens != segment["tokens"]


def test_opsd_hint_overflow_and_missing_anchor_fall_back_to_grpo():
    tokenizer = _FakeTemplateTokenizer()
    messages = [{"role": "user", "content": "task"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort=None,
    )
    pending = {
        0: {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "template_tools": None,
            "template_kwargs": {},
            "reasoning_effort": None,
        }
    }
    segment = {
        "agent_steps": [0],
        "tokens": prompt_ids + [7],
        "loss_mask": [1],
    }

    _, eligible, reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        {
            "rollout_not_fully_successful": True,
            "evidence_anchors": ["required"],
        },
        max_context_tokens=8,
        infrastructure_failure=False,
    )
    assert eligible is False
    assert reason == "hint_overflow"

    _, eligible, reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        {
            "rollout_not_fully_successful": True,
            "evidence_anchors": [],
        },
        max_context_tokens=4096,
        infrastructure_failure=False,
    )
    assert eligible is False
    assert reason == "missing_evidence_anchors"

    _, eligible, reason = _prepare_opsd_segment(
        tokenizer,
        segment,
        pending,
        {
            "rollout_not_fully_successful": False,
            "evidence_anchors": ["required"],
        },
        max_context_tokens=4096,
        infrastructure_failure=False,
    )
    assert eligible is False
    assert reason == ""


def test_exact_trajectory_limit_is_kept(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 3, 4], [5], "p1 r1 tool", "r2", -0.2),
        },
        max_tokens=5,
    )
    assert len(samples) == 1
    assert len(samples[0].tokens) == 5


def test_oversized_new_segment_is_not_submitted(monkeypatch):
    samples = _submit(
        monkeypatch,
        {
            0: _step([1, 2], [3], "p1", "r1", -0.1),
            1: _step([1, 2, 3, 4], [5], "p1 r1 tool", "r2", -0.2),
        },
        max_tokens=4,
    )
    assert [sample.tokens for sample in samples] == [[1, 2, 3]]
    assert all(len(sample.tokens) <= 4 for sample in samples)


def test_matched_placebo_pair_preserves_identical_response_span():
    tokenizer = _FakeTemplateTokenizer()
    messages = [{"role": "user", "content": "task"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tools=None,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort=None,
    )
    pending = {
        0: {
            "messages": messages,
            "prompt_ids": prompt_ids,
            "template_tools": None,
            "template_kwargs": {},
            "reasoning_effort": None,
        }
    }
    response = [41, 42, 43]
    segment = {
        "agent_steps": [0],
        "tokens": prompt_ids + response,
        "loss_mask": [1, 1, 1],
    }
    real, placebo, eligible, reason = _prepare_opsd_pair(
        tokenizer,
        segment,
        pending,
        {
            "rollout_not_fully_successful": True,
            "evidence_anchors": [{"condition": "real condition"}],
            "placebo_evidence_anchors": [{"condition": "placebo condition"}],
        },
        max_context_tokens=4096,
        infrastructure_failure=False,
        require_incomplete=True,
        matched_placebo=True,
    )

    assert eligible is True
    assert reason == ""
    assert real[-len(response):] == response
    assert placebo[-len(response):] == response
    assert len(real) - len(response) != 0
    assert len(placebo) - len(response) != 0
    assert real != placebo
