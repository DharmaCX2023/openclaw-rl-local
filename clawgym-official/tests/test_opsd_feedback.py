import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

import clawgym_rl_rollout as rollout_module
from clawgym_rl_rollout import (
    _build_opsd_feedback,
    _extract_opsd_teacher_logprobs,
    _load_opsd_evidence_anchors,
    _load_opsd_placebo,
    _score_opsd_placebo_segment,
    _score_opsd_segment,
    _score_opsd_negative_trajectories,
)
from clawgym_rl_api_server import _opsd_feedback_text
from opsd_evidence import parse_evidence_anchor_xml, render_teacher_feedback


def test_teacher_logprobs_align_to_exact_response_suffix():
    output = {
        "meta_info": {
            "input_token_logprobs": [
                [None, 10],
                [-0.3, 20],
                [-0.4, 21],
            ]
        }
    }

    assert _extract_opsd_teacher_logprobs(output, [20, 21]) == [-0.3, -0.4]


def test_teacher_logprobs_reject_token_mismatch():
    output = {
        "meta_info": {
            "input_token_logprobs": [
                [-0.3, 20],
                [-0.4, 99],
            ]
        }
    }

    with pytest.raises(ValueError, match="suffix mismatch"):
        _extract_opsd_teacher_logprobs(output, [20, 21])


def test_teacher_scoring_includes_last_prompt_token_for_finite_suffix(monkeypatch):
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 12],
                        [-0.3, 20],
                        [-0.4, 21],
                    ]
                }
            }

    def fake_post(url, *, json, headers, timeout):
        captured.update(json)
        return Response()

    monkeypatch.setattr("clawgym_rl_rollout.requests.post", fake_post)
    args = type(
        "Args",
        (),
        {"sglang_router_ip": "127.0.0.1", "sglang_router_port": 4333},
    )()
    sample = type(
        "SampleLike",
        (),
        {
            "opsd_teacher_tokens": [10, 11, 12, 20, 21],
            "response_length": 2,
            "tokens": [10, 11, 12, 20, 21],
            "metadata": {"session_id": "session"},
        },
    )()

    assert _score_opsd_segment(args, sample) == [-0.3, -0.4]
    assert captured["logprob_start_len"] == 2


def test_anchor_generator_xml_is_parsed_to_structured_storage():
    record = parse_evidence_anchor_xml(
        """
        <evidences>
          <evidence>
            <condition>output/result.json must be valid JSON.</condition>
            <artifact>output/result.json</artifact>
            <verified_fact>A verifier accepted a parseable JSON object at this path.</verified_fact>
            <explanation>This is a necessary output condition.</explanation>
          </evidence>
        </evidences>
        """,
        task_id="task_1411",
    )
    assert record["schema_version"] == "clawgym_evidence_anchor_v1"
    assert record["anchors"][0]["artifact"] == "output/result.json"


def test_anchor_xml_rejects_entities_and_missing_fields():
    with pytest.raises(ValueError, match="DTD and entities"):
        parse_evidence_anchor_xml(
            '<!DOCTYPE x [<!ENTITY y "z">]><evidences/>', task_id="task"
        )
    with pytest.raises(ValueError, match="incomplete evidence"):
        parse_evidence_anchor_xml(
            "<evidences><evidence><condition>x</condition></evidence></evidences>",
            task_id="task",
        )


def test_structured_jsonl_load_and_plain_text_teacher_render(
    tmp_path, monkeypatch
):
    anchor_file = tmp_path / "anchors.jsonl"
    anchor_file.write_text(
        json.dumps(
            {
                "schema_version": "clawgym_evidence_anchor_v1",
                "task_id": "task_1411",
                "anchors": [
                    {
                        "condition": "output/result.json must be valid JSON.",
                        "artifact": "output/result.json",
                        "verified_fact": "The artifact was parseable.",
                        "explanation": "Parsing is required.",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE", str(anchor_file)
    )
    anchors = _load_opsd_evidence_anchors("task_1411")
    rendered = _opsd_feedback_text(
        {
            "rollout_not_fully_successful": True,
            "student_final_answer": "must never be rendered",
            "evidence_anchors": anchors,
        }
    )
    assert "Teacher-only auxiliary context" in rendered
    assert "provided only as auxiliary context" in rendered
    assert "did not fully satisfy the task verifier" not in rendered
    assert "Use this evidence when solving" not in rendered
    assert "Some intermediate actions may still be valid" not in rendered
    assert "Condition: output/result.json must be valid JSON." in rendered
    assert "Relevant artifact: output/result.json" in rendered
    assert "must never be rendered" not in rendered
    assert "final answer" not in rendered.lower()
    assert "Submitted artifacts:" not in rendered
    assert "verifier score" not in rendered.lower()
    assert "<teacher_feedback>" not in rendered
    assert "<evidence>" not in rendered


def test_feedback_status_is_derived_from_full_verifier_credit(monkeypatch):
    monkeypatch.setattr(
        "clawgym_rl_rollout._load_opsd_evidence_anchors",
        lambda task_id: [{"condition": f"condition for {task_id}"}],
    )
    task = type("Task", (), {"task_id": "task_1411"})()

    failed = _build_opsd_feedback(task, 0.625)
    successful = _build_opsd_feedback(task, 1.0)

    assert failed["rollout_not_fully_successful"] is True
    assert successful["rollout_not_fully_successful"] is False
    assert "student_final_answer" not in failed


def test_sdar_teacher_feedback_is_neutral_and_accepts_success():
    rendered = render_teacher_feedback(
        {
            "teacher_guidance_mode": "sdar",
            "rollout_not_fully_successful": False,
            "evidence_anchors": [{"condition": "output/result.json must parse"}],
        }
    )
    assert "Condition: output/result.json must parse" in rendered
    assert "did not fully satisfy" not in rendered
    assert "intermediate actions" not in rendered


class _ScoredSample:
    def __init__(self, index, reward):
        self.group_index = 0
        self.index = index
        self.response_length = 2
        self.opsd_eligible = True
        self.opsd_fallback_reason = ""
        self.reward = reward
        self.loss_mask = [1, 1]
        self.opsd_step_ids = [0, 1]
        self.rollout_log_probs = [-1.0, -1.0]

    def get_reward_value(self, args):
        return self.reward


def test_sdar_scores_success_and_failure_trajectories(monkeypatch):
    samples = [_ScoredSample(0, 0.25), _ScoredSample(1, 1.0)]
    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_segment",
        lambda args, sample: [-0.2, -0.3],
    )
    args = type("Args", (), {
        "opsd_enable": False, "sdar_enable": True,
        "opsd_teacher_concurrency": 2,
    })()
    _score_opsd_negative_trajectories(args, [samples])
    assert all(sample.opsd_apply for sample in samples)
    assert all(sample.opsd_teacher_log_probs == [-0.2, -0.3] for sample in samples)


def test_opsd_only_scores_below_group_mean(monkeypatch):
    failed = _ScoredSample(0, 0.25)
    successful = _ScoredSample(1, 1.0)
    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_segment",
        lambda args, sample: [-0.2, -0.3],
    )
    args = type("Args", (), {
        "opsd_enable": True, "sdar_enable": False,
        "opsd_teacher_concurrency": 2,
    })()
    _score_opsd_negative_trajectories(args, [[failed, successful]])
    assert failed.opsd_apply is True
    assert successful.opsd_apply is False


@pytest.mark.parametrize("opsd_mode", ["agentopsd", "bb_agentopsd"])
def test_agentopsd_scores_both_sides_of_group_advantage(monkeypatch, opsd_mode):
    failed = _ScoredSample(0, 0.25)
    successful = _ScoredSample(1, 1.0)
    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_segment",
        lambda args, sample: [-0.2, -0.3],
    )
    args = type("Args", (), {
        "opsd_enable": True,
        "sdar_enable": False,
        "opsd_mode": opsd_mode,
        "opsd_teacher_concurrency": 2,
    })()
    _score_opsd_negative_trajectories(args, [[failed, successful]])
    assert failed.opsd_apply is True
    assert successful.opsd_apply is True


def test_repeat_teacher_scoring_records_stability_without_changing_primary(monkeypatch):
    sample = _ScoredSample(0, 0.5)
    monkeypatch.setenv("OPENCLAW_GUIDANCE_REPEAT_SCORE", "1")
    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_segment",
        lambda args, row: [-0.2, -0.3],
    )
    rollout_module._TRAIN_METRICS.clear()
    args = type("Args", (), {
        "opsd_enable": False, "sdar_enable": True,
        "opsd_teacher_concurrency": 1,
    })()
    _score_opsd_negative_trajectories(args, [[sample]])
    assert sample.opsd_teacher_log_probs == [-0.2, -0.3]
    assert rollout_module._TRAIN_METRICS["guidance_repeat_scored_tokens"] == 2
    assert rollout_module._TRAIN_METRICS["guidance_repeat_token_logprob_mae"] == 0.0
    assert rollout_module._TRAIN_METRICS["guidance_repeat_step_support_pearson"] == pytest.approx(1.0)
    assert rollout_module._TRAIN_METRICS["guidance_repeat_step_rank_spearman"] == pytest.approx(1.0)
    rollout_module._TRAIN_METRICS.clear()


def test_matched_placebo_mapping_loads_nonself_anchor(tmp_path, monkeypatch):
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "task_a", "anchors": [{"condition": "A"}]}),
                json.dumps({"task_id": "task_b", "anchors": [{"condition": "B"}]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "schema_version": "clawgym_matched_placebo_v1",
                "tasks": {
                    "task_a": {
                        "placebos": [
                            {
                                "task_id": "task_b",
                                "token_difference": 0,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE", str(anchors))
    monkeypatch.setenv("OPENCLAW_OPSD_PLACEBO_MAPPING_FILE", str(mapping))
    monkeypatch.setenv("OPENCLAW_OPSD_PLACEBO_INDEX", "0")

    task_id, loaded, metadata = _load_opsd_placebo("task_a")

    assert task_id == "task_b"
    assert loaded == [
        {
            "condition": "B",
            "artifact": "",
            "verified_fact": "",
            "explanation": "",
        }
    ]
    assert metadata["token_difference"] == 0


def test_real_and_placebo_teacher_scoring_share_routing_key(monkeypatch):
    headers_seen = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "meta_info": {
                    "input_token_logprobs": [
                        [None, 12],
                        [-0.3, 20],
                        [-0.4, 21],
                    ]
                }
            }

    def fake_post(url, *, json, headers, timeout):
        headers_seen.append(headers["X-SMG-Routing-Key"])
        return Response()

    monkeypatch.setattr("clawgym_rl_rollout.requests.post", fake_post)
    args = type(
        "Args",
        (),
        {"sglang_router_ip": "127.0.0.1", "sglang_router_port": 4333},
    )()
    sample = type(
        "SampleLike",
        (),
        {
            "opsd_teacher_tokens": [10, 11, 12, 20, 21],
            "opsd_placebo_teacher_tokens": [30, 31, 12, 20, 21],
            "response_length": 2,
            "tokens": [10, 11, 12, 20, 21],
            "metadata": {"session_id": "session"},
        },
    )()

    _score_opsd_segment(args, sample)
    _score_opsd_placebo_segment(args, sample)

    assert headers_seen == ["session", "session"]


def test_matched_placebo_scoring_is_atomic_per_logical_trajectory(monkeypatch):
    first = _ScoredSample(0, 0.25)
    second = _ScoredSample(0, 0.25)
    successful = _ScoredSample(1, 1.0)
    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_segment",
        lambda args, sample: [-0.2, -0.3],
    )

    def fail_placebo(args, sample):
        raise RuntimeError("placebo unavailable")

    monkeypatch.setattr(
        "clawgym_rl_rollout._score_opsd_placebo_segment", fail_placebo
    )
    args = type(
        "Args",
        (),
        {
            "opsd_enable": True,
            "sdar_enable": False,
            "opsd_matched_placebo": True,
            "opsd_teacher_concurrency": 2,
        },
    )()

    _score_opsd_negative_trajectories(args, [[first, second, successful]])

    assert first.opsd_apply is False
    assert second.opsd_apply is False
    assert "placebo_teacher_scoring_failed" in first.opsd_fallback_reason
    assert "placebo_teacher_scoring_failed" in second.opsd_fallback_reason
