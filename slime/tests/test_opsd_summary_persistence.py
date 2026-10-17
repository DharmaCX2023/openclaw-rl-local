import json
from types import SimpleNamespace

from slime.ray.rollout import RolloutManager


def test_opsd_summary_persists_scalars_without_large_arrays(tmp_path):
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager.args = SimpleNamespace(
        save_opsd_step_summary=str(tmp_path / "{rollout_id}.json")
    )
    manager.rollout_id = 3
    sample = SimpleNamespace(
        metadata={"task_id": "task_0001", "session_id": "train-task-3-7", "segment_index": 2},
        group_index=4,
        index=7,
        response_length=4,
        loss_mask=[1, 1, 0, 1],
        opsd_eligible=True,
        opsd_fallback_reason="",
        opsd_step_ids=[0, 0, -1, 3],
    )

    manager._save_opsd_step_summary(
        [sample],
        [-0.75],
        [0.25],
        [[0.5, -0.25]],
        [[0.6, 1.2]],
        [1.0],
    )

    output = tmp_path / "3.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["guidance_method"] == "opsd_absolute"
    assert payload["rollout_id"] == 3
    assert payload["records"] == [
        {
            "guidance_method": "opsd_absolute",
            "guidance_requested": True,
            "guidance_applied": True,
            "task_id": "task_0001",
            "session_id": "train-task-3-7",
            "group_index": 4,
            "trajectory_index": 7,
            "segment_index": 2,
            "raw_reward": 0.25,
            "advantage": -0.75,
            "response_token_count": 4,
            "loss_token_count": 3,
            "opsd_requested": True,
            "opsd_applied": True,
            "fallback_reason": "",
            "placebo_task_id": "",
            "steps": [
                {
                    "step_id": 0,
                    "loss_token_count": 2,
                    "teacher_minus_rollout_old_mean": 0.5,
                    "weight": 0.6000000238418579,
                    "original_advantage": -0.75,
                    "weighted_advantage": -0.45000001788139343,
                },
                {
                    "step_id": 3,
                    "loss_token_count": 1,
                    "teacher_minus_rollout_old_mean": -0.25,
                    "weight": 1.2000000476837158,
                    "original_advantage": -0.75,
                    "weighted_advantage": -0.9000000357627869,
                },
            ],
        }
    ]
    serialized = output.read_text(encoding="utf-8")
    for forbidden in ("token_ids", "log_probs", "routed_experts", "routing_tensor"):
        assert forbidden not in serialized


def test_matched_placebo_summary_persists_pair_and_advantages(tmp_path):
    manager_class = RolloutManager.__ray_metadata__.modified_class
    manager = manager_class.__new__(manager_class)
    manager.args = SimpleNamespace(
        save_opsd_step_summary=str(tmp_path / "{rollout_id}.json"),
        opsd_enable=True,
        sdar_enable=False,
        opsd_mode="centered",
        opsd_matched_placebo=True,
    )
    manager.rollout_id = 4
    sample = SimpleNamespace(
        metadata={
            "task_id": "task_real",
            "session_id": "train-task-4-0",
            "segment_index": 0,
        },
        group_index=0,
        index=1,
        response_length=2,
        loss_mask=[1, 1],
        opsd_eligible=True,
        opsd_fallback_reason="",
        opsd_placebo_task_id="task_placebo",
        opsd_step_ids=[2, 2],
    )

    manager._save_opsd_step_summary(
        [sample],
        [-0.5],
        [0.25],
        [[0.25]],
        [[0.8]],
        [1.0],
        [[-0.75]],
        [[-1.0]],
    )

    payload = json.loads((tmp_path / "4.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["guidance_method"] == "opsd_matched_placebo_centered"
    record = payload["records"][0]
    assert record["placebo_task_id"] == "task_placebo"
    assert record["steps"] == [
        {
            "step_id": 2,
            "loss_token_count": 2,
            "real_anchor_logprob_mean": -0.75,
            "placebo_anchor_logprob_mean": -1.0,
            "real_minus_placebo_mean": 0.25,
            "weight": 0.800000011920929,
            "original_advantage": -0.5,
            "weighted_advantage": -0.4000000059604645,
        }
    ]
