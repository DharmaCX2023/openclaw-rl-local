"""Load a Megatron checkpoint, sync it to SGLang, and run one full eval."""

import os

import ray

from slime.ray.placement_group import (
    create_placement_groups,
    create_rollout_manager,
    create_training_models,
)
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, init_tracking


def evaluate(args) -> None:
    assert not args.colocate, "Colocation is not supported by this eval runner."
    train_step = int(os.environ["OPENCLAW_EVAL_SEMANTIC_STEP"])
    framework_iteration = train_step - 1

    configure_logger()
    placement_groups = create_placement_groups(args)
    init_tracking(args)
    rollout_manager, _ = create_rollout_manager(
        args, placement_groups["rollout"], placement_groups.get("prm")
    )
    actor_model, _, _ = create_training_models(
        args, placement_groups, rollout_manager
    )

    try:
        if args.start_rollout_id != train_step:
            raise RuntimeError(
                "checkpoint/eval step mismatch: "
                f"start_rollout_id={args.start_rollout_id}, train_step={train_step}"
            )
        actor_model.update_weights()
        ray.get(
            rollout_manager.eval.remote(
                rollout_id=framework_iteration,
                train_step=train_step,
            )
        )
    finally:
        ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    evaluate(parse_args())
