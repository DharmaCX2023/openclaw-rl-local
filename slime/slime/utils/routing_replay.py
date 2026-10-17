import os
import torch


ROUTING_REPLAY = None


def set_routing_replay(replay):
    global ROUTING_REPLAY
    ROUTING_REPLAY = replay


class RoutingReplay:
    all_routing_replays = []

    def __init__(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.top_probs_list = []
        RoutingReplay.all_routing_replays.append(self)

    def record(self, top_indices, top_probs=None):
        # offload top_indices to CPU pinned memory
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)
        if top_probs is not None:
            probs_buf = torch.empty_like(top_probs, device="cpu", pin_memory=True)
            probs_buf.copy_(top_probs.detach())
            self.top_probs_list.append(probs_buf)

    def pop_forward(self):
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        # SGLang serializes routed expert IDs as int32 to keep rollout payloads
        # compact. torch.gather requires int64 indices, so cast only the active
        # microbatch while moving it to GPU instead of doubling the full CPU
        # replay buffer and saved rollout size.
        return top_indices.to(torch.cuda.current_device(), dtype=torch.long)

    def pop_backward(self):
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return top_indices.to(torch.cuda.current_device(), dtype=torch.long)

    def pop_probs_forward(self):
        return self.top_probs_list[self.forward_index - 1].to(torch.cuda.current_device())

    def pop_probs_backward(self):
        return self.top_probs_list[self.backward_index - 1].to(torch.cuda.current_device())

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []
        self.top_probs_list = []

    def clear_forward(self):
        self.forward_index = 0

    @staticmethod
    def clear_all():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear()

    @staticmethod
    def clear_all_forward():
        for replay in RoutingReplay.all_routing_replays:
            replay.clear_forward()


def get_routing_replay_compute_topk(old_compute_topk):
    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            routing_replay_stage = os.environ["ROUTING_REPLAY_STAGE"]
            if routing_replay_stage == "fallthrough":
                return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if routing_replay_stage == "record":
                probs, top_indices = old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
                fixed_probs = os.environ.get("SLIME_ROUTING_REPLAY_FIXED_PROBS", "0") == "1"
                ROUTING_REPLAY.record(top_indices, probs if fixed_probs else None)
            elif routing_replay_stage == "replay_forward":
                top_indices = ROUTING_REPLAY.pop_forward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"[{torch.distributed.get_rank()}] top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                if os.environ.get("SLIME_ROUTING_REPLAY_FIXED_PROBS", "0") == "1":
                    probs = ROUTING_REPLAY.pop_probs_forward()
                else:
                    probs = scores.gather(1, top_indices)
            elif routing_replay_stage == "replay_backward":
                top_indices = ROUTING_REPLAY.pop_backward()
                assert (
                    top_indices.shape[0] == scores.shape[0] and top_indices.shape[1] == topk
                ), f"top_indices shape {top_indices.shape} does not match scores shape {scores.shape} and topk {topk}"
                if os.environ.get("SLIME_ROUTING_REPLAY_FIXED_PROBS", "0") == "1":
                    probs = ROUTING_REPLAY.pop_probs_backward()
                else:
                    probs = scores.gather(1, top_indices)
            return probs, top_indices
        else:
            return old_compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

    return compute_topk


def register_routing_replay(module):
    if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
        module.routing_replay = RoutingReplay()

        def pre_forward_hook(*args, **kwargs):
            set_routing_replay(module.routing_replay)

        module.register_forward_pre_hook(pre_forward_hook)
