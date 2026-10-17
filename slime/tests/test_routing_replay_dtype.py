import torch

from slime.utils.routing_replay import RoutingReplay


def test_replayed_sglang_int32_expert_ids_become_gather_indices(monkeypatch):
    replay = RoutingReplay()
    replay.top_indices_list = [torch.tensor([[1, 3]], dtype=torch.int32)]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")

    forward = replay.pop_forward()
    replay.forward_index = 0
    replay.backward_index = 0
    backward = replay.pop_backward()

    assert forward.dtype == torch.int64
    assert backward.dtype == torch.int64
    scores = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    gathered = scores[:1].gather(1, forward)
    assert gathered.tolist() == [[1.0, 3.0]]
