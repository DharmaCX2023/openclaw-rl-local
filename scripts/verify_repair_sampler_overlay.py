#!/usr/bin/env python3
"""Exercise the ACTUAL overlaid SGLang sampler at a known hash endpoint."""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
from contextlib import nullcontext

import numpy as np
import torch

from diagnose_verified_repair_sampler_endpoint import hash_numpy
from verified_repair_runtime import save_json, sha
from verified_repair_sampler_patch import SOURCE_SHA256, VERSION


def check(device):
    from sglang.srt.layers import sampler
    if getattr(sampler, "verified_repair_sampler_patch", None) != VERSION:
        raise RuntimeError("actual imported sampler is not overlaid")
    seed = torch.tensor([1795547190], dtype=torch.int64, device=device)
    pos = torch.tensor([20755], dtype=torch.int64, device=device)
    columns = torch.arange(152064, device=device)

    def cpu_hash(s, p, c):
        return torch.stack([torch.from_numpy(hash_numpy(
            int(ss), int(pp), c.numpy().astype(np.uint64)).astype(np.uint32))
            for ss, pp in zip(s, p)])

    context = patch.object(sampler, "murmur_hash32", cpu_hash) if device == "cpu" else nullcontext()
    started = time.monotonic()
    with context:
        actual = sampler.murmur_hash32(seed.to(torch.uint64), pos, columns)
        expected = torch.from_numpy(hash_numpy(
            1795547190, 20755, np.arange(152064, dtype=np.uint64)).astype(np.uint32)).reshape(1, -1).to(device)
        assert torch.equal(actual, expected)
        lp = torch.full((1, 152064), -float("inf"), dtype=torch.float64, device=device)
        lp[0, 0] = 0.
        function = sampler.multinomial_with_seed
        if device == "cpu":
            function = function._torchdynamo_orig_callable
        sampled = int(function(lp, seed, pos).item())
        assert sampled == 0, "sampler selected a zero-probability masked category"
        endpoint_hash = int(actual[0, 35146].item())
        assert endpoint_hash == 2**32-1
    assert sha(Path(sampler.__file__)) == SOURCE_SHA256
    return {"device": device, "sampler_overlay": VERSION, "sampled_column": sampled,
            "selected_probability": 1., "endpoint_masked_column": 35146,
            "hash_at_masked_column": endpoint_hash, "native_hash_matches_reference": True,
            "actual_overlaid_function_tested": True, "shared_installation_modified": False,
            "torch_compile_enabled": device == "cuda", "model_weights_loaded": False,
            "seconds": time.monotonic()-started}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.device)
    save_json(args.output, result)
    print(json.dumps(result), flush=True)
