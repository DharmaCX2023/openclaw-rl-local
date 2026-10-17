#!/usr/bin/env python3
"""Build a complete-trajectory saved-batch subset for recomputation A/B."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-sample-tokens", type=int, default=45_000)
    args = parser.parse_args()

    payload = torch.load(args.input, weights_only=False)
    samples = payload["samples"]
    by_trajectory = defaultdict(list)
    for sample in samples:
        key = (int(sample.get("group_index", -1)), int(sample.get("index", -1)))
        by_trajectory[key].append(sample)

    kept_trajectory_keys = {
        key
        for key, trajectory_samples in by_trajectory.items()
        if max(len(sample.get("tokens") or []) for sample in trajectory_samples)
        <= args.max_sample_tokens
    }
    kept_samples = [
        sample
        for sample in samples
        if (int(sample.get("group_index", -1)), int(sample.get("index", -1)))
        in kept_trajectory_keys
    ]
    if not kept_samples:
        raise RuntimeError("complete-trajectory token limit removed every sample")

    kept_ids = {
        (int(sample.get("group_index", -1)), int(sample.get("index", -1)))
        for sample in kept_samples
    }
    for key, trajectory_samples in by_trajectory.items():
        present = sum(
            1
            for sample in kept_samples
            if (int(sample.get("group_index", -1)), int(sample.get("index", -1)))
            == key
        )
        if key in kept_ids and present != len(trajectory_samples):
            raise AssertionError(f"partial trajectory retained: {key}")

    output_payload = dict(payload)
    output_payload["samples"] = kept_samples
    output_payload["subset_metadata"] = {
        "source": str(args.input.resolve()),
        "rule": "retain complete trajectories whose every sample fits max_sample_tokens",
        "max_sample_tokens": args.max_sample_tokens,
        "original_samples": len(samples),
        "kept_samples": len(kept_samples),
        "kept_trajectories": len(kept_trajectory_keys),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)
    print(output_payload["subset_metadata"])


if __name__ == "__main__":
    main()
