#!/usr/bin/env python3
"""Retain semantic N-step milestones plus the newest committed checkpoint."""

import argparse
import json
import re
import shutil
import time
from pathlib import Path


def committed(root: Path, iteration: int) -> bool:
    path = root / f"iter_{iteration:07d}"
    required = ("common.pt", ".metadata", "metadata.json")
    if path.is_symlink() or not path.is_dir():
        return False
    if not all((path / name).is_file() and (path / name).stat().st_size for name in required):
        return False
    try:
        json.loads((path / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    shards = list(path.glob("*.distcp"))
    return len(shards) >= 8 and all(shard.stat().st_size for shard in shards)


def prune(root: Path, retain_steps: int) -> dict:
    tracker = root / "latest_checkpointed_iteration.txt"
    try:
        latest = int(tracker.read_text().strip())
    except (OSError, ValueError):
        return {"latest": None, "removed": []}
    if not committed(root, latest):
        return {"latest": latest, "removed": []}

    removed = []
    for path in sorted(root.glob("iter_*")):
        match = re.fullmatch(r"iter_(\d{7})", path.name)
        if not match or path.is_symlink():
            continue
        iteration = int(match.group(1))
        semantic_step = iteration + 1
        # The tracker advances only after a checkpoint is fully committed.
        # Anything newer than the tracker may be actively written; never touch it.
        if iteration >= latest or semantic_step % retain_steps == 0:
            continue
        if tracker.read_text().strip() != str(latest):
            break
        shutil.rmtree(path)
        removed.append(path.name)
    return {"latest": latest, "removed": removed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--retain-steps", type=int, default=20)
    parser.add_argument("--final-iteration", type=int, default=99)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)

    while True:
        record = {"time": time.time(), **prune(args.root, args.retain_steps)}
        (args.root / "checkpoint_retention_status.json").write_text(
            json.dumps(record, indent=2)
        )
        if record["removed"]:
            print(json.dumps(record), flush=True)
        if record["latest"] == args.final_iteration and committed(
            args.root, args.final_iteration
        ):
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
