#!/usr/bin/env python3
"""Build a TensorBoard view from an old prefix and a live resumed suffix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def copy_prefix(source: Path, destination: Path, max_step: int) -> None:
    writer = EventFileWriter(str(destination))
    try:
        for record in RawEventFileLoader(str(source)).Load():
            event = Event.FromString(record)
            if not event.summary.value or event.step <= max_step:
                writer.add_event(event)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    parser.add_argument("--prefix-max-step", type=int, required=True)
    parser.add_argument("--resume-min-step", type=int, required=True)
    args = parser.parse_args()

    args.destination_dir.mkdir(parents=True, exist_ok=False)
    sources = sorted(args.source_dir.glob("events.out.tfevents*"))
    if not sources:
        raise SystemExit(f"no event files in {args.source_dir}")

    old_sources = sources[:2]
    resume_sources = sources[2:]
    for source in old_sources:
        copy_prefix(source, args.destination_dir, args.prefix_max_step)

    # Resume files stay live while training writes to them. Symlinks let
    # TensorBoard observe appended records without rebuilding the view.
    for index, source in enumerate(resume_sources):
        link = args.destination_dir / f"events.out.tfevents.live_resume_{index}"
        os.symlink(source, link)

    print(
        f"built {args.destination_dir}: {len(old_sources)} prefix files through "
        f"step {args.prefix_max_step}, {len(resume_sources)} live resume files "
        f"from expected step {args.resume_min_step}"
    )


if __name__ == "__main__":
    main()
