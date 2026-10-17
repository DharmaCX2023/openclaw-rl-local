#!/usr/bin/env python3
"""Run the existing fixed-seed ClawGym gate while retaining terminal workspaces.

This is deliberately a wrapper instead of a change to the production rollout
path.  It installs narrow audit hooks:

* after the official reward has been computed, capture the raw criterion JSON;
* immediately before the rollout container is destroyed, freeze it and stream
  a gzip-compressed terminal workspace back to the local machine.
* after tool results arrive but before the next policy generation, record an
  exact workspace manifest delta linked to the observed tool-call IDs.

The archive contains the agent-visible workspace only.  Reward code and the
temporary reward payload are excluded so later artifact crossover cannot edit
or accidentally learn from verifier assets.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = REPO_ROOT / "clawgym-official"
for path in (str(OFFICIAL_DIR), str(REPO_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import clawgym_rl_rollout as rollout  # noqa: E402
import clawgym_rl_api_server as proxy_api  # noqa: E402
import run_fixed_seed_8x8_rollout_gate as gate  # noqa: E402
from workspace_provenance import WorkspaceProvenanceObserver  # noqa: E402


_PENDING_LOCK = threading.Lock()
_PENDING: dict[str, dict[str, Any]] = {}
_ARCHIVE_ROOT: Path | None = None
_ARCHIVE_MAX_BYTES = 2 * 1024**3
_PROVENANCE: WorkspaceProvenanceObserver | None = None
_START_SEMAPHORE: threading.BoundedSemaphore | None = None
_STARTING_LOCK = threading.Lock()
_STARTING_CONTAINERS: set[str] = set()


def _release_start_slot(runtime: rollout.ContainerRuntime) -> None:
    """Release a startup slot once, after the gateway is ready or has failed."""
    if _START_SEMAPHORE is None:
        return
    with _STARTING_LOCK:
        if runtime.container_name not in _STARTING_CONTAINERS:
            return
        _STARTING_CONTAINERS.remove(runtime.container_name)
    _START_SEMAPHORE.release()


def _parse_json_stdout(stdout: bytes) -> tuple[dict[str, Any] | None, str | None]:
    text = stdout.decode(errors="replace").strip()
    if not text:
        return None, "criterion checker produced no stdout"
    candidates = [text]
    candidates.extend(reversed([line.strip() for line in text.splitlines() if line.strip()]))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, None
        return None, f"criterion checker returned {type(value).__name__}, not an object"
    return None, "criterion checker stdout was not JSON"


def _capture_criterion_json(
    task: rollout.TaskSpec,
    runtime: rollout.ContainerRuntime,
) -> tuple[dict[str, Any] | None, str | None]:
    if not (task.task_dir / "reward" / "check.py").is_file():
        return None, "reward/check.py is unavailable"

    checker = "/root/.openclaw/workspace/reward/check.py"
    workspace = "/root/.openclaw/workspace"
    if runtime.remote_root is not None:
        command = " ".join(
            shlex.quote(part)
            for part in [
                "docker",
                "exec",
                "--user",
                "root",
                runtime.container_name,
                "python3",
                checker,
                workspace,
            ]
        )
        result = rollout._remote_run(command, timeout=180, check=False)
        if result.returncode != 0:
            error = result.stderr.decode(errors="replace").strip()[:1000]
            return None, f"criterion checker exited {result.returncode}: {error}"
        return _parse_json_stdout(result.stdout)

    if runtime.use_chroot:
        result = subprocess.run(
            [
                "unshare",
                "--user",
                "--map-root-user",
                "--",
                "chroot",
                str(runtime.chroot_rootfs),
                "python3",
                checker,
                workspace,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
    else:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "--user",
                "root",
                runtime.container_name,
                "python3",
                checker,
                workspace,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
        )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()[:1000]
        return None, f"criterion checker exited {result.returncode}: {error}"
    return _parse_json_stdout(result.stdout)


def _stream_remote_workspace(runtime: rollout.ContainerRuntime, destination: Path) -> None:
    assert runtime.remote_root is not None
    quoted_workspace = shlex.quote(f"{runtime.remote_root}/workspace")
    command = (
        "tar -C "
        + quoted_workspace
        + " --exclude=./reward --exclude=./.openclaw_reward_payload.json "
        + "-czf - ."
    )
    with destination.open("wb") as output:
        result = subprocess.run(
            rollout._ssh_base() + [command],
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=600,
        )
    if result.returncode != 0:
        error = result.stderr.decode(errors="replace").strip()[:2000]
        raise RuntimeError(f"remote workspace tar failed ({result.returncode}): {error}")


def _archive_local_workspace(runtime: rollout.ContainerRuntime, destination: Path) -> None:
    with tarfile.open(destination, mode="w:gz", dereference=False) as archive:
        for child in sorted(runtime.workspace_dir.iterdir(), key=lambda item: item.name):
            if child.name in {"reward", ".openclaw_reward_payload.json"}:
                continue
            archive.add(child, arcname=child.name, recursive=True)


def _archive_workspace(runtime: rollout.ContainerRuntime, record: dict[str, Any]) -> None:
    if _ARCHIVE_ROOT is None:
        raise RuntimeError("mosaic archive root was not configured")
    _ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    safe_task = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(record["task_id"])
    )
    stem = f"{runtime.session_id}__{safe_task}"
    destination = _ARCHIVE_ROOT / f"{stem}.tar.gz"
    metadata_path = _ARCHIVE_ROOT / f"{stem}.json"
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"mosaic archive already exists for {stem}")

    started = time.monotonic()
    with tempfile.NamedTemporaryFile(
        dir=_ARCHIVE_ROOT, prefix=f".{stem}.", suffix=".tar.gz.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        if runtime.remote_root is not None:
            pause = rollout._remote_run(
                "docker pause " + shlex.quote(runtime.container_name),
                timeout=30,
                check=False,
            )
            if pause.returncode != 0:
                error = pause.stderr.decode(errors="replace").strip()[:1000]
                raise RuntimeError(f"could not freeze rollout container: {error}")
            _stream_remote_workspace(runtime, temporary)
        else:
            if not runtime.use_chroot:
                pause = subprocess.run(
                    ["docker", "pause", runtime.container_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                if pause.returncode != 0:
                    error = pause.stderr.decode(errors="replace").strip()[:1000]
                    raise RuntimeError(f"could not freeze rollout container: {error}")
            _archive_local_workspace(runtime, temporary)

        archive_bytes = temporary.stat().st_size
        if archive_bytes > _ARCHIVE_MAX_BYTES:
            raise RuntimeError(
                f"terminal workspace archive is {archive_bytes} bytes, above "
                f"the configured {_ARCHIVE_MAX_BYTES}-byte limit"
            )
        digest = hashlib.sha256()
        with temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        temporary.replace(destination)
        metadata = dict(record)
        metadata.update(
            {
                "archive": str(destination),
                "archive_bytes": archive_bytes,
                "archive_sha256": digest.hexdigest(),
                "capture_seconds": round(time.monotonic() - started, 6),
                "workspace_frozen_before_archive": True,
                "reward_assets_excluded": True,
            }
        )
        temp_metadata = metadata_path.with_suffix(".json.tmp")
        temp_metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_metadata.replace(metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def _install_hooks() -> None:
    original_reward = rollout._run_reward
    original_stop = rollout._stop_container
    original_start = rollout._start_container
    original_wait_for_chat_ready = rollout._wait_for_chat_ready
    original_handle_request = gate.ClawGymRLAPIServer._handle_request

    def provenance_start(*args, **kwargs):
        if _START_SEMAPHORE is not None:
            _START_SEMAPHORE.acquire()
        try:
            runtime = original_start(*args, **kwargs)
        except BaseException:
            if _START_SEMAPHORE is not None:
                _START_SEMAPHORE.release()
            raise
        if _START_SEMAPHORE is not None:
            with _STARTING_LOCK:
                _STARTING_CONTAINERS.add(runtime.container_name)
        if _PROVENANCE is not None:
            try:
                _PROVENANCE.register(runtime)
            except Exception as exc:
                # Instrumentation is audit-only and must never leak a live
                # container or change rollout completion semantics.
                print(
                    "[mosaic-provenance] baseline failed for "
                    f"{runtime.session_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return runtime

    def limited_wait_for_chat_ready(runtime):
        try:
            return original_wait_for_chat_ready(runtime)
        finally:
            # The slot protects only prepare/docker-run/gateway-init and is
            # returned once this container is healthy or definitively failed.
            _release_start_slot(runtime)

    async def provenance_handle_request(self, body, session_id, turn_type):
        if _PROVENANCE is not None and turn_type == "main":
            try:
                messages = proxy_api._require_list(body["messages"], "messages")
                tool_results = proxy_api._collect_new_tool_results(messages)
                if tool_results:
                    await asyncio.to_thread(
                        _PROVENANCE.observe,
                        self,
                        session_id,
                        tool_results,
                    )
            except Exception as exc:
                # A recording failure is visible in stderr, but cannot affect
                # the policy request or the environment transition.
                print(
                    "[mosaic-provenance] event failed for "
                    f"{session_id}: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return await original_handle_request(self, body, session_id, turn_type)

    def recording_reward(
        task: rollout.TaskSpec,
        runtime: rollout.ContainerRuntime,
        final_message: str,
    ) -> float:
        reward = original_reward(task, runtime, final_message)
        criterion_vector, criterion_error = _capture_criterion_json(task, runtime)
        with _PENDING_LOCK:
            _PENDING[runtime.container_name] = {
                "task_id": task.task_id,
                "task_dir": str(task.task_dir),
                "session_id": runtime.session_id,
                "container_name": runtime.container_name,
                "official_reward": reward,
                "criterion_vector": criterion_vector,
                "criterion_capture_error": criterion_error,
                "captured_at_unix": time.time(),
            }
        return reward

    def archiving_stop(
        runtime: rollout.ContainerRuntime,
        log_dir: Path | None,
    ) -> None:
        with _PENDING_LOCK:
            record = _PENDING.pop(runtime.container_name, None)
        archive_error: Exception | None = None
        try:
            if record is None:
                # The task failed before reward calculation. Raising from this
                # cleanup hook would mask the original rollout/HTTP exception
                # with an uninformative "missing reward metadata" error.
                print(
                    "[mosaic] no reward metadata; terminal workspace was not "
                    f"archived for {runtime.container_name}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                _archive_workspace(runtime, record)
        except Exception as exc:  # cleanup must still run
            archive_error = exc
        finally:
            # Fallback for a future path that stops a container before its
            # readiness check. _release_start_slot is intentionally idempotent.
            _release_start_slot(runtime)
            original_stop(runtime, log_dir)
        if archive_error is not None:
            raise archive_error

    rollout._run_reward = recording_reward
    rollout._stop_container = archiving_stop
    rollout._start_container = provenance_start
    rollout._wait_for_chat_ready = limited_wait_for_chat_ready
    gate.ClawGymRLAPIServer._handle_request = provenance_handle_request


def _parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mosaic-archive-dir", type=Path, required=True)
    parser.add_argument(
        "--mosaic-archive-max-gib", type=float, default=2.0
    )
    return parser.parse_known_args()


def main() -> int:
    global _ARCHIVE_ROOT, _ARCHIVE_MAX_BYTES, _PROVENANCE, _START_SEMAPHORE
    wrapper_args, remaining = _parse_wrapper_args()
    _ARCHIVE_ROOT = wrapper_args.mosaic_archive_dir.resolve()
    _ARCHIVE_MAX_BYTES = int(wrapper_args.mosaic_archive_max_gib * 1024**3)
    if _ARCHIVE_MAX_BYTES <= 0:
        raise ValueError("--mosaic-archive-max-gib must be positive")
    _ARCHIVE_ROOT.mkdir(parents=True, exist_ok=False)
    _PROVENANCE = WorkspaceProvenanceObserver(
        _ARCHIVE_ROOT.parent / "workspace_provenance.jsonl",
        rollout._remote_run,
    )
    start_concurrency = int(
        os.environ.get("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "4")
    )
    if start_concurrency <= 0:
        raise ValueError(
            "OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY must be positive"
        )
    _START_SEMAPHORE = threading.BoundedSemaphore(start_concurrency)
    print(
        "[mosaic-start-limit] at most "
        f"{start_concurrency} containers may start before gateway readiness",
        flush=True,
    )
    # Future artifact-credit extraction must use the exact IDs sampled by
    # SGLang, never silently substitute a tokenizer round-trip.
    os.environ.setdefault("OPENCLAW_RECORD_RESPONSE_IDS", "1")
    _install_hooks()
    sys.argv = [sys.argv[0], *remaining]
    return gate.main()


if __name__ == "__main__":
    raise SystemExit(main())
