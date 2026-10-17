#!/usr/bin/env python3
"""Search for strict file-level mosaics in retained ClawGym workspaces.

The Phase-0 test is intentionally conservative: an artifact atom is one exact
terminal file version (bytes + mode) produced by an on-policy rollout.  A
candidate may choose at most one observed version per path.  The analyzer never
asks an LLM to repair a conflict and never invents file contents.  This gives a
clean lower bound on cross-rollout complementarity before implementing JSON,
CSV, Markdown, or AST-level atoms.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any


@dataclass(frozen=True)
class FileVersion:
    key: str
    kind: str
    mode: int
    source_path: Path | None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        safe_members = []
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"unsafe archive member {member.name!r}")
            if member.issym() or member.islnk():
                relative = member.name.removeprefix("./")
                if relative == "output" or relative.startswith("output/"):
                    raise ValueError(
                        "output symlink/hardlink is outside conservative Phase-0: "
                        f"{member.name}"
                    )
                # Virtual environments and other runtime scaffolding are not
                # artifact atoms. Skipping their links is safer than resolving
                # host-absolute targets and still permits verifier replay from
                # the retained regular task inputs and output files.
                continue
            safe_members.append(member)
        archive.extractall(destination, members=safe_members, filter="data")


def _copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, copy_function=shutil.copy2)
        elif child.is_file():
            shutil.copy2(child, target)


def _build_initial_workspace(task_dir: Path, destination: Path) -> dict[str, Any]:
    entry = json.loads((task_dir / "data_entry.json").read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    input_files = task_dir / "input_files"
    if input_files.is_dir():
        mount = entry.get("input_mount_dir", "/root/.openclaw/workspace")
        root = PurePosixPath("/root/.openclaw/workspace")
        mount_path = PurePosixPath(mount)
        if mount_path == root:
            target = destination
        else:
            try:
                relative = mount_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"input_mount_dir outside workspace: {mount}") from exc
            target = destination.joinpath(*relative.parts)
        _copy_tree_contents(input_files, target)
    return entry


def _manifest(root: Path, max_file_bytes: int) -> dict[str, FileVersion]:
    result: dict[str, FileVersion] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if not relative or relative == "reward" or relative.startswith("reward/"):
            continue
        if relative == ".openclaw_reward_payload.json":
            continue
        if path.is_symlink():
            raise ValueError(f"workspace contains symlink: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"workspace contains special file: {relative}")
        size = path.stat().st_size
        if size > max_file_bytes:
            raise ValueError(
                f"workspace file {relative} is {size} bytes, above limit {max_file_bytes}"
            )
        mode = path.stat().st_mode & 0o777
        digest = _sha256_file(path)
        key = f"file:{mode:o}:{size}:{digest}"
        result[relative] = FileVersion(key, "file", mode, path)
    return result


def _absent_version() -> FileVersion:
    return FileVersion("absent", "absent", 0, None)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _materialize(
    initial_root: Path,
    destination: Path,
    paths: list[str],
    state: tuple[str, ...],
    versions: dict[str, dict[str, FileVersion]],
) -> None:
    shutil.copytree(initial_root, destination, copy_function=shutil.copy2)
    for relative, version_key in zip(paths, state, strict=True):
        target = destination / relative
        if target.exists() or target.is_symlink():
            _remove_path(target)
        version = versions[relative][version_key]
        if version.kind == "absent":
            continue
        assert version.source_path is not None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(version.source_path, target)
        target.chmod(version.mode)


def _parse_checker_json(stdout: str) -> dict[str, float]:
    text = stdout.strip()
    candidates = [text]
    candidates.extend(reversed([line.strip() for line in text.splitlines() if line.strip()]))
    value: Any = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        break
    if not isinstance(value, dict) or not value:
        raise ValueError("checker did not return a non-empty JSON object")
    result = {}
    for key, item in value.items():
        if isinstance(item, bool):
            result[str(key)] = float(item)
        elif isinstance(item, (int, float)) and math.isfinite(float(item)):
            result[str(key)] = float(item)
        else:
            raise ValueError(f"criterion {key!r} is not finite numeric/bool")
    return result


def _run_checker(
    task_dir: Path,
    workspace: Path,
    timeout: int,
    workspace_mode: str = "absolute",
) -> tuple[float, dict[str, float], float]:
    # The checker runs with ``cwd=task_dir`` because some task-local verifiers
    # load auxiliary assets relative to their task directory. Always pass an
    # absolute workspace path: Phase-0 output directories may be supplied as
    # relative paths, which would otherwise be reinterpreted below task_dir and
    # silently score a nonexistent workspace as all-false.
    task_dir = task_dir.resolve()
    workspace = workspace.resolve()
    checker = task_dir / "reward" / "check.py"
    if not checker.is_file():
        raise FileNotFoundError(f"missing checker: {checker}")
    started = time.monotonic()
    if workspace_mode == "absolute":
        checker_cwd = task_dir
        workspace_argument = str(workspace)
    elif workspace_mode == "relative":
        # Some online artifacts legitimately record canonical absolute paths
        # under /root/.openclaw/workspace. Passing an offline temp directory as
        # an absolute checker argument can make a path-sensitive verifier fail
        # even though the exact artifact passed online. Running from the
        # materialized workspace and passing "." preserves relative-path
        # semantics without rewriting a candidate byte.
        checker_cwd = workspace
        workspace_argument = "."
    else:
        raise ValueError(f"unknown checker workspace mode: {workspace_mode}")
    result = subprocess.run(
        [sys.executable, str(checker), workspace_argument],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        cwd=checker_cwd,
        env={**os.environ, "PYTHONHASHSEED": "0"},
    )
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"checker exited {result.returncode}: {result.stderr.strip()[:1000]}"
        )
    vector = _parse_checker_json(result.stdout)
    reward = sum(vector.values()) / len(vector)
    return reward, vector, elapsed


def _load_gate_rows(gate_output: Path) -> dict[int, dict[str, Any]]:
    report_path = gate_output / "gate_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("trajectories") or report.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"cannot find trajectory rows in {report_path}")
    result = {}
    for row in rows:
        if isinstance(row, dict) and "task_index" in row:
            result[int(row["task_index"])] = row
    if not result:
        raise ValueError(f"no task_index rows in {report_path}")
    return result


def _load_archives(archive_dir: Path, gate_rows: dict[int, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metadata_path in sorted(archive_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        session_id = str(metadata["session_id"])
        try:
            task_index = int(session_id.rsplit("-", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"cannot parse task index from {session_id}") from exc
        row = gate_rows.get(task_index)
        if row is None:
            raise ValueError(f"archive {session_id} has no gate row")
        archive = Path(metadata["archive"])
        if not archive.is_file():
            raise FileNotFoundError(archive)
        if _sha256_file(archive) != metadata["archive_sha256"]:
            raise ValueError(f"archive checksum mismatch: {archive}")
        item = {**metadata, **row, "archive_path": archive}
        grouped.setdefault(str(metadata["task_id"]), []).append(item)
    return grouped


def _minimum_source_cover(
    paths: list[str],
    state: tuple[str, ...],
    initial_state: tuple[str, ...],
    variant_sources: dict[str, dict[str, set[int]]],
    parent_count: int,
) -> tuple[int | None, list[int]]:
    requirements = []
    for relative, key, initial_key in zip(paths, state, initial_state, strict=True):
        if key == initial_key:
            continue
        sources = variant_sources[relative].get(key, set())
        if not sources:
            return None, []
        requirements.append(sources)
    if not requirements:
        return 0, []
    for size in range(1, parent_count + 1):
        for subset in itertools.combinations(range(parent_count), size):
            chosen = set(subset)
            if all(chosen & sources for sources in requirements):
                return size, list(subset)
    return None, []


def _analyze_group(
    task_id: str,
    items: list[dict[str, Any]],
    output_dir: Path,
    beam_width: int,
    max_queries: int,
    checker_timeout: int,
    max_file_bytes: int,
) -> dict[str, Any]:
    started = time.monotonic()
    items = sorted(items, key=lambda item: int(item["task_index"]))
    task_dir = Path(items[0]["task_dir"])
    group_dir = output_dir / "extracted" / task_id
    group_dir.mkdir(parents=True, exist_ok=False)
    initial_root = group_dir / "initial"
    entry = _build_initial_workspace(task_dir, initial_root)
    parent_roots = []
    for parent_index, item in enumerate(items):
        root = group_dir / f"parent_{parent_index:02d}"
        root.mkdir()
        _safe_extract(Path(item["archive_path"]), root)
        parent_roots.append(root)

    initial_manifest = _manifest(initial_root, max_file_bytes)
    parent_manifests = [_manifest(root, max_file_bytes) for root in parent_roots]
    observed_paths = set(initial_manifest).union(*(set(value) for value in parent_manifests))
    injected_context_files = {
        "AGENTS.md",
        "BOOTSTRAP.md",
        "HEARTBEAT.md",
        "IDENTITY.md",
        "MEMORY.md",
        "SOUL.md",
        "TOOLS.md",
        "USER.md",
    }

    def is_runtime_scaffolding(relative: str) -> bool:
        if relative in initial_manifest:
            return False
        if relative in injected_context_files:
            return True
        if relative.startswith((".git/", ".openclaw/", ".venv/")):
            return True
        parts = PurePosixPath(relative).parts
        return "__pycache__" in parts or relative.endswith((".pyc", ".pyo"))

    # Preserve every task-provided path and every other model-created artifact,
    # including legitimate root-level deliverables. Exclude only known
    # OpenClaw/Python execution scaffolding, which is not agent task output and
    # otherwise pollutes source-cover accounting.
    all_paths = sorted(
        relative
        for relative in observed_paths
        if not is_runtime_scaffolding(relative)
    )
    changed_paths = [
        relative
        for relative in all_paths
        if any(
            manifest.get(relative, _absent_version()).key
            != initial_manifest.get(relative, _absent_version()).key
            for manifest in parent_manifests
        )
    ]
    versions: dict[str, dict[str, FileVersion]] = {}
    variant_sources: dict[str, dict[str, set[int]]] = {}
    initial_keys = []
    parent_states: list[tuple[str, ...]] = []
    for relative in changed_paths:
        initial_version = initial_manifest.get(relative, _absent_version())
        versions[relative] = {initial_version.key: initial_version}
        variant_sources[relative] = {initial_version.key: set()}
        initial_keys.append(initial_version.key)
        for parent_index, manifest in enumerate(parent_manifests):
            version = manifest.get(relative, _absent_version())
            versions[relative].setdefault(version.key, version)
            variant_sources[relative].setdefault(version.key, set()).add(parent_index)
    initial_state = tuple(initial_keys)
    for manifest in parent_manifests:
        parent_states.append(
            tuple(manifest.get(relative, _absent_version()).key for relative in changed_paths)
        )

    query_root = output_dir / "query_workspaces" / task_id
    query_root.mkdir(parents=True, exist_ok=False)
    cache: dict[tuple[str, ...], dict[str, Any]] = {}
    verifier_seconds = 0.0
    checker_workspace_mode = "absolute"

    def evaluate(state: tuple[str, ...]) -> dict[str, Any]:
        nonlocal verifier_seconds
        cached = cache.get(state)
        if cached is not None:
            return cached
        workspace = query_root / f"q_{len(cache):04d}"
        _materialize(initial_root, workspace, changed_paths, state, versions)
        reward, vector, elapsed = _run_checker(
            task_dir,
            workspace,
            checker_timeout,
            workspace_mode=checker_workspace_mode,
        )
        verifier_seconds += elapsed
        row = {"reward": reward, "criterion_vector": vector, "workspace": workspace}
        cache[state] = row
        return row

    seed_states = list(dict.fromkeys([initial_state, *parent_states]))
    for state in seed_states:
        evaluate(state)

    def replay_parents() -> tuple[list[dict[str, Any]], bool]:
        parent_replay = []
        replay_match = True
        for parent_index, (item, state) in enumerate(
            zip(items, parent_states, strict=True)
        ):
            calculated = evaluate(state)["reward"]
            official = float(item["official_reward"])
            matches = abs(calculated - official) <= 1e-6
            replay_match = replay_match and matches
            parent_replay.append(
                {
                    "parent_index": parent_index,
                    "task_index": int(item["task_index"]),
                    "official_reward": official,
                    "offline_reward": calculated,
                    "matches": matches,
                    "react_steps": item.get("react_steps"),
                    "archive_bytes": item.get("archive_bytes"),
                    "criterion_capture_error": item.get("criterion_capture_error"),
                }
            )
        return parent_replay, replay_match

    parent_replay, replay_match = replay_parents()
    if not replay_match:
        checker_workspace_mode = "relative"
        cache.clear()
        shutil.rmtree(query_root)
        query_root.mkdir(parents=True)
        for state in seed_states:
            evaluate(state)
        parent_replay, replay_match = replay_parents()

    if not replay_match:
        return {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "supported": False,
            "reason": "offline checker does not reproduce official parent reward",
            "parent_replay": parent_replay,
            "checker_workspace_mode": checker_workspace_mode,
            "changed_path_count": len(changed_paths),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }

    def rank_key(state: tuple[str, ...]) -> tuple[float, int, tuple[str, ...]]:
        source_cover, _ = _minimum_source_cover(
            changed_paths, state, initial_state, variant_sources, len(items)
        )
        return (
            evaluate(state)["reward"],
            -(source_cover if source_cover is not None else len(items) + 1),
            state,
        )

    beam = sorted(seed_states, key=rank_key, reverse=True)[:beam_width]
    seen = set(seed_states)
    while len(cache) < max_queries:
        proposals = []
        for state in beam:
            for path_index, relative in enumerate(changed_paths):
                for version_key in sorted(versions[relative]):
                    if version_key == state[path_index]:
                        continue
                    candidate = list(state)
                    candidate[path_index] = version_key
                    candidate_tuple = tuple(candidate)
                    if candidate_tuple not in seen:
                        seen.add(candidate_tuple)
                        proposals.append(candidate_tuple)
        if not proposals:
            break
        for state in proposals:
            if len(cache) >= max_queries:
                break
            evaluate(state)
        new_beam = sorted(cache, key=rank_key, reverse=True)[:beam_width]
        if new_beam == beam:
            break
        beam = new_beam

    best_parent_reward = max(evaluate(state)["reward"] for state in parent_states)
    best_state = max(cache, key=rank_key)
    best = evaluate(best_state)
    gain = best["reward"] - best_parent_reward
    strict = gain > 1e-9 and best_state not in parent_states
    source_cover_size, source_cover = _minimum_source_cover(
        changed_paths, best_state, initial_state, variant_sources, len(items)
    )
    # A strict super-parent can arise by reverting/deleting a harmful artifact
    # from just one rollout. That is useful selection evidence, but it does not
    # validate cross-rollout complementarity. Keep the notions separate and
    # require at least two source parents for a genuine Mosaic hit.
    cross_parent_complementarity = bool(
        strict and source_cover_size is not None and source_cover_size >= 2
    )
    chosen = []
    for relative, key, initial_key in zip(
        changed_paths, best_state, initial_state, strict=True
    ):
        if key == initial_key:
            continue
        chosen.append(
            {
                "path": relative,
                "version": key,
                "sources": sorted(variant_sources[relative].get(key, set())),
            }
        )

    mosaic_path = None
    if strict:
        mosaic_path = output_dir / "strict_mosaics" / task_id
        mosaic_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(best["workspace"], mosaic_path, copy_function=shutil.copy2)

    semantic_vector = {
        key: value for key, value in best["criterion_vector"].items() if key != "reward"
    }
    all_parents_fail = all(float(item["official_reward"]) < 1.0 - 1e-9 for item in items)
    mosaic_success = best["reward"] >= 1.0 - 1e-9
    return {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "prompt": entry.get("user_query", ""),
        "supported": True,
        "parent_count": len(items),
        "parent_replay": parent_replay,
        "checker_workspace_mode": checker_workspace_mode,
        "changed_path_count": len(changed_paths),
        "variant_count": sum(len(value) for value in versions.values()),
        "queries": len(cache),
        "verifier_seconds": round(verifier_seconds, 6),
        "best_parent_reward": best_parent_reward,
        "best_mosaic_reward": best["reward"],
        "strict_super_parent": strict,
        "cross_parent_complementarity": cross_parent_complementarity,
        "gain_over_best_parent": gain,
        "minimum_source_cover": source_cover_size,
        "source_cover_example": source_cover,
        "chosen_noninitial_files": chosen,
        "all_parents_fail": all_parents_fail,
        "all_fail_to_success": bool(strict and all_parents_fail and mosaic_success),
        "cross_parent_all_fail_to_success": bool(
            cross_parent_complementarity and all_parents_fail and mosaic_success
        ),
        "best_criterion_vector": semantic_vector,
        "mosaic_workspace": str(mosaic_path) if mosaic_path else None,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    supported = [row for row in rows if row.get("supported")]
    strict = [row for row in supported if row.get("strict_super_parent")]
    cross_parent = [
        row for row in supported if row.get("cross_parent_complementarity")
    ]
    all_fail = [row for row in supported if row.get("all_parents_fail")]
    converted = [row for row in supported if row.get("all_fail_to_success")]
    cross_parent_converted = [
        row for row in supported if row.get("cross_parent_all_fail_to_success")
    ]
    query_counts = [int(row["queries"]) for row in supported]
    gains = [float(row["gain_over_best_parent"]) for row in strict]
    return {
        "task_groups": len(rows),
        "supported_groups": len(supported),
        "supported_fraction": len(supported) / len(rows) if rows else 0.0,
        "strict_super_parent_groups": len(strict),
        "strict_super_parent_fraction_supported": (
            len(strict) / len(supported) if supported else 0.0
        ),
        "cross_parent_complementarity_groups": len(cross_parent),
        "cross_parent_complementarity_fraction_supported": (
            len(cross_parent) / len(supported) if supported else 0.0
        ),
        "all_parent_fail_groups": len(all_fail),
        "all_fail_to_success_groups": len(converted),
        "all_fail_to_success_fraction": (
            len(converted) / len(all_fail) if all_fail else 0.0
        ),
        "cross_parent_all_fail_to_success_groups": len(cross_parent_converted),
        "cross_parent_all_fail_to_success_fraction": (
            len(cross_parent_converted) / len(all_fail) if all_fail else 0.0
        ),
        "mean_strict_gain": statistics.mean(gains) if gains else 0.0,
        "median_strict_gain": statistics.median(gains) if gains else 0.0,
        "total_verifier_queries": sum(query_counts),
        "mean_verifier_queries_per_supported_group": (
            statistics.mean(query_counts) if query_counts else 0.0
        ),
        "total_verifier_seconds": sum(
            float(row.get("verifier_seconds", 0.0)) for row in supported
        ),
        "phase0_scope": "exact whole-file atoms; conservative lower bound",
        "decision": (
            "promising-small-pilot"
            if len(supported) >= 4 and len(cross_parent) / len(supported) >= 0.05
            else "insufficient-or-no-go-at-file-level"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-output-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-queries-per-task", type=int, default=192)
    parser.add_argument("--checker-timeout", type=int, default=120)
    parser.add_argument("--max-file-mib", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    gate_rows = _load_gate_rows(args.gate_output_dir)
    grouped = _load_archives(args.archive_dir, gate_rows)
    rows = []
    for task_id, items in sorted(grouped.items()):
        try:
            row = _analyze_group(
                task_id,
                items,
                args.output_dir,
                args.beam_width,
                args.max_queries_per_task,
                args.checker_timeout,
                args.max_file_mib * 1024**2,
            )
        except Exception as exc:
            row = {
                "task_id": task_id,
                "supported": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        print(
            f"{task_id}: supported={row.get('supported')} "
            f"strict={row.get('strict_super_parent')} "
            f"gain={row.get('gain_over_best_parent')} "
            f"reason={row.get('reason', '')}",
            flush=True,
        )
    report = {
        "summary": _summary(rows),
        "tasks": rows,
        "configuration": {
            "beam_width": args.beam_width,
            "max_queries_per_task": args.max_queries_per_task,
            "checker_timeout": args.checker_timeout,
            "max_file_mib": args.max_file_mib,
        },
    }
    (args.output_dir / "mosaic_phase0_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
