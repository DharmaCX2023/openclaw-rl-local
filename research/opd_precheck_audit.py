#!/usr/bin/env python3
"""Auditable Phase-0 checks for the ClawGym success-preserving OPD idea.

This script deliberately separates three notions that are easy to conflate:

* schema factorization: a verifier exposes multiple named scalar outcomes;
* outcome mixture: a final workspace passes some factors and fails others;
* causal mixed credit: one action is necessary for a passed factor while also
  being responsible for a failed factor (not established by this script).

The first two can be measured without an LLM judge.  Cross-checkpoint factor
transitions are also reported, but they are observational and must not be
presented as the regression rate of a targeted OPD update.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AGGREGATE_KEYS = {
    "reward",
    "score",
    "total",
    "total_score",
    "overall",
    "overall_score",
    "final_score",
}


def _scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _factor_vector(payload: Any) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    factors: dict[str, float] = {}
    for key, value in payload.items():
        if str(key).strip().lower() in AGGREGATE_KEYS:
            continue
        scalar = _scalar(value)
        if scalar is not None:
            factors[str(key)] = scalar
    return factors


def _safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"unsafe archive member: {member.name}")
        handle.extractall(destination)


def _run_checker(checker: Path, workspace: Path, timeout: float) -> dict[str, Any]:
    # The checker runs with reward/ as cwd because some ClawGym verifiers use
    # paths relative to their own directory. Consequently both argv paths
    # must be absolute; otherwise a repository-relative checker path is
    # interpreted a second time underneath reward/.
    checker = checker.resolve(strict=True)
    workspace = workspace.resolve(strict=True)
    proc = subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(checker), str(workspace)],
        cwd=str(checker.parent),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    parsed: Any = None
    error = ""
    if proc.returncode == 0:
        try:
            parsed = json.loads(proc.stdout.strip())
        except Exception as exc:  # noqa: BLE001 - diagnostics must retain failures
            error = f"json:{type(exc).__name__}:{exc}"
    else:
        error = f"exit:{proc.returncode}:{proc.stderr[-500:]}"
    factors = _factor_vector(parsed)
    return {
        "returncode": proc.returncode,
        "payload": parsed,
        "factors": factors,
        "error": error,
    }


def _payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _audit_workspace(
    task_id: str,
    checker: Path,
    workspace: Path,
    timeout: float,
    repeat: bool,
) -> dict[str, Any]:
    first = _run_checker(checker, workspace, timeout)
    second = _run_checker(checker, workspace, timeout) if repeat else first
    stable = (
        not first["error"]
        and not second["error"]
        and _payload_digest(first["payload"]) == _payload_digest(second["payload"])
    )
    factors: dict[str, float] = first["factors"]
    has_pass = any(value >= 1.0 - 1e-9 for value in factors.values())
    has_fail = any(value <= 1e-9 for value in factors.values())
    has_fractional = any(1e-9 < value < 1.0 - 1e-9 for value in factors.values())
    return {
        "task_id": task_id,
        "checker_ok": not first["error"],
        "error": first["error"],
        "stable": stable,
        "factor_count": len(factors),
        "factorized": len(factors) >= 2,
        "binary_factors": all(
            abs(value) <= 1e-9 or abs(value - 1.0) <= 1e-9
            for value in factors.values()
        ),
        "mixed_outcome": has_pass and has_fail,
        "has_fractional_factor": has_fractional,
        "factors": factors,
        "raw_payload": first["payload"],
    }


def _initial_task_worker(args: tuple[str, str, float, bool]) -> dict[str, Any]:
    task_dir_s, task_id, timeout, repeat = args
    task_dir = Path(task_dir_s)
    checker = task_dir / "reward" / "check.py"
    if not checker.is_file():
        return {
            "task_id": task_id,
            "checker_ok": False,
            "error": "missing reward/check.py",
            "stable": False,
            "factor_count": 0,
            "factorized": False,
            "binary_factors": False,
            "mixed_outcome": False,
            "has_fractional_factor": False,
            "factors": {},
            "raw_payload": None,
        }
    with tempfile.TemporaryDirectory(prefix="opd-schema-") as tmp:
        workspace = Path(tmp)
        input_files = task_dir / "input_files"
        if input_files.is_dir():
            shutil.copytree(input_files, workspace, dirs_exist_ok=True)
        shutil.copytree(task_dir / "reward", workspace / "reward", dirs_exist_ok=True)
        return _audit_workspace(task_id, checker, workspace, timeout, repeat)


def audit_training_schema(
    task_root: Path,
    sample_size: int,
    seed: int,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    tasks = sorted(path for path in task_root.iterdir() if path.is_dir())
    if sample_size and sample_size < len(tasks):
        rng = random.Random(seed)
        tasks = sorted(rng.sample(tasks, sample_size))
    jobs = [(str(path), path.name, timeout, True) for path in tasks]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_initial_task_worker, jobs))
    return {
        "task_root": str(task_root),
        "seed": seed,
        "requested_sample_size": sample_size,
        "records": records,
        "summary": _summarize_records(records),
    }


def _workspace_worker(
    args: tuple[str, str, str, str, float, bool]
) -> dict[str, Any]:
    archive_s, checker_s, task_id, session_id, timeout, repeat = args
    archive = Path(archive_s)
    checker = Path(checker_s)
    with tempfile.TemporaryDirectory(prefix="opd-workspace-") as tmp:
        workspace = Path(tmp)
        _safe_extract(archive, workspace)
        result = _audit_workspace(task_id, checker, workspace, timeout, repeat)
    result["session_id"] = session_id
    result["workspace_archive"] = str(archive)
    return result


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    ok = [record for record in records if record["checker_ok"]]
    factors = [record["factor_count"] for record in ok]
    hist = Counter(factors)
    return {
        "n": count,
        "checker_ok": len(ok),
        "checker_ok_rate": len(ok) / count if count else 0.0,
        "stable": sum(record["stable"] for record in records),
        "stable_rate": sum(record["stable"] for record in records) / count if count else 0.0,
        "factorized": sum(record["factorized"] for record in records),
        "factorized_rate": sum(record["factorized"] for record in records) / count if count else 0.0,
        "mixed_outcome": sum(record["mixed_outcome"] for record in records),
        "mixed_outcome_rate": sum(record["mixed_outcome"] for record in records) / count if count else 0.0,
        "factor_count_mean": sum(factors) / len(factors) if factors else 0.0,
        "factor_count_median": sorted(factors)[len(factors) // 2] if factors else 0,
        "factor_count_histogram": {str(key): hist[key] for key in sorted(hist)},
        "errors": Counter(record["error"] for record in records if record["error"]),
    }


def audit_eval_step(
    step_dir: Path,
    task_root: Path,
    workers: int,
    timeout: float,
) -> dict[str, Any]:
    results_path = step_dir / "results.jsonl"
    result_rows = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    archives = {path.name.split("__", 1)[1].removesuffix(".tar.gz"): path
                for path in (step_dir / "workspaces").glob("*.tar.gz")}
    jobs: list[tuple[str, str, str, str, float, bool]] = []
    scalar_rewards: dict[str, float] = {}
    result_metadata: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        task_id = row["task_id"]
        archive = archives.get(task_id)
        checker = task_root / task_id / "reward" / "check.py"
        if archive is None or not checker.is_file():
            continue
        jobs.append((
            str(archive),
            str(checker),
            task_id,
            str(row.get("session_id", "")),
            timeout,
            True,
        ))
        scalar_rewards[task_id] = float(row["reward"])
        result_metadata[task_id] = {
            "reward": float(row["reward"]),
            "correct": bool(row.get("correct", False)),
            "agent_steps": (
                int(row["agent_steps"]) if row.get("agent_steps") is not None else None
            ),
            "termination_reason": row.get("termination_reason"),
            "instance_log": row.get("instance_log"),
        }
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(_workspace_worker, jobs))
    for record in records:
        record.update(result_metadata[record["task_id"]])
    partial = [value for value in scalar_rewards.values() if 1e-9 < value < 1.0 - 1e-9]
    summary = _summarize_records(records)
    summary.update({
        "result_rows": len(result_rows),
        "workspace_records": len(records),
        "partial_scalar_reward": len(partial),
        "partial_scalar_reward_rate": len(partial) / len(result_rows) if result_rows else 0.0,
        "reward_mean": sum(scalar_rewards.values()) / len(scalar_rewards) if scalar_rewards else 0.0,
    })
    return {
        "step_dir": str(step_dir),
        "summary": summary,
        "records": records,
    }


def factor_transitions(steps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    vectors: dict[str, dict[str, dict[str, float]]] = {}
    for step, payload in steps.items():
        vectors[step] = {record["task_id"]: record["factors"] for record in payload["records"]}
    ordered = sorted(vectors, key=int)
    transitions: dict[str, Any] = {}
    pairs = list(zip(ordered, ordered[1:]))
    if len(ordered) >= 2:
        pairs.append((ordered[0], ordered[-1]))
    for left, right in pairs:
        correction_eligible = correction = regression_eligible = regression = unchanged = 0
        task_overlap = set(vectors[left]) & set(vectors[right])
        for task_id in task_overlap:
            a = vectors[left][task_id]
            b = vectors[right][task_id]
            for key in set(a) & set(b):
                av, bv = a[key], b[key]
                if av <= 1e-9:
                    correction_eligible += 1
                    correction += bv >= 1.0 - 1e-9
                if av >= 1.0 - 1e-9:
                    regression_eligible += 1
                    regression += bv <= 1e-9
                unchanged += abs(av - bv) <= 1e-9
        transitions[f"{left}->{right}"] = {
            "task_overlap": len(task_overlap),
            "correction_eligible": correction_eligible,
            "corrections_0_to_1": correction,
            "correction_rate": correction / correction_eligible if correction_eligible else None,
            "regression_eligible": regression_eligible,
            "regressions_1_to_0": regression,
            "regression_rate": regression / regression_eligible if regression_eligible else None,
            "unchanged_equal": unchanged,
            "interpretation": "observational checkpoint transition; not a targeted-update causal rate",
        }
    return transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--train-sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Resolve once before values cross ProcessPool boundaries. This also
    # makes the JSON provenance independent of the caller's cwd.
    args.train_root = args.train_root.resolve(strict=True)
    args.eval_root = args.eval_root.resolve(strict=True)
    args.artifact_root = args.artifact_root.resolve(strict=True)
    args.output = args.output.resolve()

    training = audit_training_schema(
        args.train_root,
        args.train_sample_size,
        args.seed,
        args.workers,
        args.timeout,
    )
    steps: dict[str, dict[str, Any]] = {}
    for step_dir in sorted(args.artifact_root.glob("train_step_*")):
        if not (step_dir / "results.jsonl").is_file():
            continue
        step = str(int(step_dir.name.rsplit("_", 1)[1]))
        steps[step] = audit_eval_step(step_dir, args.eval_root, args.workers, args.timeout)
    output = {
        "schema_version": 1,
        "definitions": {
            "factorized": "at least two named non-aggregate scalar verifier outputs",
            "mixed_outcome": "same final workspace has >=1 factor equal to 1 and >=1 factor equal to 0",
            "causal_warning": "mixed_outcome is not mixed-credit turn attribution",
        },
        "training_schema_audit": training,
        "eval_steps": steps,
        "checkpoint_factor_transitions": factor_transitions(steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "training": training["summary"],
        "eval": {step: payload["summary"] for step, payload in steps.items()},
        "transitions": output["checkpoint_factor_transitions"],
    }, ensure_ascii=False, indent=2, default=lambda value: dict(value)))


if __name__ == "__main__":
    main()
