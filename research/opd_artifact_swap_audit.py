#!/usr/bin/env python3
"""Causal artifact-level precheck for success-preserving OPD on ClawGym.

For every adjacent-checkpoint task pair whose earlier workspace has at least
one failed factor, start from the earlier workspace and transplant exactly one
differing artifact from the later workspace. The official verifier is replayed
after each intervention. Candidate selection therefore does not condition on
whether the complete later workspace regresses an already-passed factor.

This establishes a file-level causal effect, not token/action-level credit.
The distinction is retained explicitly in every output record.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from opd_precheck_audit import _audit_workspace, _run_checker, _safe_extract


CHECKPOINT_PAIRS = (("0", "20"), ("20", "40"), ("40", "60"), ("60", "80"), ("80", "100"))

# Runtime scaffolding and caches are not policy deliverables.  Task-provided
# input/config/code and generated output directories remain eligible because
# editing those is often exactly what ClawGym asks the agent to do.
EXCLUDED_ROOT_FILES = {
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}
EXCLUDED_PARTS = {
    ".git",
    ".openclaw",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log"}


def _file_state(path: Path) -> tuple[str, str] | None:
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return ("file", digest.hexdigest())


def _eligible(relative: PurePosixPath) -> bool:
    if not relative.parts:
        return False
    if relative.name in EXCLUDED_ROOT_FILES and len(relative.parts) == 1:
        return False
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return True


def _states(root: Path) -> dict[str, tuple[str, str]]:
    states: dict[str, tuple[str, str]] = {}
    for path in root.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if not _eligible(relative):
            continue
        state = _file_state(path)
        if state is not None:
            states[str(relative)] = state
    return states


def _copy_tree_hardlinked(source: Path, destination: Path) -> None:
    """Cheap private tree; transplanted paths are unlinked before replacement."""
    shutil.copytree(source, destination, copy_function=os.link, symlinks=True)


def _transplant(source_root: Path, destination_root: Path, relative_s: str) -> None:
    relative = PurePosixPath(relative_s)
    source = source_root.joinpath(*relative.parts)
    destination = destination_root.joinpath(*relative.parts)
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.is_dir():
        shutil.rmtree(destination)
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.readlink(source))
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _strict_deltas(
    baseline: dict[str, float], candidate: dict[str, float]
) -> tuple[list[str], list[str], list[str]]:
    shared = baseline.keys() & candidate.keys()
    corrections = sorted(
        key for key in shared
        if baseline[key] <= 1e-9 and candidate[key] >= 1.0 - 1e-9
    )
    regressions = sorted(
        key for key in shared
        if baseline[key] >= 1.0 - 1e-9 and candidate[key] <= 1e-9
    )
    changed = sorted(key for key in shared if abs(baseline[key] - candidate[key]) > 1e-9)
    return corrections, regressions, changed


def _pair_worker(job: dict[str, Any]) -> dict[str, Any]:
    left_archive = Path(job["left_archive"])
    right_archive = Path(job["right_archive"])
    checker = Path(job["checker"])
    timeout = float(job["timeout"])
    with tempfile.TemporaryDirectory(prefix="opd-swap-") as temporary_s:
        temporary = Path(temporary_s)
        left_root = temporary / "left"
        right_root = temporary / "right"
        _safe_extract(left_archive, left_root)
        _safe_extract(right_archive, right_root)

        left_check = _audit_workspace(job["task_id"], checker, left_root, timeout, True)
        right_check = _audit_workspace(job["task_id"], checker, right_root, timeout, True)
        result: dict[str, Any] = {
            "task_id": job["task_id"],
            "left_step": job["left_step"],
            "right_step": job["right_step"],
            "left_archive": str(left_archive),
            "right_archive": str(right_archive),
            "checker": str(checker),
            "baseline_checks_ok": left_check["checker_ok"] and right_check["checker_ok"],
            "baseline_checks_stable": left_check["stable"] and right_check["stable"],
            "left_factors": left_check["factors"],
            "right_factors": right_check["factors"],
            "candidate_artifacts": [],
            "interventions": [],
            "causal_unit": "single final-workspace artifact replacement; not an action/token update",
        }
        if not result["baseline_checks_ok"]:
            result["error"] = f"left={left_check['error']}; right={right_check['error']}"
            return result

        full_corrections, full_regressions, full_changed = _strict_deltas(
            left_check["factors"], right_check["factors"]
        )
        result.update({
            "full_workspace_corrections": full_corrections,
            "full_workspace_regressions": full_regressions,
            "full_workspace_changed": full_changed,
        })
        left_states = _states(left_root)
        right_states = _states(right_root)
        candidates = sorted(
            path for path in left_states.keys() | right_states.keys()
            if left_states.get(path) != right_states.get(path)
        )
        result["candidate_artifacts"] = candidates

        for ordinal, relative in enumerate(candidates):
            intervention_root = temporary / f"intervention-{ordinal:04d}"
            _copy_tree_hardlinked(left_root, intervention_root)
            _transplant(right_root, intervention_root, relative)
            checked = _run_checker(checker, intervention_root, timeout)
            intervention: dict[str, Any] = {
                "artifact": relative,
                "checker_ok": not checked["error"],
                "error": checked["error"],
                "factors": checked["factors"],
            }
            if not checked["error"]:
                corrections, regressions, changed = _strict_deltas(
                    left_check["factors"], checked["factors"]
                )
                intervention.update({
                    "corrections": corrections,
                    "regressions": regressions,
                    "changed": changed,
                    "strict_mixed_effect": bool(corrections and regressions),
                })
                # Repeat only positive findings to rule out a nondeterministic
                # verifier without doubling the full experiment's cost.
                if corrections and regressions:
                    repeated = _run_checker(checker, intervention_root, timeout)
                    intervention["repeat_stable"] = (
                        not repeated["error"]
                        and repeated["factors"] == checked["factors"]
                    )
                    intervention["repeat_error"] = repeated["error"]
            result["interventions"].append(intervention)
            shutil.rmtree(intervention_root)
        return result


def _records_by_task(audit: dict[str, Any], step: str) -> dict[str, dict[str, Any]]:
    return {
        record["task_id"]: record
        for record in audit["eval_steps"][step]["records"]
        if record["checker_ok"]
    }


def _build_jobs(
    audit: dict[str, Any], eval_root: Path, timeout: float
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for left_step, right_step in CHECKPOINT_PAIRS:
        left = _records_by_task(audit, left_step)
        right = _records_by_task(audit, right_step)
        for task_id in sorted(left.keys() & right.keys()):
            # Select only on the pre-intervention state. Conditioning on an
            # observed later regression would inflate mixed-credit prevalence.
            if not any(value <= 1e-9 for value in left[task_id]["factors"].values()):
                continue
            jobs.append({
                "task_id": task_id,
                "left_step": left_step,
                "right_step": right_step,
                "left_archive": left[task_id]["workspace_archive"],
                "right_archive": right[task_id]["workspace_archive"],
                "checker": str((eval_root / task_id / "reward" / "check.py").resolve()),
                "timeout": timeout,
            })
    return jobs


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    interventions = [item for record in records for item in record["interventions"]]
    ok = [item for item in interventions if item["checker_ok"]]
    effective = [item for item in ok if item.get("changed")]
    mixed = [item for item in ok if item.get("strict_mixed_effect")]
    stable_mixed = [item for item in mixed if item.get("repeat_stable")]
    tasks_with_mixed = {
        (record["left_step"], record["right_step"], record["task_id"])
        for record in records
        if any(item.get("strict_mixed_effect") and item.get("repeat_stable")
               for item in record["interventions"])
    }
    pair_counts = Counter(
        f"{record['left_step']}->{record['right_step']}" for record in records
    )
    mixed_pair_counts = Counter(f"{left}->{right}" for left, right, _ in tasks_with_mixed)
    return {
        "candidate_task_pairs": len(records),
        "candidate_task_pairs_by_transition": dict(sorted(pair_counts.items())),
        "baseline_checks_ok": sum(record["baseline_checks_ok"] for record in records),
        "baseline_checks_stable": sum(record["baseline_checks_stable"] for record in records),
        "artifact_interventions": len(interventions),
        "artifact_interventions_ok": len(ok),
        "effective_artifact_interventions": len(effective),
        "strict_mixed_artifact_interventions": len(mixed),
        "repeat_stable_mixed_interventions": len(stable_mixed),
        "task_pairs_with_repeat_stable_mixed_artifact": len(tasks_with_mixed),
        "task_pair_mixed_rate": len(tasks_with_mixed) / len(records) if records else 0.0,
        "task_pairs_with_mixed_by_transition": dict(sorted(mixed_pair_counts.items())),
        "interpretation": (
            "A positive result proves a single final artifact replacement causally changes "
            "both passed and failed verifier factors. It does not yet prove that one model "
            "turn or token span has the same mixed effect."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor-audit", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.factor_audit.resolve(strict=True).read_text())
    eval_root = args.eval_root.resolve(strict=True)
    jobs = _build_jobs(audit, eval_root, args.timeout)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_pair_worker, jobs))
    payload = {
        "schema_version": 1,
        "method": (
            "For every adjacent-checkpoint pair whose earlier workspace has a failed "
            "factor, replace one differing eligible artifact at a time in the earlier "
            "workspace and replay the official verifier. Selection does not use the "
            "later workspace's correction or regression outcomes."
        ),
        "exclusions": {
            "root_files": sorted(EXCLUDED_ROOT_FILES),
            "path_parts": sorted(EXCLUDED_PARTS),
            "suffixes": sorted(EXCLUDED_SUFFIXES),
        },
        "summary": _summary(records),
        "records": records,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "summary": payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
