#!/usr/bin/env python3
"""Same-policy peer-replacement audit for ClawGym mixed-credit actions.

Within each 8-rollout task group, replace one artifact in a recipient final
workspace with a distinct version produced by a peer rollout from the same
checkpoint.  Replay the official verifier and collapse results by the original
recipient action.  This approximates a targeted corrective rewrite while
holding task, checkpoint, and all non-target workspace artifacts fixed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from opd_artifact_swap_audit import (
    _copy_tree_hardlinked,
    _eligible,
    _states,
    _strict_deltas,
    _transplant,
)
from opd_map_action_provenance import _actions
from opd_precheck_audit import _audit_workspace, _factor_vector, _run_checker, _safe_extract


def _task_index(sidecar: dict[str, Any]) -> int:
    return int(str(sidecar["session_id"]).rsplit("-", 1)[1])


def _exact_write(log: Path, relative: str, root: Path) -> dict[str, Any] | None:
    target = root / relative
    if not target.is_file() or target.is_symlink() or not log.is_file():
        return None
    data = target.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    matching = [action for action in _actions(log) if action["path"] == relative]
    if len(matching) != 1 or matching[0]["tool"] != "write":
        return None
    action = matching[0]
    if action.get("content_sha256") != digest or action.get("content_bytes") != len(data):
        return None
    return {
        "line": action["line"],
        "content_sha256": digest,
        "content_bytes": len(data),
    }


def _group_worker(job: dict[str, Any]) -> dict[str, Any]:
    task_id = job["task_id"]
    checker = Path(job["checker"])
    timeout = float(job["timeout"])
    parents = job["parents"]
    with tempfile.TemporaryDirectory(prefix="opd-peer-") as temporary_s:
        temporary = Path(temporary_s)
        roots: dict[int, Path] = {}
        states: dict[int, dict[str, tuple[str, str]]] = {}
        factors: dict[int, dict[str, float]] = {}
        baseline_rows = []
        for parent in parents:
            index = int(parent["task_index"])
            root = temporary / f"parent-{index}"
            _safe_extract(Path(parent["archive"]), root)
            checked = _audit_workspace(task_id, checker, root, timeout, True)
            expected = _factor_vector(parent["criterion_vector"])
            baseline_rows.append({
                "task_index": index,
                "checker_ok": checked["checker_ok"],
                "stable": checked["stable"],
                "matches_sidecar": checked["factors"] == expected,
                "error": checked["error"],
            })
            roots[index] = root
            states[index] = _states(root)
            factors[index] = checked["factors"]

        interventions = []
        for recipient in parents:
            recipient_index = int(recipient["task_index"])
            recipient_root = roots[recipient_index]
            if not any(value <= 1e-9 for value in factors[recipient_index].values()):
                continue
            # Deduplicate peers that produced the same version of a path.
            variants: dict[tuple[str, tuple[str, str] | None], list[int]] = defaultdict(list)
            for donor in parents:
                donor_index = int(donor["task_index"])
                if donor_index == recipient_index:
                    continue
                for relative in states[recipient_index].keys() | states[donor_index].keys():
                    from pathlib import PurePosixPath
                    if not _eligible(PurePosixPath(relative)):
                        continue
                    donor_state = states[donor_index].get(relative)
                    if states[recipient_index].get(relative) == donor_state:
                        continue
                    variants[(relative, donor_state)].append(donor_index)

            ordered_variants = sorted(
                variants.items(), key=lambda item: (item[0][0], repr(item[0][1]))
            )
            for ordinal, ((relative, _), donor_indices) in enumerate(ordered_variants):
                donor_index = donor_indices[0]
                intervention_root = temporary / f"q-{recipient_index}-{ordinal:04d}"
                _copy_tree_hardlinked(recipient_root, intervention_root)
                _transplant(roots[donor_index], intervention_root, relative)
                checked = _run_checker(checker, intervention_root, timeout)
                row: dict[str, Any] = {
                    "task_id": task_id,
                    "recipient_task_index": recipient_index,
                    "donor_task_indices": donor_indices,
                    "artifact": relative,
                    "checker_ok": not checked["error"],
                    "error": checked["error"],
                }
                if not checked["error"]:
                    corrections, regressions, changed = _strict_deltas(
                        factors[recipient_index], checked["factors"]
                    )
                    row.update({
                        "corrections": corrections,
                        "regressions": regressions,
                        "changed": changed,
                        "strict_mixed_effect": bool(corrections and regressions),
                    })
                    if corrections:
                        recipient_log = Path(job["log_dir"]) / f"instance-gate-{recipient_index}.log"
                        target_write = _exact_write(recipient_log, relative, recipient_root)
                        donor_writes = []
                        for candidate_index in donor_indices:
                            donor_log = Path(job["log_dir"]) / f"instance-gate-{candidate_index}.log"
                            mapped = _exact_write(donor_log, relative, roots[candidate_index])
                            if mapped:
                                donor_writes.append({"task_index": candidate_index, **mapped})
                        row["target_exact_write"] = target_write
                        row["donor_exact_writes"] = donor_writes
                        if corrections and regressions:
                            repeated = _run_checker(checker, intervention_root, timeout)
                            row["repeat_stable"] = (
                                not repeated["error"] and repeated["factors"] == checked["factors"]
                            )
                            row["repeat_error"] = repeated["error"]
                interventions.append(row)
                shutil.rmtree(intervention_root)
        return {
            "task_id": task_id,
            "parent_count": len(parents),
            "baseline": baseline_rows,
            "interventions": interventions,
        }


def _load_groups(run_dir: Path, train_root: Path, timeout: float) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sidecar_path in sorted((run_dir / "archives").glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        archive = Path(sidecar["archive"])
        if not archive.is_file():
            fallback = sidecar_path.with_suffix(".tar.gz")
            if not fallback.is_file():
                continue
            archive = fallback
        sidecar["archive"] = str(archive.resolve())
        sidecar["task_index"] = _task_index(sidecar)
        groups[str(sidecar["task_id"])].append(sidecar)
    log_dir = run_dir / "gate" / "openclaw_logs" / "rollout_0"
    return [{
        "task_id": task_id,
        "checker": str((train_root / task_id / "reward" / "check.py").resolve()),
        "parents": sorted(parents, key=lambda row: row["task_index"]),
        "log_dir": str(log_dir.resolve()),
        "timeout": timeout,
    } for task_id, parents in sorted(groups.items()) if len(parents) >= 2]


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    interventions = [row for record in records for row in record["interventions"]]
    correction = [row for row in interventions if row.get("corrections")]
    target_exact = [row for row in correction if row.get("target_exact_write")]
    both_exact = [row for row in target_exact if row.get("donor_exact_writes")]

    def collapse(rows: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
        targets: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        for row in rows:
            write = row["target_exact_write"]
            key = (
                row["task_id"], int(row["recipient_task_index"]),
                row["artifact"], int(write["line"]),
            )
            entry = targets.setdefault(key, {
                "task_id": row["task_id"],
                "recipient_task_index": row["recipient_task_index"],
                "artifact": row["artifact"],
                "target_write_line": write["line"],
                "corrective_variants": 0,
                "has_mixed_correction": False,
                "mixed_examples": [],
            })
            entry["corrective_variants"] += 1
            if row.get("regressions") and row.get("repeat_stable"):
                entry["has_mixed_correction"] = True
                entry["mixed_examples"].append({
                    "donor_task_indices": row["donor_task_indices"],
                    "corrections": row["corrections"],
                    "regressions": row["regressions"],
                    "donor_exact_writes": row.get("donor_exact_writes", []),
                })
        values = list(targets.values())
        mixed = sum(row["has_mixed_correction"] for row in values)
        return len(values), mixed, values

    target_n, target_mixed, target_rows = collapse(target_exact)
    both_n, both_mixed, both_rows = collapse(both_exact)
    return {
        "task_groups": len(records),
        "rollouts": sum(record["parent_count"] for record in records),
        "baseline_checks_ok": sum(row["checker_ok"] for record in records for row in record["baseline"]),
        "baseline_checks_stable": sum(row["stable"] for record in records for row in record["baseline"]),
        "baseline_sidecar_matches": sum(row["matches_sidecar"] for record in records for row in record["baseline"]),
        "peer_artifact_interventions": len(interventions),
        "peer_artifact_interventions_ok": sum(row["checker_ok"] for row in interventions),
        "correction_interventions": len(correction),
        "correction_interventions_target_exact_write": len(target_exact),
        "correction_interventions_target_and_donor_exact_write": len(both_exact),
        "failure_relevant_target_actions": target_n,
        "mixed_credit_target_actions": target_mixed,
        "rho_mixed_target_action": target_mixed / target_n if target_n else 0.0,
        "failure_relevant_target_actions_both_versions_exact": both_n,
        "mixed_credit_target_actions_both_versions_exact": both_mixed,
        "rho_mixed_both_versions_exact": both_mixed / both_n if both_n else 0.0,
        "target_actions": target_rows,
        "both_exact_target_actions": both_rows,
        "interpretation": (
            "A target action is counted once even if several peer versions correct it. Mixed means "
            "at least one same-checkpoint peer replacement fixes a failed factor and regresses a "
            "passed factor while all other recipient artifacts remain fixed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(strict=True)
    train_root = args.train_root.resolve(strict=True)
    jobs = _load_groups(run_dir, train_root, args.timeout)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(_group_worker, jobs))
    summary = _summarize(records)
    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "selection": "all available same-checkpoint task groups with at least two archived rollouts",
        "summary": summary,
        "records": records,
    }
    destination = args.output.resolve()
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    compact = {key: value for key, value in summary.items() if key not in {"target_actions", "both_exact_target_actions"}}
    print(json.dumps({"output": str(destination), "summary": compact}, indent=2))


if __name__ == "__main__":
    main()
