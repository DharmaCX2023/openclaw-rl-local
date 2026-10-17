#!/usr/bin/env python3
"""Uncertainty and stratification for same-policy OPD peer-swap audits."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from opd_precheck_audit import _factor_vector


STEP_RE = re.compile(r"^Steps:\s*(\d+)\s*$", re.MULTILINE)


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - radius, center + radius]


def _load_metadata(run_dir: Path) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    log_dir = run_dir / "gate" / "openclaw_logs" / "rollout_0"
    for path in (run_dir / "archives").glob("*.json"):
        sidecar = json.loads(path.read_text())
        index = int(str(sidecar["session_id"]).rsplit("-", 1)[1])
        log = log_dir / f"instance-gate-{index}.log"
        steps = None
        if log.is_file():
            match = STEP_RE.search(log.read_text(errors="replace"))
            if match:
                steps = int(match.group(1))
        metadata[index] = {
            "factor_count": len(_factor_vector(sidecar["criterion_vector"])),
            "agent_steps": steps,
            "instance_log": str(log.resolve()) if log.is_file() else None,
        }
    return metadata


def _step_bin(value: Any) -> str:
    if value is None:
        return "unknown"
    if value <= 10:
        return "<=10"
    if value <= 20:
        return "11-20"
    if value <= 40:
        return "21-40"
    return ">40"


def _factor_bin(value: int) -> str:
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    return ">=17"


def _rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mixed = sum(bool(row["has_mixed_correction"]) for row in rows)
    return {
        "target_actions": len(rows),
        "mixed_actions": mixed,
        "rate": mixed / len(rows) if rows else 0.0,
        "wilson_95": _wilson(mixed, len(rows)),
        "task_group_clusters": len({(row["seed"], row["task_id"]) for row in rows}),
    }


def _stratify(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: _rate(values) for name, values in sorted(groups.items())}


def _cluster_bootstrap(rows: list[dict[str, Any]], seed: int, repeats: int) -> list[float]:
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        clusters[(row["seed"], row["task_id"])].append(row)
    keys = sorted(clusters)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(repeats):
        sampled = [rng.choice(keys) for _ in keys]
        denominator = sum(len(clusters[key]) for key in sampled)
        numerator = sum(
            bool(row["has_mixed_correction"])
            for key in sampled for row in clusters[key]
        )
        values.append(numerator / denominator)
    values.sort()
    return [values[int(0.025 * repeats)], values[min(repeats - 1, int(0.975 * repeats))]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="append", nargs=3, metavar=("SEED", "JSON", "RUN_DIR"), required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    all_rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    excluded_mismatch_groups: list[dict[str, str]] = []
    for seed_name, audit_s, run_dir_s in args.audit:
        audit = json.loads(Path(audit_s).resolve(strict=True).read_text())
        run_dir = Path(run_dir_s).resolve(strict=True)
        metadata = _load_metadata(run_dir)
        rows = []
        for row in audit["summary"]["both_exact_target_actions"]:
            enriched = dict(row)
            enriched["seed"] = seed_name
            enriched.update(metadata.get(int(row["recipient_task_index"]), {}))
            enriched["factor_bin"] = _factor_bin(int(enriched["factor_count"]))
            enriched["step_bin"] = _step_bin(enriched.get("agent_steps"))
            rows.append(enriched)
        mismatch_tasks = {
            record["task_id"]
            for record in audit["records"]
            if any(not item["matches_sidecar"] or not item["checker_ok"] or not item["stable"]
                   for item in record["baseline"])
        }
        excluded_mismatch_groups.extend(
            {"seed": seed_name, "task_id": task_id} for task_id in sorted(mismatch_tasks)
        )
        per_seed[seed_name] = {
            **_rate(rows),
            "run_dir": str(run_dir),
            "rollouts": audit["summary"]["rollouts"],
            "task_groups": audit["summary"]["task_groups"],
            "baseline_checks_ok": audit["summary"]["baseline_checks_ok"],
            "baseline_checks_stable": audit["summary"]["baseline_checks_stable"],
            "baseline_sidecar_matches": audit["summary"]["baseline_sidecar_matches"],
        }
        all_rows.extend(rows)
    clean_rows = [
        row for row in all_rows
        if {"seed": row["seed"], "task_id": row["task_id"]} not in excluded_mismatch_groups
    ]
    mixed_examples = [row for row in all_rows if row["has_mixed_correction"]]
    examples = []
    for row in sorted(mixed_examples, key=lambda x: (-len(x["mixed_examples"]), x["seed"], x["task_id"]))[:30]:
        examples.append({
            "seed": row["seed"],
            "task_id": row["task_id"],
            "recipient_task_index": row["recipient_task_index"],
            "artifact": row["artifact"],
            "target_write_line": row["target_write_line"],
            "factor_count": row["factor_count"],
            "agent_steps": row.get("agent_steps"),
            "instance_log": row.get("instance_log"),
            "mixed_examples": row["mixed_examples"],
        })
    payload = {
        "schema_version": 1,
        "unit": (
            "A recipient's byte-exact single-write action, counted once. It is failure-relevant "
            "if at least one same-checkpoint peer's byte-exact write version fixes a failed factor. "
            "It is mixed if any such replacement also regresses a passed factor."
        ),
        "per_seed": per_seed,
        "combined": {
            **_rate(all_rows),
            "task_group_cluster_bootstrap_95": _cluster_bootstrap(
                all_rows, args.seed, args.bootstrap_repeats
            ),
            "by_factor_count": _stratify(all_rows, "factor_bin"),
            "by_trajectory_steps": _stratify(all_rows, "step_bin"),
        },
        "mismatch_sensitivity": {
            "excluded_groups": excluded_mismatch_groups,
            "after_exclusion": _rate(clean_rows),
        },
        "gate_assessment": (
            "borderline-supported: combined point estimate is near the prespecified 25% threshold, "
            "replicates across two independent task batches, but uncertainty includes materially "
            "lower values and the intervention is a peer rewrite rather than an OPD update"
        ),
        "qualitative_examples": examples,
    }
    destination = args.output.resolve()
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(destination),
        "per_seed": per_seed,
        "combined": {k: v for k, v in payload["combined"].items() if not k.startswith("by_")},
        "mismatch_sensitivity": payload["mismatch_sensitivity"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
