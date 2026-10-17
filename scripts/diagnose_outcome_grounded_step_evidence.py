#!/usr/bin/env python3
"""Score verifiable action-outcome evidence on saved error/recovery pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from analyze_matched_placebo_phase0 import (
    _action_type,
    _active_runs,
    _extract_tools,
    _normalize_messages_for_template,
    _score,
    _tool_result_failed,
)
from clawgym_rl_api_server import _inject_opsd_feedback


def _feedback(action_type: str) -> str:
    common = [
        "Teacher-only verified action-outcome evidence",
        "=============================================",
        "",
        "Judge the recorded candidate action using its verified tool outcome.",
    ]
    if action_type == "tool_error":
        common.append(
            "The candidate action returned an error and did not complete its "
            "intended operation; it is not progress unless a later action "
            "corrects the cause."
        )
    elif action_type == "recovery_after_error":
        common.append(
            "The preceding action failed, while this candidate action returned "
            "success and restored execution; treat the successful correction "
            "as progress."
        )
    else:
        raise ValueError(action_type)
    common.extend(["", "Do not quote or discuss this teacher-only evidence."])
    return "\n".join(common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-records", type=Path, required=True)
    ap.add_argument("--step-summary", type=Path, required=True)
    ap.add_argument("--debug-train", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--generate-url", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    urls = [x.strip() for x in args.generate_url.split(",") if x.strip()]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    summary = json.loads(args.step_summary.read_text())["records"]
    debug = torch.load(args.debug_train, weights_only=True)["rollout_data"]
    proxy = {}
    for line in args.proxy_records.open():
        row = json.loads(line)
        if "messages" in row and "step" in row:
            proxy[(str(row["session_id"]), int(row["step"]))] = row

    action_types = {}
    for session in sorted({key[0] for key in proxy}):
        previous_failed = False
        for key in sorted((k for k in proxy if k[0] == session), key=lambda x: x[1]):
            row = proxy[key]
            action_types[key] = _action_type(row, previous_failed)
            previous_failed = _tool_result_failed(row)

    jobs = []
    for index, record in enumerate(summary):
        if not record.get("opsd_applied"):
            continue
        response_length = int(debug["response_lengths"][index])
        response_ids = debug["tokens"][index].tolist()[-response_length:]
        runs = _active_runs(debug["loss_masks"][index].tolist())
        for step, (start, end) in zip(record["steps"], runs, strict=True):
            key = (str(record["session_id"]), int(step["step_id"]))
            action_type = action_types.get(key)
            if action_type not in {"tool_error", "recovery_after_error"}:
                continue
            row = proxy[key]
            tools = _extract_tools(row["prompt_text"])
            messages = _inject_opsd_feedback(row["messages"], _feedback(action_type))
            prompt_ids = [int(x) for x in tokenizer.apply_chat_template(
                _normalize_messages_for_template(messages), tools=tools,
                tokenize=True, add_generation_prompt=True, reasoning_effort=None,
            )]
            jobs.append({
                "session_id": key[0], "step_id": key[1],
                "action_type": action_type,
                "prompt_ids": prompt_ids,
                "response_ids": [int(x) for x in response_ids[start:end]],
                "old_logprob": statistics.fmean(
                    float(x) for x in debug["rollout_log_probs"][index].tolist()[start:end]
                ),
            })

    locks = [threading.Semaphore(1) for _ in urls]
    def run(job):
        engine = int(hashlib.sha256(job["session_id"].encode()).hexdigest(), 16) % len(urls)
        with locks[engine]:
            values = _score(
                generate_url=urls[engine], routing_key=job["session_id"],
                prompt_ids=job["prompt_ids"], response_ids=job["response_ids"],
                timeout=600,
            )
        return {**{k:v for k,v in job.items() if not k.endswith("_ids")},
                "teacher_logprob": statistics.fmean(values),
                "delta": statistics.fmean(values) - job["old_logprob"]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, jobs))
    by = {(r["session_id"], r["step_id"]): r for r in rows}
    pairs = []
    for row in rows:
        if row["action_type"] != "tool_error":
            continue
        recovery = by.get((row["session_id"], row["step_id"] + 1))
        if recovery and recovery["action_type"] == "recovery_after_error":
            pairs.append({
                "session_id": row["session_id"],
                "error_step": row["step_id"],
                "error_delta": row["delta"],
                "recovery_delta": recovery["delta"],
                "recovery_minus_error": recovery["delta"] - row["delta"],
            })
    report = {
        "rows": rows,
        "pairs": pairs,
        "summary": {
            "steps": len(rows), "pairs": len(pairs),
            "recovery_ranked_higher_fraction": sum(p["recovery_minus_error"] > 0 for p in pairs) / len(pairs),
            "mean_recovery_minus_error": statistics.fmean(p["recovery_minus_error"] for p in pairs),
            "median_recovery_minus_error": statistics.median(p["recovery_minus_error"] for p in pairs),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
