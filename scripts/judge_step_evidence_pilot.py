#!/usr/bin/env python3
"""LLM pilot for explicit, outcome-grounded step evidence labels."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import statistics
from pathlib import Path

import requests

from analyze_matched_placebo_phase0 import _action_type, _tool_result_failed


SYSTEM = """You are a conservative evidence judge for a tool-agent trajectory.
Judge only the candidate assistant action and its observed tool result relative to the original task and prior state.

Labels:
- progress: the action obtains relevant missing evidence, creates/corrects a required artifact, validates it, or successfully recovers from a preceding failure.
- error: the action fails, repeats a no-op, corrupts relevant state, or contradicts observed task evidence.
- neutral: the evidence is insufficient or the action has no verifiable effect.

Hard precedence rules:
+- If the candidate result contains a traceback, explicit error/failure/timeout, or non-zero command exit, label error even when the failure is informative or easy to fix.
+- A later correction is progress only when this candidate action itself completes successfully and has a task-relevant result.
+- Judge the current candidate, never a hypothetical future fix.
+
Do not infer final task success. Do not reward prose or intent without an observed useful state transition. The first characters of your response must be <step_evidence>. Return only:
<step_evidence><label>progress|error|neutral</label><evidence>one concise observation-grounded reason</evidence></step_evidence>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-records", type=Path, required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--model", default="anchor-repair")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    urls = [x.strip() for x in args.endpoints.split(",") if x.strip()]

    proxy = {}
    for line in args.proxy_records.open():
        row = json.loads(line)
        if "messages" in row and "step" in row:
            proxy[(str(row["session_id"]), int(row["step"]))] = row
    typed = {}
    for session in sorted({k[0] for k in proxy}):
        previous_failed = False
        for key in sorted((k for k in proxy if k[0] == session), key=lambda x: x[1]):
            row = proxy[key]
            typed[key] = _action_type(row, previous_failed)
            previous_failed = _tool_result_failed(row)
    selected = [(key, proxy[key], kind) for key, kind in typed.items()
                if kind in {"tool_error", "recovery_after_error"}]

    def run(item):
        key, row, kind = item
        user_text = ""
        for message in row.get("messages") or []:
            if message.get("role") == "user":
                user_text = str(message.get("content") or "")
                break
        action = str(row.get("response_text") or row.get("response") or "")
        result = str((row.get("next_state") or {}).get("content") or "")
        prompt = (f"ORIGINAL TASK:\n{user_text[-10000:]}\n\n"
                  f"CANDIDATE ACTION:\n{action[-6000:]}\n\n"
                  f"OBSERVED TOOL RESULT:\n{result[-6000:]}")
        index = int(hashlib.sha256(key[0].encode()).hexdigest(), 16) % len(urls)
        response = requests.post(urls[index].rstrip("/") + "/v1/chat/completions", json={
            "model": args.model,
            "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
            "temperature": 0.0, "max_tokens": 512, "seed": 20260909 + key[1],
            "chat_template_kwargs": {"enable_thinking": False},
        }, timeout=600)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        text = str(message.get("content") or message.get("reasoning_content") or "")
        match = re.search(r"<label>\s*(progress|error|neutral)\s*</label>", text, re.I)
        return {"session_id":key[0],"step_id":key[1],"action_type":kind,
                "label":match.group(1).lower() if match else "parse_error","raw":text}

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, selected))
    by = {(r["session_id"],r["step_id"]):r for r in rows}
    pairs=[]
    for r in rows:
        if r["action_type"] != "tool_error": continue
        q=by.get((r["session_id"],r["step_id"]+1))
        if q and q["action_type"] == "recovery_after_error":
            pairs.append({"session_id":r["session_id"],"error":r["label"],"recovery":q["label"]})
    summary={
        "steps":len(rows),"pairs":len(pairs),
        "error_labeled_error_fraction":sum(r["label"]=="error" for r in rows if r["action_type"]=="tool_error")/sum(r["action_type"]=="tool_error" for r in rows),
        "recovery_labeled_progress_fraction":sum(r["label"]=="progress" for r in rows if r["action_type"]=="recovery_after_error")/sum(r["action_type"]=="recovery_after_error" for r in rows),
        "pair_correct_fraction":sum(p["error"]=="error" and p["recovery"]=="progress" for p in pairs)/len(pairs),
        "parse_errors":sum(r["label"]=="parse_error" for r in rows),
    }
    args.output.write_text(json.dumps({"summary":summary,"pairs":pairs,"rows":rows},indent=2)+"\n")
    print(json.dumps(summary,sort_keys=True))


if __name__ == "__main__":
    main()
