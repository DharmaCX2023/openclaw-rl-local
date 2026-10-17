#!/usr/bin/env python3
"""Join Centered OPSD weights to executed actions without emitting content."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import orjson
import torch

from slime.utils.opsd_utils import compute_centered_opsd_group_weights

SESSION_RE = re.compile(r"^train-task-(\d+)-(\d+)$")
ERROR_MARKERS = (
    "traceback (most recent call last)", "no such file or directory",
    "command exited with code", "process exited with code", "permission denied",
    "timed out", "timeout", "error:", "iserror\":true",
)


def build_weight_map(summary_dir: Path, beta: float):
    groups = defaultdict(list)
    for path in sorted(summary_dir.glob("*.json"), key=lambda p: int(p.stem)):
        payload = json.loads(path.read_text())
        rollout_id = int(payload["rollout_id"])
        for row in payload.get("records", []):
            # Schema v2 persists the exact gateway session id. Older summaries
            # only had completion-order trajectory_index and require the
            # conservative group/step-set recovery below.
            identity = row.get("session_id")
            if not identity:
                identity = int(row["trajectory_index"])
            key = (rollout_id, int(row["group_index"]), identity)
            groups[key].append(row)
    trajectories = {}
    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r.get("segment_index", 0)))
        if not rows or not all(r.get("opsd_applied") for r in rows):
            continue
        steps = [step for row in rows for step in row.get("steps", [])]
        if len(steps) < 2:
            continue
        supports = torch.tensor([s["teacher_minus_rollout_old_mean"] for s in steps])
        counts = torch.tensor([s["loss_token_count"] for s in steps], dtype=torch.float32)
        weights, diag = compute_centered_opsd_group_weights(
            supports=supports, token_counts=counts, beta=beta,
            clip_lower=.8, clip_upper=1.2, epsilon=1e-6, min_steps=2,
        )
        if not diag["applied"]:
            continue
        trajectories[key] = {
            "weights": {int(step["step_id"]): float(weight) for step, weight in zip(steps, weights.tolist(), strict=True)},
            "step_ids": frozenset(int(step["step_id"]) for step in steps),
            "session_id": rows[0].get("session_id"),
        }
    return trajectories


def tool_names(row):
    names=[]
    for call in row.get("tool_calls") or []:
        fn=call.get("function") or {}
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def categories(row):
    names=tool_names(row)
    cats=set()
    if not names: cats.add("final_or_text")
    for name in names: cats.add("tool_"+name)
    if int(row.get("raw_tool_call_count",0)) > int(row.get("parsed_tool_call_count",0)):
        cats.add("tool_parse_failure")
    if row.get("incomplete_tool_call_at_limit"):
        cats.add("incomplete_tool_at_limit")
    if row.get("repetition_degenerated") or int(row.get("max_ngram_repeat",0)) >= 8:
        cats.add("repetition_degenerated")
    if int(row.get("consecutive_identical_tool_call_turns",0)) >= 2:
        cats.add("identical_tool_repeat")
    content=str((row.get("next_state") or {}).get("content","")).lower()
    if any(marker in content for marker in ERROR_MARKERS): cats.add("tool_result_error")
    if any(name in {"write","edit"} for name in names): cats.add("workspace_mutation")
    if "exec" in names:
        args=" ".join(str((call.get("function") or {}).get("arguments","")).lower() for call in row.get("tool_calls") or [])
        if any(x in args for x in ("pytest", "unittest", "npm test", "json.load", "jq ", "validate", "check")):
            cats.add("validation_exec")
    return cats


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary-dir",type=Path,required=True)
    ap.add_argument("--proxy-records",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--beta",type=float,default=.02)
    ap.add_argument("--permutations",type=int,default=200)
    args=ap.parse_args()
    trajectories=build_weight_map(args.summary_dir,args.beta)
    sessions=defaultdict(dict)
    with args.proxy_records.open("rb") as handle:
        for line in handle:
            row=orjson.loads(line)
            match=SESSION_RE.match(str(row.get("session_id","")))
            if not match: continue
            rollout_id=int(match.group(1))
            if "step" not in row:
                continue
            local_id=int(match.group(2))
            group_id=local_id // 8
            step_id=int(row["step"])
            session_rows = sessions[(rollout_id,group_id)].setdefault(local_id,{})
            # The JSONL can contain multiple launcher attempts with reused session ids.
            # A repeated step 1 starts a new occurrence; retain the latest complete one.
            if step_id == 1 and 1 in session_rows:
                session_rows.clear()
            session_rows[step_id]=categories(row)

    # trajectory_index is completion-order, while session local_id is dispatch-order.
    # Only retain an exact, unique full-step-set match inside the same task group.
    matched={}
    ambiguous=missing=0
    for (rollout_id,group_id,trajectory_id), meta in trajectories.items():
        exact_session_id = meta.get("session_id")
        if exact_session_id:
            match = SESSION_RE.match(str(exact_session_id))
            local_id = int(match.group(2)) if match else None
            step_map = sessions.get((rollout_id,group_id),{}).get(local_id, {})
            if local_id is not None and meta["step_ids"].issubset(step_map):
                matched[(rollout_id,group_id,trajectory_id)]=local_id
            else:
                missing += 1
            continue
        candidates=[]
        for local_id, step_map in sessions.get((rollout_id,group_id),{}).items():
            if frozenset(step_map) == meta["step_ids"]:
                candidates.append(local_id)
        if len(candidates)==1:
            matched[(rollout_id,group_id,trajectory_id)]=candidates[0]
        elif candidates:
            ambiguous += 1
        else:
            missing += 1

    observations=[]
    trajectory_weights={}
    for key,local_id in matched.items():
        rollout_id,group_id,trajectory_id=key
        step_map=sessions[(rollout_id,group_id)][local_id]
        weights=trajectories[key]["weights"]
        trajectory_key=(rollout_id,group_id,trajectory_id)
        trajectory_weights[trajectory_key]=weights
        for step_id,weight in weights.items():
            observations.append((trajectory_key,step_id,weight,step_map[step_id]))

    category_names=sorted({c for *_,cats in observations for c in cats})
    result={}
    rng=random.Random(20260906)
    for cat in category_names:
        yes=[w for _,_,w,cats in observations if cat in cats]
        no=[w for _,_,w,cats in observations if cat not in cats]
        observed=sum(yes)/len(yes)-sum(no)/len(no) if yes and no else 0.0
        perm=[]
        for _ in range(args.permutations):
            shuffled={}
            for traj,mapping in trajectory_weights.items():
                ids=list(mapping); vals=list(mapping.values()); rng.shuffle(vals)
                shuffled.update({(traj,sid):v for sid,v in zip(ids,vals)})
            y=[shuffled[(traj,sid)] for traj,sid,_,cats in observations if cat in cats]
            n=[shuffled[(traj,sid)] for traj,sid,_,cats in observations if cat not in cats]
            perm.append(sum(y)/len(y)-sum(n)/len(n) if y and n else 0.0)
        p=(1+sum(abs(x)>=abs(observed) for x in perm))/(1+len(perm))
        result[cat]={
            "count":len(yes), "fraction":len(yes)/max(1,len(observations)),
            "mean_weight":sum(yes)/max(1,len(yes)),
            "fraction_weight_gt_1":sum(w>1 for w in yes)/max(1,len(yes)),
            "mean_weight_difference_vs_other":observed,
            "within_trajectory_permutation_p_two_sided":p,
        }
    payload={
        "schema_version":2, "beta":args.beta,
        "summary_trajectories":len(trajectories), "uniquely_matched_trajectories":len(matched),
        "ambiguous_trajectories_excluded":ambiguous, "missing_trajectories_excluded":missing,
        "matched_steps":len(observations),
        "summary_steps":sum(len(meta["weights"]) for meta in trajectories.values()),
        "categories":result,
        "note":"Associational diagnostic on exact unique group/step-set joins only; content is never emitted.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True))

if __name__=="__main__": main()
