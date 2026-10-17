#!/usr/bin/env python3
"""Conservative Phase-0 audit for environment-grounded failure/recovery pairs."""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


STEP_RE = re.compile(r"^--- Step (\d+) ---$")
ACTION_RE = re.compile(r"^ACTION: ([^ ]+) (.*)$")
REWARD_RE = re.compile(r"^Reward: ([0-9.eE+-]+)$")
TASK_RE = re.compile(r"^=== TRAIN Task (.+) ===$")

INFRA_PATTERNS = (
    "serper request failed", "invalid endpoint url", "gateway", "pairing required",
    "http 401", "http 403", "http 429", "timed out", "timeout", "connection refused",
    "temporary failure in name resolution", "no route to host", "externally-managed-environment",
    "command not found", "not found: python", "sh: 1: python: not found", "xxd: not found",
    "dig: not found", "/app/skills/serper", "web_fetch: not found",
)
FAIL_PATTERNS = (
    "traceback (most recent call last)", "syntaxerror", "syntax error", "assertionerror",
    " failed", "failure", "error:", '"status": "error"', "validation failed",
    "no such file", "not found", "unterminated", "unexpected eof", "tests=[<unittest.loader._failedtest",
)
PASS_PATTERNS = (
    "passed", "all tests passed", "success", "valid json", "validation passed", "ok",
)
MUTATION_TOOLS = {"edit", "write"}


@dataclass
class Event:
    step: int
    order: int
    tool: str
    args: dict
    obs: str

    @property
    def path(self) -> str:
        raw = self.args.get("path") or self.args.get("file_path") or ""
        return str(raw).replace("/root/.openclaw/workspace/", "").lstrip("./")

    @property
    def command(self) -> str:
        raw = self.args.get("command") or self.args.get("cmd") or ""
        return " ".join(str(raw).split())


def parse_log(path: str):
    lines = Path(path).read_text(errors="replace").splitlines()
    task = "unknown"
    reward = None
    for line in lines[:30]:
        if m := TASK_RE.match(line):
            task = m.group(1)
        if m := REWARD_RE.match(line):
            reward = float(m.group(1))

    events: list[Event] = []
    step = -1
    i = 0
    while i < len(lines):
        if m := STEP_RE.match(lines[i]):
            step = int(m.group(1))
            j = i + 1
            while j < len(lines) and not STEP_RE.match(lines[j]):
                j += 1
            block = lines[i + 1:j]
            acts = []
            obs = []
            k = 0
            while k < len(block):
                if m := ACTION_RE.match(block[k]):
                    try:
                        args = json.loads(m.group(2))
                    except Exception:
                        args = {}
                    acts.append((m.group(1), args))
                elif block[k].startswith("OBSERVATION:"):
                    chunk = [block[k][len("OBSERVATION:"):].strip()]
                    k += 1
                    while k < len(block) and not block[k].startswith("OBSERVATION:"):
                        if ACTION_RE.match(block[k]):
                            break
                        chunk.append(block[k])
                        k += 1
                    obs.append("\n".join(chunk).strip())
                    continue
                k += 1
            for n, (tool, args) in enumerate(acts):
                events.append(Event(step, n, tool, args, obs[n] if n < len(obs) else ""))
            i = j
            continue
        i += 1
    return task, reward, events


def is_infra(obs: str) -> bool:
    text = obs.lower()
    return any(p in text for p in INFRA_PATTERNS)


def is_fail(obs: str) -> bool:
    text = " " + obs.lower()
    return any(p in text for p in FAIL_PATTERNS)


def is_pass(obs: str) -> bool:
    text = obs.lower()
    return not is_fail(obs) and any(p in text for p in PASS_PATTERNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    paths = []
    for root in args.roots:
        paths.extend(glob.glob(str(Path(root) / "checkpoints/openclaw_logs/rollout_*/instance-train-*.log")))
    paths = sorted(set(paths))

    counts = Counter()
    by_run = defaultdict(Counter)
    examples = defaultdict(list)
    rewards = defaultdict(list)
    for path in paths:
        run = "opsd" if "8b-opsd" in path else "grpo"
        task, reward, evs = parse_log(path)
        counts["trajectories"] += 1
        by_run[run]["trajectories"] += 1
        traj_tags = set()

        for idx, ev in enumerate(evs):
            if is_fail(ev.obs):
                counts["failure_events_broad"] += 1
                traj_tags.add("broad_failure")
                if is_infra(ev.obs):
                    counts["infra_failure_events"] += 1
                    traj_tags.add("infra_failure")
                else:
                    counts["noninfra_failure_events"] += 1
                    traj_tags.add("noninfra_failure")

            # High precision: invalid edit invocation, then later same path edit succeeds.
            if ev.tool == "edit" and "validation failed for tool \"edit\"" in ev.obs.lower():
                counts["edit_schema_failure_events"] += 1
                later = [x for x in evs[idx + 1:] if x.tool == "edit" and x.path == ev.path]
                recovered = next((x for x in later if "successfully replaced" in x.obs.lower()), None)
                if recovered:
                    counts["edit_schema_recovered_pairs"] += 1
                    traj_tags.add("edit_schema_recovered")
                    if len(examples["edit_schema_recovered"]) < args.examples:
                        examples["edit_schema_recovered"].append({
                            "file": path, "task": task, "reward": reward, "path": ev.path,
                            "failure_step": ev.step, "recovery_step": recovered.step,
                        })
                else:
                    counts["edit_schema_unrecovered_events"] += 1
                    traj_tags.add("edit_schema_unrecovered")

        # High precision: exact same validator/test command fails then passes; exactly one
        # intervening file mutation makes its recovery attribution unambiguous.
        exec_by_cmd = defaultdict(list)
        for idx, ev in enumerate(evs):
            if ev.tool == "exec" and ev.command:
                exec_by_cmd[ev.command].append((idx, ev))
        for cmd, seq in exec_by_cmd.items():
            for a, b in zip(seq, seq[1:]):
                ai, first = a
                bi, second = b
                if is_infra(first.obs) or not is_fail(first.obs) or not is_pass(second.obs):
                    continue
                muts = [x for x in evs[ai + 1:bi] if x.tool in MUTATION_TOOLS]
                counts["exact_recheck_fail_to_pass_pairs"] += 1
                traj_tags.add("exact_recheck_recovered")
                if len(muts) == 1:
                    counts["exact_recheck_single_mutation_pairs"] += 1
                    traj_tags.add("single_mutation_recovered")
                    if len(examples["single_mutation_recovered"]) < args.examples:
                        examples["single_mutation_recovered"].append({
                            "file": path, "task": task, "reward": reward,
                            "failed_check_step": first.step, "mutation_step": muts[0].step,
                            "mutation_tool": muts[0].tool, "mutation_path": muts[0].path,
                            "passed_check_step": second.step, "command": cmd[:300],
                        })

        for tag in traj_tags:
            counts[f"trajectories_{tag}"] += 1
            by_run[run][f"trajectories_{tag}"] += 1
            if reward is not None:
                rewards[tag].append(reward)

    out = {
        "counts": counts,
        "by_run": by_run,
        "reward_means": {k: sum(v) / len(v) for k, v in rewards.items() if v},
        "examples": examples,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
