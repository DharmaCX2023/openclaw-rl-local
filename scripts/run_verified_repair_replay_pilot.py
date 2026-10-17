#!/usr/bin/env python3
"""Fixed-task v3 pilot. No optimizer updates and no per-step file snapshots."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import run_verified_repair_pilot as old
from verified_repair_runtime import ROOT, command, rollout, save_json, sha, paired_accept
from verified_repair_replay_proxy import ReplayProxy, ReplayRemoteState
from verified_repair_sampler_patch import prepare_environment, SOURCE_SHA256
from verified_repair_teacher import review_trajectory
from verified_repair_state_review import review_runtime_drift

ORIGINAL = ROOT / "outputs/verified-repair-v2-20260907-pilot-a3/selection.json"
AGENT_SECONDS = 1000
TRANSPORT_SECONDS = 1200
_RUNTIME_OPTIONS = threading.local()
_ORIGINAL_GATEWAY_CONFIG = rollout._build_gateway_config


def gateway_config(runtime, port):
    _ORIGINAL_GATEWAY_CONFIG(runtime, port)
    if getattr(_RUNTIME_OPTIONS, "replay", False):
        # This allows physical prefix replay overhead only. The v3 Proxy and
        # controller independently enforce 1000 - source_prefix_seconds.
        path = runtime.config_dir / "openclaw.json"
        config = json.loads(path.read_text())
        config["agents"]["defaults"]["timeoutSeconds"] = 2400
        # This fresh run-owned config contains a Gateway token. Never print it.
        path.write_text(json.dumps(config))


def manifest(output):
    original = json.loads(ORIGINAL.read_text())
    if len(original["candidate_tasks"]) != 16 or original["samples_per_task"] != 4:
        raise ValueError("fixed v2 16x4 selection missing")
    value = copy.deepcopy(original)
    for key in ("branch_continuation_turns", "branch_seconds", "branch_assistant_tokens"):
        value.pop(key, None)
    value.update(
        version=3, protocol="recorded-prefix-full-continuation-v3",
        original_selection=str(ORIGINAL), original_selection_sha256=sha(ORIGINAL),
        agent_wall_seconds=AGENT_SECONDS, source_transport_seconds=TRANSPORT_SECONDS,
        replay_transport_seconds=2400, max_agent_steps=96, max_compactions=8,
        live_context_tokens=65536, per_step_response_tokens=8192,
        gate_min_train_pairs=16, gate_min_train_tasks=8,
        paired_repeat_identities=["paired-314159", "paired-271828"],
        replay_state_review="openclaw-initialization-metadata-only-v1",
        teacher_call_index="attempted_tool_blocks_including_incomplete",
        replay_overhead_is_not_policy_budget=True, optimizer_updates=0,
    )
    path = output / "selection.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("existing protocol differs: use a fresh output")
    save_json(path, value)
    return value


def preflight_ports(g):
    if len({g.backend_port, g.proxy_port, g.reverse_port}) != 3:
        raise ValueError("backend, proxy and reverse ports must be distinct")
    sockets = []
    try:
        for port in (g.backend_port, g.proxy_port):
            handle = socket.socket()
            sockets.append(handle)
            handle.bind(("127.0.0.1", port))
        code = ("import socket; s=socket.socket(); "
                f"s.bind(('127.0.0.1',{g.reverse_port})); s.close(); print('port_free')")
        result = rollout._remote_run(command(["python3", "-c", code]), timeout=30)
        if result.stdout.strip() != b"port_free":
            raise RuntimeError("remote port check failed")
    finally:
        for handle in sockets:
            handle.close()


def post_full_task(runtime, task, *, replay):
    """Same Gateway OpenAI request, with a per-request transport envelope."""
    payload = json.dumps({"model": "openclaw", "stream": False,
                          "messages": [{"role": "user", "content": task.user_query}]},
                         ensure_ascii=False).encode()
    seconds = 2400 if replay else TRANSPORT_SECONDS
    port = int(runtime.gateway_url.rsplit(":", 1)[1])
    cmd = command([
        "curl", "--fail-with-body", "-sS", "--max-time", str(seconds),
        "-H", "Authorization: Bearer " + runtime.token,
        "-H", "Content-Type: application/json",
        "-H", "X-OpenClaw-Scopes: operator.admin,operator.read,operator.write",
        "--data-binary", "@-", f"http://127.0.0.1:{port}/v1/chat/completions",
    ])
    result = rollout._remote_run(cmd, input_data=payload, timeout=seconds + 120, check=False)
    # No command/token/header is included in diagnostics.
    if result.returncode:
        raise RuntimeError("gateway_curl_exit_" + str(result.returncode))
    return json.loads(result.stdout)


def execute(row, sample, sid, index, g, state, proxy, branch=None):
    destination = g.output_dir / ("branches" if branch else "sources") / (sid + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    task = rollout._load_task_spec(Path(row["entry"]))
    record = {"task_id": task.task_id, "sample": sample, "session_id": sid,
              "split": row.get("split", "engineering_smoke"), "started_at": time.time(),
              "protocol": "recorded-prefix-full-continuation-v3", "optimizer_updates": 0}
    runtime = None
    worker = None
    stopped = False
    try:
        _RUNTIME_OPTIONS.replay = branch is not None
        runtime, startup = old.start_runtime(task, index, sid, g, state, proxy, branch=branch)
        record["startup_seconds"] = startup
        proxy.begin_execution(sid, AGENT_SECONDS)
        result_queue = queue.Queue()

        def work():
            try:
                result_queue.put(("completed", post_full_task(runtime, task, replay=branch is not None)))
            except BaseException as exc:
                result_queue.put(("error", type(exc).__name__ + ":" + str(exc)[:160]))

        worker = threading.Thread(target=work, name="gateway-" + sid, daemon=True)
        worker.start()
        began = time.monotonic()
        physical_deadline = began + (2400 if branch else TRANSPORT_SECONDS)
        while True:
            deadline = min(physical_deadline, proxy.deadlines.get(sid, physical_deadline))
            if branch and not branch.get("replacement_done"):
                # Prefix replay timeout is infrastructure, not a 1200s policy grant.
                deadline = min(deadline, began + TRANSPORT_SECONDS)
            try:
                result_kind, value = result_queue.get(timeout=min(1., max(.001, deadline-time.monotonic())))
                if result_kind != "completed":
                    raise RuntimeError(value)
                record["gateway_response"] = value
                record["final_message"] = rollout._extract_final_message(value)
                record["status"] = "completed"
                record["termination"] = copy.deepcopy(
                    proxy._session_budget_terminations.get(sid, {"reason": "gateway_completion"}))
                break
            except queue.Empty:
                if time.monotonic() < deadline:
                    continue
                proxy.closed_sessions.add(sid)
                policy_cutoff = sid in proxy.deadlines and time.monotonic() >= proxy.deadlines[sid]
                record["status"] = "completed" if policy_cutoff else "infrastructure_error"
                record["termination"] = {"reason": "original_task_wall_budget" if policy_cutoff
                                         else "replay_or_transport_timeout"}
                break
        record["rollout_seconds"] = time.monotonic() - began
        record["steps"] = proxy.export_trace(sid)
        record["request_event_count"] = len(proxy.events.get(sid, []))
        record["request_events_dir"] = str(g.output_dir / "request_events" / sid)
        record["source_execution_started_at"] = proxy.execution_started[sid]["wall"]
        # Only the terminal workspace is archived. Stop the exact owned Gateway
        # immediately after capture, before running its isolated verifier.
        terminal = state.snapshot(runtime, g.output_dir / "terminal" / (sid + ".tar.gz"),
                                  audit_external=False)
        record["terminal_snapshot"] = terminal
        proxy.closed_sessions.add(sid)
        state.stop(runtime)
        stopped = True
        record["verifier"] = state.verify(task, terminal["archive"], sid + "-terminal")
        if branch is not None:
            record["replay"] = {k: copy.deepcopy(v) for k, v in branch.items()
                               if k not in {"source_events", "source_steps", "replacement"}}
            if branch.get("error") or not branch.get("replacement_done"):
                record["status"] = "replay_failure"
            # Observation differences require explicit review, not blind acceptance.
            record["state_comparable_without_review"] = (
                branch.get("state_comparability") == "match"
                and not branch.get("observation_differences"))
    except Exception as exc:
        record.update(status="infrastructure_error", error_type=type(exc).__name__,
                      error=str(exc)[:300])
        record.setdefault("steps", proxy.export_trace(sid))
    finally:
        _RUNTIME_OPTIONS.replay = False
        proxy.closed_sessions.add(sid)
        if runtime is not None and not stopped:
            try:
                state.stop(runtime)
            except Exception as exc:
                record["cleanup_error_type"] = type(exc).__name__
        if worker is not None:
            worker.join(timeout=5)
            record["chat_worker_exited"] = not worker.is_alive()
        record.update(finished_at=time.time(), seconds=time.time()-record["started_at"])
        save_json(destination, record)
        print(json.dumps({k: record.get(k) for k in ("session_id", "status", "seconds")}
                         | {"agent_steps": len(record.get("steps", []))}), flush=True)
    return record


def source_events(source):
    events = [json.loads(p.read_text()) for p in sorted(Path(source["request_events_dir"]).glob("*.json"))]
    if len(events) != source["request_event_count"]:
        raise ValueError("incomplete source request-event journal")
    return events


def make_branch(source, proposal, identity, *, original=False):
    events = source_events(source)
    chosen = next(e for e in events if e.get("agent_step") == proposal["agent_step"])
    step = chosen["step_data"]
    wire = copy.deepcopy(chosen["wire_response"])
    if original:
        replacement = {"wire_response": wire, "text": step["response_text"], "ids": step["response_ids"]}
    else:
        wire["choices"][0]["message"] = {
            "role": "assistant", "content": proposal["content"], "tool_calls": proposal["tool_calls"]}
        wire["choices"][0]["finish_reason"] = "tool_calls" if proposal["tool_calls"] else "stop"
        replacement = {"wire_response": wire, "text": proposal["text"], "ids": proposal["ids"]}
    return {"source_events": events, "source_steps": source["steps"],
            "target_step": proposal["agent_step"], "replacement": replacement,
            "pair_identity": identity, "original_budget_seconds": AGENT_SECONDS}


def comparison(original, repaired, *, source=None):
    if original.get("status") != "completed" or repaired.get("status") != "completed":
        return {"improved": False, "reason": "infrastructure_or_replay_failure"}
    source = source or original
    reviews = {"repair": review_runtime_drift(source, repaired)}
    if not reviews["repair"]["eligible"]:
        return {"improved": False, "reason": "replay_difference_requires_review",
                "state_reviews": reviews,
                "reward_delta_unverified": repaired["verifier"]["reward"] - original["verifier"]["reward"]}
    if "replay" in original:
        reviews["original"] = review_runtime_drift(source, original)
        if not reviews["original"]["eligible"]:
            return {"improved": False, "reason": "original_replay_difference_requires_review",
                    "state_reviews": reviews}
    accepted, reason = paired_accept(original["verifier"], repaired["verifier"])
    if reason == "no_strict_short_window_improvement":
        reason = "no_strict_full_task_improvement"
    return {"improved": accepted, "reason": reason, "state_reviews": reviews,
            "original_reward": original["verifier"]["reward"],
            "repaired_reward": repaired["verifier"]["reward"],
            "criterion_changes": {key: [value, repaired["verifier"]["criteria"].get(key)]
                                  for key, value in original["verifier"]["criteria"].items()
                                  if value != repaired["verifier"]["criteria"].get(key)}}


def repair_one(row, source, index, anchors, g, state, proxy):
    sid = source["session_id"]
    path = g.output_dir / "pairs" / (sid + ".json")
    if path.exists():
        return json.loads(path.read_text())
    result = {"session_id": sid, "task_id": source["task_id"], "split": source["split"],
              "accepted": False, "comparisons": [], "started_at": time.time()}
    if source.get("status") != "completed" or not source.get("steps"):
        result["reason"] = "incomplete_source"
    elif source["verifier"]["reward"] >= 1:
        result["reason"] = "source_already_satisfies_all_criteria"
    else:
        proposal = review_trajectory(source, anchors[source["task_id"]], proxy, g,
                                     g.output_dir / "teacher" / (sid + ".json"))
        result["teacher_status"] = proposal["status"]
        if proposal["status"] != "proposed":
            result["reason"] = "teacher_" + proposal["status"]
        else:
            branch = make_branch(source, proposal, sid)
            repaired = execute(row, source["sample"], sid + "-repair-screen", 1000+index,
                               g, state, proxy, branch)
            screen = comparison(source, repaired)
            result["comparisons"].append(screen | {"phase": "existing_full_source_screen"})
            result["reason"] = screen["reason"]
            if screen["improved"]:
                for repeat, seed_name in enumerate(("paired-314159", "paired-271828")):
                    identity = sid + ":" + seed_name
                    variants = {}
                    # Alternate the two execution orders, retaining a paired seed.
                    for side in (("original", "repair") if repeat % 2 == 0 else ("repair", "original")):
                        variants[side] = execute(
                            row, source["sample"], sid + f"-repeat{repeat}-{side}",
                            2000+index*4+repeat*2+(side == "repair"), g, state, proxy,
                            make_branch(source, proposal, identity, original=side == "original"))
                    check = comparison(variants["original"], variants["repair"], source=source)
                    result["comparisons"].append(check | {"phase": seed_name})
                result["accepted"] = all(x["improved"] for x in result["comparisons"])
                result["reason"] = "reliable_repeated_improvement" if result["accepted"] else "repeat_not_confirmed"
            # Exact action span and native sampler support audit for each proposal.
            try:
                from verified_repair_sampler_probe import audit_action
                selected = source["steps"][proposal["agent_step"]-1]
                saved_state = {"step": selected}
                actions = {"original": {"text": selected["response_text"], "ids": selected["response_ids"]},
                           "repair": proposal}
                result["sampler_audits"] = {
                    side: audit_action(g, saved_state, action, sid+"-"+side) for side, action in actions.items()}
                result["auxiliary_span_validated"] = True
                save_json(g.output_dir / "training_states" / (sid + ".json"), {
                    "task_id": source["task_id"], "session_id": sid, "split": source["split"],
                    "agent_step": proposal["agent_step"], "step": selected,
                    "repair": proposal, "is_policy_rollout": False,
                })
            except Exception as exc:
                result.update(accepted=False, reason="native_sampler_audit_unresolved",
                              auxiliary_span_validated=False, audit_error_type=type(exc).__name__)
    result["seconds"] = time.time() - result["started_at"]
    save_json(path, result)
    print(json.dumps({"pair": sid, "accepted": result["accepted"], "reason": result["reason"]}), flush=True)
    return result


def run(args):
    args.output = args.output.resolve()
    if args.output.parent != ROOT / "outputs" or not args.output.name.startswith("verified-repair-"):
        raise ValueError("output must be a new run-owned outputs/verified-repair-* directory")
    args.output.mkdir(parents=True, exist_ok=True)
    selection = manifest(args.output)
    source_output = getattr(args, "source_output", None)
    if source_output is not None:
        source_output = source_output.resolve()
        if source_output == args.output or source_output.parent != ROOT/"outputs":
            raise ValueError("source reuse requires a different original run-owned output")
        original_manifest = json.loads((source_output/"selection.json").read_text())
        for key in ("protocol", "original_selection_sha256", "source_seed", "temperature",
                    "top_p", "top_k", "student_model", "agent_wall_seconds"):
            if original_manifest.get(key) != selection[key]:
                raise ValueError("source protocol mismatch: " + key)
    if args.stage == "prepare":
        print(json.dumps({"output": str(args.output), "protocol": selection["protocol"],
                          "fixed_tasks": len(selection["candidate_tasks"])}))
        return
    if args.stage == "data" and not (args.output / "smoke_passed.json").exists():
        raise RuntimeError("reviewed end-to-end engineering gate required before fixed 64")
    args.backend_url = ""
    g = old.configure(args)
    rollout._build_gateway_config = gateway_config
    os.environ["OPENCLAW_MAX_COMPACTION_EVENTS"] = "8"
    g.engine_environment = prepare_environment(args.output, ROOT / "scripts")
    state = ReplayRemoteState(args.output, g.remote_base, old.gate.PINNED_IMAGE)
    preflight_ports(g)
    process = handle = proxy = tunnel = None
    identity = {"started_at": time.time(), "stage": args.stage, "workers": args.workers,
                "ports": [args.backend_port, args.proxy_port, args.reverse_port],
                "gpus": args.gpus, "optimizer_updates": 0,
                "code_sha256": {p.name: sha(p) for p in sorted((ROOT/"scripts").glob("*verified_repair*.py"))},
                "selection_sha256": sha(args.output / "selection.json"), "completed": False}
    identity["source_output"] = str(source_output) if source_output is not None else None
    save_json(args.output / (args.stage + "-execution.json"), identity)
    try:
        probe_env = dict(g.engine_environment, CUDA_VISIBLE_DEVICES=g.gpus)
        probe = subprocess.run([str(g.python), str(ROOT/"scripts/verify_repair_sampler_overlay.py"),
            "--device", "cuda", "--output", str(args.output/"sampler_overlay_cuda_check.json")],
            env=probe_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
        (args.output/"sampler_overlay_cuda_check.log").write_bytes(probe.stdout + probe.stderr)
        if probe.returncode:
            raise RuntimeError("isolated CUDA sampler endpoint check failed")
        process, handle = old.start_model(g)
        audit_file = args.output / "sampler_overlay_imports.jsonl"
        if not audit_file.exists() or not audit_file.read_text().strip():
            raise RuntimeError("no real SGLang worker imported the isolated sampler overlay")
        imports = [json.loads(line) for line in audit_file.read_text().splitlines()]
        engine_started = json.loads((args.output/"engine-command.json").read_text())["started_at"]
        if (not all(row["source_sha256"] == SOURCE_SHA256 for row in imports)
                or not any(row["time"] >= engine_started for row in imports)):
            raise RuntimeError("sampler import evidence changed")
        enabled = threading.Event()
        enabled.set()
        proxy = ReplayProxy(SimpleNamespace(hf_checkpoint=str(old.MODEL),
            sglang_router_ip="127.0.0.1", sglang_router_port=g.backend_port,
            rollout_temperature=.7, rollout_top_p=.8, rollout_top_k=20),
            queue.Queue(), enabled, remote_state=state)
        proxy.start()
        proxy.wait_until_ready(900)
        tunnel = old.gate._start_reverse_tunnel(g)
        rows = selection["smoke"] if args.stage == "smoke" else selection["candidate_tasks"]
        repeats = 1 if args.stage == "smoke" else 4
        jobs = [(row, sample, i*repeats+sample) for i, row in enumerate(rows) for sample in range(repeats)]
        if source_output is not None:
            sources, source_index = {}, {}
            for row, sample, _ in jobs:
                sid = f"source-{row['task_id']}-{sample}"
                path = source_output/"sources"/(sid+".json")
                record = json.loads(path.read_text())
                if (record["session_id"] != sid or record["task_id"] != row["task_id"]
                        or record["sample"] != sample
                        or record["split"] != row.get("split", "engineering_smoke")
                        or record["protocol"] != selection["protocol"]):
                    raise ValueError("source identity/protocol mismatch")
                sources[sid] = record
                source_index[sid] = {"path": str(path), "sha256": sha(path)}
            save_json(args.output/"reused_sources.json", source_index)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(execute, row, sample, f"source-{row['task_id']}-{sample}",
                                       index, g, state, proxy) for row, sample, index in jobs]
                sources = {r["session_id"]: r for f in as_completed(futures) if (r := f.result())}
        anchors = {r["task_id"]: r["anchors"] for line in old.ANCHORS.open() if (r := json.loads(line))}
        pairs = []
        for row, sample, index in jobs:
            source = sources[f"source-{row['task_id']}-{sample}"]
            pairs.append(repair_one(row, source, index, anchors, g, state, proxy))
        summary = {"source_count": len(sources), "pairs": pairs,
                   "accepted_train_pairs": sum(p["accepted"] and p["split"] == "train" for p in pairs),
                   "accepted_train_tasks": sorted({p["task_id"] for p in pairs
                                                  if p["accepted"] and p["split"] == "train"}),
                   "optimizer_updates": 0, "smoke_gate_requires_manual_engineering_review": True}
        save_json(args.output / (args.stage + "-summary.json"), summary)
        identity["completed"] = True
    finally:
        if proxy:
            proxy.closed_sessions.update(proxy.runtimes)
        if tunnel:
            old.gate._stop_process(tunnel, 10)
        if proxy:
            proxy.stop()
        if process:
            old.gate._stop_process(process, 30)
        if handle:
            handle.close()
        identity.update(finished_at=time.time(), seconds=time.time()-identity["started_at"])
        save_json(args.output / (args.stage + "-execution.json"), identity)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage", choices=["prepare", "smoke", "data"], required=True)
    p.add_argument("--source-output", type=Path,
                   help="Reuse unchanged v3 full source trajectories; write all new evidence to --output")
    p.add_argument("--gpus", default="0")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    p.add_argument("--backend-port", type=int, default=31780)
    p.add_argument("--proxy-port", type=int, default=31781)
    p.add_argument("--reverse-port", type=int, default=39781)
    run(p.parse_args())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
