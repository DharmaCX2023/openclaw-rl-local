#!/usr/bin/env python3
"""Inference-only rollouts through the official ClawGym RL harness shape.

Baseline: ClawGym/ClawGym-Agents commit 0e8526d8f5e0999896c74cee4d6b998fb8a0429e.
Only required adaptations are applied: six tools, pinned skills, and remote CPU
Docker. No trainer or Ray job is started.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "clawgym-official"
sys.path.insert(0, str(OFFICIAL))
from clawgym_rl_api_server import ClawGymRLAPIServer  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "diagnostic_helpers", ROOT / "scripts/run_openclaw_8rollout_inference.py"
)
helpers = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(helpers)

TOOLS = ["read", "edit", "write", "exec", "process", "web_fetch"]
SKILLS = [
    "clawflow", "clawflow-inbox-triage", "healthcheck",
    "node-connect", "skill-creator", "weather",
]
MODEL = Path("/root/OpenClaw-RL/hf_ckpt/qwen3-30b-a3b-instruct-clawgym-rl100-iter99-20260730")
TASKS = Path("/root/OpenClaw-RL/data/clawgym-rl/tasks-smoke.jsonl")
IMAGE = "clawgym-rl:v0.1"
COMMIT = "0e8526d8f5e0999896c74cee4d6b998fb8a0429e"


def ssh(args: argparse.Namespace) -> list[str]:
    return [
        "ssh", "-i", str(args.ssh_key), "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no", f"{args.remote_user}@{args.remote_host}",
    ]


def run(cmd: list[str], *, data: bytes | None = None, timeout: int = 120, check: bool = True):
    return subprocess.run(
        cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, check=check,
    )


def post_json(url: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def official_config(gateway_port: int, token: str, model_id: str, reverse_port: int) -> dict[str, Any]:
    """Official _build_gateway_config with only tools/skills/connectivity adapted."""
    model_parts = model_id.split("__", 1)
    if len(model_parts) != 2 or not model_parts[1]:
        raise ValueError("model_id must be '<turn_type>__<session_id>'")
    compaction_model_id = f"compact__{model_parts[1]}"

    def model_entry(entry_id: str, name: str) -> dict[str, Any]:
        return {
            "id": entry_id,
            "name": name,
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 65536,
            "maxTokens": 8192,
        }

    return {
        "gateway": {
            "port": gateway_port,
            "mode": "local",
            "bind": "loopback",
            "auth": {"mode": "token", "token": token},
            "http": {"endpoints": {"chatCompletions": {"enabled": True}}},
            "tailscale": {"mode": "off", "resetOnExit": False},
            "nodes": {"denyCommands": [
                "camera.snap", "camera.clip", "screen.record", "contacts.add",
                "calendar.add", "reminders.add", "sms.send",
            ]},
        },
        "models": {
            "mode": "merge",
            "providers": {
                "rl-server": {
                    "baseUrl": f"http://localhost:{reverse_port}/v1",
                    "api": "openai-completions",
                    "models": [
                        model_entry(model_id, "RL Model"),
                        model_entry(compaction_model_id, "RL Compaction Model"),
                    ],
                }
            },
        },
        "agents": {"defaults": {
            "model": {"primary": f"rl-server/{model_id}"},
            "workspace": "/root/.openclaw/workspace",
            "userTimezone": "Asia/Shanghai",
            "maxConcurrent": 4,
            "subagents": {"maxConcurrent": 4},
            "timeoutSeconds": 1000,
            "llm": {"idleTimeoutSeconds": 0},
            "heartbeat": {"every": "35791m"},
            "compaction": {"model": f"rl-server/{compaction_model_id}"},
        }},
        "tools": {
            "allow": TOOLS,
            "deny": ["apply_patch", "image", "web_search", "memory_search"],
            "exec": {"host": "gateway", "security": "full"},
        },
        "skills": {
            "allowBundled": SKILLS,
            "load": {"watch": False},
            "entries": {"serper": {"enabled": True}},
        },
        "commands": {
            "native": "auto", "nativeSkills": "auto",
            "restart": True, "ownerDisplay": "raw",
        },
        "session": {"dmScope": "per-channel-peer"},
    }


def payload(row: dict[str, Any], gateway_port: int, token: str, model_id: str, reverse_port: int) -> bytes:
    with tempfile.TemporaryDirectory(prefix="clawgym-official-8-") as name:
        root = Path(name)
        state = root / "state"
        workspace = root / "workspace"
        (state / "skills/serper").mkdir(parents=True)
        workspace.mkdir()
        shutil.copy2(
            ROOT / "sft_contract/recovered_skills/serper/SKILL.md",
            state / "skills/serper/SKILL.md",
        )
        for item in (row.get("metadata") or {}).get("input_files") or []:
            relative = Path(str(item["file_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe input path: {relative}")
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(item.get("content", "")), encoding="utf-8")
        (state / "openclaw.json").write_text(
            json.dumps(official_config(gateway_port, token, model_id, reverse_port), indent=2),
            encoding="utf-8",
        )
        prompt = str(row.get("task") or (row.get("metadata") or {}).get("prompt") or "")
        (root / "chat.json").write_text(
            json.dumps({"model": "openclaw", "stream": False,
                        "messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "task.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        archive = root / "payload.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            for child in (state, workspace, root / "chat.json", root / "task.json"):
                tf.add(child, arcname=child.name)
        return archive.read_bytes()


def one(args: argparse.Namespace, row: dict[str, Any], index: int, out: Path, batch: str) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    task_id = str(meta.get("task_id") or meta.get("task_name") or f"task-{index}")
    session_id = f"task-{batch}-{index}"
    model_id = f"main__{session_id}"
    gateway_port = args.gateway_port_base + index
    token = uuid.uuid4().hex
    remote_root = f"{args.remote_base}/{batch}-{index:02d}"
    container = f"clawgym-rl-{batch}-{index:02d}"
    archive = payload(row, gateway_port, token, model_id, args.reverse_port)
    prepare = f"mkdir -p {shlex.quote(remote_root)} && tar -xzf - -C {shlex.quote(remote_root)}"
    run(ssh(args) + [prepare], data=archive)

    docker = [
        "docker", "run", "-d", "--name", container, "--network", "host", "--user", "root",
        "-e", f"OPENCLAW_GATEWAY_TOKEN={token}", "-e", "HOME=/root",
        "-e", "TZ=Asia/Shanghai",
        "-v", f"{remote_root}/state:/root/.openclaw",
        "-v", f"{remote_root}/workspace:/root/.openclaw/workspace",
        IMAGE, "node", "dist/index.js", "gateway", "--bind", "loopback",
        "--port", str(gateway_port), "--token", token,
    ]
    run(ssh(args) + ["docker rm -f " + shlex.quote(container)], check=False)
    started = run(ssh(args) + [" ".join(shlex.quote(x) for x in docker)], check=False)
    result: dict[str, Any] = {
        "index": index, "task_id": task_id, "session_id": session_id,
        "container": container, "gateway_port": gateway_port,
        "start_exit_code": started.returncode,
    }
    t0 = time.time()
    try:
        wait_and_chat = (
            f"for i in $(seq 1 120); do curl -fsS --max-time 2 "
            f"http://127.0.0.1:{gateway_port}/health >/dev/null && break; sleep 1; done; "
            f"curl -sS --fail --max-time {args.rollout_timeout} "
            f"-H {shlex.quote('Authorization: Bearer ' + token)} "
            f"-H 'Content-Type: application/json' "
            f"-H 'X-OpenClaw-Scopes: operator.admin,operator.read,operator.write' "
            f"--data-binary @{shlex.quote(remote_root + '/chat.json')} "
            f"http://127.0.0.1:{gateway_port}/v1/chat/completions"
        )
        chat = run(ssh(args) + [wait_and_chat], timeout=args.rollout_timeout + 180, check=False)
        result.update({
            "exit_code": chat.returncode,
            "status": "ok" if chat.returncode == 0 else "failed",
            "chat_response": chat.stdout.decode(errors="replace"),
            "stderr": chat.stderr.decode(errors="replace"),
            "elapsed_seconds": round(time.time() - t0, 3),
        })
        try:
            conversation = post_json(
                f"http://127.0.0.1:{args.proxy_port}/get_conversation_log",
                {"session_id": session_id},
            )
            result["react_steps"] = conversation.get("steps", [])
            result["raw_prompts"] = conversation.get("raw_prompts", {})
        except Exception as exc:
            result["conversation_error"] = f"{type(exc).__name__}: {exc}"

        hook = str(meta.get("hook_code") or "").strip()
        if hook:
            # The flattened smoke JSONL carries only a hook string; unlike the
            # official task-directory format, it does not carry reward/.
            result["verifier"] = {
                "status": "not_run",
                "reason": "flattened smoke JSONL has no official reward/ assets",
                "hook": hook,
            }
        run(ssh(args) + [f"docker logs {shlex.quote(container)}"], check=False)
        local = out / "rollouts" / f"{index:02d}-{task_id}"
        local.mkdir(parents=True, exist_ok=True)
        fetched = run(
            ssh(args) + [f"tar -czf - -C {shlex.quote(remote_root)} state workspace task.json"],
            timeout=180, check=False,
        )
        if fetched.returncode == 0:
            artifact = local / "artifacts.tar.gz"
            artifact.write_bytes(fetched.stdout)
            with tarfile.open(artifact, "r:gz") as tf:
                tf.extractall(local, filter="data")
        (local / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{index + 1}/{args.rollout_count}] {task_id}: {result['status']} steps={len(result.get('react_steps', []))}", flush=True)
        return result
    finally:
        run(ssh(args) + ["docker rm -f " + shlex.quote(container)], check=False)


def report(out: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_system = (ROOT / "sft_contract/system_prompt.txt").read_text(encoding="utf-8")
    expected_tools = json.loads((ROOT / "sft_contract/six_tool_schemas.json").read_text())
    calls: list[str] = []
    observations: list[str] = []
    prompt_checks = []
    for result in results:
        raw = result.get("raw_prompts") or {}
        actual_system = str(raw.get("system_prompt") or "")
        try:
            actual_tools = json.loads(raw.get("tools") or "[]")
        except json.JSONDecodeError:
            actual_tools = []
        for step in result.get("react_steps") or []:
            calls.extend(str(x.get("name") or "") for x in step.get("tool_calls") or [])
            observations.extend(str(x) for x in step.get("observations") or [])
        prompt_checks.append({
            "task_id": result["task_id"],
            "system_prompt_exact_sft": actual_system == expected_system,
            "tool_schemas_exact_sft": helpers.canonical_json(actual_tools) == helpers.canonical_json(expected_tools),
            "tool_names": helpers.extract_tool_names(actual_tools),
            "skills_present": {name: name in actual_system for name in SKILLS + ["serper"]},
        })
    data = {
        "baseline": {"repo": "ClawGym/ClawGym-Agents", "commit": COMMIT},
        "inference_only": True,
        "rollout_count": len(results),
        "necessary_adaptations": [
            "six-tool allowlist", "fixed six bundled skills plus pinned Serper SKILL.md",
            "reverse tunnel from remote CPU Docker host to local policy server",
        ],
        "unchanged_official_semantics": {
            "container_is_sandbox": True,
            "workspace": "/root/.openclaw/workspace",
            "exec": {"host": "gateway", "security": "full"},
            "model_reasoning": False,
            "sglang_tool_call_parser": "qwen25",
            "sglang_reasoning_parser": None,
            "gateway_chat_api": "/v1/chat/completions",
        },
        "summary": {
            "rollouts_ok": sum(x.get("status") == "ok" for x in results),
            "react_steps": sum(len(x.get("react_steps") or []) for x in results),
            "tool_calls": len(calls),
            "observations": len(observations),
            "tool_calls_only_six": set(calls).issubset(TOOLS),
            "observed_tool_names": sorted(set(calls)),
            "all_tool_schemas_exact_sft": bool(prompt_checks) and all(x["tool_schemas_exact_sft"] for x in prompt_checks),
            "all_system_prompts_exact_sft": bool(prompt_checks) and all(x["system_prompt_exact_sft"] for x in prompt_checks),
            "all_fixed_skills_present": bool(prompt_checks) and all(all(x["skills_present"].values()) for x in prompt_checks),
            "observations_are_strings": all(isinstance(x, str) for x in observations),
        },
        "prompt_checks": prompt_checks,
        "rollouts": results,
    }
    (out / "alignment_report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/clawgym-official-8rollout")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--tasks", type=Path, default=TASKS)
    p.add_argument("--rollout-count", type=int, default=8)
    p.add_argument("--python", default="/root/OpenClaw-RL/.venv-openclaw/bin/python")
    p.add_argument("--served-model", default="clawgym-rl100")
    p.add_argument("--backend-url", default="")
    p.add_argument("--backend-port", type=int, default=30080)
    p.add_argument("--proxy-port", type=int, default=30081)
    p.add_argument("--reverse-port", type=int, default=39081)
    p.add_argument("--gateway-port-base", type=int, default=19200)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--data-parallel-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--serve-timeout", type=int, default=600)
    p.add_argument("--rollout-timeout", type=int, default=900)
    p.add_argument("--verifier-timeout", type=int, default=120)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--sglang-tool-call-parser", default="qwen25")
    p.add_argument("--sglang-reasoning-parser", default="")
    p.add_argument("--ssh-key", type=Path, default=Path("/root/openclaw_cpu_key"))
    p.add_argument("--remote-host", default="159.138.50.147")
    p.add_argument("--remote-user", default="root")
    p.add_argument("--remote-base", default="/root/clawgym_official_eval")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.rollout_count < 1:
        raise ValueError("rollout-count must be positive")
    if args.parallel < 1 or args.parallel > args.rollout_count:
        raise ValueError("parallel must be between 1 and rollout-count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with args.tasks.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == args.rollout_count:
                break
    if len(rows) != args.rollout_count:
        raise ValueError(f"exactly {args.rollout_count} tasks are required")
    (args.output_dir / "selected_tasks.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8"
    )

    sglang = helpers.start_sglang(args, args.output_dir)
    os.environ.update({
        "HOST": "127.0.0.1", "PORT": str(args.proxy_port), "SGLANG_API_KEY": "",
        "OPENCLAW_RECORD_ENABLED": "1",
        "OPENCLAW_TOKEN_LEDGER_MODE": "enforce",
        "OPENCLAW_RECORD_FILE": str(args.output_dir / "official_proxy_records.jsonl"),
    })
    enabled = threading.Event(); enabled.set()
    proxy_args = SimpleNamespace(
        hf_checkpoint=str(args.model), sglang_router_ip="127.0.0.1",
        sglang_router_port=args.backend_port,
    )
    server = ClawGymRLAPIServer(proxy_args, queue.Queue(), enabled)
    server.start(); server.wait_until_ready(120)

    tunnel = subprocess.Popen(
        ssh(args)[:-1] + ["-N", "-o", "ExitOnForwardFailure=yes", "-R",
                         f"127.0.0.1:{args.reverse_port}:127.0.0.1:{args.proxy_port}", ssh(args)[-1]],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    time.sleep(3)
    if tunnel.poll() is not None:
        raise RuntimeError(tunnel.stderr.read().decode(errors="replace"))

    batch = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    results = []
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(one, args, row, i, args.output_dir, batch): i for i, row in enumerate(rows)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    error = {"index": i, "task_id": str((rows[i].get("metadata") or {}).get("task_id")),
                             "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    results.append(error)
                    print(f"[{i + 1}/ {args.rollout_count}] {error['task_id']}: error {error['error']}", flush=True)
    finally:
        tunnel.send_signal(signal.SIGTERM)
        try: tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired: tunnel.kill()
        server.stop()
        if sglang is not None:
            sglang.send_signal(signal.SIGTERM)
            try: sglang.wait(timeout=30)
            except subprocess.TimeoutExpired: sglang.kill()
    results.sort(key=lambda x: x["index"])
    data = report(args.output_dir, results)
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    return 0 if data["summary"]["rollouts_ok"] == args.rollout_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
