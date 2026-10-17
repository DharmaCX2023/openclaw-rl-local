#!/usr/bin/env python3
"""Run exactly eight inference-only rollouts through real OpenClaw 2026.4.1.

The GPU node serves an RL checkpoint with SGLang.  A recording reverse proxy
captures the exact OpenAI requests emitted by OpenClaw.  Each task runs in its
own remote OpenClaw session and Docker sandbox.  No trainer is imported or
started.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_COUNT = 8
SIX_TOOLS = ["read", "write", "edit", "exec", "process", "web_fetch"]
SIX_SKILLS = [
    "clawflow",
    "clawflow-inbox-triage",
    "healthcheck",
    "node-connect",
    "skill-creator",
    "weather",
]
DEFAULT_MODEL = Path(
    "/root/OpenClaw-RL/hf_ckpt/"
    "qwen3-30b-a3b-instruct-clawgym-rl100-iter99-20260730"
)
DEFAULT_TASKS = Path("/root/OpenClaw-RL/data/clawgym-rl/tasks-smoke.jsonl")
OPENCLAW_IMAGE = "ghcr.io/openclaw/openclaw:2026.4.1"
SANDBOX_IMAGE = "clawgym-rl:v0.1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def run(
    cmd: list[str],
    *,
    check: bool = True,
    timeout: int | None = None,
    input_bytes: bytes | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        cmd,
        input=input_bytes,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=check,
        timeout=timeout,
    )


def wait_http(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    last = "not attempted"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"endpoint did not become ready: {url}: {last}")


class TraceStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self.lock:
            self.records.append(record)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.records)


def start_trace_proxy(
    host: str, port: int, backend_host: str, backend_port: int, store: TraceStore
) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _forward(self) -> None:
            request_id = uuid.uuid4().hex
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                parsed_body = json.loads(body) if body else None
            except json.JSONDecodeError:
                parsed_body = body.decode("utf-8", errors="replace")
            store.append(
                {
                    "kind": "request",
                    "request_id": request_id,
                    "time": time.time(),
                    "method": self.command,
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": parsed_body,
                }
            )

            connection = http.client.HTTPConnection(backend_host, backend_port, timeout=900)
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"host", "content-length", "connection"}
            }
            if body:
                headers["Content-Length"] = str(len(body))
            response_bytes = bytearray()
            try:
                connection.request(self.command, self.path, body=body or None, headers=headers)
                upstream = connection.getresponse()
                self.send_response(upstream.status)
                for key, value in upstream.getheaders():
                    if key.lower() not in {
                        "content-length",
                        "transfer-encoding",
                        "connection",
                    }:
                        self.send_header(key, value)
                self.end_headers()
                while True:
                    chunk = upstream.read(65536)
                    if not chunk:
                        break
                    response_bytes.extend(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                store.append(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "time": time.time(),
                        "status": upstream.status,
                        "body": response_bytes.decode("utf-8", errors="replace"),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                store.append(
                    {
                        "kind": "proxy_error",
                        "request_id": request_id,
                        "time": time.time(),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                try:
                    self.send_error(502, str(exc))
                except Exception:  # noqa: BLE001
                    pass
            finally:
                connection.close()

        do_GET = _forward
        do_POST = _forward

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def start_sglang(args: argparse.Namespace, output_dir: Path) -> subprocess.Popen[bytes] | None:
    if args.backend_url:
        wait_http(args.backend_url.rstrip("/") + "/health", 30)
        return None
    log_path = output_dir / "sglang.log"
    log_handle = log_path.open("wb")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.model),
        "--served-model-name",
        args.served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.backend_port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--data-parallel-size",
        str(getattr(args, "data_parallel_size", 1)),
        "--context-length",
        "65536",
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--random-seed",
        str(args.seed),
        "--trust-remote-code",
        "--disable-cuda-graph",
    ]
    if args.sglang_tool_call_parser:
        cmd.extend(["--tool-call-parser", args.sglang_tool_call_parser])
    if args.sglang_reasoning_parser:
        cmd.extend(["--reasoning-parser", args.sglang_reasoning_parser])
    print("Starting inference server (no trainer):", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    try:
        wait_http(f"http://127.0.0.1:{args.backend_port}/health", args.serve_timeout)
    except Exception:
        process.terminate()
        raise
    return process


def ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-i",
        str(args.ssh_key),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        f"{args.remote_user}@{args.remote_host}",
    ]


def make_payload(
    row: dict[str, Any],
    run_id: str,
    remote_root: str,
    reverse_port: int,
    served_model: str,
) -> bytes:
    metadata = row.get("metadata") or {}
    with tempfile.TemporaryDirectory(prefix="oc8-payload-") as temp_name:
        temp = Path(temp_name)
        state = temp / "state"
        workspace = temp / "workspace"
        (state / "skills" / "serper").mkdir(parents=True)
        workspace.mkdir(parents=True)
        shutil.copy2(
            ROOT / "sft_contract/recovered_skills/serper/SKILL.md",
            state / "skills/serper/SKILL.md",
        )
        for item in metadata.get("input_files") or []:
            relative = Path(str(item["file_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe task input path: {relative}")
            destination = workspace / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(str(item.get("content", "")), encoding="utf-8")
        (temp / "task.txt").write_text(str(row.get("task") or metadata.get("prompt") or ""), encoding="utf-8")
        (temp / "task.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

        config = {
            "agents": {
                "defaults": {
                    "workspace": f"{remote_root}/workspace",
                    "model": {"primary": f"vllm/{served_model}"},
                    "userTimezone": "Asia/Shanghai",
                    "sandbox": {
                        "mode": "all",
                        "backend": "docker",
                        "scope": "session",
                        "workspaceAccess": "rw",
                        "docker": {
                            "image": SANDBOX_IMAGE,
                            "network": "none",
                            "containerPrefix": f"oc8-{run_id}-",
                        },
                    },
                }
            },
            "models": {
                "mode": "merge",
                "providers": {
                    "vllm": {
                        "baseUrl": f"http://127.0.0.1:{reverse_port}/v1",
                        "apiKey": "openclaw-local-eval",
                        "api": "openai-completions",
                        "models": [
                            {
                                "id": served_model,
                                "name": served_model,
                                "reasoning": False,
                                "input": ["text"],
                                "cost": {
                                    "input": 0,
                                    "output": 0,
                                    "cacheRead": 0,
                                    "cacheWrite": 0,
                                },
                                "contextWindow": 65536,
                                "maxTokens": 4096,
                            }
                        ],
                    }
                },
            },
            "tools": {
                "allow": SIX_TOOLS,
                "deny": ["apply_patch", "image"],
                "exec": {"host": "sandbox", "security": "full", "ask": "off"},
                "elevated": {"enabled": False},
                "sandbox": {"tools": {"allow": SIX_TOOLS, "deny": ["apply_patch", "image"]}},
            },
            "skills": {
                "allowBundled": SIX_SKILLS,
                "load": {"watch": False},
                "entries": {"serper": {"enabled": True}},
            },
        }
        (state / "openclaw.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        archive_path = temp / "payload.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in ["state", "workspace", "task.txt", "task.json"]:
                archive.add(temp / name, arcname=name)
        return archive_path.read_bytes()


def remote_rollout(
    args: argparse.Namespace,
    row: dict[str, Any],
    index: int,
    output_dir: Path,
    batch_id: str,
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    task_id = str(metadata.get("task_id") or metadata.get("task_name") or f"task-{index}")
    run_id = f"{batch_id}-{index:02d}-{uuid.uuid4().hex[:8]}"
    remote_root = f"{args.remote_base}/{run_id}"
    result: dict[str, Any] = {
        "index": index,
        "task_id": task_id,
        "run_id": run_id,
        "remote_root": remote_root,
        "status": "starting",
    }
    payload = make_payload(row, run_id, remote_root, args.reverse_port, args.served_model)
    prepare = f"mkdir -p {shlex.quote(remote_root)} && tar -xzf - -C {shlex.quote(remote_root)}"
    run(ssh_base(args) + [prepare], input_bytes=payload, timeout=120)

    session_id = f"oc8-{run_id}"
    docker_args = [
        "docker",
        "run",
        "--rm",
        "--user",
        "root",
        "--network",
        "host",
        "--hostname",
        "openclaw-batch-9-f2aec758-master-0",
        "-v",
        "/usr/bin/docker:/usr/local/bin/docker:ro",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-v",
        f"{remote_root}:{remote_root}",
        "-e",
        f"OPENCLAW_STATE_DIR={remote_root}/state",
        "-e",
        f"OPENCLAW_CONFIG_PATH={remote_root}/state/openclaw.json",
        "-e",
        "HOME=/root",
        OPENCLAW_IMAGE,
        "node",
        "openclaw.mjs",
        "agent",
        "--local",
        "--session-id",
        session_id,
        "--message",
    ]
    task_file = f"{remote_root}/task.txt"
    quoted_prefix = " ".join(shlex.quote(part) for part in docker_args)
    remote_command = (
        f"{quoted_prefix} \"$(cat {shlex.quote(task_file)})\" "
        f"--thinking low --timeout {int(args.rollout_timeout)} --json"
    )
    started = time.time()
    completed = run(
        ssh_base(args) + [remote_command],
        check=False,
        timeout=args.rollout_timeout + 120,
    )
    result.update(
        {
            "elapsed_seconds": round(time.time() - started, 3),
            "exit_code": completed.returncode,
            "stdout": completed.stdout.decode("utf-8", errors="replace"),
            "stderr": completed.stderr.decode("utf-8", errors="replace"),
            "status": "ok" if completed.returncode == 0 else "failed",
        }
    )

    hook = str(metadata.get("hook_code") or "").strip()
    if hook:
        verify_command = " ".join(
            shlex.quote(part)
            for part in [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{remote_root}/workspace:/root/.openclaw/workspace",
                SANDBOX_IMAGE,
                "sh",
                "-lc",
                hook,
            ]
        )
        verified = run(
            ssh_base(args) + [verify_command], check=False, timeout=args.verifier_timeout
        )
        result["verifier"] = {
            "exit_code": verified.returncode,
            "stdout": verified.stdout.decode("utf-8", errors="replace"),
            "stderr": verified.stderr.decode("utf-8", errors="replace"),
        }

    local_rollout = output_dir / "rollouts" / f"{index:02d}-{task_id}"
    local_rollout.mkdir(parents=True, exist_ok=True)
    archive_result = run(
        ssh_base(args)
        + [f"tar -czf - -C {shlex.quote(remote_root)} state workspace task.json"],
        check=False,
        timeout=180,
    )
    if archive_result.returncode == 0:
        archive_path = local_rollout / "artifacts.tar.gz"
        archive_path.write_bytes(archive_result.stdout)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(local_rollout, filter="data")
    else:
        result["artifact_error"] = archive_result.stderr.decode("utf-8", errors="replace")

    cleanup = (
        f"docker ps -aq --filter name={shlex.quote('oc8-' + run_id + '-')} "
        "| xargs -r docker rm -f"
    )
    run(ssh_base(args) + [cleanup], check=False, timeout=60)
    (local_rollout / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{index + 1}/{ROLLOUT_COUNT}] {task_id}: {result['status']}", flush=True)
    return result


def extract_tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    for tool in tools or []:
        if isinstance(tool, dict):
            function = tool.get("function") or {}
            if function.get("name"):
                names.append(str(function["name"]))
    return names


def build_alignment_report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_system = (ROOT / "sft_contract/system_prompt.txt").read_text(encoding="utf-8")
    expected_tools = json.loads(
        (ROOT / "sft_contract/six_tool_schemas.json").read_text(encoding="utf-8")
    )
    requests = [
        record
        for record in records
        if record.get("kind") == "request"
        and record.get("path", "").endswith("/chat/completions")
        and isinstance(record.get("body"), dict)
    ]
    task_prompts = [str(row.get("task") or (row.get("metadata") or {}).get("prompt") or "") for row in rows]
    per_request: list[dict[str, Any]] = []
    observed_call_names: list[str] = []
    for record in requests:
        body = record["body"]
        messages = body.get("messages") or []
        system_messages = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "system"
        ]
        user_contents = [
            str(message.get("content") or "")
            for message in messages
            if message.get("role") == "user"
        ]
        tools = body.get("tools") or []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                name = str((tool_call.get("function") or {}).get("name") or "")
                if name:
                    observed_call_names.append(name)
        matched_task = next(
            (index for index, prompt in enumerate(task_prompts) if prompt in user_contents),
            None,
        )
        actual_system = system_messages[0] if system_messages else ""
        per_request.append(
            {
                "request_id": record["request_id"],
                "matched_task_index": matched_task,
                "message_count": len(messages),
                "tool_names": extract_tool_names(tools),
                "six_tool_names_exact": set(extract_tool_names(tools)) == set(SIX_TOOLS) and len(extract_tool_names(tools)) == len(SIX_TOOLS),
                "tool_schemas_exact_sft": canonical_json(tools) == canonical_json(expected_tools),
                "system_prompt_exact_sft": actual_system == expected_system,
                "system_prompt_sha256": sha256_text(actual_system),
                "skills_present": {
                    skill: skill in actual_system for skill in SIX_SKILLS + ["serper"]
                },
            }
        )
    report = {
        "inference_only": True,
        "rollout_limit": ROLLOUT_COUNT,
        "openclaw_image": OPENCLAW_IMAGE,
        "sandbox": {
            "mode": "all",
            "scope": "session",
            "image": SANDBOX_IMAGE,
            "host_or_elevated_exec": False,
        },
        "sft_contract": {
            "system_prompt_sha256": sha256_text(expected_system),
            "tool_schema_sha256": sha256_text(canonical_json(expected_tools)),
            "tool_names": extract_tool_names(expected_tools),
        },
        "summary": {
            "rollouts_requested": ROLLOUT_COUNT,
            "rollouts_completed": len(results),
            "rollouts_ok": sum(result.get("status") == "ok" for result in results),
            "model_requests": len(requests),
            "all_requests_six_tools_exact": bool(per_request)
            and all(item["six_tool_names_exact"] for item in per_request),
            "all_tool_schemas_exact_sft": bool(per_request)
            and all(item["tool_schemas_exact_sft"] for item in per_request),
            "all_system_prompts_exact_sft": bool(per_request)
            and all(item["system_prompt_exact_sft"] for item in per_request),
            "assistant_tool_calls_only_six": set(observed_call_names).issubset(SIX_TOOLS),
            "observed_assistant_tool_names": sorted(set(observed_call_names)),
        },
        "requests": per_request,
        "rollouts": results,
        "interpretation": (
            "Exact=true means the real OpenClaw request matches the filtered SFT contract "
            "byte-for-byte for that field. Exact=false is a migration difference to inspect; "
            "it is not silently normalized by this script."
        ),
    }
    (output_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--python", default="/root/OpenClaw-RL/.venv-openclaw/bin/python")
    parser.add_argument("--served-model", default="MiniMaxAI/MiniMax-M2.5")
    parser.add_argument("--backend-url", default="")
    parser.add_argument("--backend-port", type=int, default=30080)
    parser.add_argument("--proxy-port", type=int, default=30081)
    parser.add_argument("--reverse-port", type=int, default=39081)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sglang-tool-call-parser", default="", help="Empty returns raw model text to OpenClaw.")
    parser.add_argument("--sglang-reasoning-parser", default="")
    parser.add_argument("--serve-timeout", type=int, default=600)
    parser.add_argument("--rollout-timeout", type=int, default=900)
    parser.add_argument("--verifier-timeout", type=int, default=120)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--ssh-key", type=Path, default=Path("/root/openclaw_cpu_key"))
    parser.add_argument("--remote-host", default="159.138.50.147")
    parser.add_argument("--remote-user", default="root")
    parser.add_argument("--remote-base", default="/root/openclaw_harness_eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.tasks.is_file():
        raise FileNotFoundError(args.tasks)
    if args.parallel < 1 or args.parallel > ROLLOUT_COUNT:
        raise ValueError("--parallel must be between 1 and 8")
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = args.output_dir or ROOT / "outputs" / f"openclaw-8rollout-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    with args.tasks.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == ROLLOUT_COUNT:
                break
    if len(rows) != ROLLOUT_COUNT:
        raise ValueError(f"need exactly 8 tasks, found {len(rows)}")
    (output_dir / "selected_tasks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    trace_store = TraceStore(output_dir / "openai_trace.jsonl")
    sglang = start_sglang(args, output_dir)
    proxy = start_trace_proxy(
        "127.0.0.1", args.proxy_port, "127.0.0.1", args.backend_port, trace_store
    )
    tunnel = subprocess.Popen(
        ssh_base(args)[:-1]
        + [
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"127.0.0.1:{args.reverse_port}:127.0.0.1:{args.proxy_port}",
            ssh_base(args)[-1],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    if tunnel.poll() is not None:
        stderr = tunnel.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"reverse SSH tunnel failed: {stderr}")

    batch_id = f"oc8-{timestamp}"
    results: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(remote_rollout, args, row, index, output_dir, batch_id): index
                for index, row in enumerate(rows)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    failure = {
                        "index": index,
                        "task_id": str(
                            (rows[index].get("metadata") or {}).get("task_id") or f"task-{index}"
                        ),
                        "status": "orchestrator_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results.append(failure)
                    print(f"[{index + 1}/8] {failure['task_id']}: orchestrator_error", flush=True)
    finally:
        tunnel.send_signal(signal.SIGTERM)
        try:
            tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
        proxy.shutdown()
        if sglang is not None:
            sglang.send_signal(signal.SIGTERM)
            try:
                sglang.wait(timeout=30)
            except subprocess.TimeoutExpired:
                sglang.kill()

    results.sort(key=lambda item: int(item["index"]))
    report = build_alignment_report(output_dir, rows, results, trace_store.snapshot())
    summary = report["summary"]
    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))
    return 0 if summary["rollouts_ok"] == ROLLOUT_COUNT else 2


if __name__ == "__main__":
    raise SystemExit(main())
