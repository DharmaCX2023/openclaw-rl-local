#!/usr/bin/env python3
"""Exercise OpenClaw multi-tool execution against a deterministic fake model.

This test intentionally does not load or forward .env. It runs the pinned
ClawGym image on the configured CPU rollout host and sends model traffic
through a temporary reverse SSH tunnel.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "clawgym-official"))
sys.path.insert(0, str(REPO_ROOT / "slime"))

os.environ.setdefault("OPENCLAW_REMOTE_HOST", "159.138.50.147")
os.environ.setdefault("OPENCLAW_REMOTE_USER", "root")
os.environ.setdefault("OPENCLAW_REMOTE_SSH_KEY", "/root/openclaw_cpu_key")
os.environ.setdefault("OPENCLAW_REMOTE_BASE", "/root/clawgym_rl_runs")
os.environ.setdefault(
    "OPENCLAW_GATEWAY_IMAGE",
    "clawgym-rl@sha256:33a30ab107cb7fef6a84f17b3fb70a0e849654e9201c61dc9f20baeac21385a3",
)
os.environ.setdefault(
    "CLAWGYM_SERPER_SKILL_DIR",
    str(REPO_ROOT / "sft_contract/recovered_skills/serper"),
)

import clawgym_rl_rollout as rollout


SCENARIOS: dict[str, list[dict[str, Any]]] = {
    "two_read": [
        {
            "id": "callreada",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/a.txt"}
                ),
            },
        },
        {
            "id": "callreadb",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/b.txt"}
                ),
            },
        },
    ],
    "three_write": [
        {
            "id": f"callwrite{index}",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps(
                    {
                        "path": f"/root/.openclaw/workspace/out/{name}",
                        "content": content,
                    }
                ),
            },
        }
        for index, (name, content) in enumerate(
            (
                ("report.json", '{"ok":true}\n'),
                ("intent.json", '{"intent":"test"}\n'),
                ("plan.md", "# Plan\n"),
            )
        )
    ],
    "mixed": [
        {
            "id": "callmixedread",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/config.txt"}
                ),
            },
        },
        {
            "id": "callmixedexec",
            "type": "function",
            "function": {
                "name": "exec",
                "arguments": json.dumps(
                    {
                        "command": (
                            "test -f /root/.openclaw/workspace/input/config.txt "
                            "&& printf validated > /root/.openclaw/workspace/validated.txt"
                        )
                    }
                ),
            },
        },
        {
            "id": "callmixedwrite",
            "type": "function",
            "function": {
                "name": "write",
                "arguments": json.dumps(
                    {
                        "path": "/root/.openclaw/workspace/result.txt",
                        "content": "written",
                    }
                ),
            },
        },
    ],
    "partial_failure": [
        {
            "id": "callpartiala",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/a.txt"}
                ),
            },
        },
        {
            "id": "callpartialmissing",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/missing.txt"}
                ),
            },
        },
        {
            "id": "callpartialb",
            "type": "function",
            "function": {
                "name": "read",
                "arguments": json.dumps(
                    {"path": "/root/.openclaw/workspace/input/b.txt"}
                ),
            },
        },
    ],
}


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _scenario_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = _message_text(message)
        marker = "scenario:"
        if marker in text:
            return text.split(marker, 1)[1].split()[0].strip()
    raise AssertionError("scenario marker missing from model request")


class FakeModelState:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests: dict[str, list[dict[str, Any]]] = {}

    def record(self, scenario: str, body: dict[str, Any]):
        with self.lock:
            self.requests.setdefault(scenario, []).append(body)


STATE = FakeModelState()


class FakeModelHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        messages = body["messages"]
        scenario = _scenario_from_messages(messages)
        STATE.record(scenario, body)
        has_tool_results = any(message.get("role") == "tool" for message in messages)
        if has_tool_results:
            message = {
                "role": "assistant",
                "content": f"completed {scenario}",
            }
            finish_reason = "stop"
        else:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": SCENARIOS[scenario],
            }
            finish_reason = "tool_calls"
        if body.get("stream"):
            self._send_stream(message, finish_reason)
        else:
            self._send_json(message, finish_reason)

    def _send_json(self, message: dict[str, Any], finish_reason: str) -> None:
        payload = {
            "id": "fake-completion",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_stream(self, message: dict[str, Any], finish_reason: str) -> None:
        first = {
            "id": "fake-completion",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "delta": message,
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
        }
        final = {
            "id": "fake-completion",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "fake-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }
        payload = (
            f"data: {json.dumps(first)}\n\n"
            f"data: {json.dumps(final)}\n\n"
            "data: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _tool_messages(scenario: str) -> list[dict[str, Any]]:
    requests = STATE.requests.get(scenario, [])
    assert len(requests) >= 2, f"{scenario}: expected a follow-up model request"
    messages = requests[-1]["messages"]
    assistant_positions = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert assistant_positions, f"{scenario}: assistant tool-call message missing"
    return [
        message
        for message in messages[assistant_positions[-1] + 1 :]
        if message.get("role") == "tool"
    ]


def _assert_tool_linkage(scenario: str) -> list[dict[str, Any]]:
    results = _tool_messages(scenario)
    expected_ids = [call["id"] for call in SCENARIOS[scenario]]
    observed_ids = [result.get("tool_call_id") for result in results]
    assert observed_ids == expected_ids, (
        f"{scenario}: tool result IDs/order differ: "
        f"expected={expected_ids}, observed={observed_ids}"
    )
    return results


def _make_task(root: Path, scenario: str) -> rollout.TaskSpec:
    task_dir = root / scenario
    input_files = task_dir / "input_files"
    input_files.mkdir(parents=True)
    (input_files / "a.txt").write_text("alpha", encoding="utf-8")
    (input_files / "b.txt").write_text("bravo", encoding="utf-8")
    (input_files / "config.txt").write_text("enabled=true", encoding="utf-8")
    return rollout.TaskSpec(
        task_dir=task_dir,
        task_id=f"e2e-{scenario}",
        user_query=f"scenario:{scenario} execute every requested tool",
        input_mount_dir="/root/.openclaw/workspace/input",
        metadata={},
    )


def _run_scenario(root: Path, scenario: str, task_index: int, local_port: int) -> None:
    task = _make_task(root, scenario)
    runtime = rollout._start_container(
        task,
        rollout_id=991,
        task_index=task_index,
        rl_port=local_port,
        session_id=f"e2e-{scenario}",
        model_id=f"side__e2e-{scenario}",
    )
    try:
        rollout._wait_for_chat_ready(runtime)
        response = rollout._post_chat_turn(
            runtime,
            turn_type="side",
            session_done=True,
            messages=[{"role": "user", "content": task.user_query}],
        )
        assert rollout._extract_final_message(response) == f"completed {scenario}"
        results = _assert_tool_linkage(scenario)
        assert len(results) == len(SCENARIOS[scenario])

        if scenario == "two_read":
            rendered = [_message_text(result) for result in results]
            assert "alpha" in rendered[0]
            assert "bravo" in rendered[1]
        elif scenario == "three_write":
            assert (
                rollout._read_container_text_file(
                    runtime, "/root/.openclaw/workspace/out/report.json"
                )
                == '{"ok":true}'
            )
            assert (
                rollout._read_container_text_file(
                    runtime, "/root/.openclaw/workspace/out/intent.json"
                )
                == '{"intent":"test"}'
            )
            assert (
                rollout._read_container_text_file(
                    runtime, "/root/.openclaw/workspace/out/plan.md"
                )
                == "# Plan"
            )
        elif scenario == "mixed":
            assert (
                rollout._read_container_text_file(
                    runtime, "/root/.openclaw/workspace/validated.txt"
                )
                == "validated"
            )
            assert (
                rollout._read_container_text_file(
                    runtime, "/root/.openclaw/workspace/result.txt"
                )
                == "written"
            )
        elif scenario == "partial_failure":
            rendered = [_message_text(result) for result in results]
            assert "alpha" in rendered[0]
            assert rendered[1]
            assert "bravo" in rendered[2]
    finally:
        rollout._stop_container(runtime, None)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModelHandler)
    local_port = int(server.server_address[1])
    remote_port = 41000 + (os.getpid() % 10000)
    if remote_port > 55000:
        remote_port -= 10000
    os.environ["OPENCLAW_REMOTE_PROXY_PORT"] = str(remote_port)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    tunnel = subprocess.Popen(
        [
            "ssh",
            "-i",
            os.environ["OPENCLAW_REMOTE_SSH_KEY"],
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-N",
            "-R",
            f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}",
            f"{os.environ['OPENCLAW_REMOTE_USER']}@{os.environ['OPENCLAW_REMOTE_HOST']}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(2)
        if tunnel.poll() is not None:
            stderr = tunnel.stderr.read().decode(errors="replace")
            raise RuntimeError(f"reverse tunnel failed: {stderr[:500]}")
        with tempfile.TemporaryDirectory(prefix="openclaw-multitool-e2e-") as tmp:
            root = Path(tmp)
            for index, scenario in enumerate(SCENARIOS):
                _run_scenario(root, scenario, index, local_port)
                print(
                    f"{scenario}: calls={len(SCENARIOS[scenario])} "
                    f"results={len(_tool_messages(scenario))} order=ok",
                    flush=True,
                )
    finally:
        tunnel.terminate()
        try:
            tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:
            tunnel.kill()
            tunnel.wait(timeout=5)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

    print("multi-tool OpenClaw end-to-end test passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
