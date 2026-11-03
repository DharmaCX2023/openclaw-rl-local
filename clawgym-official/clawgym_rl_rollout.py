import asyncio
import atexit
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import shlex
import shutil
import socket
import string
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import requests

from clawgym_rl_api_server import ClawGymRLAPIServer
from opsd_evidence import (
    load_evidence_anchor_index,
    load_evidence_anchors,
    load_placebo_mapping,
)
from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample


_WEB_CREDENTIAL_KEYS = ("SERPER_URL", "SERPER_API_KEY")
_WEB_CREDENTIAL_ALIASES = {"SERPER_API_KEY": ("SERPER_KEY",)}


def _load_web_credentials() -> dict[str, str]:
    env_path = Path(
        os.environ.get(
            "OPENCLAW_ENV_FILE",
            str(Path(__file__).resolve().parents[1] / ".env"),
        )
    )
    parsed: dict[str, str] = {}
    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[7:].strip()
            accepted_keys = set(_WEB_CREDENTIAL_KEYS)
            for aliases in _WEB_CREDENTIAL_ALIASES.values():
                accepted_keys.update(aliases)
            if key not in accepted_keys:
                continue
            value = value.strip()
            if (
                len(value) >= 2
                and value[0] == value[-1]
                and value[0] in {"'", '"'}
            ):
                value = value[1:-1]
            parsed[key] = value

    credentials = {}
    for key in _WEB_CREDENTIAL_KEYS:
        value = os.environ.get(key, parsed.get(key, "")).strip()
        if not value:
            for alias in _WEB_CREDENTIAL_ALIASES.get(key, ()):
                value = os.environ.get(alias, parsed.get(alias, "")).strip()
                if value:
                    break
        credentials[key] = value
    missing = [key for key, value in credentials.items() if not value]
    if missing:
        raise RuntimeError(
            "missing required web credentials: " + ", ".join(missing)
        )
    invalid = [key for key, value in credentials.items() if "\n" in value or "\r" in value]
    if invalid:
        raise RuntimeError(
            "web credentials contain unsupported newlines: " + ", ".join(invalid)
        )
    return credentials


_OPTIONAL_GATEWAY_RUNTIME_ENV_KEYS = ("OPENCLAW_BUNDLED_PLUGINS_DIR",)


def _load_optional_gateway_runtime_env() -> dict[str, str]:
    runtime_env: dict[str, str] = {}
    for key in _OPTIONAL_GATEWAY_RUNTIME_ENV_KEYS:
        value = os.environ.get(key, "").strip()
        if not value:
            continue
        if "\n" in value or "\r" in value or "\x00" in value:
            raise RuntimeError(f"{key} contains unsupported control characters")
        if key == "OPENCLAW_BUNDLED_PLUGINS_DIR" and not value.startswith("/"):
            raise RuntimeError(f"{key} must be an absolute container path")
        runtime_env[key] = value
    return runtime_env


def _write_secret_env_file(
    container_name: str,
    credentials: dict[str, str],
    runtime_env: dict[str, str] | None = None,
) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"{container_name}.",
        suffix=".runtime.env",
        dir="/tmp",
        delete=False,
    ) as file_handle:
        file_handle.write(
            "".join(f"{key}={credentials[key]}\n" for key in _WEB_CREDENTIAL_KEYS)
        )
        file_handle.write(
            "".join(
                f"{key}={value}\n"
                for key, value in (runtime_env or {}).items()
            )
        )
        return Path(file_handle.name)

_CHAT_READY_TIMEOUT_SECONDS = 120
# Per-turn HTTP read timeout for chat/completions. Small models (4B) loop on
# malformed tool-call JSON and can push a single task past 900s; override via
# OPENCLAW_CHAT_TURN_TIMEOUT env so this only bumps for the runs that need it.
_CHAT_TURN_TIMEOUT_SECONDS = int(os.environ.get("OPENCLAW_CHAT_TURN_TIMEOUT", "900"))
_REWARD_TIMEOUT_SECONDS = 120
_CHAT_READY_POLL_SECONDS = 3
_CHAT_READY_SETTLE_SECONDS = float(
    os.environ.get("OPENCLAW_CHAT_READY_SETTLE_SECONDS", "2")
)
_GATEWAY_IMAGE = os.environ.get("OPENCLAW_GATEWAY_IMAGE", "clawgym-rl:v0.1")
_POLICY_READY_TIMEOUT_SECONDS = 600
_GLOBAL_WORKER = None
_GLOBAL_WORKER_LOCK = threading.Lock()
_TOOL_CALL_MARKER: str | None = None
_REMOTE_RUN_ID = f"{int(time.time())}-{os.getpid()}"
_TRAIN_ATTEMPTS: dict[int, int] = {}

def _train_session_id(rollout_id,task_index):
    base=f"train-task-{rollout_id}-{task_index}"
    attempt=_TRAIN_ATTEMPTS.get(rollout_id,int(os.environ.get("OPENCLAW_TRAIN_ATTEMPT_BASE","0")))
    return base if attempt==0 else f"train-task-{rollout_id}-retry{attempt}-{task_index}"

_REMOTE_CONTAINER_PREFIX = os.environ.get("OPENCLAW_CONTAINER_PREFIX", "clawgym-rl")
if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", _REMOTE_CONTAINER_PREFIX):
    raise ValueError("Invalid OPENCLAW_CONTAINER_PREFIX")

# ── Train rollout metrics (reset each rollout step) ──────────────────────
_TRAIN_METRICS_LOCK = threading.Lock()
_TRAIN_METRICS: dict = {}


def _record_train_startup_timing(name: str, seconds: float) -> None:
    """Record one low-overhead startup phase observation for train rollouts."""
    with _TRAIN_METRICS_LOCK:
        _TRAIN_METRICS.setdefault(f"startup_{name}_seconds", []).append(
            max(0.0, float(seconds))
        )


def _increment_train_startup_metric(name: str, value: int | float = 1) -> None:
    with _TRAIN_METRICS_LOCK:
        key = f"startup_{name}"
        _TRAIN_METRICS[key] = _TRAIN_METRICS.get(key, 0) + value


def _get_tool_call_marker() -> str:
    """Get the tool-call start token for the configured SGLang parser (cached).

    Different parsers expose it under different attribute names:
      - qwen25 / many others: ``bot_token`` (e.g. ``<tool_call>\\n``)
      - qwen3_coder: ``tool_call_start_token`` (``<tool_call>``); its
        ``bot_token`` is an empty string.
    We probe several attributes and fall back to ``<tool_call>``.
    """
    global _TOOL_CALL_MARKER
    if _TOOL_CALL_MARKER is not None:
        return _TOOL_CALL_MARKER
    parser_name = os.environ.get("TOOL_CALL_PARSER", "qwen25")
    marker = ""
    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        parser_cls = FunctionCallParser.ToolCallParserEnum.get(parser_name)
        if parser_cls:
            inst = parser_cls()
            for attr in ("tool_call_start_token", "bot_token"):
                val = getattr(inst, attr, "") or ""
                val = val.strip()
                if val:
                    marker = val
                    break
    except Exception:
        marker = ""
    if not marker:
        marker = "<tool_call>"  # safe fallback
    _TOOL_CALL_MARKER = marker
    return _TOOL_CALL_MARKER


@dataclass(frozen=True)
class TaskSpec:
    task_dir: Path
    task_id: str
    user_query: str
    input_mount_dir: str | None
    metadata: dict


@dataclass(frozen=True)
class TaskSource:
    dataset_name: str
    tasks: list[TaskSpec]


@dataclass(frozen=True)
class ContainerRuntime:
    container_name: str
    gateway_url: str
    token: str
    config_dir: Path
    workspace_dir: Path
    session_id: str
    model_id: str
    remote_root: str | None = None
    use_chroot: bool = False
    chroot_rootfs: Path | None = None
    chroot_process: subprocess.Popen | None = field(default=None, hash=False, compare=False)
    chroot_log_path: Path | None = None


@dataclass(frozen=True)
class TaskRunResult:
    reward: float
    final_message: str
    react_steps: list[dict]
    raw_prompts: dict[str, str]
    compaction_count: int = 0
    compaction_model_call_count: int = 0
    tool_result_pruning_events: int = 0
    tool_results_pruned: int = 0
    tool_result_truncation_events: int = 0
    tool_results_truncated: int = 0
    unclassified_history_rewrites: int = 0
    context_overflow_count: int = 0
    budget_termination: dict | None = None
    termination_reason: str = "natural_completion"
    session_id: str = ""
    duration_seconds: float = 0.0
    sglang_stats: dict = field(default_factory=dict)
    workspace_artifact: dict | None = None


class AsyncRolloutWorker:
    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue = queue.Queue(maxsize=100000)
        self.worker_thread = None
        self.submission_enabled = threading.Event()
        self.server = ClawGymRLAPIServer(
            args=args,
            output_queue=self.output_queue,
            submission_enabled=self.submission_enabled,
        )

    async def _worker_loop(self):
        while self.running:
            await asyncio.sleep(1.0)

    def _thread_main(self):
        asyncio.run(self._worker_loop())

    def start(self):
        self.server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        self.submission_enabled.clear()
        self.server.stop()
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self.submission_enabled.is_set():
            self.submission_enabled.clear()
            print("[ClawGym-RL] submission paused", flush=True)

    def resume_submission(self):
        if not self.submission_enabled.is_set():
            self.server.wait_until_ready(_POLICY_READY_TIMEOUT_SECONDS)
            self.submission_enabled.set()
            print("[ClawGym-RL] submission resumed", flush=True)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


def _random_token(length: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _remote_enabled() -> bool:
    return bool(os.environ.get("OPENCLAW_REMOTE_HOST", "").strip())


def _remote_proxy_port() -> int:
    return int(os.environ.get("OPENCLAW_REMOTE_PROXY_PORT", "39081"))


_SSH_SLOT_LOCK = threading.Lock()
_SSH_SLOT_COUNTER = 0
_SSH_THREAD_STATE = threading.local()
_SSH_MASTER_LOCKS: dict[str, threading.Lock] = {}
_REMOTE_DOCKER_START_LOCK = threading.Lock()
_REMOTE_DOCKER_START_LIMIT: int | None = None
_REMOTE_DOCKER_START_SEMAPHORE: threading.BoundedSemaphore | None = None
_REMOTE_GATEWAY_PORT_LOCK = threading.Lock()
_REMOTE_GATEWAY_PORTS_IN_USE: set[int] = set()


class RemoteOpenClawChatError(OSError):
    """Recoverable remote gateway failure with secret-safe diagnostics."""


def _remote_docker_start_semaphore() -> threading.BoundedSemaphore:
    """Limit only remote container startup; running agents remain concurrent."""
    global _REMOTE_DOCKER_START_LIMIT, _REMOTE_DOCKER_START_SEMAPHORE
    limit = int(os.environ.get("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY", "4"))
    if limit <= 0:
        raise ValueError("OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY must be positive")
    with _REMOTE_DOCKER_START_LOCK:
        if _REMOTE_DOCKER_START_SEMAPHORE is None or _REMOTE_DOCKER_START_LIMIT != limit:
            _REMOTE_DOCKER_START_LIMIT = limit
            _REMOTE_DOCKER_START_SEMAPHORE = threading.BoundedSemaphore(limit)
        return _REMOTE_DOCKER_START_SEMAPHORE


class _GatewayPrewarmBarrier:
    """Accumulate task-bound ready gateways before releasing their prompts."""

    def __init__(self, target: int, timeout_seconds: float):
        if target <= 0:
            raise ValueError("gateway prewarm target must be positive")
        if timeout_seconds <= 0:
            raise ValueError("gateway prewarm timeout must be positive")
        self.target = target
        self.timeout_seconds = timeout_seconds
        self.started_at = time.monotonic()
        self.ready = 0
        self.released = False
        self.aborted = False
        self.release_reason = ""
        self.release_seconds = 0.0
        self._condition = threading.Condition()

    def _release_locked(self, reason: str) -> None:
        if self.released:
            return
        self.released = True
        self.aborted = reason != "ready"
        self.release_reason = reason
        self.release_seconds = time.monotonic() - self.started_at
        print(
            "[ClawGym-RL] gateway prewarm released "
            f"ready={self.ready}/{self.target}, reason={reason}, "
            f"elapsed={self.release_seconds:.3f}s",
            flush=True,
        )
        self._condition.notify_all()

    def _wait_locked(self) -> None:
        deadline = self.started_at + self.timeout_seconds
        while not self.released:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._release_locked("timeout")
                break
            self._condition.wait(timeout=remaining)

    def wait_for_release(self) -> None:
        with self._condition:
            self._wait_locked()

    def arrive_and_wait(self, task_index: int) -> None:
        with self._condition:
            if not self.released:
                self.ready += 1
                print(
                    "[ClawGym-RL] gateway prewarm ready "
                    f"task_index={task_index}, ready={self.ready}/{self.target}",
                    flush=True,
                )
                if self.ready >= self.target:
                    self._release_locked("ready")
            self._wait_locked()

    def abort(self, reason: str) -> None:
        with self._condition:
            if not self.released:
                self._release_locked(f"abort:{reason}")

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "target": self.target,
                "ready": self.ready,
                "released": self.released,
                "aborted": self.aborted,
                "reason": self.release_reason,
                "release_seconds": self.release_seconds,
            }


def _remote_gateway_port_mode() -> str:
    mode = os.environ.get(
        "OPENCLAW_REMOTE_GATEWAY_PORT_MODE", "run_scoped"
    ).strip().lower()
    if mode not in {"run_scoped", "static"}:
        raise ValueError(
            "OPENCLAW_REMOTE_GATEWAY_PORT_MODE must be run_scoped or static"
        )
    return mode


def _remote_port_is_listening(port: int) -> bool:
    result = _remote_run(
        f"ss -H -ltn 'sport = :{port}'",
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(
            "remote port probe failed "
            f"(port={port}, exit={result.returncode})"
        )
    return bool(result.stdout.strip())


def _allocate_remote_gateway_port(
    rollout_id: int, task_index: int, is_train: bool
) -> int:
    """Choose a run-scoped free port instead of reusing a static slot.

    The run ID and task identity produce a deterministic candidate sequence for
    reproducible diagnostics. A remote listener probe rejects stale containers
    and other jobs, while the process-local set prevents sibling rollout
    threads from selecting the same candidate.
    """
    if _remote_gateway_port_mode() == "static":
        return _gateway_port_for_task(task_index, is_train=is_train)

    port_min = int(os.environ.get("OPENCLAW_REMOTE_GATEWAY_PORT_MIN", "20000"))
    port_max = int(os.environ.get("OPENCLAW_REMOTE_GATEWAY_PORT_MAX", "62000"))
    max_attempts = int(
        os.environ.get("OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS", "512")
    )
    if not 1024 <= port_min <= port_max <= 65535:
        raise ValueError(
            "remote gateway port range must satisfy "
            "1024 <= min <= max <= 65535"
        )
    if max_attempts <= 0:
        raise ValueError(
            "OPENCLAW_REMOTE_GATEWAY_PORT_MAX_ATTEMPTS must be positive"
        )

    span = port_max - port_min + 1
    role = "train" if is_train else "eval"
    seed_material = (
        f"{_REMOTE_RUN_ID}:{rollout_id}:{role}:{task_index}"
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    proxy_port = _remote_proxy_port()
    # 7919 is prime. It walks broadly through the configured range instead of
    # clustering retries around one stale port block.
    with _REMOTE_GATEWAY_PORT_LOCK:
        for attempt in range(max_attempts):
            candidate = port_min + ((seed + attempt * 7919) % span)
            if candidate == proxy_port or candidate in _REMOTE_GATEWAY_PORTS_IN_USE:
                continue
            if _remote_port_is_listening(candidate):
                continue
            _REMOTE_GATEWAY_PORTS_IN_USE.add(candidate)
            return candidate
    raise OSError(
        "unable to allocate a free run-scoped remote gateway port "
        f"after {max_attempts} attempts in [{port_min}, {port_max}]"
    )


def _release_remote_gateway_port(port: int) -> None:
    with _REMOTE_GATEWAY_PORT_LOCK:
        _REMOTE_GATEWAY_PORTS_IN_USE.discard(int(port))


def _ssh_connection_settings() -> tuple[str, str, str]:
    host = os.environ.get("OPENCLAW_REMOTE_HOST", "").strip()
    if not host:
        raise RuntimeError("OPENCLAW_REMOTE_HOST is required for remote Docker transport")
    user = os.environ.get("OPENCLAW_REMOTE_USER", "root").strip()
    key = os.environ.get("OPENCLAW_REMOTE_SSH_KEY", "/root/openclaw_cpu_key").strip()
    return host, user, key


def _ssh_common_options(key: str) -> list[str]:
    return [
        "ssh",
        "-i", key,
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
    ]


def _ssh_pool_size() -> int:
    value = int(os.environ.get("OPENCLAW_REMOTE_SSH_POOL_SIZE", "8"))
    if not 1 <= value <= 8:
        raise ValueError("OPENCLAW_REMOTE_SSH_POOL_SIZE must be between 1 and 8")
    return value


def _ssh_slot() -> int:
    slot = getattr(_SSH_THREAD_STATE, "slot", None)
    if slot is not None:
        return slot
    global _SSH_SLOT_COUNTER
    with _SSH_SLOT_LOCK:
        slot = _SSH_SLOT_COUNTER % _ssh_pool_size()
        _SSH_SLOT_COUNTER += 1
    _SSH_THREAD_STATE.slot = slot
    return slot


def _ssh_control_path(host: str, user: str, key: str, slot: int) -> Path:
    identity = hashlib.sha256(f"{user}@{host}\\0{key}".encode()).hexdigest()[:16]
    directory = Path(f"/tmp/clawgym-ssh-{os.getuid()}-{identity}")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory / f"master-{slot}.sock"


def _ssh_master_lock(control_path: Path) -> threading.Lock:
    lock_key = str(control_path)
    with _SSH_SLOT_LOCK:
        return _SSH_MASTER_LOCKS.setdefault(lock_key, threading.Lock())


def _ensure_ssh_master(
    host: str, user: str, key: str, control_path: Path
) -> None:
    target = f"{user}@{host}"
    common = _ssh_common_options(key)
    lock = _ssh_master_lock(control_path)
    with lock:
        check = subprocess.run(
            common + ["-S", str(control_path), "-O", "check", target],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if check.returncode == 0:
            return

        # Starting a master never executes a remote task command, so retrying
        # a pre-authentication reset is safe. Eight slots keep the initial
        # handshake burst below OpenSSH's default MaxStartups threshold of ten.
        last_result: subprocess.CompletedProcess | None = None
        for delay in (0.0, 0.5, 1.5):
            if delay:
                time.sleep(delay)
            last_result = subprocess.run(
                common
                + [
                    "-S", str(control_path),
                    "-o", "ControlMaster=yes",
                    "-o", "ControlPersist=120",
                    "-N", "-f", target,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            if last_result.returncode == 0:
                return
        assert last_result is not None
        raise subprocess.CalledProcessError(
            last_result.returncode,
            last_result.args,
            output=last_result.stdout,
            stderr=last_result.stderr.decode(errors="replace"),
        )


def _ssh_base() -> list[str]:
    host, user, key = _ssh_connection_settings()
    control_path = _ssh_control_path(host, user, key, _ssh_slot())
    _ensure_ssh_master(host, user, key, control_path)
    return _ssh_common_options(key) + [
        "-S", str(control_path),
        "-o", "ControlMaster=no",
        f"{user}@{host}",
    ]


def _remote_run(
    command: str,
    *,
    input_data: bytes | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        _ssh_base() + [command],
        input=input_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr.decode(errors="replace"),
        )
    return result


def _tar_bytes(paths: list[tuple[Path, str]], *, gzip: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if gzip else "w", dereference=False) as archive:
        for source, arcname in paths:
            archive.add(source, arcname=arcname, recursive=True)
    return buffer.getvalue()


def _remote_state_archive(
    runtime: ContainerRuntime, secret_env_file: Path | None = None
) -> bytes:
    paths = [
        (runtime.config_dir / "openclaw.json", "state/openclaw.json"),
        (runtime.workspace_dir, "workspace"),
    ]
    skills_dir = runtime.config_dir / "skills"
    if skills_dir.is_dir():
        paths.append((skills_dir, "state/skills"))
    if secret_env_file is not None:
        paths.append((secret_env_file, "runtime.env"))
    return _tar_bytes(paths, gzip=True)


_GATEWAY_PORT_BASE = 16000
_GATEWAY_PORT_STRIDE = 10
_GATEWAY_PORT_TRAIN_OFFSET = 5000
_GATEWAY_PORT_MAX_SLOTS = 4000  # 4000 slots × 10 stride = 40000 ports (16000-56000)


def _get_rootfs() -> str | None:
    val = os.environ.get("OPENCLAW_ROOTFS", "").strip()
    return val if val else None


def _setup_fake_proc(container_rootfs: Path):
    """Populate a static /proc inside the chroot for tools that need procfs."""
    proc_dir = container_rootfs / "proc"
    proc_dir.mkdir(exist_ok=True)
    for child in proc_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            shutil.rmtree(child, ignore_errors=True)
    self_dir = proc_dir / "self"
    if self_dir.exists():
        shutil.rmtree(self_dir)
    self_dir.mkdir()
    (self_dir / "status").write_text(
        "Name:\tbash\nState:\tS (sleeping)\nPid:\t1\nPPid:\t0\n"
        "Uid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\nThreads:\t1\n"
    )
    (self_dir / "stat").write_text(
        "1 (bash) S 0 0 0 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 0 8192000 1024 "
        "18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0\n"
    )
    (self_dir / "comm").write_text("bash\n")
    (self_dir / "cmdline").write_text("\x00")
    (self_dir / "maps").write_text("")
    (self_dir / "environ").write_bytes(
        b"HOME=/root\x00PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\x00"
    )
    (self_dir / "fd").mkdir(exist_ok=True)
    (self_dir / "exe").symlink_to("/bin/bash")
    (self_dir / "cwd").symlink_to("/root/.openclaw/workspace")
    mounts = proc_dir / "mounts"
    if not mounts.exists():
        mounts.write_text(
            "rootfs / rootfs rw 0 0\n"
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0\n"
            "tmpfs /tmp tmpfs rw,nosuid,nodev 0 0\n"
        )


def _gateway_port_for_task(task_index: int, is_train: bool = False) -> int:
    base = int(os.environ.get("OPENCLAW_GATEWAY_PORT_BASE", str(_GATEWAY_PORT_BASE)))
    stride = int(
        os.environ.get("OPENCLAW_GATEWAY_PORT_STRIDE", str(_GATEWAY_PORT_STRIDE))
    )
    train_offset = int(
        os.environ.get(
            "OPENCLAW_GATEWAY_PORT_TRAIN_OFFSET",
            str(_GATEWAY_PORT_TRAIN_OFFSET),
        )
    )
    max_slots = int(
        os.environ.get("OPENCLAW_GATEWAY_PORT_MAX_SLOTS", str(_GATEWAY_PORT_MAX_SLOTS))
    )
    if base < 1 or stride < 1 or train_offset < 0 or max_slots < 1:
        raise ValueError("gateway port configuration values are out of range")
    offset = train_offset if is_train else 0
    slot = task_index % max_slots
    port = base + offset + slot * stride
    if port > 65535:
        raise ValueError(f"computed gateway port exceeds 65535: {port}")
    return port


def _gateway_healthcheck_command(gateway_port: int) -> str:
    """Return the image healthcheck with the per-task gateway port."""
    return (
        "node -e \"fetch('http://127.0.0.1:"
        f"{gateway_port}"
        "/healthz').then((r)=>process.exit(r.ok?0:1))"
        ".catch(()=>process.exit(1))\""
    )


def _wait_for_port_free(port: int, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return
        try:
            subprocess.run(
                f"lsof -ti :{port} | xargs kill -9 2>/dev/null; true",
                shell=True, capture_output=True, timeout=5,
            )
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Port {port} still occupied after {timeout}s")


def _require_dict(value, field_name: str) -> dict:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict")
    return value


def _require_list(value, field_name: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "type" in item and item["type"] == "text" and "text" in item:
                parts.append(_require_string(item["text"], "message content text"))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _task_sources_from_env(env_name: str) -> list[TaskSource]:
    if env_name not in os.environ:
        return []
    raw_value = os.environ[env_name].strip()
    if not raw_value:
        return []
    sources: list[TaskSource] = []
    for raw_path in raw_value.split(","):
        source_path = Path(raw_path.strip())
        if not source_path.exists():
            raise FileNotFoundError(f"task source not found: {source_path}")
        task_entries = _discover_task_entries(source_path)
        tasks = [_load_task_spec(entry_path) for entry_path in task_entries]
        if env_name == "OPENCLAW_EVAL_TASKS":
            raw_task_ids = os.environ.get("OPENCLAW_EVAL_TASK_IDS", "").strip()
            if raw_task_ids:
                requested = [item.strip() for item in raw_task_ids.split(",") if item.strip()]
                requested_set = set(requested)
                available = {task.task_id for task in tasks}
                missing = sorted(requested_set - available)
                if missing:
                    raise ValueError(
                        "OPENCLAW_EVAL_TASK_IDS contains unknown task ids: "
                        + ", ".join(missing)
                    )
                tasks = [task for task in tasks if task.task_id in requested_set]
        dataset_name = source_path.name
        sources.append(TaskSource(dataset_name=dataset_name, tasks=tasks))
    return sources


def _discover_task_entries(source_path: Path) -> list[Path]:
    if not source_path.is_dir():
        raise ValueError(f"task source must be a dataset directory, got: {source_path}")

    entries: list[Path] = []
    for child in sorted(source_path.iterdir()):
        if not child.is_dir():
            raise ValueError(f"dataset directory must contain only task directories, got file: {child}")
        entry_path = child / "data_entry.json"
        if not entry_path.exists():
            raise ValueError(f"task directory missing data_entry.json: {child}")
        entries.append(entry_path)
    if not entries:
        raise ValueError(f"no task folders with data_entry.json found under: {source_path}")
    return entries


def _load_task_spec(entry_path: Path) -> TaskSpec:
    with entry_path.open(encoding="utf-8") as file_handle:
        entry = _require_dict(json.load(file_handle), str(entry_path))

    task_dir = entry_path.parent
    metadata = _require_dict(entry["metadata"], "metadata")

    input_mount_dir: str | None = None
    if "input_mount_dir" in entry:
        input_mount_dir = _require_string(entry["input_mount_dir"], "input_mount_dir")
        input_files_dir = task_dir / "input_files"
        if not input_files_dir.exists():
            raise FileNotFoundError(f"input_mount_dir is set but input_files/ not found: {input_files_dir}")

    reward_sh = task_dir / "reward" / "reward.sh"
    if not reward_sh.exists():
        raise FileNotFoundError(f"reward/reward.sh not found: {reward_sh}")

    return TaskSpec(
        task_dir=task_dir,
        task_id=_require_string(entry["task_id"], "task_id"),
        user_query=_require_string(entry["user_query"], "user_query"),
        input_mount_dir=input_mount_dir,
        metadata=metadata,
    )


def _container_path_to_host_path(workspace_dir: Path, container_path: str) -> Path:
    container_pure = PurePosixPath(container_path)
    working_pure = PurePosixPath("/root/.openclaw/workspace")
    if not str(container_pure).startswith(str(working_pure)):
        raise ValueError(f"path {container_path} is outside /root/.openclaw/workspace")
    if container_pure == working_pure:
        return workspace_dir
    relative = container_pure.relative_to(working_pure)
    return workspace_dir / Path(str(relative))


def _copy_tree_contents(source_dir: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def _prepare_workspace(task: TaskSpec, workspace_dir: Path):
    workspace_dir.mkdir(parents=True, exist_ok=True)
    if task.input_mount_dir is not None:
        input_dir = _container_path_to_host_path(workspace_dir, task.input_mount_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        input_files_dir = task.task_dir / "input_files"
        _copy_tree_contents(input_files_dir, input_dir)


def _build_official_gateway_config(runtime: ContainerRuntime, rl_port: int):
    """Gateway config from the public ClawGym 4B rollout."""
    model_proxy_port = _remote_proxy_port() if runtime.remote_root is not None else rl_port
    config = {
        "gateway": {
            "port": int(runtime.gateway_url.rsplit(":", 1)[1]),
            "mode": "local",
            "bind": "loopback",
            "auth": {"mode": "token", "token": runtime.token},
            "http": {"endpoints": {"chatCompletions": {"enabled": True}}},
            "tailscale": {"mode": "off", "resetOnExit": False},
            "nodes": {
                "denyCommands": [
                    "camera.snap", "camera.clip", "screen.record",
                    "contacts.add", "calendar.add", "reminders.add", "sms.send",
                ],
            },
        },
        "models": {
            "mode": "merge",
            "providers": {
                "rl-server": {
                    "baseUrl": f"http://localhost:{model_proxy_port}/v1",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": runtime.model_id,
                            "name": "RL Model",
                            "reasoning": False,
                            "input": ["text"],
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                            "contextWindow": int(
                                os.environ.get("OPENCLAW_CONTEXT_WINDOW", "128000")
                            ),
                            "maxTokens": int(
                                os.environ.get(
                                    "OPENCLAW_MAX_RESPONSE_TOKENS",
                                    os.environ.get("OPENCLAW_MAX_TOKENS", "8192"),
                                )
                            ),
                        }
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": f"rl-server/{runtime.model_id}"},
                "workspace": "/root/.openclaw/workspace",
                "maxConcurrent": 4,
                "subagents": {"maxConcurrent": 4},
                "timeoutSeconds": 1000,
                "llm": {"idleTimeoutSeconds": 0},
                "heartbeat": {"every": "3680000m"},
            }
        },
        "tools": {
            "profile": "coding",
            "deny": ["web_search", "memory_search"],
            "exec": {"host": "gateway", "security": "full"},
        },
        "commands": {
            "native": "auto",
            "nativeSkills": "auto",
            "restart": True,
            "ownerDisplay": "raw",
        },
        "session": {"dmScope": "per-channel-peer"},
    }
    with (runtime.config_dir / "openclaw.json").open("w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle)


def _build_gateway_config(runtime: ContainerRuntime, rl_port: int):
    if os.environ.get("OPENCLAW_GATEWAY_CONFIG", "") == "official":
        _build_official_gateway_config(runtime, rl_port)
        return
    # Official ClawGym reference config, with only the migration-required
    # six-tool allowlist and pinned skills policy added.
    model_proxy_port = _remote_proxy_port() if runtime.remote_root is not None else rl_port
    model_parts = runtime.model_id.split("__", 1)
    if len(model_parts) != 2 or not model_parts[1]:
        raise ValueError("runtime model_id must be '<turn_type>__<session_id>'")
    compaction_model_id = f"compact__{model_parts[1]}"
    context_window = int(os.environ.get("OPENCLAW_CONTEXT_WINDOW", "128000"))
    max_response_tokens = int(
        os.environ.get("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")
    )

    def model_entry(model_id: str, name: str) -> dict:
        return {
            "id": model_id,
            "name": name,
            "reasoning": False,
            "input": ["text"],
            "cost": {
                "input": 0,
                "output": 0,
                "cacheRead": 0,
                "cacheWrite": 0,
            },
            "contextWindow": context_window,
            "maxTokens": max_response_tokens,
        }

    config = {
        "gateway": {
            "port": int(runtime.gateway_url.rsplit(":", 1)[1]),
            "mode": "local",
            "bind": "loopback",
            "auth": {"mode": "token", "token": runtime.token},
            "http": {"endpoints": {"chatCompletions": {"enabled": True}}},
            "tailscale": {"mode": "off", "resetOnExit": False},
            "nodes": {
                "denyCommands": [
                    "camera.snap", "camera.clip", "screen.record",
                    "contacts.add", "calendar.add", "reminders.add", "sms.send",
                ],
            },
        },
        "models": {
            "mode": "merge",
            "providers": {
                "rl-server": {
                    "baseUrl": f"http://localhost:{model_proxy_port}/v1",
                    "api": "openai-completions",
                    "models": [
                        model_entry(runtime.model_id, "RL Model"),
                        model_entry(compaction_model_id, "RL Compaction Model"),
                    ],
                }
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": f"rl-server/{runtime.model_id}"},
                "workspace": "/root/.openclaw/workspace",
                "userTimezone": "Asia/Shanghai",
                "maxConcurrent": 4,
                "subagents": {"maxConcurrent": 4},
                "timeoutSeconds": 1000,
                "llm": {"idleTimeoutSeconds": 0},
                "heartbeat": {"every": "35791m"},
                "compaction": {
                    "model": f"rl-server/{compaction_model_id}",
                },
            }
        },
        "tools": {
            "allow": ["read", "edit", "write", "exec", "process", "web_fetch"],
            "deny": ["apply_patch", "image", "web_search", "memory_search"],
            "exec": {
                "host": "gateway",
                "security": "full",
                # Keep background completion notifications out of the policy
                # trajectory. The model can still query jobs via process.poll.
                "notifyOnExit": False,
            },
        },
        "skills": {
            "allowBundled": ["clawflow", "clawflow-inbox-triage", "healthcheck", "node-connect", "skill-creator", "weather"],
            "load": {"watch": False},
            "entries": {"serper": {"enabled": True}},
        },
        "commands": {
            "native": "auto",
            "nativeSkills": "auto",
            "restart": True,
            "ownerDisplay": "raw",
        },
        "session": {"dmScope": "per-channel-peer"},
    }
    default_serper_dir = (
        Path(__file__).resolve().parents[1] / "sft_contract/recovered_skills/serper"
    )
    serper_source_dir = Path(
        os.environ.get("CLAWGYM_SERPER_SKILL_DIR", str(default_serper_dir))
    )
    required_serper_files = (
        serper_source_dir / "SKILL.md",
        serper_source_dir / "scripts" / "search.py",
    )
    if not all(path.is_file() for path in required_serper_files):
        raise RuntimeError(
            "Serper skill directory is incomplete: SKILL.md and "
            "scripts/search.py are required"
        )
    shutil.copytree(
        serper_source_dir, runtime.config_dir / "skills" / "serper", dirs_exist_ok=True
    )
    with (runtime.config_dir / "openclaw.json").open("w", encoding="utf-8") as file_handle:
        json.dump(config, file_handle)


def _start_container(task: TaskSpec, rollout_id: int, task_index: int, rl_port: int, session_id: str, model_id: str, is_train: bool = False) -> ContainerRuntime:
    remote = _remote_enabled()
    gateway_runtime_env = _load_optional_gateway_runtime_env()
    port_allocate_started = time.perf_counter()
    try:
        gateway_port = (
            _allocate_remote_gateway_port(rollout_id, task_index, is_train)
            if remote
            else _gateway_port_for_task(task_index, is_train=is_train)
        )
    finally:
        if is_train:
            _record_train_startup_timing(
                "port_allocate", time.perf_counter() - port_allocate_started
            )
    token = _random_token(32)
    local_name = f"{_REMOTE_CONTAINER_PREFIX}-{_REMOTE_RUN_ID}-{rollout_id}-{task_index}" if remote else f"clawgym-rl-{rollout_id}-{task_index}"
    config_dir = Path(f"/tmp/{local_name}")
    workspace_dir = config_dir / "workspace"
    shutil.rmtree(config_dir, ignore_errors=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    rootfs = None if remote else _get_rootfs()
    use_chroot = rootfs is not None
    remote_base = os.environ.get("OPENCLAW_REMOTE_BASE", "/root/clawgym_rl_runs").rstrip("/")
    remote_root = f"{remote_base}/{_REMOTE_RUN_ID}/{rollout_id}/{task_index}" if remote else None

    runtime = ContainerRuntime(
        container_name=local_name,
        gateway_url=f"http://127.0.0.1:{gateway_port}",
        token=token,
        config_dir=config_dir,
        workspace_dir=workspace_dir,
        session_id=session_id,
        model_id=model_id,
        remote_root=remote_root,
        use_chroot=use_chroot,
    )
    workspace_prepare_started = time.perf_counter()
    try:
        _prepare_workspace(task, workspace_dir)
        _build_gateway_config(runtime, rl_port)
        credentials = _load_web_credentials()
    except Exception:
        if remote:
            _release_remote_gateway_port(gateway_port)
        raise
    finally:
        if is_train:
            _record_train_startup_timing(
                "workspace_prepare", time.perf_counter() - workspace_prepare_started
            )

    if remote:
        if not remote_root or remote_root in {"/", "/root"}:
            _release_remote_gateway_port(gateway_port)
            raise ValueError(f"unsafe OPENCLAW_REMOTE_BASE produced {remote_root!r}")
        quoted_container = shlex.quote(runtime.container_name)
        quoted_root = shlex.quote(remote_root)
        prepare = (
            f"docker rm -f {quoted_container} >/dev/null 2>&1 || true; "
            f"rm -rf -- {quoted_root}; mkdir -p {quoted_root}; "
            f"tar -xzf - -C {quoted_root}"
        )
        secret_env_file = None
        try:
            secret_env_file = _write_secret_env_file(
                runtime.container_name, credentials, gateway_runtime_env
            )
            archive_started = time.perf_counter()
            try:
                archive = _remote_state_archive(runtime, secret_env_file)
            finally:
                if is_train:
                    _record_train_startup_timing(
                        "archive_build", time.perf_counter() - archive_started
                    )
            if is_train:
                _increment_train_startup_metric("archive_bytes", len(archive))
                _increment_train_startup_metric("archive_count")
            upload_started = time.perf_counter()
            try:
                _remote_run(prepare, input_data=archive, timeout=180)
            finally:
                if is_train:
                    _record_train_startup_timing(
                        "upload_extract", time.perf_counter() - upload_started
                    )
        except Exception:
            _release_remote_gateway_port(gateway_port)
            raise
        finally:
            if secret_env_file is not None:
                secret_env_file.unlink(missing_ok=True)
        remote_env_file = f"{remote_root}/runtime.env"
        docker_command = [
            "docker", "run", "-d",
            "--name", runtime.container_name,
            "--network", "host",
            "--user", "root",
            "--health-cmd", _gateway_healthcheck_command(gateway_port),
            "-e", f"OPENCLAW_GATEWAY_TOKEN={runtime.token}",
            "-e", "HOME=/root",
            "-e", "TZ=Asia/Shanghai",
            "--env-file", remote_env_file,
            "-v", f"{remote_root}/state:/root/.openclaw",
            "-v", f"{remote_root}/workspace:/root/.openclaw/workspace",
            _GATEWAY_IMAGE,
            "node", "dist/index.js", "gateway",
            "--bind", "loopback",
            "--port", str(gateway_port),
            "--token", runtime.token,
        ]
        quoted_env_file = shlex.quote(remote_env_file)
        docker_shell = (
            " ".join(shlex.quote(part) for part in docker_command)
            + f"; status=$?; rm -f -- {quoted_env_file}; exit $status"
        )
        docker_run_started = time.perf_counter()
        try:
            _remote_run(docker_shell, timeout=60)
        except Exception:
            _release_remote_gateway_port(gateway_port)
            _remote_run(
                f"rm -f -- {quoted_env_file}",
                timeout=30,
                check=False,
            )
            raise
        finally:
            if is_train:
                _record_train_startup_timing(
                    "docker_run", time.perf_counter() - docker_run_started
                )
        return runtime

    subprocess.run(["chmod", "-R", "777", str(config_dir)], check=True)
    _wait_for_port_free(gateway_port)

    if use_chroot:
        container_rootfs = config_dir / "rootfs"
        log_path = config_dir / "gateway.log"

        subprocess.run(
            ["cp", "-al", rootfs, str(container_rootfs)],
            check=True, capture_output=True, text=True,
        )
        for writable in ("tmp", "var/tmp", "var/log", "var/cache", "root", "run"):
            p = container_rootfs / writable
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
        for apt_dir in ("var/cache/apt/archives/partial", "var/lib/apt/lists/partial"):
            (container_rootfs / apt_dir).mkdir(parents=True, exist_ok=True)
        for mutable_dir in (
            "var/lib/dpkg", "var/lib/apt",
            "usr/local/lib/python3.11/dist-packages", "usr/local/bin", "etc",
        ):
            p = container_rootfs / mutable_dir
            if p.is_dir():
                tmp = p.with_name(p.name + ".tmp")
                shutil.copytree(p, tmp, symlinks=True)
                shutil.rmtree(p)
                tmp.rename(p)
        etc_hosts = container_rootfs / "etc" / "hosts"
        if not etc_hosts.exists():
            etc_hosts.write_text("127.0.0.1 localhost\n::1 localhost\n")

        oc_home = container_rootfs / "root" / ".openclaw"
        oc_home.mkdir(parents=True, exist_ok=True)
        for f in config_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, oc_home / f.name)

        oc_workspace = oc_home / "workspace"
        if oc_workspace.exists():
            shutil.rmtree(oc_workspace)
        shutil.copytree(workspace_dir, oc_workspace, dirs_exist_ok=True)
        object.__setattr__(runtime, "workspace_dir", oc_workspace)

        # Create /workspace symlink for backward compatibility
        legacy_workspace = container_rootfs / "workspace"
        if legacy_workspace.exists() and not legacy_workspace.is_symlink():
            shutil.rmtree(legacy_workspace)
        if not legacy_workspace.exists():
            legacy_workspace.symlink_to("/root/.openclaw/workspace")

        _setup_fake_proc(container_rootfs)

        process_env = os.environ.copy()
        process_env.update(credentials)
        process_env.update(gateway_runtime_env)
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [
                "unshare", "--user", "--map-root-user", "--",
                "chroot", str(container_rootfs),
                "/bin/bash", "-c",
                f"export HOME=/root TZ=Asia/Shanghai "
                f"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
                f"OPENCLAW_GATEWAY_TOKEN={token} "
                f"LC_ALL=C.UTF-8; "
                f"cd /app && exec node dist/index.js gateway "
                f"--bind loopback --port {gateway_port} --token {token}",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=process_env,
        )
        object.__setattr__(runtime, "chroot_rootfs", container_rootfs)
        object.__setattr__(runtime, "chroot_process", proc)
        object.__setattr__(runtime, "chroot_log_path", log_path)
    else:
        subprocess.run(["docker", "rm", "-f", runtime.container_name], capture_output=True, text=True)
        secret_env_file = _write_secret_env_file(
            runtime.container_name, credentials, gateway_runtime_env
        )
        try:
            subprocess.run(
                [
                    "docker", "run", "-d",
                    "--name", runtime.container_name,
                    "--network", "host",
                    "--user", "root",
                    "--health-cmd", _gateway_healthcheck_command(gateway_port),
                    "-e", f"OPENCLAW_GATEWAY_TOKEN={runtime.token}",
                    "-e", "HOME=/root",
                    "-e", "TZ=Asia/Shanghai",
                    "--env-file", str(secret_env_file),
                    "-v", f"{runtime.config_dir}:/root/.openclaw",
                    "-v", f"{runtime.workspace_dir}:/root/.openclaw/workspace",
                    _GATEWAY_IMAGE,
                    "node", "dist/index.js", "gateway",
                    "--bind", "loopback",
                    "--port", str(gateway_port),
                    "--token", runtime.token,
                ],
                check=True, capture_output=True, text=True,
            )
        finally:
            secret_env_file.unlink(missing_ok=True)
    return runtime


def _stop_container(runtime: ContainerRuntime, log_dir: Path | None):
    if runtime.remote_root is not None:
        gateway_port = int(runtime.gateway_url.rsplit(":", 1)[1])
        # Release the process-local reservation first. If the remote container
        # still owns the listener during cleanup, the next allocation's remote
        # probe will reject it; this also guarantees release when SSH cleanup
        # itself fails.
        _release_remote_gateway_port(gateway_port)
        quoted_container = shlex.quote(runtime.container_name)
        quoted_root = shlex.quote(runtime.remote_root)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            logs_result = _remote_run(
                f"docker logs {quoted_container}",
                timeout=30,
                check=False,
            )
            with (log_dir / f"gateway-{runtime.container_name}.log").open("w", encoding="utf-8") as file_handle:
                stdout = logs_result.stdout.decode(errors="replace")
                stderr = logs_result.stderr.decode(errors="replace")
                if stdout:
                    file_handle.write(stdout)
                if stderr:
                    file_handle.write("\n--- STDERR ---\n")
                    file_handle.write(stderr)
        _remote_run(
            f"docker rm -f {quoted_container} >/dev/null 2>&1 || true; "
            f"rm -rf -- {quoted_root}",
            timeout=60,
            check=False,
        )
        shutil.rmtree(runtime.config_dir, ignore_errors=True)
        return

    if runtime.use_chroot:
        proc = runtime.chroot_process
        if proc and proc.poll() is None:
            # NOTE: inside `unshare --user`, node becomes PID 1 of the user
            # namespace and the kernel ignores SIGTERM to PID-1-in-ns (no
            # default handler). Go straight to SIGKILL on the whole process
            # group; otherwise we wait 10s each rollout and accumulate
            # zombies when training dies before proc.wait() completes.
            try:
                subprocess.run(["kill", "-9", "--", f"-{proc.pid}"], capture_output=True, timeout=5)
                proc.wait(timeout=5)
            except Exception:
                pass
        # Reap defunct child to prevent zombie accumulation
        if proc is not None:
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
        # Fallback: pkill any leftover gateway procs identified by our config_dir path
        # (covers the case where the chroot parent died without reaping its children).
        try:
            subprocess.run(
                ["pkill", "-9", "-f", str(runtime.config_dir)],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        if log_dir and runtime.chroot_log_path and runtime.chroot_log_path.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime.chroot_log_path, log_dir / f"{runtime.container_name}.log")
    else:
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            logs_result = subprocess.run(
                ["docker", "logs", runtime.container_name],
                capture_output=True,
                text=True,
            )
            with (log_dir / f"gateway-{runtime.container_name}.log").open("w", encoding="utf-8") as file_handle:
                if logs_result.stdout:
                    file_handle.write(logs_result.stdout)
                if logs_result.stderr:
                    file_handle.write("\n--- STDERR ---\n")
                    file_handle.write(logs_result.stderr)
        subprocess.run(["docker", "rm", "-f", runtime.container_name], capture_output=True, text=True)
    shutil.rmtree(runtime.config_dir, ignore_errors=True)


def _wait_for_chat_ready(runtime: ContainerRuntime):
    if runtime.remote_root is not None:
        gateway_port = int(runtime.gateway_url.rsplit(":", 1)[1])
        attempts = max(1, int(_CHAT_READY_TIMEOUT_SECONDS / _CHAT_READY_POLL_SECONDS))
        command = (
            "ready=0; "
            f"for i in $(seq 1 {attempts}); do "
            f"if ! docker inspect -f '{{{{.State.Running}}}}' "
            f"{shlex.quote(runtime.container_name)} 2>/dev/null | grep -qx true; "
            "then exit 2; fi; "
            f"if curl -fsS --max-time 2 http://127.0.0.1:{gateway_port}/health >/dev/null; "
            "then ready=1; break; fi; "
            f"sleep {_CHAT_READY_POLL_SECONDS}; "
            'done; test "$ready" = 1'
        )
        result = _remote_run(
            command,
            timeout=_CHAT_READY_TIMEOUT_SECONDS + 30,
            check=False,
        )
        if result.returncode == 0:
            time.sleep(_CHAT_READY_SETTLE_SECONDS)
            return
        raise TimeoutError(
            f"OpenClaw chat endpoint not ready for {runtime.container_name}: "
            + result.stderr.decode(errors="replace").strip()
        )

    deadline = time.time() + _CHAT_READY_TIMEOUT_SECONDS
    last_error = "no response"
    while time.time() < deadline:
        try:
            response = requests.get(
                f"{runtime.gateway_url}/health",
                timeout=3,
            )
            if response.status_code == 200:
                time.sleep(_CHAT_READY_SETTLE_SECONDS)
                return
            last_error = f"health status={response.status_code}"
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(_CHAT_READY_POLL_SECONDS)
    raise TimeoutError(f"OpenClaw chat endpoint not ready for {runtime.container_name}: {last_error}")


def _post_chat_turn(runtime: ContainerRuntime, turn_type: str, session_done: bool, messages: list[dict]) -> dict:
    if runtime.remote_root is not None:
        gateway_port = int(runtime.gateway_url.rsplit(":", 1)[1])
        payload = json.dumps(
            {
                "model": "openclaw",
                "stream": False,
                "messages": messages,
            },
            ensure_ascii=False,
        ).encode()
        command = (
            f"curl --fail-with-body -sS --max-time {_CHAT_TURN_TIMEOUT_SECONDS} "
            f"-H {shlex.quote('Authorization: Bearer ' + runtime.token)} "
            "-H 'Content-Type: application/json' "
            "-H 'X-OpenClaw-Scopes: operator.admin,operator.read,operator.write' "
            "--data-binary @- "
            f"http://127.0.0.1:{gateway_port}/v1/chat/completions"
        )
        result = _remote_run(
            command,
            input_data=payload,
            timeout=_CHAT_TURN_TIMEOUT_SECONDS + 120,
            check=False,
        )
        if result.returncode != 0:
            # Do not serialize the SSH command: it contains the gateway bearer
            # token. Curl's stderr and the HTTP error body do not contain the
            # request headers and are enough to distinguish failure classes.
            stderr = result.stderr.decode(errors="replace").strip()[:1000]
            response = result.stdout.decode(errors="replace").strip()[:1000]
            raise RemoteOpenClawChatError(
                "remote OpenClaw chat request failed "
                f"(curl_exit={result.returncode}, stderr={stderr!r}, "
                f"response={response!r})"
            )
        try:
            return _require_dict(json.loads(result.stdout), "chat response")
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"invalid OpenClaw response for {runtime.container_name}: "
                + result.stdout[:500].decode(errors="replace")
            ) from exc

    response = requests.post(
        f"{runtime.gateway_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {runtime.token}",
            "Content-Type": "application/json",
            "X-OpenClaw-Scopes": "operator.admin,operator.read,operator.write",
        },
        json={
            "model": "openclaw",
            "stream": False,
            "messages": messages,
        },
        timeout=_CHAT_TURN_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return _require_dict(response.json(), "chat response")


def _extract_final_message(chat_response: dict) -> str:
    choices = _require_list(chat_response["choices"], "chat response choices")
    if not choices:
        raise ValueError("chat response choices must not be empty")
    choice = _require_dict(choices[0], "chat response choice")
    message = _require_dict(choice["message"], "chat response message")
    return _flatten_content(message["content"]).strip()


def _run_turns(runtime: ContainerRuntime, task: TaskSpec, max_turns: int, turn_type: str) -> str:
    messages = [{"role": "user", "content": task.user_query}]
    final_message = ""
    for turn_index in range(max_turns):
        session_done = turn_index == max_turns - 1
        chat_response = _post_chat_turn(runtime, turn_type=turn_type, session_done=session_done, messages=messages)
        final_message = _extract_final_message(chat_response)
        if session_done:
            break
        messages = [{"role": "user", "content": "Continue working on the task."}]
    return final_message


def _register_session_metadata(
    rl_port: int,
    session_id: str,
    task: TaskSpec,
    rollout_id: int,
    task_index: int,
    group_index: int | None,
    sampling_identity: str | None = None,
) -> None:
    metadata = {
        "task_id": task.task_id,
        "phase": "train" if group_index is not None else "eval",
        "rollout_id": rollout_id,
        "task_index": task_index,
        "group_index": group_index,
        "task_selection_seed": int(
            os.environ.get("OPENCLAW_TRAIN_SHUFFLE_SEED", "0")
        ),
        "fixed_rollout_seed": None,
    }
    if sampling_identity is not None:
        metadata["sampling_identity"] = sampling_identity
    response = requests.post(
        f"http://127.0.0.1:{rl_port}/register_session",
        json={
            "session_id": session_id,
            "metadata": metadata,
        },
        timeout=10,
    )
    response.raise_for_status()


def _fetch_react_steps(
    rl_port: int, session_id: str
) -> tuple[list[dict], dict[str, str], int, int, int, int, int, int, int, int, dict | None, dict]:
    response = requests.post(
        f"http://127.0.0.1:{rl_port}/get_conversation_log",
        json={"session_id": session_id},
        timeout=10,
    )
    response.raise_for_status()
    payload = _require_dict(response.json(), "conversation log response")
    steps = _require_list(payload["steps"], "conversation log steps")
    raw_prompts = payload.get("raw_prompts", {})
    compaction_count = int(payload.get("compaction_count", 0))
    compaction_model_call_count = int(payload.get("compaction_model_call_count", 0))
    tool_result_pruning_events = int(payload.get("tool_result_pruning_events", 0))
    tool_results_pruned = int(payload.get("tool_results_pruned", 0))
    tool_result_truncation_events = int(payload.get("tool_result_truncation_events", 0))
    tool_results_truncated = int(payload.get("tool_results_truncated", 0))
    unclassified_history_rewrites = int(payload.get("unclassified_history_rewrites", 0))
    context_overflow_count = len(payload.get("exact_context_overflows", []))
    budget_termination = payload.get("budget_termination")
    sglang_stats = _require_dict(payload.get("sglang_stats", {}), "sglang_stats")
    return (
        steps,
        raw_prompts,
        compaction_count,
        compaction_model_call_count,
        context_overflow_count,
        tool_result_pruning_events,
        tool_results_pruned,
        tool_result_truncation_events,
        tool_results_truncated,
        unclassified_history_rewrites,
        budget_termination,
        sglang_stats,
    )


def _classify_termination_reason(
    react_steps: list[dict], budget_termination: dict | None
) -> str:
    if budget_termination:
        reason = str(budget_termination.get("reason", ""))
        return {
            "max_agent_steps": "max_agent_steps",
            "max_trajectory_tokens": "max_trajectory_tokens",
            "max_compaction_events": "session_compaction_limit",
            "outer_timeout": "outer_timeout",
        }.get(reason, reason or "budget_termination")
    if react_steps:
        final_step = react_steps[-1]
        if bool(final_step.get("hit_response_limit", False)):
            if bool(final_step.get("incomplete_tool_call_at_limit", False)):
                return "length_limited_incomplete_tool_call"
            return "length_limited_valid_response"
    return "natural_completion"


def _safe_artifact_component(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _eval_train_step(args, rollout_id: int) -> int:
    override = getattr(args, "eval_train_step", None)
    if override is not None:
        return int(override)
    return 0 if rollout_id == 0 else rollout_id + 1


def _eval_runtime_task_index(task_id: str, default_index: int) -> int:
    raw_mapping = os.environ.get("OPENCLAW_EVAL_TASK_INDEX_MAP", "").strip()
    if not raw_mapping:
        return default_index
    mapping = json.loads(raw_mapping)
    if not isinstance(mapping, dict):
        raise TypeError("OPENCLAW_EVAL_TASK_INDEX_MAP must be a JSON object")
    if task_id not in mapping:
        raise ValueError(
            f"OPENCLAW_EVAL_TASK_INDEX_MAP has no entry for {task_id}"
        )
    runtime_index = int(mapping[task_id])
    if runtime_index < 0:
        raise ValueError("OPENCLAW_EVAL_TASK_INDEX_MAP values must be non-negative")
    return runtime_index


def _eval_sampling_identity(
    dataset_name: str, task_id: str, sample_index: int
) -> str:
    """Stable seed/tool-ID namespace shared by every checkpoint eval."""
    if sample_index < 0:
        raise ValueError("eval sample_index must be non-negative")
    return f"eval:v1:{dataset_name}:{task_id}:sample:{sample_index}"


def _eval_artifact_root(args, train_step: int) -> Path | None:
    explicit = os.environ.get("OPENCLAW_EVAL_ARTIFACT_DIR", "").strip()
    if explicit:
        base = Path(explicit)
    elif getattr(args, "save", None):
        base = Path(args.save) / "eval_artifacts"
    elif os.environ.get("SAVE_CKPT", "").strip():
        base = Path(os.environ["SAVE_CKPT"]) / "eval_artifacts"
    else:
        return None
    return base / f"train_step_{train_step:06d}"


def _workspace_archive_manifest(path: Path) -> dict:
    members: list[dict] = []
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.name in {".", "./"}:
                continue
            kind = (
                "file"
                if member.isfile()
                else "dir"
                if member.isdir()
                else "symlink"
                if member.issym()
                else "other"
            )
            members.append(
                {
                    "path": member.name.removeprefix("./"),
                    "size": int(member.size),
                    "kind": kind,
                }
            )
    return {
        "file_count": sum(member["kind"] == "file" for member in members),
        "total_uncompressed_file_bytes": sum(
            member["size"] for member in members if member["kind"] == "file"
        ),
        "members": members,
    }


def _eval_workspace_capture_retry_settings() -> tuple[int, float]:
    attempts = int(os.environ.get("OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS", "3"))
    retry_delay = float(
        os.environ.get("OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS", "2")
    )
    if attempts <= 0:
        raise ValueError("OPENCLAW_EVAL_WORKSPACE_CAPTURE_ATTEMPTS must be positive")
    if retry_delay < 0:
        raise ValueError(
            "OPENCLAW_EVAL_WORKSPACE_RETRY_DELAY_SECONDS must be non-negative"
        )
    return attempts, retry_delay


def _pause_remote_eval_container(container_name: str) -> None:
    attempts, retry_delay = _eval_workspace_capture_retry_settings()
    command = "docker pause " + shlex.quote(container_name)
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        try:
            result = _remote_run(command, timeout=30, check=False)
            if result.returncode == 0:
                return
            stderr = result.stderr.decode(errors="replace").strip()[:1000]
            last_error = f"exit={result.returncode}, stderr={stderr or '[empty]'}"
        except Exception as exc:
            last_error = _environment_error_summary(exc)
        if attempt < attempts:
            print(
                "[ClawGym-RL] eval workspace freeze retry "
                f"(container={container_name}, attempt={attempt}/{attempts}, "
                f"error={last_error})",
                flush=True,
            )
            time.sleep(retry_delay)
    raise RuntimeError(
        "could not freeze remote eval container after "
        f"{attempts} attempts: {last_error}"
    )


def _archive_terminal_workspace(
    args,
    runtime: ContainerRuntime,
    task: TaskSpec,
    rollout_id: int,
    train_step: int,
) -> dict | None:
    if os.environ.get("OPENCLAW_EVAL_SAVE_WORKSPACES", "1") != "1":
        return None
    artifact_root = _eval_artifact_root(args, train_step)
    if artifact_root is None:
        return None
    workspace_root = artifact_root / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{_safe_artifact_component(runtime.session_id)}__"
        f"{_safe_artifact_component(task.task_id)}"
    )
    destination = workspace_root / f"{stem}.tar.gz"
    max_bytes = int(
        os.environ.get("OPENCLAW_EVAL_WORKSPACE_MAX_BYTES", str(128 * 1024 * 1024))
    )
    if max_bytes <= 0:
        raise ValueError("OPENCLAW_EVAL_WORKSPACE_MAX_BYTES must be positive")

    started = time.monotonic()
    paused = False
    with tempfile.NamedTemporaryFile(
        dir=workspace_root,
        prefix=f".{stem}.",
        suffix=".tar.gz.tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        if runtime.remote_root is not None:
            _pause_remote_eval_container(runtime.container_name)
            paused = True
            remote_workspace = shlex.quote(f"{runtime.remote_root}/workspace")
            command = (
                f"tar -C {remote_workspace} "
                "--exclude=./reward "
                "--exclude=./.openclaw_reward_payload.json "
                "-czf - ."
            )
            attempts, retry_delay = _eval_workspace_capture_retry_settings()
            archive_timeout = int(
                os.environ.get("OPENCLAW_EVAL_WORKSPACE_TIMEOUT", "600")
            )
            last_error = "unknown error"
            for attempt in range(1, attempts + 1):
                try:
                    with temporary.open("wb") as output:
                        result = subprocess.run(
                            _ssh_base() + [command],
                            stdout=output,
                            stderr=subprocess.PIPE,
                            timeout=archive_timeout,
                        )
                    if result.returncode == 0:
                        break
                    stderr = result.stderr.decode(errors="replace").strip()[:1000]
                    last_error = (
                        f"exit={result.returncode}, stderr={stderr or '[empty]'}"
                    )
                except Exception as exc:
                    last_error = _environment_error_summary(exc)
                if attempt < attempts:
                    print(
                        "[ClawGym-RL] eval workspace archive retry "
                        f"(container={runtime.container_name}, "
                        f"attempt={attempt}/{attempts}, error={last_error})",
                        flush=True,
                    )
                    time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    "remote workspace archive failed after "
                    f"{attempts} attempts: {last_error}"
                )
        else:
            if not runtime.use_chroot:
                pause = subprocess.run(
                    ["docker", "pause", runtime.container_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
                if pause.returncode != 0:
                    raise RuntimeError("could not freeze local eval container")
                paused = True
            with tarfile.open(temporary, mode="w:gz", dereference=False) as archive:
                for child in sorted(runtime.workspace_dir.iterdir(), key=lambda item: item.name):
                    if child.name in {"reward", ".openclaw_reward_payload.json"}:
                        continue
                    archive.add(child, arcname=child.name, recursive=True)

        archive_bytes = temporary.stat().st_size
        if archive_bytes > max_bytes:
            raise RuntimeError(
                f"workspace archive is {archive_bytes} bytes, above limit {max_bytes}"
            )
        digest = hashlib.sha256()
        with temporary.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        archive_manifest = _workspace_archive_manifest(temporary)
        temporary.replace(destination)
        return {
            "archive": str(destination),
            "archive_relative": str(destination.relative_to(artifact_root)),
            "archive_bytes": archive_bytes,
            "archive_sha256": digest.hexdigest(),
            "capture_seconds": round(time.monotonic() - started, 6),
            "workspace_frozen_before_verifier": True,
            "reward_assets_excluded": True,
            **archive_manifest,
        }
    finally:
        if paused:
            if runtime.remote_root is not None:
                _remote_run(
                    "docker unpause " + shlex.quote(runtime.container_name),
                    timeout=30,
                    check=False,
                )
            elif not runtime.use_chroot:
                subprocess.run(
                    ["docker", "unpause", runtime.container_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                )
        temporary.unlink(missing_ok=True)


def _read_container_text_file(runtime: ContainerRuntime, container_path: str) -> str:
    if runtime.remote_root is not None:
        result = _remote_run(
            f"docker exec {shlex.quote(runtime.container_name)} "
            f"cat {shlex.quote(container_path)}",
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.decode(errors="replace").strip()

    target_path = runtime.config_dir / ".copied_output.txt"
    if target_path.exists():
        target_path.unlink()
    copy_result = subprocess.run(
        ["docker", "cp", f"{runtime.container_name}:{container_path}", str(target_path)],
        capture_output=True,
        text=True,
    )
    if copy_result.returncode != 0:
        return ""
    if not target_path.exists():
        return ""
    return target_path.read_text(encoding="utf-8").strip()


def _copy_reward_assets_to_container(task: TaskSpec, runtime: ContainerRuntime) -> str:
    source_path = task.task_dir / "reward"
    if runtime.remote_root is not None:
        _remote_run(
            f"docker cp - {shlex.quote(runtime.container_name)}:/root/.openclaw/workspace",
            input_data=_tar_bytes([(source_path, "reward")]),
            timeout=120,
        )
    elif runtime.use_chroot:
        reward_dst = runtime.workspace_dir / "reward"
        if reward_dst.exists():
            shutil.rmtree(reward_dst)
        shutil.copytree(source_path, reward_dst)
    else:
        subprocess.run(
            ["docker", "cp", str(source_path), f"{runtime.container_name}:/root/.openclaw/workspace"],
            check=True, capture_output=True, text=True,
        )
    return "/root/.openclaw/workspace/reward/reward.sh"


def _read_session_transcript(runtime: ContainerRuntime) -> list[dict]:
    """Read the OpenClaw agent session JSONL from inside the container."""
    if runtime.remote_root is not None:
        result = _remote_run(
            f"docker cp {shlex.quote(runtime.container_name)}:"
            "/root/.openclaw/agents/main/sessions -",
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            return []
        transcript = []
        try:
            with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:*") as archive:
                for member in archive.getmembers():
                    if not member.isfile() or not member.name.endswith(".jsonl"):
                        continue
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    for raw_line in handle.read().decode(errors="replace").splitlines():
                        try:
                            transcript.append(json.loads(raw_line))
                        except json.JSONDecodeError:
                            pass
        except tarfile.TarError:
            return []
        return transcript
    elif runtime.use_chroot:
        sessions_dir = runtime.chroot_rootfs / "root" / ".openclaw" / "agents" / "main" / "sessions"
    else:
        tmp_sessions = runtime.config_dir / ".sessions_copy"
        result = subprocess.run(
            ["docker", "cp",
             f"{runtime.container_name}:/root/.openclaw/agents/main/sessions",
             str(tmp_sessions)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return []
        sessions_dir = tmp_sessions

    if not sessions_dir.is_dir():
        return []

    transcript = []
    for jsonl_file in sorted(sessions_dir.glob("*.jsonl")):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    transcript.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # skip truncated/corrupt lines
    return transcript


def _write_reward_payload(
    task: TaskSpec,
    runtime: ContainerRuntime,
    final_message: str,
) -> str:
    transcript = _read_session_transcript(runtime)
    payload = {
        "final_message": final_message,
        "metadata": task.metadata,
        "transcript": transcript,
    }
    payload_bytes = json.dumps(payload, ensure_ascii=False).encode()
    if runtime.remote_root is not None:
        container_path = "/root/.openclaw/workspace/.openclaw_reward_payload.json"
        _remote_run(
            f"docker exec -i {shlex.quote(runtime.container_name)} "
            f"sh -c {shlex.quote('cat > ' + container_path)}",
            input_data=payload_bytes,
            timeout=30,
        )
    else:
        payload_path = runtime.workspace_dir / ".openclaw_reward_payload.json"
        payload_path.write_bytes(payload_bytes)
    return str(PurePosixPath("/root/.openclaw/workspace/.openclaw_reward_payload.json"))


def _reward_exec_command(container_script_path: str) -> list[str]:
    return ["bash", container_script_path]


def _parse_reward_stdout(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        print("[ClawGym-RL] WARNING: reward script produced no stdout, treating as 0.0", flush=True)
        return 0.0
    try:
        value = float(lines[-1])
    except ValueError:
        print(f"[ClawGym-RL] WARNING: reward script output not a number: {lines[-1]!r}, treating as 0.0", flush=True)
        return 0.0
    if not math.isfinite(value):
        print(f"[ClawGym-RL] WARNING: non-finite reward {value!r}, treating as 0.0", flush=True)
        return 0.0
    return value


def _run_reward(
    task: TaskSpec,
    runtime: ContainerRuntime,
    final_message: str,
) -> float:
    container_reward_path = _copy_reward_assets_to_container(task, runtime)
    container_payload_path = _write_reward_payload(task, runtime, final_message)

    if runtime.remote_root is not None:
        result = _remote_run(
            " ".join(
                shlex.quote(part)
                for part in [
                    "docker", "exec", "--user", "root",
                    "-e", f"OPENCLAW_REWARD_PAYLOAD={container_payload_path}",
                    runtime.container_name,
                ] + _reward_exec_command(container_reward_path)
            ),
            timeout=_REWARD_TIMEOUT_SECONDS + 30,
            check=False,
        )
        result.stdout = result.stdout.decode(errors="replace")
        result.stderr = result.stderr.decode(errors="replace")
    elif runtime.use_chroot:
        result = subprocess.run(
            [
                "unshare", "--user", "--map-root-user", "--",
                "chroot", str(runtime.chroot_rootfs),
                "bash", "-c",
                f"export OPENCLAW_REWARD_PAYLOAD={container_payload_path} "
                f"PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
                f"HOME=/root LC_ALL=C.UTF-8; "
                f"bash {container_reward_path}",
            ],
            capture_output=True, text=True, timeout=_REWARD_TIMEOUT_SECONDS,
        )
    else:
        result = subprocess.run(
            [
                "docker", "exec", "--user", "root",
                "-e", f"OPENCLAW_REWARD_PAYLOAD={container_payload_path}",
                runtime.container_name,
            ] + _reward_exec_command(container_reward_path),
            capture_output=True, text=True, timeout=_REWARD_TIMEOUT_SECONDS,
        )

    if result.returncode != 0:
        stderr_preview = (result.stderr or "").strip()[:500]
        print(
            f"[ClawGym-RL] reward.sh failed for {task.task_id} "
            f"(exit={result.returncode}): {stderr_preview}",
            flush=True,
        )
        return 0.0
    return _parse_reward_stdout(result.stdout)


def _load_opsd_evidence_anchors(task_id: str) -> list[dict[str, str]]:
    path = os.environ.get("OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE", "").strip()
    if not path:
        return []
    try:
        return load_evidence_anchors(path, task_id)
    except Exception as exc:
        print(
            f"[ClawGym-RL] OPSD anchor load failed for {task_id}: {exc}",
            flush=True,
        )
        return []


def _load_opsd_placebo(
    task_id: str,
) -> tuple[str, list[dict[str, str]], dict]:
    mapping_path = os.environ.get(
        "OPENCLAW_OPSD_PLACEBO_MAPPING_FILE", ""
    ).strip()
    anchor_path = os.environ.get(
        "OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE", ""
    ).strip()
    if not mapping_path or not anchor_path:
        return "", [], {}
    try:
        row = load_placebo_mapping(mapping_path).get(task_id) or {}
        placebos = row.get("placebos") or []
        if not placebos:
            return "", [], {}
        index = int(os.environ.get("OPENCLAW_OPSD_PLACEBO_INDEX", "0"))
        if index < 0 or index >= len(placebos):
            raise ValueError(
                f"placebo index {index} is unavailable for {task_id}"
            )
        selected = dict(placebos[index])
        placebo_task_id = str(selected.get("task_id") or "")
        if not placebo_task_id or placebo_task_id == task_id:
            raise ValueError("invalid self/empty placebo assignment")
        anchors = load_evidence_anchor_index(anchor_path).get(
            placebo_task_id, []
        )
        return placebo_task_id, anchors, selected
    except Exception as exc:
        print(
            f"[ClawGym-RL] matched placebo load failed for {task_id}: {exc}",
            flush=True,
        )
        return "", [], {}


def _build_opsd_feedback(
    task: TaskSpec,
    reward: float,
) -> dict:
    feedback = {
        "rollout_not_fully_successful": float(reward) < 1.0 - 1e-8,
        "evidence_anchors": _load_opsd_evidence_anchors(task.task_id),
    }
    if os.environ.get("OPENCLAW_OPSD_MATCHED_PLACEBO", "0") == "1":
        placebo_task_id, placebo_anchors, metadata = _load_opsd_placebo(
            task.task_id
        )
        feedback.update(
            {
                "placebo_task_id": placebo_task_id,
                "placebo_evidence_anchors": placebo_anchors,
                "placebo_match_metadata": metadata,
            }
        )
    return feedback


def _submit_reward(
    rl_port: int,
    session_id: str,
    reward: float,
    group_index: int,
    *,
    opsd_feedback: dict | None = None,
    infrastructure_failure: bool = False,
):
    # Retry transient network/server errors so a single blip doesn't drop the
    # sample from the GRPO group.  3 attempts, exponential backoff.
    timeout_seconds = float(
        os.environ.get("OPENCLAW_REWARD_SUBMIT_TIMEOUT_SECONDS", "120")
    )
    if timeout_seconds <= 0:
        raise ValueError("OPENCLAW_REWARD_SUBMIT_TIMEOUT_SECONDS must be positive")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.post(
                f"http://127.0.0.1:{rl_port}/set_reward",
                json={
                    "session_id": session_id,
                    "reward": reward,
                    "group_index": group_index,
                    "opsd_feedback": opsd_feedback,
                    "infrastructure_failure": infrastructure_failure,
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s
    assert last_exc is not None
    raise last_exc


def _extract_opsd_teacher_logprobs(
    output: dict,
    response_ids: list[int],
) -> list[float]:
    meta_info = output.get("meta_info") or {}
    pairs = meta_info.get("input_token_logprobs") or []
    parsed: list[tuple[float | None, int]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        parsed.append((None if pair[0] is None else float(pair[0]), int(pair[1])))
    if len(parsed) < len(response_ids):
        raise ValueError(f"teacher returned {len(parsed)} logprobs for " f"{len(response_ids)} response tokens")
    suffix = parsed[-len(response_ids) :] if response_ids else []
    if [token_id for _, token_id in suffix] != response_ids:
        raise ValueError("teacher response-token suffix mismatch")
    if any(logprob is None or not math.isfinite(logprob) for logprob, _ in suffix):
        raise ValueError("teacher returned missing/non-finite response logprob")
    return [float(logprob) for logprob, _ in suffix]


def _score_opsd_tokens(
    args,
    sample: Sample,
    teacher_tokens: list[int],
) -> list[float]:
    response_length = int(sample.response_length)
    prompt_length = len(teacher_tokens) - response_length
    if prompt_length <= 0:
        raise ValueError("invalid OPSD teacher prompt length")
    payload = {
        "input_ids": teacher_tokens,
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        # Include the final prompt token so SGLang can return a finite
        # conditional logprob for the first response token.
        "logprob_start_len": max(0, prompt_length - 1),
    }
    session_id = str((sample.metadata or {}).get("session_id", ""))
    response = requests.post(
        f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        json=payload,
        headers={"X-SMG-Routing-Key": session_id},
        timeout=float(os.environ.get("OPENCLAW_OPSD_TEACHER_TIMEOUT", "600")),
    )
    response.raise_for_status()
    output = response.json()
    response_ids = list(sample.tokens[-response_length:])
    return _extract_opsd_teacher_logprobs(output, response_ids)


def _score_opsd_segment(args, sample: Sample) -> list[float]:
    return _score_opsd_tokens(
        args,
        sample,
        list(sample.opsd_teacher_tokens),
    )


def _score_opsd_placebo_segment(args, sample: Sample) -> list[float]:
    return _score_opsd_tokens(
        args,
        sample,
        list(sample.opsd_placebo_teacher_tokens),
    )


def _score_opsd_negative_trajectories(
    args,
    completed_samples: list[list[Sample]],
) -> None:
    if not (
        getattr(args, "opsd_enable", False)
        or getattr(args, "sdar_enable", False)
    ):
        return
    flat = [sample for group in completed_samples for sample in group]
    matched_placebo = bool(getattr(args, "opsd_matched_placebo", False))
    for sample in flat:
        sample.opsd_apply = False
        sample.opsd_teacher_log_probs = [0.0] * int(sample.response_length)
        sample.opsd_placebo_log_probs = [0.0] * int(sample.response_length)

    group_to_trajectories: dict[int, dict[int, list[Sample]]] = {}
    for sample in flat:
        group_id = int(sample.group_index)
        trajectory_id = int(sample.index)
        group_to_trajectories.setdefault(group_id, {}).setdefault(
            trajectory_id, []
        ).append(sample)
    agentopsd = str(getattr(args, "opsd_mode", "absolute")) in {"agentopsd", "bb_agentopsd"}

    eligible_trajectories: list[list[Sample]] = []
    for trajectories in group_to_trajectories.values():
        trajectory_rows = list(trajectories.values())
        rewards = [
            float(rows[0].get_reward_value(args)) for rows in trajectory_rows
        ]
        group_mean = sum(rewards) / len(rewards)
        for rows, reward in zip(trajectory_rows, rewards, strict=True):
            if (
                not agentopsd
                and not getattr(args, "sdar_enable", False)
                and reward >= group_mean
            ):
                continue
            if not all(
                bool(getattr(sample, "opsd_eligible", False))
                for sample in rows
            ):
                for sample in rows:
                    sample.opsd_fallback_reason = (
                        getattr(sample, "opsd_fallback_reason", "")
                        or "trajectory_segment_ineligible"
                    )
                continue
            eligible_trajectories.append(rows)

    segment_rows = [
        (trajectory_rows, sample)
        for trajectory_rows in eligible_trajectories
        for sample in trajectory_rows
    ]
    if not segment_rows:
        return

    real_results: dict[int, list[float]] = {}
    placebo_results: dict[int, list[float]] = {}
    failures: set[int] = set()
    jobs = []
    for trajectory_rows, sample in segment_rows:
        jobs.append(("real", trajectory_rows, sample, _score_opsd_segment))
        if matched_placebo:
            jobs.append(
                (
                    "placebo",
                    trajectory_rows,
                    sample,
                    _score_opsd_placebo_segment,
                )
            )
    with ThreadPoolExecutor(
        max_workers=int(args.opsd_teacher_concurrency)
    ) as executor:
        future_to_row = {
            executor.submit(score_fn, args, sample): (
                condition,
                trajectory_rows,
                sample,
            )
            for condition, trajectory_rows, sample, score_fn in jobs
        }
        for future in as_completed(future_to_row):
            condition, trajectory_rows, sample = future_to_row[future]
            trajectory_id = int(sample.index)
            try:
                target = (
                    real_results
                    if condition == "real"
                    else placebo_results
                )
                target[id(sample)] = future.result()
            except Exception as exc:
                failures.add(trajectory_id)
                for row in trajectory_rows:
                    row.opsd_fallback_reason = (
                        f"{condition}_teacher_scoring_failed:{exc}"
                    )

    for trajectory_rows in eligible_trajectories:
        trajectory_id = int(trajectory_rows[0].index)
        if trajectory_id in failures:
            continue
        if not all(id(sample) in real_results for sample in trajectory_rows):
            continue
        if matched_placebo and not all(
            id(sample) in placebo_results for sample in trajectory_rows
        ):
            continue
        for sample in trajectory_rows:
            sample.opsd_teacher_log_probs = real_results[id(sample)]
            if matched_placebo:
                sample.opsd_placebo_log_probs = placebo_results[id(sample)]
            sample.opsd_apply = True

    if os.environ.get("OPENCLAW_GUIDANCE_REPEAT_SCORE", "0") == "1":
        repeated: dict[int, list[float]] = {}
        repeat_failures = 0
        repeat_rows = [
            (trajectory_rows, sample)
            for trajectory_rows, sample in segment_rows
            if int(trajectory_rows[0].index) not in failures
        ]
        with ThreadPoolExecutor(
            max_workers=int(args.opsd_teacher_concurrency)
        ) as executor:
            future_to_sample = {
                executor.submit(_score_opsd_segment, args, sample): sample
                for _, sample in repeat_rows
            }
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    repeated[id(sample)] = future.result()
                except Exception:
                    repeat_failures += 1

        token_abs_diffs = []
        first_step_supports = []
        second_step_supports = []
        for trajectory_rows in eligible_trajectories:
            if not all(
                sample.opsd_apply and id(sample) in repeated
                for sample in trajectory_rows
            ):
                continue
            first_sums: dict[int, float] = {}
            second_sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            for sample in trajectory_rows:
                first = sample.opsd_teacher_log_probs
                second = repeated[id(sample)]
                old = sample.rollout_log_probs
                for first_lp, second_lp, old_lp, active, step_id in zip(
                    first,
                    second,
                    old,
                    sample.loss_mask,
                    sample.opsd_step_ids,
                    strict=True,
                ):
                    if not active or int(step_id) < 0:
                        continue
                    step_id = int(step_id)
                    token_abs_diffs.append(abs(float(first_lp) - float(second_lp)))
                    first_sums[step_id] = first_sums.get(step_id, 0.0) + (
                        float(first_lp) - float(old_lp)
                    )
                    second_sums[step_id] = second_sums.get(step_id, 0.0) + (
                        float(second_lp) - float(old_lp)
                    )
                    counts[step_id] = counts.get(step_id, 0) + 1
            for step_id in sorted(counts):
                first_step_supports.append(first_sums[step_id] / counts[step_id])
                second_step_supports.append(second_sums[step_id] / counts[step_id])

        def correlation(left, right):
            if len(left) < 2 or len(left) != len(right):
                return 0.0
            left_mean = sum(left) / len(left)
            right_mean = sum(right) / len(right)
            numerator = sum(
                (x - left_mean) * (y - right_mean)
                for x, y in zip(left, right, strict=True)
            )
            left_var = sum((x - left_mean) ** 2 for x in left)
            right_var = sum((y - right_mean) ** 2 for y in right)
            denominator = math.sqrt(left_var * right_var)
            return numerator / denominator if denominator > 0 else 0.0

        def average_ranks(values):
            order = sorted(range(len(values)), key=values.__getitem__)
            ranks = [0.0] * len(values)
            position = 0
            while position < len(order):
                end = position + 1
                while end < len(order) and values[order[end]] == values[order[position]]:
                    end += 1
                rank = (position + end - 1) / 2.0
                for offset in range(position, end):
                    ranks[order[offset]] = rank
                position = end
            return ranks

        with _TRAIN_METRICS_LOCK:
            _TRAIN_METRICS["guidance_repeat_score_failures"] = repeat_failures
            _TRAIN_METRICS["guidance_repeat_scored_tokens"] = len(token_abs_diffs)
            _TRAIN_METRICS["guidance_repeat_token_logprob_mae"] = (
                sum(token_abs_diffs) / len(token_abs_diffs)
                if token_abs_diffs
                else 0.0
            )
            _TRAIN_METRICS["guidance_repeat_step_support_pearson"] = correlation(
                first_step_supports, second_step_supports
            )
            _TRAIN_METRICS["guidance_repeat_step_rank_spearman"] = correlation(
                average_ranks(first_step_supports),
                average_ranks(second_step_supports),
            )


def _build_log_lines(
    mode_label: str,
    task: TaskSpec,
    result: TaskRunResult,
    group_index: int | None,
) -> list[str]:
    lines = [f"=== {mode_label} Task {task.task_id} ==="]
    if group_index is not None:
        lines.append(f"Group Index: {group_index}")
    lines.append(f"Reward: {result.reward:.6f}")
    lines.append(f"User Query (from raw data): {task.user_query}")
    # lines.append(f"Final Message: {result.final_message}")
    lines.append("")
    if result.raw_prompts:
        if "system_prompt" in result.raw_prompts:
            first_line = result.raw_prompts["system_prompt"].split("\n", 1)[0]
            lines.append(f"System Prompt: {first_line} [truncated, see main log for full]")
        if "user_prompt" in result.raw_prompts:
            lines.append(f"User Prompt (from OpenClaw): {result.raw_prompts['user_prompt']}")
        lines.append("")
    lines.append(f"Steps: {len(result.react_steps)}")
    lines.append(f"Compactions: {result.compaction_count}")
    lines.append(f"Compaction model calls: {result.compaction_model_call_count}")
    lines.append(f"Tool-result pruning events: {result.tool_result_pruning_events}")
    lines.append(f"Tool results pruned: {result.tool_results_pruned}")
    lines.append(f"Tool-result truncation events: {result.tool_result_truncation_events}")
    lines.append(f"Tool results truncated: {result.tool_results_truncated}")
    lines.append(f"Unclassified history rewrites: {result.unclassified_history_rewrites}")
    lines.append(f"Context Overflows: {result.context_overflow_count}")
    lines.append(f"Termination Reason: {result.termination_reason}")
    if result.budget_termination:
        lines.append(
            "Budget Termination: "
            + json.dumps(result.budget_termination, ensure_ascii=False, sort_keys=True)
        )
    lines.append("")
    for step in result.react_steps:
        step_number = step["step"]
        lines.append(f"--- Step {step_number} ---")
        if step.get("reasoning_content"):
            lines.append(f"REASONING_CONTENT: {step['reasoning_content']}")
        lines.append(f"Content: {step['content']}")
        tool_calls = step["tool_calls"]
        observations = step["observations"]

        if tool_calls:
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_arguments = tool_call["arguments"]
                lines.append(f"ACTION: {tool_name} {tool_arguments}")
        else:
            lines.append("ACTION: [none]")

        if observations:
            for observation in observations:
                lines.append(f"OBSERVATION: {observation}")
        else:
            lines.append("OBSERVATION: [none]")
        lines.append("")
    return lines


def _log_root(args, rollout_id: int) -> Path | None:
    if args.save:
        return Path(args.save) / "openclaw_logs" / f"rollout_{rollout_id}"
    if "SAVE_CKPT" in os.environ and os.environ["SAVE_CKPT"]:
        return Path(os.environ["SAVE_CKPT"]) / "openclaw_logs" / f"rollout_{rollout_id}"
    return None


def _environment_error_summary(exc: Exception) -> str:
    """Format environment failures without serializing secret-bearing commands."""
    if isinstance(exc, subprocess.CalledProcessError):
        return (
            "CalledProcessError: command failed with exit status "
            f"{exc.returncode}"
        )
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"TimeoutExpired: command exceeded {exc.timeout} seconds"
    return f"{type(exc).__name__}: {exc}"


def _write_instance_log(
    args,
    rollout_id: int,
    file_name: str,
    mode_label: str,
    task: TaskSpec,
    result: TaskRunResult,
    group_index: int | None,
):
    log_dir = _log_root(args, rollout_id)
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    log_lines = _build_log_lines(mode_label, task, result, group_index)
    with (log_dir / file_name).open("w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(log_lines) + "\n")



def _rubric_entry_for_task(task: TaskSpec) -> tuple[dict, str, int]:
    dataset_value = os.environ.get("OPENCLAW_RUBRIC_DATASET", "").strip()
    if not dataset_value:
        raise RuntimeError("OPENCLAW_RUBRIC_DATASET is required for rubric archival")
    dataset_path = Path(dataset_value)
    raw_bytes = dataset_path.read_bytes()
    rows = [json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
    try:
        ordinal = int(task.task_id.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cannot map rubric task id to benchmark row: {task.task_id}") from exc
    if not 0 <= ordinal < len(rows):
        raise IndexError(f"rubric task ordinal out of range: {task.task_id}")
    entry = _require_dict(rows[ordinal], f"rubric row {ordinal}")
    if entry.get("prompt") != task.user_query:
        raise ValueError(f"rubric prompt mismatch for {task.task_id}")
    rules = entry.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"rubric task has no rules: {task.task_id}")
    return entry, hashlib.sha256(raw_bytes).hexdigest(), ordinal


def _safe_rubric_target(file_path: str) -> PurePosixPath:
    target = PurePosixPath(file_path)
    if target.is_absolute() or not target.parts or ".." in target.parts:
        raise ValueError(f"unsafe rubric file_path: {file_path!r}")
    return target


def _read_rubric_target(
    runtime: ContainerRuntime, file_path: str, max_bytes: int = 65536
) -> dict:
    target = _safe_rubric_target(file_path)
    if runtime.remote_root is not None:
        container_path = "/root/.openclaw/workspace/" + target.as_posix()
        command = " ".join(
            shlex.quote(part)
            for part in [
                "docker", "exec", runtime.container_name,
                "head", "-c", str(max_bytes), "--", container_path,
            ]
        )
        result = _remote_run(command, timeout=30, check=False)
        if result.returncode != 0:
            return {"exists": False, "content": ""}
        raw = result.stdout
    else:
        host_path = runtime.workspace_dir.joinpath(*target.parts)
        if not host_path.is_file():
            return {"exists": False, "content": ""}
        with host_path.open("rb") as file_handle:
            raw = file_handle.read(max_bytes)
    text = raw.decode("utf-8", errors="replace")
    return {
        "exists": True,
        "content": text[:4000],
        "bytes_read": len(raw),
        "characters_archived": min(len(text), 4000),
    }


def _archive_rubric_artifact(
    task: TaskSpec,
    runtime: ContainerRuntime,
    final_message: str,
    code_score: float,
) -> None:
    archive_value = os.environ.get("OPENCLAW_RUBRIC_ARCHIVE_DIR", "").strip()
    if not archive_value:
        return
    entry, dataset_sha256, ordinal = _rubric_entry_for_task(task)
    rules = entry["rules"]
    target_paths = sorted(
        {
            _require_string(rule["file_path"], "rubric rule file_path")
            for rule in rules
            if isinstance(rule, dict) and "file_path" in rule
        }
    )
    target_files = {
        file_path: _read_rubric_target(runtime, file_path)
        for file_path in target_paths
    }
    payload = {
        "task_id": task.task_id,
        "session_id": runtime.session_id,
        "benchmark_ordinal": ordinal,
        "benchmark_sha256": dataset_sha256,
        "prompt": task.user_query,
        "rules": rules,
        "target_files": target_files,
        "code_score": code_score,
        "final_message": final_message,
        "max_characters_per_file": 4000,
    }
    archive_dir = Path(archive_value)
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / f"{task.task_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)
def _run_single_task(
    args,
    task: TaskSpec,
    rollout_id: int,
    task_index: int,
    turn_type: str,
    max_turns: int,
    rl_port: int,
    group_index: int | None,
    log_prefix: str,
    eval_train_step: int | None = None,
    sampling_identity: str | None = None,
    prewarm_barrier: _GatewayPrewarmBarrier | None = None,
    prewarm_task: bool = False,
) -> TaskRunResult:
    task_start_time = time.time()
    phase = "train" if log_prefix == "train" else "eval"
    session_id = _train_session_id(rollout_id,task_index) if phase=='train' else f"{phase}-task-{rollout_id}-{task_index}"
    if phase=='train' and sampling_identity is None:
        sampling_identity=f"train-task-{rollout_id}-{task_index}"
    if prewarm_barrier is not None and not prewarm_task:
        prewarm_barrier.wait_for_release()
    try:
        _register_session_metadata(
            rl_port,
            session_id,
            task,
            rollout_id,
            task_index,
            group_index,
            sampling_identity,
        )
        log_dir = _log_root(args, rollout_id)
    except Exception as exc:
        if prewarm_barrier is not None and prewarm_task:
            prewarm_barrier.abort(_environment_error_summary(exc))
        raise
    # Resolve every potentially failing non-runtime attribute before creating
    # the container.  Once _start_container returns, all subsequent work must
    # remain inside the cleanup guard so a driver contract error cannot leak a
    # live remote container and fan out through the executor queue.
    start_args = (
        task,
        rollout_id,
        task_index,
        rl_port,
        session_id,
        f"{turn_type}__{session_id}",
    )
    startup_attempts = (
        max(1, int(os.environ.get("OPENCLAW_REMOTE_START_ATTEMPTS", "8")))
        if _remote_enabled()
        else 1
    )
    runtime = None
    for startup_attempt in range(1, startup_attempts + 1):
        attempt_started = time.perf_counter()
        try:
            if _remote_enabled():
                start_semaphore = _remote_docker_start_semaphore()
                queue_started = time.perf_counter()
                with start_semaphore:
                    if turn_type == "main":
                        _record_train_startup_timing(
                            "queue_wait", time.perf_counter() - queue_started
                        )
                        slot_started = time.perf_counter()
                    try:
                        runtime = _start_container(
                            *start_args, is_train=(turn_type == "main")
                        )
                        # Keep the startup slot until the gateway is actually
                        # serving requests. Limiting only ``docker run`` lets a
                        # large worker pool still create an initialization storm.
                        health_started = time.perf_counter()
                        try:
                            _wait_for_chat_ready(runtime)
                        finally:
                            if turn_type == "main":
                                _record_train_startup_timing(
                                    "health_ready",
                                    time.perf_counter() - health_started,
                                )
                    finally:
                        if turn_type == "main":
                            _record_train_startup_timing(
                                "slot_held", time.perf_counter() - slot_started
                            )
            else:
                runtime = _start_container(
                    *start_args, is_train=(turn_type == "main")
                )
                _wait_for_chat_ready(runtime)
            if turn_type == "main":
                _increment_train_startup_metric("successful_attempts")
            break
        except (
            TimeoutError,
            OSError,
            subprocess.SubprocessError,
            requests.exceptions.ConnectionError,
        ) as exc:
            if runtime is not None:
                failed_gateway_port = (
                    int(runtime.gateway_url.rsplit(":", 1)[1])
                    if runtime.remote_root is not None
                    else None
                )
                try:
                    _stop_container(runtime, log_dir)
                except Exception as cleanup_exc:
                    print(
                        "[ClawGym-RL] remote startup cleanup warning "
                        f"(attempt={startup_attempt}/{startup_attempts}, "
                        f"error={_environment_error_summary(cleanup_exc)})",
                        flush=True,
                    )
                # Another host-network container can claim a port between the
                # free-port probe and docker run. Keep failed candidates out of
                # this RolloutManager's pool even after normal cleanup releases
                # their live reservation.
                if failed_gateway_port is not None:
                    with _REMOTE_GATEWAY_PORT_LOCK:
                        _REMOTE_GATEWAY_PORTS_IN_USE.add(failed_gateway_port)
                runtime = None
            if startup_attempt == startup_attempts:
                if turn_type == "main":
                    _increment_train_startup_metric("failed_attempts")
                if prewarm_barrier is not None and prewarm_task:
                    prewarm_barrier.abort(_environment_error_summary(exc))
                raise
            if turn_type == "main":
                _increment_train_startup_metric("retries")
            print(
                "[ClawGym-RL] remote gateway startup retry "
                f"(attempt={startup_attempt}/{startup_attempts}, "
                f"error={_environment_error_summary(exc)})",
                flush=True,
            )
        finally:
            if turn_type == "main":
                _increment_train_startup_metric("attempts")
                _record_train_startup_timing(
                    "attempt", time.perf_counter() - attempt_started
                )
    assert runtime is not None
    try:
        if prewarm_barrier is not None and prewarm_task:
            prewarm_barrier.arrive_and_wait(task_index)
        final_message = _run_turns(runtime, task, max_turns=max_turns, turn_type=turn_type)
        (
            react_steps,
            raw_prompts,
            compaction_count,
            compaction_model_call_count,
            context_overflow_count,
            tool_result_pruning_events,
            tool_results_pruned,
            tool_result_truncation_events,
            tool_results_truncated,
            unclassified_history_rewrites,
            budget_termination,
            sglang_stats,
        ) = _fetch_react_steps(rl_port, session_id)
        termination_reason = _classify_termination_reason(
            react_steps, budget_termination
        )
        workspace_artifact = None
        if log_prefix == "eval":
            try:
                workspace_artifact = _archive_terminal_workspace(
                    args,
                    runtime,
                    task,
                    rollout_id,
                    int(eval_train_step if eval_train_step is not None else _eval_train_step(args, rollout_id)),
                )
            except Exception as exc:
                workspace_artifact = {
                    "error": _environment_error_summary(exc),
                    "workspace_frozen_before_verifier": False,
                }
        reward = _run_reward(task, runtime, final_message)

        # Format penalty: if any step's content contains the tool call bot_token,
        # it means the model tried to make a tool call but SGLang failed to parse it.
        # We detect this by checking for the bot_token of the configured parser.
        #
        # Default DISABLED: previous runs shipped with a broken marker detector
        # (qwen3_coder's bot_token=''), so no penalty ever fired. Analysis showed
        # the fixed version mostly punishes successful trajectories with a trailing
        # `<tool_call>` sentinel (reward>0 → 0), which GRPO already handles via
        # group normalization. Set OPENCLAW_FORMAT_PENALTY=1 to opt in.
        _tool_call_marker = _get_tool_call_marker() if os.environ.get("OPENCLAW_FORMAT_PENALTY", "0") == "1" else ""
        if _tool_call_marker:
            for step in react_steps:
                if _tool_call_marker in step.get("content", ""):
                    print(
                        f"[ClawGym-RL] {log_prefix} {task.task_id}: format penalty "
                        f"(step {step.get('step')}: unparsed tool_call in content), reward {reward:.4f} → 0.0",
                        flush=True,
                    )
                    if log_prefix == "train":
                        with _TRAIN_METRICS_LOCK:
                            _TRAIN_METRICS["format_penalty"] = _TRAIN_METRICS.get("format_penalty", 0) + 1
                            if reward > 0:
                                _TRAIN_METRICS["format_penalty_nonzero_to_zero"] = _TRAIN_METRICS.get("format_penalty_nonzero_to_zero", 0) + 1
                    reward = 0.0
                    break

        _archive_rubric_artifact(task, runtime, final_message, reward)
        if group_index is not None:
            _submit_reward(
                rl_port,
                session_id,
                reward,
                group_index,
                opsd_feedback=(
                    _build_opsd_feedback(task, reward)
                    if (
                        os.environ.get("OPENCLAW_OPSD_ENABLE", "0") == "1"
                        or os.environ.get("OPENCLAW_SDAR_ENABLE", "0") == "1"
                    )
                    else None
                ),
            )

        # Track per-trajectory metrics for wandb
        if log_prefix == "train":
            duration = time.time() - task_start_time
            with _TRAIN_METRICS_LOCK:
                m = _TRAIN_METRICS
                m["submitted"] = m.get("submitted", 0) + 1
                m.setdefault("rewards", []).append(reward)
                m["steps_total"] = m.get("steps_total", 0) + len(react_steps)
                for metric_name in (
                    "requests",
                    "prompt_tokens",
                    "cache_hit_tokens",
                    "prefill_compute_tokens",
                    "request_e2e_seconds",
                    "engine_switches",
                    "affinity_used",
                    "affinity_fallback",
                ):
                    total_name = f"sglang_{metric_name}_total"
                    m[total_name] = m.get(total_name, 0) + sglang_stats.get(metric_name, 0)
                m.setdefault("durations", []).append(duration)
                m["compactions_total"] = m.get("compactions_total", 0) + compaction_count
                m["compaction_model_calls_total"] = (
                    m.get("compaction_model_calls_total", 0) + compaction_model_call_count
                )
                m["tool_result_pruning_events_total"] = (
                    m.get("tool_result_pruning_events_total", 0)
                    + tool_result_pruning_events
                )
                m["tool_results_pruned_total"] = (
                    m.get("tool_results_pruned_total", 0) + tool_results_pruned
                )
                m["tool_result_truncation_events_total"] = (
                    m.get("tool_result_truncation_events_total", 0)
                    + tool_result_truncation_events
                )
                m["tool_results_truncated_total"] = (
                    m.get("tool_results_truncated_total", 0) + tool_results_truncated
                )
                m["unclassified_history_rewrites_total"] = (
                    m.get("unclassified_history_rewrites_total", 0)
                    + unclassified_history_rewrites
                )
                m["context_overflows_total"] = (
                    m.get("context_overflows_total", 0) + context_overflow_count
                )
                m["assistant_turns_total"] = m.get("assistant_turns_total", 0) + len(react_steps)
                turns_hit_response_limit = sum(
                    bool(step.get("hit_response_limit", False))
                    for step in react_steps
                )
                m["turns_hit_response_limit"] = (
                    m.get("turns_hit_response_limit", 0)
                    + turns_hit_response_limit
                )
                if turns_hit_response_limit:
                    m["trajectories_hit_response_limit"] = (
                        m.get("trajectories_hit_response_limit", 0) + 1
                    )
                m["incomplete_tool_calls_at_limit"] = (
                    m.get("incomplete_tool_calls_at_limit", 0)
                    + sum(
                        bool(step.get("incomplete_tool_call_at_limit", False))
                        for step in react_steps
                    )
                )
                m["response_limit_tokens"] = max(
                    m.get("response_limit_tokens", 0),
                    max(
                        (
                            int(step.get("configured_response_limit_tokens", 0))
                            for step in react_steps
                        ),
                        default=0,
                    ),
                )
                m.setdefault("repeated_ngram_ratios", []).extend(
                    float(step.get("repeated_ngram_ratio", 0.0))
                    for step in react_steps
                )
                m["max_ngram_repeat"] = max(
                    m.get("max_ngram_repeat", 0),
                    max(
                        (int(step.get("max_ngram_repeat", 0)) for step in react_steps),
                        default=0,
                    ),
                )
                if any(
                    bool(step.get("repetition_degenerated", False))
                    for step in react_steps
                ):
                    m["repetition_degenerated_trajectories"] = (
                        m.get("repetition_degenerated_trajectories", 0) + 1
                    )
                termination_key = f"termination_{termination_reason}"
                m[termination_key] = m.get(termination_key, 0) + 1
                m["multi_tool_turns_total"] = m.get("multi_tool_turns_total", 0) + sum(
                    int(step.get("returned_tool_call_count", 0)) > 1
                    for step in react_steps
                )
                m["raw_tool_calls_total"] = m.get("raw_tool_calls_total", 0) + sum(
                    int(step.get("raw_tool_call_count", 0)) for step in react_steps
                )
                m["parsed_tool_calls_total"] = m.get("parsed_tool_calls_total", 0) + sum(
                    int(step.get("parsed_tool_call_count", 0)) for step in react_steps
                )
                m["returned_tool_calls_total"] = m.get("returned_tool_calls_total", 0) + sum(
                    int(step.get("returned_tool_call_count", 0)) for step in react_steps
                )
                m["max_identical_tool_call_turns"] = max(
                    m.get("max_identical_tool_call_turns", 0),
                    max(
                        (int(step.get("consecutive_identical_tool_call_turns", 0)) for step in react_steps),
                        default=0,
                    ),
                )
                if budget_termination:
                    reason = str(budget_termination.get("reason", "unknown"))
                    m["budget_terminated"] = m.get("budget_terminated", 0) + 1
                    key = f"budget_terminated_{reason}"
                    m[key] = m.get(key, 0) + 1

        result = TaskRunResult(
            reward=reward,
            final_message=final_message,
            react_steps=react_steps,
            raw_prompts=raw_prompts,
            compaction_count=compaction_count,
            compaction_model_call_count=compaction_model_call_count,
            tool_result_pruning_events=tool_result_pruning_events,
            tool_results_pruned=tool_results_pruned,
            tool_result_truncation_events=tool_result_truncation_events,
            tool_results_truncated=tool_results_truncated,
            unclassified_history_rewrites=unclassified_history_rewrites,
            context_overflow_count=context_overflow_count,
            budget_termination=budget_termination,
            termination_reason=termination_reason,
            session_id=session_id,
            duration_seconds=time.time() - task_start_time,
            sglang_stats=sglang_stats,
            workspace_artifact=workspace_artifact,
        )
        _write_instance_log(
            args,
            rollout_id,
            f"instance-{log_prefix}-{task_index}.log",
            log_prefix.upper(),
            task,
            result,
            group_index,
        )
        return result
    finally:
        _stop_container(runtime, log_dir)


def _train_batch(args, tasks: list[TaskSpec], rollout_id: int) -> list[tuple[int, TaskSpec]]:
    """Compute the (group_index, task) batch for this rollout.

    Cursor is derived deterministically from rollout_id so that resume picks up
    where we left off (instead of restarting from task 0). This requires
    rollout_batch_size to stay constant across resume.
    """
    n_prompts = int(args.rollout_batch_size)
    n_samples = int(args.n_samples_per_prompt)
    n_tasks = len(tasks)
    base_seed = int(os.environ.get("OPENCLAW_TRAIN_SHUFFLE_SEED", "0"))
    batch: list[tuple[int, TaskSpec]] = []
    cursor = rollout_id * n_prompts
    for prompt_offset in range(n_prompts):
        global_idx = cursor + prompt_offset
        epoch, pos = divmod(global_idx, n_tasks)
        rng = random.Random(base_seed + epoch)
        perm = list(range(n_tasks))
        rng.shuffle(perm)
        task = tasks[perm[pos]]
        for _ in range(n_samples):
            batch.append((prompt_offset, task))
    return batch


def _train_submission_indices(
    batch: list[tuple[int, TaskSpec]], scheduling: str | None = None
) -> list[int]:
    """Return submission order without changing stable task/sample indices.

    ``group_interleaved`` starts one sample from every GRPO group before the
    second sample from any group. This prevents a whole late group from
    waiting behind earlier groups when the executor has fewer workers than
    trajectories. Returned values are original batch indices, so session
    IDs, request seeds, rewards, and GRPO grouping remain unchanged.
    """
    mode = scheduling or os.environ.get("OPENCLAW_TRAIN_SCHEDULING", "sequential")
    if mode == "sequential":
        return list(range(len(batch)))
    if mode != "group_interleaved":
        raise ValueError(f"unsupported OPENCLAW_TRAIN_SCHEDULING: {mode}")

    indices_by_group: dict[int, list[int]] = {}
    for task_index, (group_index, _) in enumerate(batch):
        indices_by_group.setdefault(group_index, []).append(task_index)

    order: list[int] = []
    max_group_size = max((len(indices) for indices in indices_by_group.values()), default=0)
    for sample_offset in range(max_group_size):
        for indices in indices_by_group.values():
            if sample_offset < len(indices):
                order.append(indices[sample_offset])
    return order


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_source_manifest(source: TaskSource) -> dict:
    digest = hashlib.sha256()
    verifier_files: list[dict] = []
    for task in source.tasks:
        stable_task = {
            "task_id": task.task_id,
            "user_query": task.user_query,
            "input_mount_dir": task.input_mount_dir,
            "metadata": task.metadata,
        }
        digest.update(
            json.dumps(stable_task, ensure_ascii=False, sort_keys=True).encode("utf-8")
        )
        reward_root = task.task_dir / "reward"
        if reward_root.is_dir():
            for path in sorted(reward_root.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(task.task_dir).as_posix()
                file_digest = _sha256_file(path)
                verifier_files.append(
                    {
                        "task_id": task.task_id,
                        "path": relative,
                        "bytes": path.stat().st_size,
                        "sha256": file_digest,
                    }
                )
                digest.update(relative.encode("utf-8"))
                digest.update(file_digest.encode("ascii"))
    return {
        "dataset_name": source.dataset_name,
        "task_count": len(source.tasks),
        "task_ids": [task.task_id for task in source.tasks],
        "task_and_verifier_sha256": digest.hexdigest(),
        "verifier_files": verifier_files,
    }


def _checkpoint_identity(args, rollout_id: int, train_step: int) -> dict:
    if train_step == 0:
        path = Path(args.hf_checkpoint)
        identity = {"kind": "hf_sft_initial", "path": str(path)}
        for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
            candidate = path / name
            if candidate.is_file():
                identity[f"{name}_sha256"] = _sha256_file(candidate)
        return identity

    save_root = Path(args.save) if getattr(args, "save", None) else None
    path = save_root / f"iter_{rollout_id:07d}" if save_root is not None else None
    identity = {
        "kind": "megatron_full_resume",
        "framework_iteration": rollout_id,
        "semantic_train_step": train_step,
        "path": str(path) if path is not None else None,
        "exists": bool(path is not None and path.is_dir()),
    }
    if path is not None and path.is_dir():
        files = sorted(candidate for candidate in path.iterdir() if candidate.is_file())
        identity["files"] = [
            {"name": candidate.name, "bytes": candidate.stat().st_size}
            for candidate in files
        ]
        for name in (".metadata", "common.pt", "metadata.json"):
            candidate = path / name
            if candidate.is_file():
                identity[f"{name}_sha256"] = _sha256_file(candidate)
    return identity


def _repository_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "--", "clawgym-official", "slime"],
        cwd=root,
        capture_output=True,
        timeout=30,
    )
    return {
        "root": str(root),
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(diff.stdout),
        "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
    }


def _write_eval_artifact_bundle(
    args,
    rollout_id: int,
    train_step: int,
    sources: list[TaskSource],
    rows: list[dict],
    summaries: dict[str, dict],
) -> Path | None:
    root = _eval_artifact_root(args, train_step)
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.jsonl"
    temporary_results = results_path.with_suffix(".jsonl.tmp")
    with temporary_results.open("w", encoding="utf-8") as output:
        for row in sorted(rows, key=lambda item: (item["dataset_name"], item["flat_index"])):
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary_results.replace(results_path)

    manifest = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "rollout_id": rollout_id,
        "semantic_train_step": train_step,
        "checkpoint": _checkpoint_identity(args, rollout_id, train_step),
        "repository": _repository_identity(),
        "datasets": [_task_source_manifest(source) for source in sources],
        "summaries": summaries,
        "results_jsonl": str(results_path),
        "results_sha256": _sha256_file(results_path),
        "configuration": {
            "eval_num_workers": int(os.environ.get("OPENCLAW_EVAL_NUM_WORKERS", "32")),
            "eval_n_samples": int(os.environ.get("OPENCLAW_EVAL_N_SAMPLES", "1")),
            "sampling_seed_base": os.environ.get("OPENCLAW_DETERMINISTIC_SAMPLING_SEED_BASE", ""),
            "eval_sampling_identity_scheme": "eval:v1:<dataset>:<task_id>:sample:<sample_index>",
            "deterministic_tool_call_ids": os.environ.get("OPENCLAW_DETERMINISTIC_TOOL_CALL_IDS", "0"),
            "sglang_deterministic_inference": os.environ.get("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "0"),
            "megatron_deterministic_mode": os.environ.get("MEGATRON_DETERMINISTIC_MODE", "0"),
            "context_window": int(os.environ.get("OPENCLAW_CONTEXT_WINDOW", "65536")),
            "max_response_tokens": int(os.environ.get("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")),
            "max_trajectory_tokens": int(os.environ.get("OPENCLAW_TRAJECTORY_MAX_TOKENS", "65536")),
            "max_agent_steps": int(os.environ.get("OPENCLAW_MAX_AGENT_STEPS", "96")),
            "max_compaction_events": int(os.environ.get("OPENCLAW_MAX_COMPACTION_EVENTS", "8")),
        },
    }
    manifest_path = root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def _run_openclaw_eval(args, rollout_id: int) -> dict[str, dict] | None:
    sources = _task_sources_from_env("OPENCLAW_EVAL_TASKS")
    if not sources:
        print("[ClawGym-RL] OPENCLAW_EVAL_TASKS not set, skipping eval", flush=True)
        return None

    if "PORT" not in os.environ or not os.environ["PORT"]:
        raise ValueError("PORT must be set for OpenClaw eval")
    rl_port = int(os.environ["PORT"])
    max_turns = int(os.environ["OPENCLAW_EVAL_MAX_TURNS"]) if "OPENCLAW_EVAL_MAX_TURNS" in os.environ else 1
    requested_workers = int(os.environ["OPENCLAW_EVAL_NUM_WORKERS"]) if "OPENCLAW_EVAL_NUM_WORKERS" in os.environ else 32
    n_eval_samples = int(os.environ.get("OPENCLAW_EVAL_N_SAMPLES", "1"))
    train_step = _eval_train_step(args, rollout_id)

    all_results: dict[str, dict] = {}
    artifact_rows: list[dict] = []
    artifact_summaries: dict[str, dict] = {}
    for source in sources:
        n_tasks = len(source.tasks)
        n_total = n_tasks * n_eval_samples
        rewards = [0.0] * n_total
        worker_count = min(requested_workers, n_total)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_index = {}
            for sample_idx in range(n_eval_samples):
                for task_index, task in enumerate(source.tasks):
                    flat_index = sample_idx * n_tasks + task_index
                    default_runtime_task_index = int(
                        os.environ.get("OPENCLAW_EVAL_TASK_INDEX_OFFSET") or "0"
                    ) + flat_index
                    runtime_task_index = _eval_runtime_task_index(
                        task.task_id, default_runtime_task_index
                    )
                    sampling_identity = _eval_sampling_identity(
                        source.dataset_name, task.task_id, sample_idx
                    )
                    future = executor.submit(
                        _run_single_task,
                        args,
                        task,
                        rollout_id,
                        runtime_task_index,
                        "side",
                        max_turns,
                        rl_port,
                        None,
                        "eval",
                        train_step,
                        sampling_identity,
                    )
                    future_to_index[future] = (
                        flat_index,
                        task_index,
                        sample_idx,
                        runtime_task_index,
                        sampling_identity,
                    )

            n_skipped = 0
            for future in as_completed(future_to_index):
                (
                    flat_index,
                    task_index,
                    sample_idx,
                    runtime_task_index,
                    sampling_identity,
                ) = future_to_index[future]
                task = source.tasks[task_index]
                try:
                    result = future.result()
                    rewards[flat_index] = result.reward
                    artifact_rows.append(
                        {
                            "dataset_name": source.dataset_name,
                            "task_id": task.task_id,
                            "task_index": task_index,
                            "runtime_task_index": runtime_task_index,
                            "sampling_identity": sampling_identity,
                            "sample_index": sample_idx,
                            "flat_index": flat_index,
                            "session_id": result.session_id,
                            "rollout_id": rollout_id,
                            "semantic_train_step": train_step,
                            "status": "completed",
                            "reward": result.reward,
                            "correct": result.reward > 0.5,
                            "termination_reason": result.termination_reason,
                            "duration_seconds": result.duration_seconds,
                            "agent_steps": len(result.react_steps),
                            "compaction_count": result.compaction_count,
                            "compaction_model_call_count": result.compaction_model_call_count,
                            "tool_result_pruning_events": result.tool_result_pruning_events,
                            "tool_results_pruned": result.tool_results_pruned,
                            "context_overflow_count": result.context_overflow_count,
                            "sglang_stats": result.sglang_stats,
                            "model_requests": [
                                {
                                    "step": step.get("step"),
                                    "sampling_seed": step.get("sampling_seed"),
                                    "sampling_identity": step.get("sampling_identity"),
                                    "sampling_params": step.get("sampling_params", {}),
                                    "finish_reason": step.get("finish_reason"),
                                    "prompt_tokens": step.get("prompt_tokens"),
                                    "response_tokens": step.get("response_tokens"),
                                }
                                for step in result.react_steps
                            ],
                            "workspace": result.workspace_artifact,
                            "instance_log": str(
                                (_log_root(args, rollout_id) or Path("."))
                                / f"instance-eval-{runtime_task_index}.log"
                            ),
                        }
                    )
                    print(
                        f"[ClawGym-RL] eval task {source.tasks[task_index].task_id}: reward={result.reward:.4f}",
                        flush=True,
                    )
                except (TimeoutError, ValueError, requests.exceptions.HTTPError, requests.exceptions.ConnectionError, OSError, subprocess.SubprocessError) as exc:
                    error_summary = _environment_error_summary(exc)
                    print(
                        f"[ClawGym-RL] eval task {task.task_id}: "
                        f"SKIPPED (env error: {error_summary})",
                        flush=True,
                    )
                    rewards[flat_index] = 0.0
                    n_skipped += 1
                    artifact_rows.append(
                        {
                            "dataset_name": source.dataset_name,
                            "task_id": task.task_id,
                            "task_index": task_index,
                            "runtime_task_index": runtime_task_index,
                            "sampling_identity": sampling_identity,
                            "sample_index": sample_idx,
                            "flat_index": flat_index,
                            "session_id": f"eval-task-{rollout_id}-{runtime_task_index}",
                            "rollout_id": rollout_id,
                            "semantic_train_step": train_step,
                            "status": "skipped",
                            "reward": 0.0,
                            "correct": False,
                            "error": error_summary,
                        }
                    )

        # Average rewards per task across n_eval_samples
        task_rewards = []
        for task_index in range(n_tasks):
            task_samples = [rewards[s * n_tasks + task_index] for s in range(n_eval_samples)]
            task_rewards.append(sum(task_samples) / len(task_samples))

        n_total_runs = n_tasks * n_eval_samples
        n_collected = n_total_runs - n_skipped
        average_reward = sum(task_rewards) / n_tasks
        avg_reward_collected_only = sum(rewards) / n_collected if n_collected > 0 else 0.0
        correct_count = sum(1 for r in task_rewards if r > 0.5)
        print(
            (
                f"[ClawGym-RL] eval dataset {source.dataset_name}: "
                f"{correct_count}/{n_tasks} correct, avg={average_reward:.4f}, "
                f"collected={n_collected}/{n_total_runs}, avg_collected_only={avg_reward_collected_only:.4f}"
                + (f", n_samples={n_eval_samples}" if n_eval_samples > 1 else "")
            ),
            flush=True,
        )
        all_results[source.dataset_name] = {
            "rewards": task_rewards,
            "n_collected": n_collected,
            "n_skipped": n_skipped,
            "avg_reward_collected_only": avg_reward_collected_only,
        }
        artifact_summaries[source.dataset_name] = {
            "task_count": n_tasks,
            "run_count": n_total_runs,
            "correct_count": correct_count,
            "correct_rate": correct_count / n_tasks,
            "average_reward": average_reward,
            "n_collected": n_collected,
            "n_skipped": n_skipped,
            "avg_reward_collected_only": avg_reward_collected_only,
        }

    manifest_path = _write_eval_artifact_bundle(
        args, rollout_id, train_step, sources, artifact_rows, artifact_summaries
    )
    if manifest_path is not None:
        print(f"[ClawGym-RL] eval artifact manifest: {manifest_path}", flush=True)
    workspace_errors = [
        row
        for row in artifact_rows
        if row.get("status") == "completed"
        and (row.get("workspace") or {}).get("error")
    ]
    if workspace_errors:
        print(
            "[ClawGym-RL] WARNING: "
            f"{len(workspace_errors)} eval workspaces failed to archive; "
            f"evaluation evidence is incomplete; see {manifest_path}",
            flush=True,
        )
    if (
        workspace_errors
        and os.environ.get("OPENCLAW_EVAL_ARTIFACT_STRICT", "1") == "1"
    ):
        raise RuntimeError(
            f"{len(workspace_errors)} eval workspaces failed to archive; "
            f"see {manifest_path}"
        )
    return all_results


def _build_eval_metrics(result: dict) -> dict | None:
    """Build wandb-ready eval metrics from _run_openclaw_eval result."""
    eval_metrics = {}
    for ds_name, ds_data in result.items():
        rewards = [float(reward) for reward in ds_data.get("rewards", [])]
        if rewards:
            correct_count = sum(reward > 0.5 for reward in rewards)
            eval_metrics[f"eval/{ds_name}-avg_reward"] = sum(rewards) / len(rewards)
            eval_metrics[f"eval/{ds_name}-correct_count"] = correct_count
            eval_metrics[f"eval/{ds_name}-correct_rate"] = correct_count / len(rewards)
            eval_metrics[f"eval/{ds_name}-task_count"] = len(rewards)
        if "n_collected" in ds_data:
            eval_metrics[f"eval/{ds_name}-n_collected"] = ds_data["n_collected"]
            eval_metrics[f"eval/{ds_name}-n_skipped"] = ds_data["n_skipped"]
            eval_metrics[f"eval/{ds_name}-avg_reward_collected_only"] = ds_data["avg_reward_collected_only"]
    return eval_metrics or None


def _active_train_submission_indices(batch):
    excluded = {x.strip() for x in os.environ.get("OPENCLAW_TRAIN_EXCLUDE_TASK_IDS", "").split(",") if x.strip()}
    return [i for i in _train_submission_indices(batch) if batch[i][1].task_id not in excluded]


def _run_openclaw_train_tasks(args, rollout_id: int):
    """Run all train tasks synchronously. Blocks until all tasks complete."""
    sources = _task_sources_from_env("OPENCLAW_TRAIN_TASKS")
    if not sources:
        print("[ClawGym-RL] OPENCLAW_TRAIN_TASKS not set, no training data", flush=True)
        return

    if len(sources) < 1:
        print("[ClawGym-RL] OPENCLAW_TRAIN_TASKS has no valid sources", flush=True)
        return
    tasks = []
    for source in sources:
        tasks.extend(source.tasks)
    print(f"[ClawGym-RL] Loaded {len(tasks)} train tasks from {len(sources)} source(s)", flush=True)

    if "PORT" not in os.environ or not os.environ["PORT"]:
        raise ValueError("PORT must be set for OpenClaw training")
    rl_port = int(os.environ["PORT"])
    max_turns = int(os.environ["OPENCLAW_TRAIN_MAX_TURNS"]) if "OPENCLAW_TRAIN_MAX_TURNS" in os.environ else 1
    requested_workers = int(os.environ["OPENCLAW_TRAIN_NUM_WORKERS"]) if "OPENCLAW_TRAIN_NUM_WORKERS" in os.environ else 128
    requested_prewarm = int(os.environ.get("OPENCLAW_GATEWAY_PREWARM_COUNT", "0"))
    if requested_prewarm < 0:
        raise ValueError("OPENCLAW_GATEWAY_PREWARM_COUNT must be non-negative")
    batch = _train_batch(args, tasks, rollout_id)
    submission_indices = _active_train_submission_indices(batch)
    if not submission_indices:
        raise RuntimeError("All train tasks in batch were excluded")
    worker_count = min(requested_workers, len(submission_indices))
    prewarm_count = min(requested_prewarm, worker_count, len(submission_indices))
    prewarm_barrier = None
    if prewarm_count:
        prewarm_timeout = float(
            os.environ.get("OPENCLAW_GATEWAY_PREWARM_TIMEOUT_SECONDS", "600")
        )
        prewarm_barrier = _GatewayPrewarmBarrier(prewarm_count, prewarm_timeout)

    # Record how many we dispatched
    with _TRAIN_METRICS_LOCK:
        _TRAIN_METRICS["dispatched"] = len(submission_indices)
        _TRAIN_METRICS["gateway_prewarm_target"] = prewarm_count

    prewarm_indices = set(submission_indices[:prewarm_count])
    print(
        "[ClawGym-RL] train scheduling="
        f"{os.environ.get('OPENCLAW_TRAIN_SCHEDULING', 'sequential')}, "
        f"workers={worker_count}, trajectories={len(submission_indices)}, excluded={len(batch)-len(submission_indices)}, "
        f"gateway_prewarm={prewarm_count}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_meta = {}
        for task_index in submission_indices:
            group_index, task = batch[task_index]
            future = executor.submit(
                _run_single_task,
                args,
                task,
                rollout_id,
                task_index,
                "main",
                max_turns,
                rl_port,
                group_index,
                "train",
                prewarm_barrier=prewarm_barrier,
                prewarm_task=task_index in prewarm_indices,
            )
            future_to_meta[future] = (task_index, group_index)
        for future in as_completed(future_to_meta):
            task_index, group_index = future_to_meta[future]
            try:
                result = future.result()
                print(f"[ClawGym-RL] train reward submitted: {result.reward:.4f}", flush=True)
            except (TimeoutError, ValueError, requests.exceptions.HTTPError, requests.exceptions.ConnectionError, OSError, subprocess.SubprocessError) as exc:
                # Submit reward=0 so GRPO sees the failure (otherwise the group is
                # short and long-loop trajectories silently disappear from gradients).
                session_id = _train_session_id(rollout_id,task_index)
                recovered = False
                try:
                    _submit_reward(
                        rl_port,
                        session_id,
                        0.0,
                        group_index,
                        infrastructure_failure=True,
                    )
                    recovered = True
                    print(
                        "[ClawGym-RL] train task SKIPPED reward=0 submitted "
                        f"(error: {_environment_error_summary(exc)})",
                        flush=True,
                    )
                except Exception as submit_exc:
                    print(
                        f"[ClawGym-RL] train task SKIPPED and reward=0 submit FAILED "
                        f"(orig: {_environment_error_summary(exc)}; "
                        f"submit: {_environment_error_summary(submit_exc)})",
                        flush=True,
                    )
                with _TRAIN_METRICS_LOCK:
                    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)):
                        _TRAIN_METRICS["termination_outer_timeout"] = (
                            _TRAIN_METRICS.get("termination_outer_timeout", 0) + 1
                        )
                    _TRAIN_METRICS["skipped"] = _TRAIN_METRICS.get("skipped", 0) + 1
                    if recovered:
                        _TRAIN_METRICS["skipped_recovered"] = _TRAIN_METRICS.get("skipped_recovered", 0) + 1
                    else:
                        _TRAIN_METRICS["skipped_lost"] = _TRAIN_METRICS.get("skipped_lost", 0) + 1

    if prewarm_barrier is not None:
        snapshot = prewarm_barrier.snapshot()
        with _TRAIN_METRICS_LOCK:
            _TRAIN_METRICS["gateway_prewarm_ready"] = int(snapshot["ready"])
            _TRAIN_METRICS["gateway_prewarm_release_seconds"] = float(
                snapshot["release_seconds"]
            )
            _TRAIN_METRICS["gateway_prewarm_aborted"] = int(snapshot["aborted"])


def get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        if _GLOBAL_WORKER is None or _GLOBAL_WORKER.worker_thread is None or not _GLOBAL_WORKER.worker_thread.is_alive():
            _GLOBAL_WORKER = AsyncRolloutWorker(args, data_buffer)
            _GLOBAL_WORKER.start()
        return _GLOBAL_WORKER


def stop_global_worker():
    global _GLOBAL_WORKER
    with _GLOBAL_WORKER_LOCK:
        if _GLOBAL_WORKER is not None:
            _GLOBAL_WORKER.stop()
            _GLOBAL_WORKER = None


def _cleanup_rollout_resources(rollout_id: int):
    """Remove leftover containers (Docker or chroot) and temp dirs from this rollout step."""
    if _remote_enabled():
        try:
            prefix = f"{_REMOTE_CONTAINER_PREFIX}-{_REMOTE_RUN_ID}-{rollout_id}-"
            listed = _remote_run(
                f"docker ps -a --format '{{{{.Names}}}}' "
                f"--filter {shlex.quote('name=^/' + prefix)}",
                timeout=30,
                check=False,
            )
            names = [
                name for name in listed.stdout.decode(errors="replace").split()
                if name.startswith(prefix)
            ]
            if names:
                _remote_run(
                    "docker rm -f " + " ".join(shlex.quote(name) for name in names),
                    timeout=60,
                    check=False,
                )
            remote_base = os.environ.get("OPENCLAW_REMOTE_BASE", "/root/clawgym_rl_runs").rstrip("/")
            rollout_root = f"{remote_base}/{_REMOTE_RUN_ID}/{rollout_id}"
            _remote_run(f"rm -rf -- {shlex.quote(rollout_root)}", timeout=60, check=False)
            import glob
            for tmp_dir in glob.glob(f"/tmp/{prefix}*"):
                shutil.rmtree(tmp_dir, ignore_errors=True)
            if names:
                print(f"[ClawGym-RL] cleaned {len(names)} remote containers for rollout {rollout_id}", flush=True)
        except Exception as exc:
            print(f"[ClawGym-RL] remote cleanup warning: {exc}", flush=True)
        return

    try:
        if _get_rootfs() is None:
            # Docker mode: kill leftover containers
            result = subprocess.run(
                ["docker", "ps", "-aq", "--filter", f"name=clawgym-rl-{rollout_id}-"],
                capture_output=True, text=True, timeout=10,
            )
            container_ids = result.stdout.strip().split()
            if container_ids:
                subprocess.run(
                    ["docker", "rm", "-f"] + container_ids,
                    capture_output=True, text=True, timeout=30,
                )
                print(f"[ClawGym-RL] cleaned up {len(container_ids)} stale containers from rollout {rollout_id}", flush=True)
        else:
            # Chroot mode: kill leftover gateway processes tied to THIS rollout's
            # temp dir pattern. unshare --user + PID-1-in-ns means SIGTERM is
            # ignored; we go straight to SIGKILL.
            subprocess.run(
                ["pkill", "-9", "-f", f"clawgym-rl-{rollout_id}-"],
                capture_output=True, text=True, timeout=10,
            )

        # Remove temp dirs
        import glob
        tmp_dirs = glob.glob(f"/tmp/clawgym-rl-{rollout_id}-*")
        for d in tmp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        if tmp_dirs:
            print(f"[ClawGym-RL] cleaned up {len(tmp_dirs)} temp dirs from rollout {rollout_id}", flush=True)

        if _get_rootfs() is None:
            # Prune stopped containers to reclaim Docker disk/overlay resources
            subprocess.run(
                ["docker", "container", "prune", "-f"],
                capture_output=True, text=True, timeout=30,
            )

        # Gateway census: if alive (non-zombie) gateway procs exceed expected,
        # force-kill strays. This catches gateways whose parent died without
        # reaping them and which are still holding ports / hung on a request.
        try:
            ps_out = subprocess.run(
                ["ps", "-eo", "pid,stat,comm"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            alive_gateways = 0
            for line in ps_out.splitlines():
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                _, stat, comm = parts
                if "openclaw-gatewa" in comm and "Z" not in stat:
                    alive_gateways += 1
            # Expected upper bound: 2x concurrent workers (eval + train overlap)
            eval_workers = int(os.environ.get("OPENCLAW_EVAL_NUM_WORKERS", "16"))
            train_workers = int(os.environ.get("OPENCLAW_TRAIN_NUM_WORKERS", "32"))
            expected_max = (eval_workers + train_workers) * 2
            if alive_gateways > expected_max:
                subprocess.run(
                    ["pkill", "-9", "-f", "openclaw-gatewa"],
                    capture_output=True, timeout=10,
                )
                print(
                    f"[ClawGym-RL] gateway census: {alive_gateways} alive "
                    f"(expected <= {expected_max}), force-killed all",
                    flush=True,
                )
            elif alive_gateways > 0:
                print(
                    f"[ClawGym-RL] gateway census: {alive_gateways} alive (ok, <= {expected_max})",
                    flush=True,
                )
        except Exception as exc:
            print(f"[ClawGym-RL] gateway census warning: {exc}", flush=True)
    except Exception as exc:
        print(f"[ClawGym-RL] cleanup warning: {exc}", flush=True)


def generate_rollout_clawgym_rl(args, rollout_id, data_buffer, evaluation=False):
    if not evaluation and os.environ.get("OPENCLAW_SERPER_FAIL_CLOSED") == "1":
        max_attempts = int(os.environ.get("OPENCLAW_SERPER_FAIL_CLOSED_ATTEMPTS", "3"))
        for attempt in range(max_attempts):
            _TRAIN_ATTEMPTS[rollout_id] = attempt
            result = _generate_rollout_clawgym_rl_once(
                args, rollout_id, data_buffer, False
            )
            try:
                _assert_no_serper_credential_failures(result.samples)
            except RuntimeError as exc:
                audit_dir = (
                    Path(os.environ["OPENCLAW_SPAN_TRACE_ROOT"]).parent
                    / "serper_fail_closed"
                )
                audit_dir.mkdir(parents=True, exist_ok=True)
                (audit_dir / f"rollout_{rollout_id}_attempt_{attempt}.json").write_text(
                    json.dumps(
                        {
                            "rollout_id": rollout_id,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                if attempt + 1 >= max_attempts:
                    raise
                print(
                    f"[ClawGym-RL] {exc}; retrying SAME step, no optimizer update",
                    flush=True,
                )
                time.sleep(5)
                continue
            return result
    if evaluation or os.environ.get('OPENCLAW_ARTIFACT_GRPO_STRICT')!='1':
        return _generate_rollout_clawgym_rl_once(args,rollout_id,data_buffer,evaluation)
    from artifact_grpo_bridge import prepare,IncompleteArtifactBatch
    attempt=0
    # Preserve groups that already passed artifact verification across retries.
    # A retry should replace only missing/damaged groups, never duplicate a
    # sample or mix samples between groups.
    cached_groups={}
    while True:
        _TRAIN_ATTEMPTS[rollout_id]=attempt
        result=_generate_rollout_clawgym_rl_once(args,rollout_id,data_buffer,False)
        groups=list(result.samples)
        if cached_groups:
            groups=[cached_groups.get(getattr(group[0], "group_index", i), group)
                    for i, group in enumerate(groups)]
            result.samples=groups
        flat=[s for group in groups for s in group]
        try:
            # Validate native evidence and all 64 A scores before releasing
            # this rollout to the learner or advancing the task cursor.
            prepare(flat)
        except IncompleteArtifactBatch as exc:
            worker=get_global_worker(args,data_buffer)
            sessions=[_train_session_id(rollout_id,i) for i in range(int(args.rollout_batch_size)*int(args.n_samples_per_prompt))]
            worker.server._finalized_sessions.update(sessions)
            out=Path(os.environ['OPENCLAW_SPAN_TRACE_ROOT']).parent/'rollout_retries'
            out.mkdir(exist_ok=True)
            (out/f'rollout_{rollout_id}_attempt_{attempt}.json').write_text(json.dumps(dict(rollout_id=rollout_id,attempt=attempt,reason=str(exc),sessions=sessions),ensure_ascii=False,indent=2))
            # The bridge writes diagnostics before raising. Retain only groups
            # with all eight records and no infrastructure/scoring errors.
            audit_path=Path(os.environ['OPENCLAW_SPAN_TRACE_ROOT']).parent/'artifact_advantage_audit'/'INCOMPLETE_BATCH.json'
            try:
                audit=json.loads(audit_path.read_text())
                by_group={}
                for rec in audit.get('records',[]):
                    gid=rec.get('group_id')
                    if gid is not None:
                        by_group.setdefault(gid,[]).append(rec)
                expected=int(getattr(args, 'n_samples_per_prompt', 8))
                for i, group in enumerate(groups):
                    gid=getattr(group[0], 'group_index', i) if group else i
                    rs=by_group.get(gid, [])
                    if (len(group)==expected and len(rs)==expected and
                        all(not (r.get('infrastructure_error') or r.get('scoring_error')) for r in rs)):
                        cached_groups[gid]=group
                if cached_groups:
                    print(f'[ArtifactGRPO] retaining reusable groups={sorted(cached_groups)}; next attempt fills only missing groups', flush=True)
            except Exception as cache_exc:
                print(f'[ArtifactGRPO] group-cache warning: {cache_exc}', flush=True)
            print(f'[ArtifactGRPO] incomplete rollout={rollout_id} attempt={attempt}: {exc}; retrying SAME step, no update',flush=True)
            attempt+=1
            time.sleep(5)
            continue
        return result


def _assert_no_serper_credential_failures(groups):
    """Prevent credential-failed web trajectories from reaching the learner."""
    patterns = (
        re.compile(r"\ASerper request failed:\s*HTTP\s+(?:401|403)\b", re.I),
        re.compile(r"\ASerper credentials are not configured\b", re.I),
        re.compile(r"\ASerper request failed:\s*invalid endpoint URL\b", re.I),
    )
    sessions = {}
    for group in groups:
        for sample in group:
            session_id = (sample.metadata or {}).get("session_id")
            if session_id:
                sessions[session_id] = (
                    Path(os.environ["OPENCLAW_SPAN_TRACE_ROOT"])
                    / "sessions"
                    / hashlib.sha256(session_id.encode()).hexdigest()[:24]
                    / "trajectory.json"
                )
    failures = []
    for session_id, trajectory_path in sessions.items():
        try:
            trajectory = json.loads(trajectory_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Serper fail-closed could not audit session {session_id}: "
                f"{type(exc).__name__}"
            ) from exc
        observations = (
            observation
            for step in trajectory.get("react_steps", [])
            for observation in step.get("observations", [])
            if isinstance(observation, str)
        )
        if any(
            pattern.search(observation.lstrip())
            for observation in observations
            for pattern in patterns
        ):
            failures.append(session_id)
    if failures:
        raise RuntimeError(
            "Serper credential failure in "
            f"{len(failures)} trajectory(s): {','.join(sorted(failures))}"
        )


def _generate_rollout_clawgym_rl_once(args, rollout_id, data_buffer, evaluation=False):
    worker = get_global_worker(args, data_buffer)

    if evaluation:
        worker.resume_submission()
        try:
            result = _run_openclaw_eval(args, rollout_id)
        finally:
            worker.pause_submission()
        _cleanup_rollout_resources(rollout_id)
        if result is not None:
            return RolloutFnEvalOutput(data=result, metrics=_build_eval_metrics(result))
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    # Reset per-step train metrics
    with _TRAIN_METRICS_LOCK:
        _TRAIN_METRICS.clear()

    worker.server.reset_eval_scores()
    worker.resume_submission()
    worker.server._step_counts.clear()
    worker.server._pending_step_data.clear()
    stale_items = worker.get_completed_groups()
    if stale_items:
        print(f"[ClawGym-RL] purged {len(stale_items)} stale queue items", flush=True)

    # Run all train tasks synchronously — blocks until every container finishes
    _run_openclaw_train_tasks(args, rollout_id)

    # All tasks done, pause submission and collect all samples from queue
    worker.pause_submission()

    # Collect all completed samples from the queue
    n_groups = int(args.rollout_batch_size)
    n_per_group = int(args.n_samples_per_prompt)
    groups: dict[int, list[Sample]] = {}
    for group_id, samples in worker.get_completed_groups():
        if group_id not in groups:
            groups[group_id] = []
        for sample in samples:
            if sample.status != Sample.Status.ABORTED:
                groups[group_id].append(sample)

    completed_samples: list[list[Sample]] = []
    for group_id in sorted(groups):
        group_samples = groups[group_id]
        if len(group_samples) >= 4:
            completed_samples.append(group_samples)
        else:
            print(
                f"[ClawGym-RL] dropping group {group_id}: only {len(group_samples)} sample(s), need >=4 for GRPO",
                flush=True,
            )

    if completed_samples:
        group_sizes = [len(g) for g in completed_samples]
        print(
            f"[ClawGym-RL] collected {len(completed_samples)} groups, "
            f"group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
            f"total={sum(group_sizes)} samples",
            flush=True,
        )

    if completed_samples and (
        getattr(args, "opsd_enable", False)
        or getattr(args, "sdar_enable", False)
    ):
        opsd_started = time.monotonic()
        _score_opsd_negative_trajectories(args, completed_samples)
        with _TRAIN_METRICS_LOCK:
            _TRAIN_METRICS["opsd_teacher_seconds"] = (
                time.monotonic() - opsd_started
            )

    # Cleanup stale Docker containers and temp dirs from this rollout step
    _cleanup_rollout_resources(rollout_id)

    eval_scores = worker.server.drain_eval_scores()

    # ── Build metrics dict for wandb ─────────────────────────────────────
    metrics = _build_train_metrics(args, completed_samples, eval_scores)

    return RolloutFnTrainOutput(samples=completed_samples, metrics=metrics or None)


def _build_train_metrics(args, completed_samples, eval_scores=None):
    """Build openclaw/ wandb metrics from _TRAIN_METRICS and completed samples."""
    n_target = int(args.rollout_batch_size) * int(args.n_samples_per_prompt)
    with _TRAIN_METRICS_LOCK:
        m = dict(_TRAIN_METRICS)
        _TRAIN_METRICS.clear()
    rewards = m.get("rewards", [])
    n_submitted = m.get("submitted", 0)
    n_skipped = m.get("skipped", 0)
    steps_total = m.get("steps_total", 0)
    n_groups_used = len(completed_samples)

    metrics = {}
    if eval_scores:
        metrics["rollout/prm_eval_score"] = sum(eval_scores) / len(eval_scores)

    # Completion rates
    metrics["openclaw/submitted_trajectories"] = n_submitted
    metrics["openclaw/skipped_trajectories"] = n_skipped
    metrics["openclaw/skipped_recovered"] = m.get("skipped_recovered", 0)
    metrics["openclaw/skipped_lost"] = m.get("skipped_lost", 0)
    metrics["openclaw/timed_out_trajectories"] = m.get("dispatched", n_target) - n_submitted - n_skipped
    metrics["openclaw/gateway_prewarm_target"] = m.get("gateway_prewarm_target", 0)
    metrics["openclaw/gateway_prewarm_ready"] = m.get("gateway_prewarm_ready", 0)
    metrics["openclaw/gateway_prewarm_release_seconds"] = m.get(
        "gateway_prewarm_release_seconds", 0.0
    )
    metrics["openclaw/gateway_prewarm_aborted"] = m.get("gateway_prewarm_aborted", 0)
    metrics["openclaw/startup_attempts"] = m.get("startup_attempts", 0)
    metrics["openclaw/startup_successful_attempts"] = m.get(
        "startup_successful_attempts", 0
    )
    metrics["openclaw/startup_failed_attempts"] = m.get(
        "startup_failed_attempts", 0
    )
    metrics["openclaw/startup_retries"] = m.get("startup_retries", 0)
    archive_bytes = m.get("startup_archive_bytes", 0)
    archive_count = m.get("startup_archive_count", 0)
    metrics["openclaw/startup_archive_bytes_total"] = archive_bytes
    metrics["openclaw/startup_archive_bytes_avg"] = (
        archive_bytes / archive_count if archive_count else 0.0
    )
    for phase in (
        "queue_wait",
        "port_allocate",
        "workspace_prepare",
        "archive_build",
        "upload_extract",
        "docker_run",
        "health_ready",
        "slot_held",
        "attempt",
    ):
        observations = sorted(m.get(f"startup_{phase}_seconds", []))
        if not observations:
            for statistic in ("avg", "p50", "p90", "max"):
                metrics[f"openclaw/startup_{phase}_sec_{statistic}"] = 0.0
            continue
        count = len(observations)
        metrics[f"openclaw/startup_{phase}_sec_avg"] = sum(observations) / count
        metrics[f"openclaw/startup_{phase}_sec_p50"] = observations[
            min(count - 1, int(0.50 * count))
        ]
        metrics[f"openclaw/startup_{phase}_sec_p90"] = observations[
            min(count - 1, int(0.90 * count))
        ]
        metrics[f"openclaw/startup_{phase}_sec_max"] = observations[-1]
    metrics["openclaw/groups_used"] = n_groups_used

    # Sample counts (after multi-step splitting, total_samples >> submitted_trajectories)
    total_samples = sum(len(g) for g in completed_samples) if completed_samples else 0
    metrics["openclaw/total_samples"] = total_samples
    metrics["openclaw/avg_samples_per_trajectory"] = total_samples / n_submitted if n_submitted > 0 else 0
    flat_samples = [sample for group in completed_samples for sample in group]
    if (
        getattr(args, "opsd_enable", False)
        or getattr(args, "sdar_enable", False)
    ):
        opsd_eligible = [
            sample
            for sample in flat_samples
            if bool(getattr(sample, "opsd_eligible", False))
        ]
        opsd_applied = [
            sample
            for sample in flat_samples
            if bool(getattr(sample, "opsd_apply", False))
        ]
        opsd_fallback = [
            sample
            for sample in flat_samples
            if bool(getattr(sample, "opsd_fallback_reason", ""))
        ]
        metrics["openclaw/opsd_teacher_seconds"] = m.get(
            "opsd_teacher_seconds", 0.0
        )
        metrics["openclaw/opsd_eligible_segments"] = len(opsd_eligible)
        metrics["openclaw/opsd_applied_segments"] = len(opsd_applied)
        metrics["openclaw/opsd_applied_trajectories"] = len(
            {int(sample.index) for sample in opsd_applied}
        )
        metrics["openclaw/opsd_fallback_segments"] = len(opsd_fallback)
        metrics["openclaw/opsd_hint_overflow_segments"] = sum(
            getattr(sample, "opsd_fallback_reason", "") == "hint_overflow"
            for sample in flat_samples
        )
        for repeat_metric in (
            "guidance_repeat_score_failures",
            "guidance_repeat_scored_tokens",
            "guidance_repeat_token_logprob_mae",
            "guidance_repeat_step_support_pearson",
            "guidance_repeat_step_rank_spearman",
        ):
            metrics[f"openclaw/{repeat_metric}"] = m.get(repeat_metric, 0.0)

        if getattr(args, "sdar_enable", False):
            gaps = []
            gates = []
            gate_beta = float(getattr(args, "sdar_gate_beta", 5.0))
            for sample in opsd_applied:
                teacher = getattr(sample, "opsd_teacher_log_probs", None)
                rollout_old = getattr(sample, "rollout_log_probs", None)
                loss_mask = getattr(sample, "loss_mask", None)
                if not (teacher and rollout_old and loss_mask):
                    continue
                for teacher_lp, old_lp, active in zip(
                    teacher, rollout_old, loss_mask, strict=True
                ):
                    if not active:
                        continue
                    gap = float(teacher_lp) - float(old_lp)
                    scaled = gate_beta * gap
                    if scaled >= 0:
                        gate = 1.0 / (1.0 + math.exp(-scaled))
                    else:
                        exp_scaled = math.exp(scaled)
                        gate = exp_scaled / (1.0 + exp_scaled)
                    gaps.append(gap)
                    gates.append(gate)

            def percentile(values, fraction):
                if not values:
                    return 0.0
                ordered = sorted(values)
                return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

            metrics["openclaw/sdar_teacher_seconds"] = m.get(
                "opsd_teacher_seconds", 0.0
            )
            metrics["openclaw/sdar_applied_segments"] = len(opsd_applied)
            metrics["openclaw/sdar_fallback_segments"] = len(opsd_fallback)
            metrics["openclaw/sdar_gate_mean_rollout_old"] = (
                sum(gates) / len(gates) if gates else 0.0
            )
            metrics["openclaw/sdar_gate_p10_rollout_old"] = percentile(gates, 0.10)
            metrics["openclaw/sdar_gate_p50_rollout_old"] = percentile(gates, 0.50)
            metrics["openclaw/sdar_gate_p90_rollout_old"] = percentile(gates, 0.90)
            metrics["openclaw/sdar_gate_active_fraction_rollout_old"] = (
                sum(gate > 0.5 for gate in gates) / len(gates) if gates else 0.0
            )
            metrics["openclaw/sdar_teacher_gap_mean_rollout_old"] = (
                sum(gaps) / len(gaps) if gaps else 0.0
            )
            considered_segments = {id(sample) for sample in opsd_eligible}
            considered_segments.update(id(sample) for sample in opsd_fallback)
            metrics["openclaw/sdar_anchor_application_rate"] = (
                len(opsd_applied) / max(1, len(considered_segments))
            )
            metrics["openclaw/sdar_fallback_rate"] = (
                len(opsd_fallback) / max(1, len(considered_segments))
            )

    routing_samples = [
        sample
        for sample in flat_samples
        if str(sample.metadata.get("routing_replay_scope", "")).endswith(
            "agent_requests"
        )
        or sample.metadata.get("routing_replay_scope")
        == "single_agent_request"
    ]
    routing_turns = sum(
        int(sample.metadata.get("routing_replay_turn_count", 0))
        for sample in routing_samples
    )
    metrics["openclaw/routing_replay_segments_total"] = len(routing_samples)
    metrics["openclaw/routing_replay_avg_turns_per_segment"] = (
        routing_turns / len(routing_samples) if routing_samples else 0
    )
    metrics["openclaw/routing_replay_stitched_turns_total"] = sum(
        int(sample.metadata.get("routing_replay_stitched_turns", 0))
        for sample in routing_samples
    )
    metrics["openclaw/routing_replay_route_conflicts_total"] = sum(
        sample.metadata.get("routing_replay_split_reason")
        == "route_prefix_conflict"
        for sample in routing_samples
    )
    metrics["openclaw/routing_replay_rewrite_splits_total"] = sum(
        sample.metadata.get("routing_replay_split_reason")
        not in {None, "initial", "route_prefix_conflict"}
        for sample in routing_samples
    )

    # Reward stats (only counting submitted)
    if rewards:
        avg_reward_submitted = sum(rewards) / len(rewards)
        # Treat missing as 0: total reward / target count
        avg_reward_with_missing = sum(rewards) / n_target if n_target > 0 else 0.0
        metrics["openclaw/avg_reward_submitted"] = avg_reward_submitted
        metrics["openclaw/avg_reward_with_missing_as_zero"] = avg_reward_with_missing

    # Average task completion time
    durations = m.get("durations", [])
    if durations:
        metrics["openclaw/avg_task_duration_sec"] = sum(durations) / len(durations)
        metrics["openclaw/max_task_duration_sec"] = max(durations)
        metrics["openclaw/p90_task_duration_sec"] = sorted(durations)[int(len(durations) * 0.9)]

    # Group-level reward distribution
    if completed_samples:
        n_all_zero = 0
        n_all_one = 0
        n_all_same_mid = 0
        n_mixed = 0
        group_variances = []
        for group in completed_samples:
            group_rewards = [float(s.reward.get("score", 0)) if isinstance(s.reward, dict) else 0.0 for s in group]
            mean = sum(group_rewards) / len(group_rewards) if group_rewards else 0.0
            var = sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards) if group_rewards else 0.0
            group_variances.append(var)
            if all(r == 0.0 for r in group_rewards):
                n_all_zero += 1
            elif all(r >= 1.0 for r in group_rewards):
                n_all_one += 1
            elif var == 0.0:
                n_all_same_mid += 1
            else:
                n_mixed += 1
        n_g = len(completed_samples)
        metrics["openclaw/group_all_zero_frac"] = n_all_zero / n_g
        metrics["openclaw/group_all_one_frac"] = n_all_one / n_g
        metrics["openclaw/group_all_same_mid_frac"] = n_all_same_mid / n_g
        metrics["openclaw/group_mixed_frac"] = n_mixed / n_g
        metrics["openclaw/group_reward_var_mean"] = sum(group_variances) / n_g
        # Fraction of groups with zero variance (no gradient signal)
        metrics["openclaw/group_zero_var_frac"] = sum(1 for v in group_variances if v == 0.0) / n_g

    # Average steps per trajectory
    if n_submitted > 0:
        metrics["openclaw/avg_steps_per_trajectory"] = steps_total / n_submitted

    sglang_requests = m.get("sglang_requests_total", 0)
    sglang_prompt_tokens = m.get("sglang_prompt_tokens_total", 0)
    metrics["openclaw/sglang_requests"] = sglang_requests
    metrics["openclaw/avg_sglang_request_e2e_sec"] = (
        m.get("sglang_request_e2e_seconds_total", 0.0) / sglang_requests
        if sglang_requests
        else 0.0
    )
    metrics["openclaw/cache_hit_tokens"] = m.get("sglang_cache_hit_tokens_total", 0)
    metrics["openclaw/prefill_compute_tokens"] = m.get(
        "sglang_prefill_compute_tokens_total", 0
    )
    metrics["openclaw/cache_hit_rate"] = (
        metrics["openclaw/cache_hit_tokens"] / sglang_prompt_tokens
        if sglang_prompt_tokens
        else 0.0
    )
    metrics["openclaw/session_engine_switches"] = m.get("sglang_engine_switches_total", 0)
    metrics["openclaw/affinity_used_rate"] = (
        m.get("sglang_affinity_used_total", 0) / sglang_requests if sglang_requests else 0.0
    )
    metrics["openclaw/affinity_fallback_rate"] = (
        m.get("sglang_affinity_fallback_total", 0) / sglang_requests
        if sglang_requests
        else 0.0
    )
    # Per-turn response-limit and repetition health. These are diagnostics only:
    # samples and rewards remain unchanged.
    hit_trajectories = m.get("trajectories_hit_response_limit", 0)
    metrics["rollout/turns_hit_response_limit"] = m.get(
        "turns_hit_response_limit", 0
    )
    metrics["rollout/trajectories_hit_response_limit"] = hit_trajectories
    metrics["rollout/incomplete_tool_calls_at_limit"] = m.get(
        "incomplete_tool_calls_at_limit", 0
    )
    metrics["rollout/response_limit_tokens"] = m.get(
        "response_limit_tokens",
        int(os.environ.get("OPENCLAW_MAX_RESPONSE_TOKENS", "8192")),
    )
    repeated_ngram_ratios = m.get("repeated_ngram_ratios", [])
    metrics["rollout/repeated_ngram_ratio"] = (
        sum(repeated_ngram_ratios) / len(repeated_ngram_ratios)
        if repeated_ngram_ratios
        else 0.0
    )
    metrics["rollout/max_ngram_repeat"] = m.get("max_ngram_repeat", 0)
    metrics["rollout/repetition_degenerated_trajectories"] = m.get(
        "repetition_degenerated_trajectories", 0
    )
    metrics["rollout/truncated"] = hit_trajectories
    for termination_reason in (
        "natural_completion",
        "length_limited_valid_response",
        "length_limited_incomplete_tool_call",
        "max_agent_steps",
        "max_trajectory_tokens",
        "session_compaction_limit",
        "outer_timeout",
    ):
        metrics[f"rollout/termination/{termination_reason}"] = m.get(
            f"termination_{termination_reason}", 0
        )

    # Format penalty count
    metrics["openclaw/format_penalty_count"] = m.get("format_penalty", 0)
    # Format penalty that actually killed a non-zero reward
    metrics["openclaw/format_penalty_nonzero_to_zero"] = m.get("format_penalty_nonzero_to_zero", 0)

    metrics["openclaw/budget_terminated_trajectories"] = m.get("budget_terminated", 0)
    metrics["openclaw/budget_terminated_max_agent_steps"] = m.get(
        "budget_terminated_max_agent_steps", 0
    )
    metrics["openclaw/budget_terminated_max_compaction_events"] = m.get(
        "budget_terminated_max_compaction_events", 0
    )
    metrics["openclaw/avg_compactions_per_trajectory"] = (
        m.get("compactions_total", 0) / n_submitted if n_submitted > 0 else 0
    )
    metrics["openclaw/session_compactions_total"] = m.get("compactions_total", 0)
    metrics["openclaw/compaction_model_calls_total"] = m.get(
        "compaction_model_calls_total", 0
    )
    metrics["openclaw/avg_session_compactions_per_trajectory"] = (
        m.get("compactions_total", 0) / n_submitted if n_submitted > 0 else 0
    )
    metrics["openclaw/tool_result_pruning_events_total"] = m.get(
        "tool_result_pruning_events_total", 0
    )
    metrics["openclaw/avg_tool_result_pruning_events_per_trajectory"] = (
        m.get("tool_result_pruning_events_total", 0) / n_submitted
        if n_submitted > 0 else 0
    )
    metrics["openclaw/tool_results_pruned_total"] = m.get(
        "tool_results_pruned_total", 0
    )
    metrics["openclaw/avg_tool_results_pruned_per_trajectory"] = (
        m.get("tool_results_pruned_total", 0) / n_submitted
        if n_submitted > 0 else 0
    )
    metrics["openclaw/tool_result_truncation_events_total"] = m.get(
        "tool_result_truncation_events_total", 0
    )
    metrics["openclaw/tool_results_truncated_total"] = m.get(
        "tool_results_truncated_total", 0
    )
    metrics["openclaw/unclassified_history_rewrites_total"] = m.get(
        "unclassified_history_rewrites_total", 0
    )
    metrics["openclaw/avg_unclassified_history_rewrites_per_trajectory"] = (
        m.get("unclassified_history_rewrites_total", 0) / n_submitted
        if n_submitted > 0 else 0
    )
    metrics["openclaw/context_overflows_total"] = m.get("context_overflows_total", 0)
    metrics["openclaw/avg_context_overflows_per_trajectory"] = (
        m.get("context_overflows_total", 0) / n_submitted if n_submitted > 0 else 0
    )
    assistant_turns = m.get("assistant_turns_total", 0)
    metrics["openclaw/multi_tool_turn_frac"] = (
        m.get("multi_tool_turns_total", 0) / assistant_turns
        if assistant_turns > 0
        else 0
    )
    metrics["openclaw/raw_tool_calls_total"] = m.get("raw_tool_calls_total", 0)
    metrics["openclaw/parsed_tool_calls_total"] = m.get("parsed_tool_calls_total", 0)
    metrics["openclaw/returned_tool_calls_total"] = m.get("returned_tool_calls_total", 0)
    metrics["openclaw/tool_call_parse_retention"] = (
        m.get("parsed_tool_calls_total", 0) / m.get("raw_tool_calls_total", 0)
        if m.get("raw_tool_calls_total", 0) > 0
        else 1.0
    )
    metrics["openclaw/tool_call_return_retention"] = (
        m.get("returned_tool_calls_total", 0) / m.get("parsed_tool_calls_total", 0)
        if m.get("parsed_tool_calls_total", 0) > 0
        else 1.0
    )
    metrics["openclaw/max_identical_tool_call_turns"] = m.get(
        "max_identical_tool_call_turns", 0
    )

    return metrics


atexit.register(stop_global_worker)
