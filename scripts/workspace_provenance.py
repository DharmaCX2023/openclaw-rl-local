"""Audit-only workspace mutation provenance for OpenClaw rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import stat
import threading
import time
from typing import Any, Callable


_BACKGROUND_MARKERS = (
    "command still running (session ",
    "process still running.",
)
_PROTECTED_TOP_LEVEL = {"input", "reward", ".git", ".openclaw", ".venv"}
_RUNTIME_ROOT_FILES = {
    "AGENTS.md",
    "BOOTSTRAP.md",
    "HEARTBEAT.md",
    "IDENTITY.md",
    "MEMORY.md",
    "SOUL.md",
    "TOOLS.md",
    "USER.md",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_relative_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    if not path.parts or ".." in path.parts:
        return False
    if path.parts[0] in _PROTECTED_TOP_LEVEL:
        return False
    if len(path.parts) == 1 and path.name in _RUNTIME_ROOT_FILES:
        return False
    if "__pycache__" in path.parts or path.name.endswith((".pyc", ".pyo")):
        return False
    return True


def _local_manifest(root: Path, max_file_bytes: int) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if not _allowed_relative_path(relative):
            continue
        info = path.stat()
        too_large = info.st_size > max_file_bytes
        manifest[relative] = {
            "mode": stat.S_IMODE(info.st_mode),
            "size": info.st_size,
            "sha256": None if too_large else _sha256_file(path),
            "hash_omitted": too_large,
        }
    return manifest


_REMOTE_MANIFEST_PROGRAM = r'''import hashlib,json,os,stat,sys
root=sys.argv[1]
limit=int(sys.argv[2])
protected={"input","reward",".git",".openclaw",".venv"}
runtime={"AGENTS.md","BOOTSTRAP.md","HEARTBEAT.md","IDENTITY.md","MEMORY.md","SOUL.md","TOOLS.md","USER.md"}
out={}
for base,dirs,files in os.walk(root):
    relbase=os.path.relpath(base,root)
    dirs[:]=sorted(d for d in dirs if (d not in protected if relbase=="." else True) and d!="__pycache__")
    for name in sorted(files):
        path=os.path.join(base,name)
        rel=os.path.relpath(path,root).replace(os.sep,"/")
        parts=rel.split("/")
        if parts[0] in protected or (len(parts)==1 and name in runtime) or "__pycache__" in parts or name.endswith((".pyc",".pyo")) or os.path.islink(path):
            continue
        info=os.stat(path,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            continue
        too_large=info.st_size>limit
        digest=None
        if not too_large:
            h=hashlib.sha256()
            with open(path,"rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):
                    h.update(chunk)
            digest=h.hexdigest()
        out[rel]={"mode":stat.S_IMODE(info.st_mode),"size":info.st_size,"sha256":digest,"hash_omitted":too_large}
print(json.dumps(out,sort_keys=True,separators=(",",":")))'''


def _manifest_delta(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(set(before) | set(after)):
        left = before.get(relative)
        right = after.get(relative)
        if left == right:
            continue
        change = "added" if left is None else "deleted" if right is None else "modified"
        rows.append({"path": relative, "change": change, "before": left, "after": right})
    return rows


def _starts_background(tool_results: list[dict[str, Any]]) -> bool:
    for result in tool_results:
        content = str(result.get("content") or "").lower()
        if any(marker in content for marker in _BACKGROUND_MARKERS):
            return True
    return False


@dataclass
class _SessionState:
    manifest: dict[str, dict[str, Any]]
    runtime: Any
    background_tainted: bool = False
    snapshot_gap: bool = False
    event_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class WorkspaceProvenanceObserver:
    """Snapshot workspace state at the tool-result/request boundary."""

    def __init__(
        self,
        record_file: Path,
        remote_run: Callable[..., Any],
        max_file_bytes: int = 64 * 1024**2,
    ) -> None:
        self.record_file = record_file
        self.remote_run = remote_run
        self.max_file_bytes = max_file_bytes
        self._sessions: dict[str, _SessionState] = {}
        self._write_lock = threading.Lock()
        record_file.parent.mkdir(parents=True, exist_ok=True)
        record_file.write_text("", encoding="utf-8")

    def _snapshot(self, runtime: Any) -> dict[str, dict[str, Any]]:
        if runtime.remote_root:
            root = f"{runtime.remote_root}/workspace"
            command = " ".join(
                [
                    "python3",
                    "-c",
                    shlex.quote(_REMOTE_MANIFEST_PROGRAM),
                    shlex.quote(root),
                    str(self.max_file_bytes),
                ]
            )
            result = self.remote_run(command, timeout=120)
            value = json.loads(result.stdout.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("remote manifest is not an object")
            return value
        return _local_manifest(Path(runtime.workspace_dir), self.max_file_bytes)

    def _append(self, row: dict[str, Any]) -> None:
        with self._write_lock, self.record_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def register(self, runtime: Any) -> None:
        started = time.perf_counter()
        manifest = self._snapshot(runtime)
        state = _SessionState(manifest=manifest, runtime=runtime)
        self._sessions[runtime.session_id] = state
        self._append(
            {
                "record_type": "workspace_provenance_baseline",
                "session_id": runtime.session_id,
                "file_count": len(manifest),
                "snapshot_seconds": time.perf_counter() - started,
            }
        )

    def observe(
        self,
        server: Any,
        session_id: str,
        tool_results: list[dict[str, Any]],
    ) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            self._append(
                {
                    "record_type": "workspace_provenance_error",
                    "session_id": session_id,
                    "error": "runtime_not_registered",
                }
            )
            return
        with state.lock:
            started = time.perf_counter()
            try:
                current = self._snapshot(state.runtime)
            except Exception as exc:
                state.snapshot_gap = True
                self._append(
                    {
                        "record_type": "workspace_provenance_error",
                        "session_id": session_id,
                        "event_index": state.event_index,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
                return

            conversation = server._conversation_logs.get(session_id, [])
            previous_step = conversation[-1] if conversation else {}
            tool_calls = list(previous_step.get("tool_calls") or [])
            expected_ids = [str(value.get("id") or "") for value in tool_calls]
            observed_ids = [str(value.get("tool_call_id") or "") for value in tool_results]
            mutations = _manifest_delta(state.manifest, current)
            begins_background = _starts_background(tool_results)
            inherited_taint = state.background_tainted
            exact_result_link = bool(expected_ids) and observed_ids == expected_ids
            hash_complete = all(
                not (value.get("before") or {}).get("hash_omitted", False)
                and not (value.get("after") or {}).get("hash_omitted", False)
                for value in mutations
            )
            eligibility_reasons = []
            if not exact_result_link:
                eligibility_reasons.append("tool_result_id_mismatch")
            if not mutations:
                eligibility_reasons.append("no_workspace_mutation")
            if inherited_taint:
                eligibility_reasons.append("inherited_background_taint")
            if begins_background:
                eligibility_reasons.append("background_process_started")
            if state.snapshot_gap:
                eligibility_reasons.append("prior_snapshot_gap")
            if not hash_complete:
                eligibility_reasons.append("large_file_hash_omitted")

            self._append(
                {
                    "record_type": "workspace_provenance_event",
                    "session_id": session_id,
                    "event_index": state.event_index,
                    "source_step": len(conversation) if conversation else None,
                    "tool_call_ids": expected_ids,
                    "observed_tool_call_ids": observed_ids,
                    "tool_names": [str(value.get("name") or "") for value in tool_calls],
                    "attribution_unit": (
                        "single_action" if len(tool_calls) == 1 else "action_group"
                    ),
                    "mutations": mutations,
                    "mutation_count": len(mutations),
                    "background_tainted_before": inherited_taint,
                    "background_started": begins_background,
                    "provenance_eligible": not eligibility_reasons,
                    "rejection_reasons": eligibility_reasons,
                    "tool_result_summaries": [
                        {
                            "tool_call_id": str(value.get("tool_call_id") or ""),
                            "content_bytes": len(
                                str(value.get("content") or "").encode("utf-8")
                            ),
                            "content_sha256": hashlib.sha256(
                                str(value.get("content") or "").encode("utf-8")
                            ).hexdigest(),
                        }
                        for value in tool_results
                    ],
                    "snapshot_seconds": time.perf_counter() - started,
                }
            )
            state.manifest = current
            state.background_tainted = inherited_taint or begins_background
            state.snapshot_gap = False
            state.event_index += 1
