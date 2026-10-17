"""Recorded-action replay through real OpenClaw tools, without prefix inference."""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
import time

from verified_repair_runtime import RemoteState, command, rollout, save_json
from verified_repair_trace import content_hash, text_content
import clawgym_rl_api_server as api


# Only hashes are copied. No per-step file bytes, process image or tar snapshot.
FINGERPRINT_SCRIPT = r"""
import hashlib,json,os,stat,time
from pathlib import Path
root=Path('/root/.openclaw/workspace')
started=time.monotonic(); files={}; total=0
for base,dirs,names in os.walk(root,followlinks=False):
    relative=Path(base).relative_to(root)
    if str(relative)=='.':
        dirs[:]=sorted(d for d in dirs if d not in {'.git','reward'})
    else: dirs.sort()
    for name in dirs:
        p=Path(base)/name; rel=p.relative_to(root).as_posix(); info=p.lstat()
        files[rel]={'mode':stat.S_IMODE(info.st_mode),'type':'directory'}
        if p.is_symlink():files[rel].update(type='symlink',target=os.readlink(p))
    for name in sorted(names):
        p=Path(base)/name; rel=p.relative_to(root).as_posix()
        if rel=='.openclaw_reward_payload.json':continue
        info=p.lstat()
        item={'mode':stat.S_IMODE(info.st_mode),'bytes':info.st_size}
        if p.is_symlink():item.update(type='symlink',target=os.readlink(p))
        elif p.is_file():
            total+=info.st_size
            if total>512*2**20 or len(files)>=20000:raise RuntimeError('fingerprint budget')
            h=hashlib.sha256()
            with p.open('rb') as f:
                for data in iter(lambda:f.read(2**20),b''):h.update(data)
            item.update(type='file',sha256=h.hexdigest())
        else:item.update(type='other')
        files[rel]=item
print(json.dumps({'files':files,'total_bytes_hashed':total,'seconds':time.monotonic()-started,
 'scope':'workspace files excluding .git, reward and injected reward payload; not a process snapshot'}))
"""


class ReplayRemoteState(RemoteState):
    def fingerprint(self, runtime):
        self.check_owned(runtime)
        started = time.monotonic()
        try:
            result = rollout._remote_run(
                command(["docker", "exec", runtime.container_name, "python3", "-c", FINGERPRINT_SCRIPT]),
                timeout=60,
            )
            value = json.loads(result.stdout)
            value.update(status="ok", transport_seconds=time.monotonic() - started)
            return value
        except Exception as exc:
            return {"status": "unavailable", "error_type": type(exc).__name__,
                    "transport_seconds": time.monotonic() - started}


def compare_file_state(original, current):
    if original.get("status") != "ok" or current.get("status") != "ok":
        return {"status": "unverified", "reason": "fingerprint_unavailable"}
    old, new = original["files"], current["files"]
    changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    return {"status": "match" if not changed else "material_file_divergence",
            "changed_paths": changed, "file_count": len(new),
            "process_and_external_state_proven": False}


def compare_observations(original, current):
    if original is None:
        return {"status": "unverified", "reason": "missing_source_observation"}
    if original == current:
        return {"status": "exact", "count": len(current)}
    # Never globally erase dates/numbers: they can be task-relevant data.
    # Keep executing minor/non-file differences, but don't call them exact.
    return {"status": "changed_requires_review", "source_sha256": content_hash(original),
            "replay_sha256": content_hash(current),
            "source_count": len(original), "replay_count": len(current)}


class ReplayProxy(api.ClawGymRLAPIServer):
    def __init__(self, *args, remote_state, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote_state = remote_state
        self.runtimes = {}
        self.events = {}
        self.replays = {}
        self.result_maps = {}
        self.request_locks = {}
        self.execution_started = {}
        self.deadlines = {}
        self.closed_sessions = set()

    def begin_execution(self, sid, seconds=1000):
        self.execution_started[sid] = {"wall": time.time(), "monotonic": time.monotonic()}
        if sid not in self.replays:
            self.deadlines[sid] = time.monotonic() + seconds

    def _budget_state(self, sid):
        return {name: copy.deepcopy(getattr(self, name).get(sid, 0)) for name in (
            "_session_compaction_counts", "_session_compaction_request_counts",
            "_session_pending_compaction_rewrites", "_session_tool_result_pruning_events",
            "_session_tool_results_pruned",
        )}

    def _wall_terminal(self, body, sid, kind):
        return self._budget_terminal_response(
            sid, kind, body["messages"], body["model"],
            {"reason": "original_task_wall_budget", "original_limit_seconds": 1000,
             "prefix_budget_charged": sid in self.replays},
        )

    def register_runtime(self, runtime, *, branch=None):
        self.runtimes[runtime.session_id] = runtime
        self.events[runtime.session_id] = []
        if branch is not None:
            self.replays[runtime.session_id] = branch
            branch.update(cursor=0, replacement_done=False, observation_differences=[])

    def _observe(self, sid, body, kind):
        if kind != "main":
            return
        values = self.result_maps.setdefault(sid, {})
        for item in api._collect_new_tool_results(body["messages"]):
            # Preserve the first real response before subsequent pruning rewrites.
            values.setdefault(item["tool_call_id"], copy.deepcopy(item["content"]))

    async def _handle_request(self, body, session_id, turn_type):
        lock = self.request_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            self._observe(session_id, body, turn_type)
            if (session_id in self.closed_sessions
                    or time.monotonic() >= self.deadlines.get(session_id, float("inf"))):
                return self._wall_terminal(body, session_id, turn_type)
            branch = self.replays.get(session_id)
            if branch and not branch["replacement_done"]:
                return await self._replay_request(body, session_id, turn_type, branch)
            before = self._step_counts.get(session_id, 0)
            event = {"kind": turn_type, "body": copy.deepcopy(body),
                     "started_at": time.time(), "forced": False}
            self.events.setdefault(session_id, []).append(event)
            started = time.monotonic()
            try:
                remaining = self.deadlines.get(session_id, float("inf")) - time.monotonic()
                result = await asyncio.wait_for(
                    super()._handle_request(body, session_id, turn_type),
                    timeout=max(.001, remaining) if remaining != float("inf") else None,
                )
            except asyncio.TimeoutError:
                event["wall_budget_exhausted_during_inference"] = True
                result = self._wall_terminal(body, session_id, turn_type)
            except BaseException as exc:
                event["request_error_type"] = type(exc).__name__
                self._save_event(session_id, event)
                raise
            event["model_request_seconds"] = time.monotonic() - started
            event["budget_state_after"] = self._budget_state(session_id)
            origin = self.execution_started.get(session_id)
            if origin:
                event["original_elapsed_before_request"] = max(0, event["started_at"] - origin["wall"])
            self._capture_result(event, result)
            after = self._step_counts.get(session_id, 0)
            if turn_type == "main" and after > before:
                step = self._pending_step_data[session_id][after]
                event["agent_step"] = after
                event["step_data"] = copy.deepcopy(step)
                if not branch:
                    event["file_state"] = await asyncio.to_thread(
                        self.remote_state.fingerprint, self.runtimes[session_id],
                    )
            self._save_event(session_id, event)
            return result

    @staticmethod
    def _capture_result(event, result):
        if isinstance(result, dict):
            event["wire_response"] = copy.deepcopy(result["response"])
        else:
            event["http_error"] = {
                "status": result.status_code,
                "body": json.loads(result.body),
            }

    def _save_event(self, sid, event):
        ordinal = next(i for i, existing in enumerate(self.events[sid]) if existing is event)
        event["event_index"] = ordinal
        save_json(self.remote_state.output / "request_events" / sid / f"{ordinal:03d}.json", event)

    def export_trace(self, sid):
        values = self.result_maps.get(sid, {})
        steps = []
        for event in self.events.get(sid, []):
            if "agent_step" not in event:
                continue
            if "step_data" in event:
                step = copy.deepcopy(event["step_data"])
            else:
                wire = event["actual_wire_response"]
                step = {
                    "prompt_ids": event["actual_prompt_ids"],
                    "response_ids": event["response_ids"],
                    "response_text": event["forced_text"],
                    "response_logprobs": None,
                    "tool_calls": wire["choices"][0]["message"].get("tool_calls", []),
                    "messages": event["body"]["messages"],
                    "forced_auxiliary_action": True, "is_policy_rollout": False,
                }
            ids = [call["id"] for call in step["tool_calls"]]
            observations = [values[i] for i in ids] if all(i in values for i in ids) else None
            step.update(agent_step=event["agent_step"], tools=event["body"].get("tools", []),
                        observations=observations, file_state=event.get("file_state"),
                        source_event_index=event["event_index"])
            steps.append(step)
        return steps

    def _prompt_state(self, sid, body, kind):
        chat, _, _ = api._build_generate_sampling_params(
            body, self.generation_defaults, self.tool_call_parser,
        )
        tools = [t.model_dump() for t in chat.tools] if chat.tools and chat.tool_choice != "none" else None
        kwargs = dict(chat.chat_template_kwargs or {})
        messages = body["messages"]
        ids = api._apply_chat_template_ids(
            self.tokenizer, messages, template_tools=tools, template_kwargs=kwargs,
            reasoning_effort=chat.reasoning_effort, add_generation_prompt=True,
        )
        previous = None if kind == "compact" else self._session_token_ledgers.get(sid)
        transition, detail = "initial", ""
        if previous is not None:
            transition, appended, detail = api._classify_structural_append(previous, messages, tools)
            if transition == "append_only":
                try:
                    ids, _, _ = api._build_incremental_prompt_ids(
                        self.tokenizer, previous, messages, appended,
                        template_tools=tools, template_kwargs=kwargs,
                        reasoning_effort=chat.reasoning_effort,
                    )
                except ValueError as exc:
                    transition, detail = "template_delta_invalid", str(exc)
            if self._session_pending_compaction_rewrites.pop(sid, 0) and transition == "history_rewritten":
                self._session_compaction_counts[sid] = self._session_compaction_counts.get(sid, 0) + 1
                transition = "session_compacted"
            elif transition == "history_rewritten":
                pruned = api._count_new_tool_result_markers(
                    previous.messages, messages, api._COMPACTED_TOOL_OUTPUT_MARKER,
                )
                if pruned:
                    transition = "tool_result_pruned"
                    self._session_tool_result_pruning_events[sid] = self._session_tool_result_pruning_events.get(sid, 0) + 1
                    self._session_tool_results_pruned[sid] = self._session_tool_results_pruned.get(sid, 0) + pruned
        return {"prompt_ids": ids, "template_tools": tools, "template_kwargs": kwargs,
                "reasoning_effort": chat.reasoning_effort, "transition": transition, "detail": detail}

    def _forced_response(self, sid, body, event, forced):
        kind = event["kind"]
        state = self._prompt_state(sid, body, kind)
        wire = copy.deepcopy(forced["wire_response"])
        wire.update(model=body["model"], session_id=sid, id="replay-" + sid)
        wire["choices"][0]["logprobs"] = None
        wire["metadata"] = {"forced_auxiliary_action": True, "is_policy_rollout": False}
        message = wire["choices"][0]["message"]
        response_ids = list(forced.get("ids", []))
        if kind == "main" and len(state["prompt_ids"]) + len(response_ids) >= 65536:
            raise ValueError("forced complete action exceeds actual live context")
        if kind == "main":
            wire["usage"] = {
                "prompt_tokens": len(state["prompt_ids"]), "completion_tokens": len(response_ids),
                "total_tokens": len(state["prompt_ids"]) + len(response_ids),
            }
        if kind == "compact":
            self._session_compaction_request_counts[sid] = self._session_compaction_request_counts.get(sid, 0) + 1
            self._session_pending_compaction_rewrites[sid] = self._session_pending_compaction_rewrites.get(sid, 0) + 1
        elif kind == "main":
            number = self._step_counts.get(sid, 0) + 1
            self._step_counts[sid] = number
            self._conversation_logs.setdefault(sid, []).append({
                "step": number, "content": message.get("content", ""),
                "tool_calls": [{"id": c["id"], **c["function"]} for c in message.get("tool_calls", [])],
                "observations": [], "forced_auxiliary_action": True,
                "prompt_tokens": len(state["prompt_ids"]), "response_tokens": len(response_ids),
            })
            self._session_token_ledgers[sid] = api.SessionTokenLedger(
                messages=copy.deepcopy(body["messages"]), assistant_message=copy.deepcopy(message),
                prompt_ids=state["prompt_ids"], response_ids=response_ids, response_logprobs=[],
                template_tools=state["template_tools"], template_kwargs=state["template_kwargs"],
                reasoning_effort=state["reasoning_effort"],
                transition=state["transition"], transition_detail=state["detail"],
            )
            event.update(agent_step=number, actual_prompt_ids=state["prompt_ids"],
                         ledger_transition=state["transition"])
        event.update(forced=True, is_policy_rollout=False, response_ids=response_ids,
                     actual_wire_response=wire, rollout_logprobs=None,
                     forced_text=forced.get("text", ""))
        # Forced actions never enter _pending_step_data / trainable rollout samples.
        return {"response": wire}

    def _stop_replay(self, body, sid, branch, reason):
        branch["error"] = reason
        return self._budget_terminal_response(
            sid, "main", body["messages"], body["model"],
            {"reason": "replay_state_divergence", "detail": reason, "infrastructure_or_replay_failure": True},
        )

    async def _replay_request(self, body, sid, kind, branch):
        if branch.get("error"):
            return self._stop_replay(body, sid, branch, branch["error"])
        source_events = branch["source_events"]
        cursor = branch["cursor"]
        if cursor >= len(source_events):
            return self._stop_replay(body, sid, branch, "source_event_exhausted_before_repair")
        expected = source_events[cursor]
        if expected["kind"] != kind:
            return self._stop_replay(body, sid, branch, "main_aux_request_order_diverged")
        if not api._semantic_tools_equal(expected["body"].get("tools"), body.get("tools")):
            return self._stop_replay(body, sid, branch, "native_tool_schema_changed")
        if "request_error_type" in expected:
            return self._stop_replay(body, sid, branch, "source_request_failure_not_replayable")
        event = {"kind": kind, "body": copy.deepcopy(body), "started_at": time.time(),
                 "source_event_index": cursor, "forced": True}
        self.events[sid].append(event)
        target = expected.get("agent_step") == branch["target_step"]
        if target:
            observed = await asyncio.to_thread(self.remote_state.fingerprint, self.runtimes[sid])
            comparison = compare_file_state(expected.get("file_state") or {}, observed)
            branch["pre_action_file_comparison"] = comparison
            branch["actual_pre_action_file_state"] = observed
            for step in branch["source_steps"]:
                if step["agent_step"] >= branch["target_step"]:
                    break
                ids = [c["id"] for c in step["tool_calls"]]
                values = self.result_maps.get(sid, {})
                actual = [values[i] for i in ids if i in values]
                diff = compare_observations(step.get("observations"), actual)
                if diff["status"] != "exact":
                    branch["observation_differences"].append({"agent_step": step["agent_step"], **diff})
            if comparison["status"] == "unverified":
                event["state_failure"] = comparison
                self._save_event(sid, event)
                return self._stop_replay(body, sid, branch, comparison["status"])
            # Replay file drift is evidence to audit, not a policy failure.
            # Execute the complete branch; never accept it as a reliable pair
            # while material differences remain unresolved.
            branch["state_comparability"] = comparison["status"]
            forced = branch["replacement"]
            charged = float(expected["original_elapsed_before_request"])
            remaining = max(0., float(branch["original_budget_seconds"]) - charged)
            branch.update(prefix_seconds_charged=charged, remaining_policy_seconds=remaining)
            if remaining <= 0:
                branch["error"] = "no_original_remaining_budget"
                return self._wall_terminal(body, sid, kind)
            self.deadlines[sid] = time.monotonic() + remaining
        else:
            forced = {"wire_response": expected.get("wire_response"),
                      "ids": expected.get("step_data", {}).get("response_ids", []),
                      "text": expected.get("step_data", {}).get("response_text", "")}
        if "http_error" in expected and not target:
            error = expected["http_error"]
            from fastapi.responses import JSONResponse
            result = JSONResponse(status_code=error["status"], content=error["body"])
            event["http_error"] = copy.deepcopy(error)
        else:
            result = self._forced_response(sid, body, event, forced)
        branch["cursor"] += 1
        if not target:
            for name, value in expected.get("budget_state_after", {}).items():
                if name not in self._budget_state(sid):
                    raise ValueError("unexpected replay budget state key")
                getattr(self, name)[sid] = copy.deepcopy(value)
        event["budget_state_after"] = self._budget_state(sid)
        self._save_event(sid, event)
        if target:
            branch["replacement_done"] = True
            branch["repair_started_at"] = time.time()
            branch["actual_pre_action"] = {
                "body": copy.deepcopy(body), "prompt_ids": event["actual_prompt_ids"],
                "source_agent_step": branch["target_step"],
            }
        return result
