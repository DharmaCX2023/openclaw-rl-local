"""Opt-in experimental proxy: capture live state; transplant it into branch runs.

OpenClaw still parses/executes every tool and returns the real observations.
The branch adapter substitutes the saved pre-action history for the fresh
Gateway's bootstrap history. It never appends a synthetic 'continue' to h.
It retains exact raw token prefixes via the production token ledger.
"""
from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import time

from verified_repair_runtime import save_json, action_scope_rejection
import clawgym_rl_api_server as api


class RepairProxy(api.ClawGymRLAPIServer):
    def __init__(self, *args, remote_state, **kwargs):
        super().__init__(*args, **kwargs)
        self.remote_state = remote_state
        self.runtimes = {}
        self.candidates = {}
        self.capture_errors = {}
        self.branch_states = {}

    def register_runtime(self, runtime, *, branch=None):
        self.runtimes[runtime.session_id] = runtime
        if branch is not None:
            self.branch_states[runtime.session_id] = branch

    async def _handle_request(self, body, session_id, turn_type):
        if session_id in self.branch_states:
            return await self._branch_request(body, session_id, turn_type)
        result = await super()._handle_request(body, session_id, turn_type)
        if turn_type != 'main' or not isinstance(result, dict):
            return result
        step_num = self._step_counts.get(session_id, 0)
        step = self._pending_step_data.get(session_id, {}).get(step_num)
        if not step or session_id not in self.runtimes:
            return result
        calls = step['tool_calls']
        scope_error = action_scope_rejection(calls)
        if scope_error:
            self.capture_errors.setdefault(session_id, []).append({'turn': step_num, 'error': scope_error})
            return result
        writer = any(c['function']['name'] in {'write', 'edit', 'exec'} for c in calls)
        if not (writer or not calls):
            return result
        candidate_dir = self.remote_state.output / 'states' / session_id
        snapshot_path = candidate_dir / f'turn-{step_num:03d}.tar.gz'
        try:
            history_text = json.dumps(body['messages'], ensure_ascii=False).lower()
            if ('command still running (session' in history_text
                    or 'process still running.' in history_text):
                raise ValueError('unrestorable_background_history')
            actual_calls = [call for event in self._conversation_logs.get(session_id, [])
                            for call in event.get('tool_calls', [])]
            if any(any(path in str(call.get('arguments', ''))
                       for path in ('/tmp/openclaw', '/tmp/jiti')) for call in actual_calls):
                raise ValueError('action_depends_on_uncaptured_gateway_runtime_files')
            if any(c['function']['name'] in {'process', 'web_fetch'} for c in calls):
                raise ValueError('out_of_scope_tool')
            snapshot = await asyncio.to_thread(self.remote_state.snapshot,
                                              self.runtimes[session_id], snapshot_path)
            state = {'session_id': session_id, 'turn': step_num,
                     'snapshot': snapshot, 'body': copy.deepcopy(body),
                     'step': copy.deepcopy(step),
                     'ledger_after_action': asdict(self._session_token_ledgers[session_id]),
                     'source_observations': None,
                     'captured_at': time.time(), 'pre_action': True}
            path = candidate_dir / f'turn-{step_num:03d}.json'
            save_json(path, state)
            slots = self.candidates.setdefault(session_id, [])
            slots.append(str(path))
            # Keep only three RESTORABLE recent boundaries. Metadata and
            # failures remain as audit records; only our own ephemeral tar is
            # superseded. Selected snapshots are never deleted by this path.
            if len(slots) > 3:
                old_path = Path(slots.pop(0))
                old = json.loads(old_path.read_text())
                old_tar = Path(old['snapshot']['archive'])
                if old_tar.parent != candidate_dir:
                    raise ValueError('snapshot cleanup scope mismatch')
                old_tar.unlink()
                old['snapshot']['superseded'] = True
                save_json(old_path, old)
        except Exception as exc:
            self.capture_errors.setdefault(session_id, []).append({
                'turn': step_num, 'error': type(exc).__name__ + ':' + str(exc)[:800]})
        return result

    async def _branch_request(self, body, sid, turn_type):
        branch = self.branch_states[sid]
        if turn_type != 'main':
            branch['error'] = 'branch_requires_compaction'
            return self._branch_stop(body, sid)
        if branch.get('error'):
            return self._branch_stop(body, sid)
        if 'bootstrap_count' not in branch:
            source = branch['state']
            branch['bootstrap_count'] = len(body['messages'])
            branch['bootstrap_messages'] = copy.deepcopy(body['messages'])
            branch['started_monotonic'] = time.monotonic()
            if not api._semantic_tools_equal(body.get('tools'), source['body'].get('tools')):
                branch['error'] = 'restored_tool_schema_mismatch'
                return self._branch_stop(body, sid)
            forced = branch['forced']
            _, parser, _ = api._build_generate_sampling_params(
                source['body'], self.generation_defaults, self.tool_call_parser)
            content, calls = api._parse_generated_assistant(
                forced['text'], parser, f"{branch['pair_identity']}:forced")
            message = {'role': 'assistant', 'content': content,
                       'reasoning_content': '', 'tool_calls': calls}
            old = source['ledger_after_action']
            # Empty logprobs explicitly mark a forced action, NOT fake rollout
            # logprobs. Only the structural/token ledger is needed to continue.
            self._session_token_ledgers[sid] = api.SessionTokenLedger(
                messages=copy.deepcopy(source['body']['messages']),
                assistant_message=copy.deepcopy(message),
                prompt_ids=list(source['step']['prompt_ids']),
                response_ids=list(forced['ids']), response_logprobs=[],
                template_tools=old['template_tools'], template_kwargs=old['template_kwargs'],
                reasoning_effort=old['reasoning_effort'])
            self._conversation_logs[sid] = [{'step': 1, 'content': content,
                'tool_calls': [{'id': c['id'], 'name': c['function']['name'],
                                'arguments': c['function']['arguments']} for c in calls],
                'observations': [], 'forced_auxiliary_action': True,
                'response_tokens': len(forced['ids'])}]
            branch['generated_turns'] = 0
            branch['tokens'] = len(forced['ids'])
            branch['first_action_message'] = message
            return {'response': {'id': 'repair-forced-' + sid, 'object': 'chat.completion',
                'created': int(time.time()), 'model': body['model'], 'session_id': sid,
                'choices': [{'index': 0, 'message': message, 'logprobs': None,
                             'finish_reason': 'tool_calls' if calls else 'stop'}],
                'usage': {'prompt_tokens': len(source['step']['prompt_ids']),
                          'completion_tokens': len(forced['ids']),
                          'total_tokens': len(source['step']['prompt_ids']) + len(forced['ids'])}}}
        incoming = body['messages']
        suffix = incoming[branch['bootstrap_count']:]
        if not api._semantic_messages_equal(incoming[:branch['bootstrap_count']], branch['bootstrap_messages']):
            branch['error'] = 'branch_bootstrap_history_rewritten'
            return self._branch_stop(body, sid)
        merged = copy.deepcopy(body)
        merged['messages'] = copy.deepcopy(branch['state']['body']['messages']) + suffix
        merged['tools'] = copy.deepcopy(branch['state']['body'].get('tools'))
        new_results = api._collect_new_tool_results(merged['messages'])
        if any('command still running (session' in str(r['content']).lower()
               or 'process still running.' in str(r['content']).lower() for r in new_results):
            self._update_previous_step_observations(sid, merged['messages'], 'main')
            branch['error'] = 'uncaptured_background_execution_in_branch'
            return self._branch_stop(body, sid)
        if branch['generated_turns'] >= 3:
            branch['termination'] = 'continuation_turn_budget'
            self._update_previous_step_observations(sid, merged['messages'], 'main')
            return self._branch_stop(body, sid)
        if time.monotonic() - branch['started_monotonic'] >= 300:
            branch['error'] = 'branch_wallclock_timeout'
            return self._branch_stop(body, sid)
        remaining = 32768 - branch['tokens']
        if remaining <= 0:
            branch['termination'] = 'branch_token_budget'
            return self._branch_stop(body, sid)
        merged['max_tokens'] = min(8192, remaining)
        result = await super()._handle_request(merged, sid, 'main')
        if isinstance(result, dict):
            branch['generated_turns'] += 1
            branch['tokens'] += result['response']['usage']['completion_tokens']
            calls = result['response']['choices'][0]['message'].get('tool_calls', [])
            scope_error = action_scope_rejection(calls)
            if scope_error:
                branch['error'] = scope_error
                return self._branch_stop(body, sid)
        else:
            branch['error'] = 'branch_model_http_error'
        return result

    def _branch_stop(self, body, sid):
        # A harness termination control response: excluded from action data,
        # continuation budget and final-answer supervision, never a model token.
        return {'response': {'id': 'repair-budget-' + sid, 'object': 'chat.completion',
            'created': int(time.time()), 'model': body['model'], 'session_id': sid,
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '',
                         'reasoning_content': '', 'tool_calls': []},
                         'finish_reason': 'stop', 'logprobs': None}],
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}}
