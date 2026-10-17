"""One-shot Teacher proposals and paired, bounded real OpenClaw continuations."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import time

import requests
import jsonschema

from verified_repair_runtime import save_json, paired_accept, rollout, action_scope_rejection
import clawgym_rl_api_server as api


def fixed_seed(base, identity):
    return int.from_bytes(hashlib.sha256(f'{base}:{identity}'.encode()).digest()[:4], 'little') & 0x7fffffff


def teacher_proposal(state, anchors, proxy, g, destination):
    destination = Path(destination)
    if destination.exists():
        return json.loads(destination.read_text())
    source = state['step']
    text = (
        'Propose exactly ONE complete replacement assistant turn at the saved decision state. '
        'Your replacement will be executed from the state BEFORE the original action. '
        'The environment has been REWOUND to that pre-action snapshot. The quoted original '
        'action has NOT been executed in this restored branch. Its feedback below comes '
        'from a separate original run, NOT from the current restored state. In particular, '
        'a quoted Successfully wrote result does not mean that writing has already happened '
        'in this branch. Do not simply summarize the quoted action as completed. '
        'The original task was not fully satisfied. Use the task and observations to repair '
        'the action; do not discuss this teaching instruction. '
        'If your replacement needs to create or edit artifacts, issue the actual native '
        'tool calls that perform those effects. A completion summary cannot replace those '
        'effects. End without tools only when the saved pre-action history supports that '
        'no further environment action is needed. '
        'You may include brief necessary reasoning followed by native tool calls. '
        'Stop when new tool observations are needed. Do not invent tool responses, unseen '
        'file contents, credentials or facts learned only after the original action. '
        'If information is unavailable at that state, acquire it through a real tool call. '
        'Independent calls may share this turn; dependent calls must wait for real results. '
        'Do not wrap your response in a markdown demonstration.\n\n'
        'COUNTERFACTUAL RECORD: original action, not yet applied in the restored branch:\n' + source['response_text'] + '\n\n'
        'Actual feedback after that original action (diagnostic only, not available state):\n'
        + json.dumps(state.get('source_observations'), ensure_ascii=False) + '\n\n'
        'Task evidence anchors (requirements, not claims about current artifacts):\n'
        + json.dumps(anchors, ensure_ascii=False)
    )
    messages = copy.deepcopy(state['body']['messages']) + [{'role': 'user', 'content': text}]
    previous = state['ledger_after_action']
    ids = api._apply_chat_template_ids(proxy.tokenizer, messages,
        template_tools=previous['template_tools'], template_kwargs=previous['template_kwargs'],
        reasoning_effort=previous['reasoning_effort'], add_generation_prompt=True)
    record = {'task_id': state['task_id'], 'state_turn': state['turn'],
              'prompt_version': 'pre_action_rewind_v2',
              'prompt_messages': messages, 'prompt_ids': ids, 'one_proposal_only': True,
              'policy': str(g.model), 'status': 'pending', 'started_at': time.time()}
    if len(ids) + 8192 >= 65536:
        record.update(status='rejected', reason='teacher_context_budget')
        save_json(destination, record)
        return record
    _, parser, sampling = api._build_generate_sampling_params(
        state['body'], proxy.generation_defaults, proxy.tool_call_parser)
    sampling.update(temperature=.7, top_p=.8, top_k=20, max_new_tokens=8192,
                    sampling_seed=fixed_seed(161803, state['session_id']))
    payload = {'input_ids': ids, 'sampling_params': sampling, 'return_logprob': True}
    save_json(destination.with_suffix('.request.json'), record)
    start = time.monotonic()
    try:
        response = requests.post(f'http://127.0.0.1:{g.backend_port}/generate', json=payload, timeout=900)
        response.raise_for_status()
        output = response.json()
        raw, response_ids, lp = api._extract_aligned_generate_response(output)
        content, calls = api._parse_generated_assistant(raw, parser, state['session_id'] + ':teacher')
        record.update(text=raw, ids=response_ids, teacher_logprobs=lp,
                      tool_calls=calls, sampling_params=sampling,
                      metadata=output['meta_info'], status='proposed')
        finish = output['meta_info'].get('finish_reason', {})
        if api._has_incomplete_qwen_tool_call(raw) or api._count_raw_qwen_tool_calls(raw) != len(calls):
            raise ValueError('incomplete_or_unparsed_tool_call')
        if finish.get('type') == 'length' and calls:
            raise ValueError('length_limited_action_boundary')
        if not response_ids:
            raise ValueError('empty_teacher_action')
        if any(marker in raw for marker in ('<tool_response>', '<|im_start|>tool', '<|im_start|>user')):
            raise ValueError('fabricated_observation_or_role')
        schemas = {tool['function']['name']: tool['function']['parameters']
                   for tool in state['body'].get('tools', [])}
        scope_error = action_scope_rejection(calls)
        if scope_error:
            raise ValueError(scope_error)
        for call in calls:
            function = call['function']
            name = function['name']
            if name not in {'read', 'write', 'edit', 'exec'}:
                raise ValueError('out_of_scope_repair_tool:' + name)
            arguments = json.loads(function['arguments'])
            jsonschema.validate(arguments, schemas[name])
            if name == 'exec' and arguments.get('background') is True:
                raise ValueError('background_repair_out_of_scope')
        # These are proposals until BOTH paired verifier comparisons pass.
        record['accepted_preference'] = False
    except Exception as exc:
        record.update(status='rejected', reason=type(exc).__name__ + ':' + str(exc)[:800])
    record['seconds'] = time.monotonic() - start
    save_json(destination, record)
    return record


def run_branch(state, forced, pair_index, side, index, g, remote_state, proxy, start_runtime):
    identity = f"{state['session_id']}:paired-seed-{[314159, 271828][pair_index]}"
    sid = f"branch-{state['task_id']}-{state['session_id'].split('-')[-1]}-t{state['turn']}-p{pair_index}-{side}"
    destination = g.output_dir / 'branches' / (sid + '.json')
    if destination.exists():
        return json.loads(destination.read_text())
    task = rollout._load_task_spec(Path(state['task_entry']))
    branch = {'state': state, 'forced': forced, 'pair_identity': identity}
    record = {'session_id': sid, 'source_session': state['session_id'], 'task_id': task.task_id,
              'side': side, 'paired_repeat': pair_index, 'sampling_identity': identity,
              'forced_action_ids': forced['ids'], 'forced_action_text': forced['text'],
              'forced_action_is_auxiliary_not_rollout': True, 'status': 'started'}
    runtime = None
    started = time.monotonic()
    try:
        runtime, init = start_runtime(task, index, sid, g, remote_state, proxy,
            archive=state['snapshot']['archive'], branch=branch)
        record['initialize_restore_seconds'] = init
        restored = remote_state.snapshot(runtime, g.output_dir / 'restored' / (sid + '.tar.gz'))
        record['restore_snapshot'] = restored
        record['restore_manifest_equal'] = restored['manifest'] == state['snapshot']['manifest']
        if not record['restore_manifest_equal']:
            raise ValueError('restore_manifest_mismatch')
        # Gateway bootstrap request is replaced by saved h INSIDE the proxy.
        # No synthetic instruction enters the Student's generation context.
        execute_started = time.monotonic()
        final = rollout._run_turns(runtime, task, 1, 'main')
        record['execution_seconds'] = time.monotonic() - execute_started
        record['conversation'] = copy.deepcopy(proxy._conversation_logs.get(sid, []))
        record['budget'] = {k: branch.get(k) for k in ('generated_turns', 'tokens', 'termination', 'error')}
        record['final_message'] = final
        observations = record['conversation'][0].get('observations', []) if record['conversation'] else []
        record['first_action_observations'] = observations
        if side == 'original':
            record['source_first_observations_equal'] = observations == state.get('source_observations')
        if branch.get('error') or record['execution_seconds'] >= 300:
            raise ValueError(branch.get('error') or 'branch_wallclock_timeout')
        if branch.get('generated_turns', 0) > 3 or branch.get('tokens', 0) > 32768:
            raise ValueError('branch_budget_violation')
        terminal = remote_state.snapshot(runtime, g.output_dir / 'branch_terminal' / (sid + '.tar.gz'), audit_external=False)
        record['terminal_snapshot'] = terminal
        record['verifier'] = remote_state.verify(task, terminal['archive'], sid)
        record['status'] = 'completed'
    except Exception as exc:
        record.update(status='invalid_or_censored', error=type(exc).__name__ + ':' + str(exc)[:1000])
    finally:
        record['total_seconds'] = time.monotonic() - started
        save_json(destination, record)
        if runtime is not None:
            remote_state.stop(runtime)
    return record


def verify_candidate(state, teacher, index, g, remote_state, proxy, start_runtime):
    path = g.output_dir / 'pairs' / (state['session_id'] + '.json')
    original = {'text': state['step']['response_text'], 'ids': state['step']['response_ids']}
    result = {'task_id': state['task_id'], 'state_session': state['session_id'],
              'split': state['split'], 'accepted': False, 'comparisons': []}
    # Original replay is needed even when a Teacher proposal is invalid.
    for repeat in range(2):
        branches = {}
        order = ['original', 'repair'] if (index + repeat) % 2 == 0 else ['repair', 'original']
        for side in order:
            if side == 'repair' and teacher['status'] != 'proposed':
                continue
            branches[side] = run_branch(state, original if side == 'original' else teacher,
                repeat, side, 1000 + index * 4 + repeat * 2 + (side == 'repair'),
                g, remote_state, proxy, start_runtime)
        result['comparisons'].append(branches)
        if teacher['status'] != 'proposed':
            result['reason'] = 'teacher_proposal_invalid'
            break
        if any(row['status'] != 'completed' for row in branches.values()):
            result['reason'] = 'invalid_or_budget_censored_pair'
            break
        if not branches['original'].get('source_first_observations_equal'):
            result['reason'] = 'original_action_replay_key_results_unverified'
            break
        accepted, reason = paired_accept(branches['original']['verifier'], branches['repair']['verifier'])
        result['reason'] = reason
        if not accepted:
            break
        if repeat == 1:
            result['accepted'] = True
    save_json(path, result)
    return result
