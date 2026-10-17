#!/usr/bin/env python3
"""Run-owned, resumable engineering/data pilot; never starts long RL training."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import sys
import threading
import time
import subprocess
from types import SimpleNamespace

from verified_repair_runtime import ROOT, RemoteState, rollout, save_json, sha, extract_snapshot, select_observed_boundary
import run_fixed_seed_8x8_rollout_gate as gate
from verified_repair_proxy import RepairProxy
from analyze_clawgym_verifiers import inspect as inspect_verifier

MODEL = ROOT / 'outputs/clawgym-8b-sft-formal-20260902-165551/final_model'
ANCHORS = ROOT / 'outputs/opsd-anchor-base-seed0-800-20260831/full800/anchors.jsonl'
TRAIN = ROOT / 'clawgym-official/data/clawgym_train'
_RESTORE = threading.local()
_ORIGINAL_PREPARE = rollout._prepare_workspace


def prepare_workspace(task, directory):
    _ORIGINAL_PREPARE(task, directory)
    archive = getattr(_RESTORE, 'archive', None)
    if archive:
        # The brand-new run-owned workspace has not yet been mounted. Retain
        # its pristine copy instead of merging with a snapshot containing deletions.
        if not str(directory).startswith('/tmp/clawgym-rl-'):
            raise ValueError('unexpected fresh workspace path')
        shutil.move(str(directory), str(directory.with_name('pristine_workspace')))
        extract_snapshot(archive, directory)


def select_tasks(output):
    anchor_rows = [json.loads(line) for line in ANCHORS.open()]
    anchor_ids = {row['task_id'] for row in anchor_rows}
    eval_queries = {json.loads(p.read_text())['user_query'].strip()
                    for p in (ROOT / 'clawgym-official/data/clawgym_eval').glob('*/data_entry.json')}
    forbidden = re.compile(r'serper|websocket|https?://|send (?:an? )?email|live (?:api|data)|'
                           r'connect to|telegram bot|deploy to|real.time|external api', re.I)
    eligible = []
    for entry in sorted(TRAIN.glob('*/data_entry.json')):
        value = json.loads(entry.read_text())
        if value['task_id'] not in anchor_ids or value['user_query'].strip() in eval_queries:
            continue
        info = inspect_verifier(entry.parent / 'reward/check.py')
        if (not info['exposes_criterion_vector'] or info['criterion_count'] < 2
                or info['uses_subprocess'] or forbidden.search(value['user_query'])
                or 'output/' not in value['user_query']):
            continue
        eligible.append({'task_id': value['task_id'], 'entry': str(entry),
                         'query_sha256': hashlib.sha256(value['user_query'].encode()).hexdigest(),
                         'checker_sha256': sha(entry.parent / 'reward/check.py'),
                         'criterion_keys': info['criterion_keys']})
    # Deterministic selection independent of rollout rewards and Teacher results.
    ordered = sorted(eligible, key=lambda r: hashlib.sha256(('repair-v2:20260907:' + r['task_id']).encode()).hexdigest())
    smoke_ids = ['task_0126', 'task_0133', 'task_0140', 'task_0158', 'task_0162']
    lookup = {r['task_id']: r for r in eligible}
    smoke = [lookup[t] for t in smoke_ids if t in lookup]
    for row in ordered:
        if len(smoke) == 5:
            break
        if row not in smoke:
            smoke.append(row)
    candidates = [r for r in ordered if r['task_id'] not in {x['task_id'] for x in smoke}][:16]
    if len(smoke) != 5 or len(candidates) != 16:
        raise ValueError('insufficient eligible tasks')
    for index, row in enumerate(candidates):
        row['split'] = 'heldout' if index % 4 == 3 else 'train'
    manifest = {'version': 2, 'selection_seed': 20260907, 'smoke': smoke,
                'candidate_tasks': candidates, 'candidate_states': 64, 'samples_per_task': 4,
                'eligible_tasks': len(eligible), 'selection_before_generation': True,
                'eval_query_overlap': 0, 'teacher_model': str(MODEL), 'student_model': str(MODEL),
                'anchors': str(ANCHORS), 'anchors_sha256': sha(ANCHORS),
                'source_seed': 20260907, 'branch_seed_bases': [314159, 271828],
                'teacher_seed_base': 161803, 'temperature': .7, 'top_p': .8, 'top_k': 20,
                'branch_continuation_turns': 3, 'branch_seconds': 300,
                'branch_assistant_tokens': 32768, 'beta': .1, 'margin': .1,
                'gate_min_pairs': 16, 'gate_min_tasks': 8,
                'cross_backend_sampler_agreement': 'PENDING_REAL_MODEL_CHECK'}
    save_json(output / 'selection.json', manifest)
    return manifest


def configure(args):
    g = SimpleNamespace(output_dir=args.output, stage=args.stage, model=MODEL, python=ROOT / '.venv-openclaw/bin/python',
        served_model='verified-repair-8b', backend_url=args.backend_url,
        backend_port=args.backend_port, proxy_port=args.proxy_port, reverse_port=args.reverse_port,
        gpus=args.gpus, tensor_parallel_size=1, data_parallel_size=1, mem_fraction_static=.70,
        engine_seed=20260907, deterministic=True, enable_cuda_graph=True, serve_timeout=900,
        sampling_seed_base=20260907, deterministic_tool_call_ids=True,
        task_seed=0, tasks=TRAIN, parallel=args.workers, rollout_timeout=1200,
        remote_host='159.138.50.147', remote_user='root', ssh_key=Path('/root/openclaw_cpu_key'),
        remote_base='/data/clawgym_rl_runs/' + args.output.name,
        ssh_pool_size=4, gateway_port_base=27000, gateway_train_offset=1000,
        gateway_port_stride=2, eval_mode=False)
    gate._configure_environment(g)
    os.environ['OPENCLAW_RECORD_FILE'] = str(args.output / (args.stage + '-proxy-records.jsonl'))
    os.environ.update({'OPENCLAW_BUNDLED_PLUGINS_DIR': '/app/dist/extensions',
        'OPENCLAW_REMOTE_GATEWAY_PORT_MODE': 'run_scoped',
        'OPENCLAW_REMOTE_GATEWAY_PORT_MIN': '20000', 'OPENCLAW_REMOTE_GATEWAY_PORT_MAX': '62000',
        'OPENCLAW_REMOTE_DOCKER_START_CONCURRENCY': '2', 'OPENCLAW_REMOTE_START_ATTEMPTS': '2',
        'OPENCLAW_RECORD_RESPONSE_IDS': '1', 'OPENCLAW_MAX_AGENT_STEPS': '96',
        'SGLANG_DISABLE_FLASHINFER_ALLREDUCE_FUSION': '1',
        'OPENCLAW_OPSD_ENABLE': '0', 'OPENCLAW_OPSD_EVIDENCE_ANCHORS_FILE': str(ANCHORS)})
    rollout._GATEWAY_IMAGE = gate.PINNED_IMAGE
    rollout._prepare_workspace = prepare_workspace
    return g


def start_model(g):
    if g.backend_url:
        return gate._start_sglang(g)
    # This private localhost engine additionally allows an explicit diagnostic
    # teacher-forcing processor. Ordinary rollout requests never specify it.
    handle = (g.output_dir / (g.stage + '-sglang.log')).open('wb')
    env = dict(getattr(g, 'engine_environment', None) or os.environ)
    env['CUDA_VISIBLE_DEVICES'] = g.gpus
    env['PYTHONPATH'] = str(ROOT / 'scripts') + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [str(g.python), '-m', 'sglang.launch_server', '--model-path', str(g.model),
        '--served-model-name', g.served_model, '--host', '127.0.0.1', '--port', str(g.backend_port),
        '--tensor-parallel-size', '1', '--context-length', '65536',
        '--mem-fraction-static', str(g.mem_fraction_static), '--chunked-prefill-size', '8192',
        '--max-prefill-tokens', '16384', '--max-running-requests', '32',
        '--random-seed', str(g.engine_seed), '--trust-remote-code', '--tool-call-parser', 'qwen25',
        '--enable-deterministic-inference', '--sampling-backend', 'pytorch',
        '--constrained-json-disable-any-whitespace',
        '--enable-custom-logit-processor']
    if not g.enable_cuda_graph:
        cmd.append('--disable-cuda-graph')
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    engine_record = {'command': cmd, 'pid': process.pid, 'started_at': time.time(),
                     'reserved_gpus': g.gpus.split(',')}
    save_json(g.output_dir / 'engine-command.json', engine_record)
    save_json(g.output_dir / (g.stage + '-engine-command.json'), engine_record)
    try:
        gate._wait_http(f'http://127.0.0.1:{g.backend_port}/health', 900)
    except BaseException:
        gate._stop_process(process, 30)
        handle.close()
        raise
    return process, handle


def start_runtime(task, index, sid, g, state, proxy, *, archive=None, branch=None):
    started = time.monotonic()
    runtime = None
    _RESTORE.archive = archive
    try:
        runtime = rollout._start_container(task, 907, index, g.proxy_port, sid, 'main__' + sid, is_train=True)
        state.register(runtime)
        proxy.register_runtime(runtime, branch=branch)
        rollout._register_session_metadata(g.proxy_port, sid, task, 907, index, 0,
                                            sampling_identity=branch['pair_identity'] if branch else sid)
        rollout._wait_for_chat_ready(runtime)
        return runtime, time.monotonic() - started
    except BaseException:
        if runtime is not None:
            state.stop(runtime)
        raise
    finally:
        _RESTORE.archive = None


def collect_one(row, sample, index, g, state, proxy):
    task = rollout._load_task_spec(Path(row['entry']))
    sid = f"source-{task.task_id}-{sample}"
    destination = g.output_dir / 'sources' / (sid + '.json')
    if destination.exists():
        return json.loads(destination.read_text())
    record = {'task_id': task.task_id, 'sample': sample, 'session_id': sid,
              'split': row.get('split', 'engineering_smoke'), 'started_at': time.time()}
    runtime = None
    try:
        runtime, startup = start_runtime(task, index, sid, g, state, proxy)
        record['startup_seconds'] = startup
        start = time.monotonic()
        final = rollout._run_turns(runtime, task, 1, 'main')
        record['rollout_seconds'] = time.monotonic() - start
        record['final_message'] = final
        record['steps'] = copy.deepcopy(proxy._conversation_logs.get(sid, []))
        terminal = state.snapshot(runtime, g.output_dir / 'terminal' / (sid + '.tar.gz'), audit_external=False)
        record['terminal_snapshot'] = terminal
        record['verifier'] = state.verify(task, terminal['archive'], sid + '-terminal')
        paths = proxy.candidates.get(sid, [])
        record['snapshot_candidates'] = paths
        record['capture_errors'] = proxy.capture_errors.get(sid, [])
        # Latest saved in-scope action with real matched tool observations.
        selected, observations, rejections = select_observed_boundary(paths, record['steps'])
        record['boundary_rejections'] = rejections
        if selected:
            candidate = json.loads(Path(selected).read_text())
            candidate['source_observations'] = observations
            candidate['task_id'] = task.task_id
            candidate['task_entry'] = str(task.task_dir / 'data_entry.json')
            candidate['split'] = record['split']
            candidate['source_terminal_verifier'] = record['verifier']
            candidate['selected'] = True
            save_json(selected, candidate)
        record['selected_state'] = selected
        record['status'] = 'collected' if selected else 'no_restorable_boundary'
    except Exception as exc:
        record['status'] = 'infrastructure_or_capture_error'
        record['error'] = type(exc).__name__ + ':' + str(exc)[:1200]
    finally:
        record['finished_at'] = time.time()
        save_json(destination, record)
        if runtime is not None:
            state.stop(runtime)
    print(json.dumps({k: record[k] for k in ('session_id', 'status', 'started_at', 'finished_at')}, ensure_ascii=False), flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['prepare', 'smoke-collect', 'data-collect', 'smoke-branches', 'data-branches'], required=True)
    p.add_argument('--gpus', default='0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--backend-url', default='')
    p.add_argument('--backend-port', type=int, default=30970)
    p.add_argument('--proxy-port', type=int, default=30971)
    p.add_argument('--reverse-port', type=int, default=39971)
    p.add_argument('--source-output', type=Path)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args()
    args.output = args.output.resolve()
    source_output = (args.source_output or args.output).resolve()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('invalid shard assignment')
    if args.num_shards != 1 and args.stage != 'data-branches':
        raise ValueError('only fixed data-branches may be partitioned')
    if not args.output.name.startswith('verified-repair-'):
        raise ValueError('unique verified-repair-* output name required')
    args.output.mkdir(parents=True, exist_ok=True)
    selection_path = source_output / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else select_tasks(args.output)
    if args.output != source_output:
        destination_selection = args.output / 'selection.json'
        if destination_selection.exists():
            if sha(destination_selection) != sha(selection_path):
                raise ValueError('existing shard selection differs; refusing overwrite')
        else:
            shutil.copy2(selection_path, destination_selection)
    if args.stage == 'prepare':
        print(json.dumps({'output': str(args.output), 'eligible': selection['eligible_tasks'],
                          'smoke': [x['task_id'] for x in selection['smoke']],
                          'candidate_tasks': [x['task_id'] for x in selection['candidate_tasks']]}))
        return 0
    if args.stage.startswith('data-') and not (source_output / 'smoke_passed.json').exists():
        raise RuntimeError('5-state closed-loop smoke must pass before data collection')
    identity = {'stage': args.stage, 'started_at': time.time(),
                'selection_sha256': sha(selection_path), 'model': str(MODEL),
                'model_config_sha256': sha(MODEL / 'config.json'),
                'gpus': args.gpus, 'workers': args.workers,
                'source_output': str(source_output), 'shard_index': args.shard_index,
                'num_shards': args.num_shards,
                'code_sha256': {p.name: sha(p) for p in sorted((ROOT / 'scripts').glob('*verified_repair*.py'))},
                'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    save_json(args.output / (args.stage + '-execution.json'), identity)
    g = configure(args)
    state = RemoteState(args.output, g.remote_base, gate.PINNED_IMAGE)
    process = handle = proxy = tunnel = None
    completed = False
    try:
        process, handle = start_model(g)
        enabled = threading.Event(); enabled.set()
        proxy = RepairProxy(SimpleNamespace(hf_checkpoint=str(MODEL), sglang_router_ip='127.0.0.1',
            sglang_router_port=g.backend_port, rollout_temperature=.7, rollout_top_p=.8, rollout_top_k=20),
            queue.Queue(), enabled, remote_state=state)
        proxy.start(); proxy.wait_until_ready(900)
        tunnel = gate._start_reverse_tunnel(g)
        if args.stage.endswith('-branches'):
            from verified_repair_branches import teacher_proposal, verify_candidate
            from verified_repair_sampler_probe import audit_action
            rollout._CHAT_TURN_TIMEOUT_SECONDS = 300
            source_records = [json.loads(path.read_text()) for path in sorted((source_output / 'sources').glob('*.json'))]
            source_records = [row for row in source_records if
                ((row.get('split') == 'engineering_smoke') == (args.stage == 'smoke-branches'))]
            active_tasks = {row['task_id'] for row in selection[
                'smoke' if args.stage == 'smoke-branches' else 'candidate_tasks']}
            source_records = [row for row in source_records if row['task_id'] in active_tasks]
            required_sources = 5 if args.stage == 'smoke-branches' else 64
            if len(source_records) != required_sources:
                raise RuntimeError(f'expected {required_sources} source records, found {len(source_records)}')
            if args.stage == 'data-branches':
                expected = {f"source-{row['task_id']}-{sample}" for row in selection['candidate_tasks']
                            for sample in range(selection['samples_per_task'])}
                if {r['session_id'] for r in source_records} != expected:
                    raise RuntimeError('fixed source identities differ from preregistration')
            anchors = {r['task_id']: r['anchors'] for line in ANCHORS.open() if (r := json.loads(line))}
            results = []
            for index, record in enumerate(source_records):
                if index % args.num_shards != args.shard_index:
                    continue
                if not record.get('selected_state'):
                    result = {'task_id': record['task_id'], 'state_session': record['session_id'],
                        'split': record['split'], 'source_status': record['status'],
                        'accepted': False, 'reason': 'no_restorable_state', 'comparisons': [],
                        'sampling_audit_not_applicable': 'no valid complete action and restorable prefix'}
                    save_json(args.output / 'pairs' / (record['session_id'] + '.json'), result)
                    results.append(result)
                    continue
                saved_state = json.loads(Path(record['selected_state']).read_text())
                if record['verifier']['reward'] >= 1:
                    teacher = {'task_id': record['task_id'], 'status': 'rejected',
                        'reason': 'source_task_already_complete', 'seconds': 0,
                        'model_requests': 0, 'ids': []}
                    save_json(args.output / 'teacher' / (saved_state['session_id'] + '.json'), teacher)
                else:
                    teacher = teacher_proposal(saved_state, anchors[saved_state['task_id']], proxy, g,
                        args.output / 'teacher' / (saved_state['session_id'] + '.json'))
                result = verify_candidate(saved_state, teacher, index, g, state, proxy, start_runtime)
                result['teacher_status'] = teacher['status']
                result['teacher_reason'] = teacher.get('reason')
                result['source_saved_logprob_temperature'] = .7
                result['source_saved_action_logprob'] = sum(saved_state['step']['response_logprobs'])
                actions = {'original': {'text': saved_state['step']['response_text'],
                                        'ids': saved_state['step']['response_ids']}}
                if teacher['status'] == 'proposed':
                    actions['repair'] = teacher
                try:
                    result['sampler_audits'] = {side: audit_action(g, saved_state, action,
                        saved_state['session_id'] + '-' + side) for side, action in actions.items()}
                    result['auxiliary_span_validated'] = True
                except Exception as exc:
                    result['auxiliary_span_validated'] = False
                    result['sampling_audit_error'] = type(exc).__name__ + ':' + str(exc)[:1000]
                    # A diagnostic transport failure is not a policy failure,
                    # but this sample cannot enter training until repaired.
                    result['verifier_pair_accepted_before_span_audit'] = result['accepted']
                    result['accepted'] = False
                    result['reason'] = 'native_sampler_audit_unresolved'
                save_json(args.output / 'pairs' / (saved_state['session_id'] + '.json'), result)
                results.append(result)
                print(json.dumps({'task_id': result['task_id'], 'accepted': result['accepted'], 'reason': result.get('reason')}), flush=True)
            save_json(args.output / (args.stage + '.json'), results)
            # The full smoke gate additionally requires GPU alignment/gradient
            # evidence; collecting/branching alone NEVER creates smoke_passed.
            completed = True
            return 0
        rows = selection['smoke'] if args.stage == 'smoke-collect' else selection['candidate_tasks']
        repeats = 1 if args.stage == 'smoke-collect' else 4
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(collect_one, row, sample, i * repeats + sample, g, state, proxy)
                       for i, row in enumerate(rows) for sample in range(repeats)]
            results = [future.result() for future in as_completed(futures)]
        save_json(args.output / (args.stage + '.json'), results)
        completed = True
        return 0
    finally:
        if tunnel: gate._stop_process(tunnel, 10)
        if proxy: proxy.stop()
        if process: gate._stop_process(process, 30)
        if handle: handle.close()
        finished_at = time.time()
        identity.update(finished_at=finished_at, wall_seconds=finished_at-identity['started_at'],
                        process_stage_completed=completed, optimizer_updates=0)
        save_json(args.output / (args.stage + '-execution.json'), identity)
        engine_path = args.output / (args.stage + '-engine-command.json')
        if process and engine_path.exists():
            engine = json.loads(engine_path.read_text())
            if engine['pid'] == process.pid and 'started_at' in engine:
                engine.update(exited_at=finished_at, returncode=process.poll(),
                    reserved_gpu_seconds=(finished_at-engine['started_at'])*len(engine['reserved_gpus']),
                    gpu_time_definition='reserved wall time, not measured kernel-active duration')
                save_json(engine_path, engine)


if __name__ == '__main__':
    def terminate_cleanly(signum, frame):
        raise KeyboardInterrupt('run-owned supervisor requested shutdown')
    signal.signal(signal.SIGTERM, terminate_cleanly)
    raise SystemExit(main())
