"""Native SGLang diagnostic teacher forcing, NOT a policy rollout.

For each exact target prefix, record the actual pre-sampling logits/support,
then force the supplied next token to visit the next teacher-forced prefix.
The synthetic generation's returned logprobs are deliberately NOT used.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import hashlib

import torch
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from verified_repair_losses import sampling_support


class TeacherForcedSamplerProbe(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list=None):
        if custom_param_list is None:
            raise ValueError('explicit diagnostic parameters required')
        for i, params in enumerate(custom_param_list):
            path = Path(params['audit_path']).resolve()
            base = Path('/root/openclaw-rl-local/outputs')
            relative = path.relative_to(base)
            if not relative.parts[0].startswith('verified-repair-') or relative.parts[-2] != 'sampler':
                raise ValueError('sampler probe output is not run-owned')
            req = params['__req__']
            # With overlap scheduling, CPU req.output_ids lags one model step.
            # Count invocations for THIS request rather than forcing token zero
            # twice. Validate the emitted IDs against every target afterward.
            position = getattr(req, '_verified_repair_probe_position', 0)
            targets = params['teacher_ids']
            if position >= len(targets):
                # SGLang may compute one unused look-ahead step before learning
                # that max_new_tokens was reached. It is not a target position.
                logits[i].fill_(-float('inf'))
                logits[i, int(targets[-1])] = 0
                continue
            setattr(req, '_verified_repair_probe_position', position + 1)
            target = int(targets[position])
            row = logits[i:i+1].detach().float()
            support, boundary = sampling_support(row, .7, .8, 20)
            u = row / .7
            if not torch.isfinite(row[0, target]):
                raise ValueError('target prohibited before custom forcing')
            members = support[0].nonzero().flatten()
            record = {'position': position, 'target': target,
                'blocked': not bool(support[0, target].item()),
                'candidate_ids': members.tolist(),
                'target_logit': float(row[0, target]),
                'target_sampling_logit': float(u[0, target]),
                'boundary_sampling_logit': float(boundary[0]),
                'target_logprob_T1': float(torch.log_softmax(row, -1)[0, target]),
                'target_logprob_T07': float(torch.log_softmax(u, -1)[0, target]),
                'prefix_length': len(req.origin_input_ids) + position,
                'origin_ids_sha256': hashlib.sha256(json.dumps(req.origin_input_ids).encode()).hexdigest(),
                'generated_prefix_matches_teacher': list(req.output_ids) == targets[:len(req.output_ids)],
                'committed_output_length': len(req.output_ids),
                'timestamp': time.time(), 'source': 'sglang_pre_sampling_logits',
                'is_policy_rollout': False}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            # Only AFTER recording the genuine logits, force the known target.
            # This walks exact repaired-action prefixes in one SGLang request.
            logits[i].fill_(-float('inf'))
            logits[i, target] = 0
        return logits


def audit_action(g, state, action, label):
    import requests
    path = g.output_dir / 'sampler' / (label + '.jsonl')
    done = path.with_suffix('.summary.json')
    if done.exists():
        return json.loads(done.read_text())
    if path.exists():
        raise ValueError('partial audit exists; inspect before retrying this diagnostic')
    ids = list(action['ids'])
    origin = list(state['step']['prompt_ids'])
    payload = {'input_ids': origin, 'return_logprob': True,
        'sampling_params': {'temperature': .7, 'top_p': .8, 'top_k': 20,
            'max_new_tokens': len(ids), 'sampling_seed': 123123,
            'skip_special_tokens': False,
            'custom_params': {'teacher_ids': ids, 'audit_path': str(path)}},
        'custom_logit_processor': TeacherForcedSamplerProbe.to_str()}
    started = time.monotonic()
    response = requests.post(f'http://127.0.0.1:{g.backend_port}/generate', json=payload, timeout=1200)
    response.raise_for_status()
    output = response.json()
    from verified_repair_runtime import save_json
    save_json(path.with_suffix('.response.json'), {
        'native_output': output, 'teacher_forcing_only': True,
        'seconds': time.monotonic() - started, 'expected_ids': ids})
    actual = [int(row[1]) for row in output['meta_info']['output_token_logprobs']]
    rows = [json.loads(line) for line in path.open()]
    assert actual == ids, 'native teacher forcing generated different token IDs'
    assert [r['position'] for r in rows] == list(range(len(ids)))
    assert [r['target'] for r in rows] == ids
    assert all(r['generated_prefix_matches_teacher'] for r in rows)
    origin_hash = hashlib.sha256(json.dumps(origin).encode()).hexdigest()
    assert all(r['origin_ids_sha256'] == origin_hash for r in rows)
    result = {'label': label, 'tokens': len(ids), 'blocked_tokens': sum(r['blocked'] for r in rows),
        'action_logprob_T1': sum(r['target_logprob_T1'] for r in rows),
        'action_logprob_T07': sum(r['target_logprob_T07'] for r in rows),
        'teacher_forcing_seconds': time.monotonic() - started,
        'exact_target_ids': True, 'exact_prefix_ids': True, 'raw_statistics_file': str(path),
        'synthetic_output_logprobs_discarded': True, 'is_policy_rollout': False}
    save_json(done, result)
    return result
