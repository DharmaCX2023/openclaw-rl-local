#!/usr/bin/env python3
"""Read artifacts, report unmet gates; never turn partial evidence into a pass."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from verified_repair_runtime import save_json, sha, archive_manifest, paired_accept


def read(path):
    return json.loads(Path(path).read_text())


def audit(out):
    selection = read(out / 'selection.json')
    rows = []
    for task in selection['smoke']:
        sid = 'source-' + task['task_id'] + '-0'
        source_path = out / 'sources' / (sid + '.json')
        row = {'task_id': task['task_id'], 'session_id': sid, 'unmet': []}
        if not source_path.exists():
            row['unmet'].append('source_missing'); rows.append(row); continue
        source = read(source_path)
        if not source.get('selected_state'):
            row['unmet'].append('restorable_state_missing'); rows.append(row); continue
        state = read(source['selected_state'])
        snapshot = state['snapshot']
        archive = Path(snapshot['archive'])
        row.update(state_turn=state['turn'], prompt_tokens=len(state['step']['prompt_ids']),
                   original_tokens=len(state['step']['response_ids']),
                   source_terminal_reward=source['verifier']['reward'])
        row['snapshot_valid'] = (archive.exists() and sha(archive) == snapshot['sha256']
                                 and archive_manifest(archive) == snapshot['manifest']
                                 and state.get('pre_action') is True)
        if not row['snapshot_valid']: row['unmet'].append('snapshot_invalid')
        teacher_path = out / 'teacher' / (sid + '.json')
        teacher = read(teacher_path) if teacher_path.exists() else {}
        if source['verifier']['reward'] >= 1 and not teacher:
            teacher = {'status': 'rejected', 'reason': 'source_task_already_complete'}
        row['teacher_status'] = teacher.get('status', 'pending')
        row['teacher_reason'] = teacher.get('reason')
        row['teacher_tokens'] = len(teacher.get('ids', []))
        pair_path = out / 'pairs' / (sid + '.json')
        pair = read(pair_path) if pair_path.exists() else {}
        comparisons = pair.get('comparisons', [])
        original = comparisons[0].get('original', {}) if comparisons else {}
        row['restore_verified'] = original.get('restore_manifest_equal') is True
        row['original_observations_verified'] = original.get('source_first_observations_equal') is True
        row['branch_status'] = original.get('status', 'pending')
        row['branch_error'] = original.get('error')
        row['pair_accepted'] = pair.get('accepted', False)
        row['pair_reason'] = pair.get('reason')
        if not row['restore_verified']: row['unmet'].append('restore_unverified')
        if not row['original_observations_verified']: row['unmet'].append('original_replay_unverified')
        if not pair: row['unmet'].append('pair_validation_pending')
        if row['pair_accepted']:
            valid = len(comparisons) == 2 and all(
                all(v.get('status') == 'completed' for v in c.values())
                and c.get('original', {}).get('source_first_observations_equal') is True
                and paired_accept(c['original']['verifier'], c['repair']['verifier'])[0]
                for c in comparisons)
            if not valid: row['unmet'].append('accepted_pair_not_twice_verified')
        sides = ['original'] + (['repair'] if teacher.get('status') == 'proposed' else [])
        row['sampler_audits'] = {}
        for side in sides:
            path = out / 'sampler' / (sid + '-' + side + '.summary.json')
            native = read(path) if path.exists() else {}
            ids = state['step']['response_ids'] if side == 'original' else teacher['ids']
            raw_path = out / 'sampler' / (sid + '-' + side + '.jsonl')
            raw = [json.loads(line) for line in raw_path.open()] if raw_path.exists() else []
            valid = (native.get('exact_target_ids') is True and native.get('exact_prefix_ids') is True
                     and len(raw) == len(ids) and [r['target'] for r in raw] == ids
                     and [r['position'] for r in raw] == list(range(len(ids))))
            row['sampler_audits'][side] = {'verified': valid, 'tokens': len(raw),
                'blocked': sum(r['blocked'] for r in raw)}
            if not valid: row['unmet'].append(side + '_native_sampler_audit_missing')
        path = out / 'gradient_smoke_native_support' / (sid + '.json')
        grad = read(path) if path.exists() else {}
        row['gradient_verified'] = (grad.get('finite_gradients') is True
            and grad.get('optimizer_steps') == 0 and grad.get('model_updated') is False
            and grad.get('parameters_with_gradients', 0) > 0)
        row['native_train_membership_agreement'] = grad.get('native_train_membership_agreement')
        if not row['gradient_verified']: row['unmet'].append('real_gpu_backward_missing')
        details = grad.get('token_details', [])
        row['actual_native_margin_mask_verified'] = (grad.get('actual_native_margin_mask_verified') is True
            and bool(details) and all(d.get('entry_mask_source') == 'native_sglang_candidate_ids'
                and d.get('entry_blocked') == d.get('sglang_blocked') for d in details))
        if not row['actual_native_margin_mask_verified']:
            row['unmet'].append('margin_mask_from_actual_sampler_unverified')
        rows.append(row)
    return {'time': time.time(), 'selection_sha256': sha(out / 'selection.json'),
            'stage': 'engineering_smoke', 'states': rows,
            'snapshot_valid': sum(r.get('snapshot_valid', False) for r in rows),
            'original_replay_verified': sum(r.get('original_observations_verified', False) for r in rows),
            'pair_accepted': sum(r.get('pair_accepted', False) for r in rows),
            'unmet_counts': dict(Counter(x for r in rows for x in r['unmet'])),
            'all_engineering_requirements_proven': all(not r['unmet'] for r in rows),
            'efficacy_conclusion': 'UNTESTED; engineering checks cannot demonstrate efficacy',
            'does_not_authorize_data_stage': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); out = args.output.resolve()
    result = audit(out)
    save_json(out / 'smoke_audit.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'states'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
