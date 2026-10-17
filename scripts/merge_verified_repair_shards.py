#!/usr/bin/env python3
"""Prepare fixed disjoint shards and independently audit their complete outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from verified_repair_runtime import save_json, sha, paired_accept, archive_manifest


def prepare(root):
    selection = json.loads((root / 'selection.json').read_text())
    expected = sorted(f"source-{task['task_id']}-{sample}"
        for task in selection['candidate_tasks'] for sample in range(4))
    if len(expected) != 64 or len(set(expected)) != 64:
        raise ValueError('exactly 64 prespecified source identities required')
    children = [root.with_name(root.name + '-shard-' + str(i)) for i in range(2)]
    manifest = {'source_output': str(root), 'selection_sha256': sha(root/'selection.json'),
        'num_shards': 2, 'assignment': 'lexicographic source-session index modulo 2',
        'shards': [{'index': i, 'output': str(path), 'gpu': i,
                    'backend_port': 30970+2*i, 'proxy_port': 30971+2*i,
                    'reverse_port': 39971+2*i, 'sessions': expected[i::2]}
                   for i, path in enumerate(children)]}
    existing = root / 'data_branch_shards.json'
    if existing.exists() and json.loads(existing.read_text()) != manifest:
        raise ValueError('refusing to change existing shard assignment')
    save_json(existing, manifest)
    return manifest


def validate_accepted(row):
    if not row.get('accepted'):
        return False
    comparisons = row.get('comparisons', [])
    if len(comparisons) != 2 or not row.get('auxiliary_span_validated'):
        raise ValueError('accepted pair lacks repeats or native span validation')
    identities = []
    for repeat, comparison in enumerate(comparisons):
        if set(comparison) != {'original', 'repair'}:
            raise ValueError('accepted comparison lacks both real branches')
        a, b = comparison['original'], comparison['repair']
        if not a.get('source_first_observations_equal'):
            raise ValueError('original action replay unverified')
        for branch in (a,b):
            budget = branch['budget']
            if branch.get('verifier',{}).get('raw_verifier',{}).get('internal_error') is True:
                raise ValueError('accepted pair contains a verifier internal-error flag')
            if (branch['status'] != 'completed' or not branch['restore_manifest_equal']
                    or branch['execution_seconds'] >= 300 or budget.get('error')
                    or budget['generated_turns'] > 3 or budget['tokens'] > 32768):
                raise ValueError('accepted branch violates execution protocol')
        if a['sampling_identity'] != b['sampling_identity']:
            raise ValueError('branch seeds were not paired')
        if not a['sampling_identity'].endswith(str([314159,271828][repeat])):
            raise ValueError('repeat seed differs from preregistration')
        identities.append(a['sampling_identity'])
        if not paired_accept(a['verifier'], b['verifier'])[0]:
            raise ValueError('accepted pair fails original verifier criteria')
    if identities[0] == identities[1]:
        raise ValueError('second repeat did not change predetermined seed')
    if set(row.get('sampler_audits', {})) != {'original', 'repair'}:
        raise ValueError('accepted pair lacks both action probability audits')
    return True


def audit_branch_artifacts(row, state):
    """Audit rejected comparisons too; a false acceptance flag is not evidence."""
    if row['state_session'] != state['session_id'] or row['task_id'] != state['task_id']:
        raise ValueError('comparison points to a different source state')
    comparisons = row.get('comparisons', [])
    if len(comparisons) > 2:
        raise ValueError('more than the two prespecified comparisons')
    for repeat, comparison in enumerate(comparisons):
        if not comparison or set(comparison) - {'original', 'repair'}:
            raise ValueError('unexpected comparison branch identities')
        identity = f"{state['session_id']}:paired-seed-{[314159,271828][repeat]}"
        for side, branch in comparison.items():
            if (branch['sampling_identity'] != identity or branch['side'] != side
                    or branch['source_session'] != state['session_id']
                    or branch['paired_repeat'] != repeat
                    or not branch['forced_action_is_auxiliary_not_rollout']):
                raise ValueError('branch identity or auxiliary-action provenance differs')
            restored = branch.get('restore_snapshot')
            if restored:
                if (sha(restored['archive']) != restored['sha256']
                        or archive_manifest(restored['archive']) != restored['manifest']):
                    raise ValueError('restored archive differs from its recorded evidence')
                equal = restored['manifest'] == state['snapshot']['manifest']
                if branch.get('restore_manifest_equal') != equal:
                    raise ValueError('restore success flag is inaccurate')
            if side == 'original' and 'first_action_observations' in branch:
                equal = branch['first_action_observations'] == state.get('source_observations')
                if branch.get('source_first_observations_equal') != equal:
                    raise ValueError('original replay success flag is inaccurate')
            for turn in branch.get('conversation', []):
                if turn.get('forced_auxiliary_action'):
                    if turn.get('sampling_seed') is not None:
                        raise ValueError('forced action must not claim a sampling seed')
                    continue
                material = f"20260907:{identity}:main:{turn['step']-1}"
                expected = int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], 'little') & 0x7fffffff
                if turn.get('sampling_seed') != expected:
                    raise ValueError('actual generated turn used a nonregistered seed')
            if branch['status'] != 'completed':
                continue  # A censored branch need not have a verifier result.
            if not restored or not branch['restore_manifest_equal']:
                raise ValueError('completed branch lacks a matched real restore')
            verifier = branch['verifier']
            source_v = state['source_terminal_verifier']
            if any(verifier[k] != source_v[k] for k in ('checker_sha256', 'reward_sha256')):
                raise ValueError('verifier changed after source collection')
            raw = json.loads(verifier['checker_stdout'])
            criteria = {k: int(v) for k,v in raw.items() if isinstance(v,bool)}
            if (raw != verifier['raw_verifier'] or criteria != verifier['criteria']
                    or float(verifier['reward_stdout'].strip().splitlines()[-1]) != verifier['reward']):
                raise ValueError('verifier summaries differ from real output')
            terminal = branch['terminal_snapshot']
            if (sha(terminal['archive']) != terminal['sha256']
                    or archive_manifest(terminal['archive']) != terminal['manifest']):
                raise ValueError('terminal workspace evidence changed')
    return True


def audit_action_artifacts(row, root, output):
    """Re-open real source/audit artifacts instead of trusting success flags."""
    source=json.loads((root/'sources'/(row['state_session']+'.json')).read_text())
    state=json.loads(Path(source['selected_state']).read_text())
    teacher=json.loads((output/'teacher'/(row['state_session']+'.json')).read_text())
    targets={'original':state['step']['response_ids']}
    if teacher['status']=='proposed':targets['repair']=teacher['ids']
    origin=state['step']['prompt_ids']
    origin_hash=hashlib.sha256(json.dumps(origin).encode()).hexdigest()
    if set(row['sampler_audits'])!=set(targets):
        raise ValueError('probability audit does not cover complete proposed actions')
    for comparison in row.get('comparisons', []):
        for side, branch in comparison.items():
            if branch['forced_action_ids'] != targets[side]:
                raise ValueError('executed branch action differs from audited action')
    for side,ids in targets.items():
        summary=row['sampler_audits'][side]
        path=Path(summary['raw_statistics_file']).resolve()
        if path.parent != (output/'sampler').resolve():
            raise ValueError('audit path points outside its own shard')
        samples=[json.loads(line) for line in path.open()]
        native=json.loads(path.with_suffix('.response.json').read_text())
        actual=[int(x[1]) for x in native['native_output']['meta_info']['output_token_logprobs']]
        if actual!=ids or [s['target'] for s in samples]!=ids:
            raise ValueError('native sampler audit token IDs differ from action')
        if [s['position'] for s in samples]!=list(range(len(ids))):
            raise ValueError('native audit positions are incomplete')
        if not all(s['origin_ids_sha256']==origin_hash and s['generated_prefix_matches_teacher']
                and s['prefix_length']==len(origin)+i
                and s['blocked']==(s['target'] not in s['candidate_ids']) for i,s in enumerate(samples)):
            raise ValueError('native audit prefix or support membership invalid')
        for key,field in [('action_logprob_T1','target_logprob_T1'),('action_logprob_T07','target_logprob_T07')]:
            if abs(summary[key]-sum(s[field] for s in samples))>1e-7:
                raise ValueError('action logprob summary differs from raw rows')
    if row['accepted']:
        checker=state['source_terminal_verifier']['checker_sha256']
        reward=state['source_terminal_verifier']['reward_sha256']
        for comparison in row['comparisons']:
            for branch in comparison.values():
                if (branch['verifier']['checker_sha256']!=checker
                        or branch['verifier']['reward_sha256']!=reward):
                    raise ValueError('verifier changed after source collection')
                restored=branch['restore_snapshot']
                if restored['manifest']!=state['snapshot']['manifest'] or sha(restored['archive'])!=restored['sha256']:
                    raise ValueError('restored state does not match source archive')
                for turn in branch.get('conversation',[]):
                    if turn.get('forced_auxiliary_action'):continue
                    seed_material=f"20260907:{branch['sampling_identity']}:main:{turn['step']-1}"
                    expected=int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:4],'little')&0x7fffffff
                    if turn.get('sampling_seed')!=expected:
                        raise ValueError('actual generated turn did not use the registered seed')
    return True


def information_boundary_review(row, root, output):
    """Grounding prerequisite from plan v2, separate from measured reward gains."""
    path=root/'information_boundary_reviews'/(row['state_session']+'.json')
    if not path.exists():
        return {'decision':'pending','approved':False,'reason':'visible_state_grounding_not_reviewed'}
    review=json.loads(path.read_text())
    source=json.loads((root/'sources'/(row['state_session']+'.json')).read_text())
    state_path=Path(source['selected_state'])
    teacher=json.loads((output/'teacher'/(row['state_session']+'.json')).read_text())
    ids_hash=hashlib.sha256(json.dumps(teacher['ids']).encode()).hexdigest()
    if (review.get('protocol')!='visible_state_v2' or review['state_session']!=row['state_session']
            or review['state_file_sha256']!=sha(state_path) or review['teacher_ids_sha256']!=ids_hash):
        raise ValueError('information-boundary review is stale or points to another action/state')
    if review['decision'] not in {'approved','rejected'} or not review.get('evidence'):
        raise ValueError('information-boundary decision lacks evidence')
    return {'decision':review['decision'],'approved':review['decision']=='approved',
            'reason':review['reason'],'review_file':str(path)}


def merge(root, manifest):
    rows = []
    for shard in manifest['shards']:
        output = Path(shard['output'])
        complete = json.loads((output/'data-branches.json').read_text())
        if sorted(r['state_session'] for r in complete) != shard['sessions']:
            raise ValueError('shard omitted or duplicated a fixed candidate')
        for row in complete:
            row['artifact_output'] = str(output)
            row['verifier_acceptance_independently_validated'] = validate_accepted(row)
            if row.get('comparisons'):
                source=json.loads((root/'sources'/(row['state_session']+'.json')).read_text())
                state=json.loads(Path(source['selected_state']).read_text())
                row['branch_artifacts_independently_validated']=audit_branch_artifacts(row,state)
            if row.get('auxiliary_span_validated'):
                row['native_artifacts_independently_validated']=audit_action_artifacts(row,root,output)
            row['verifier_pair_accepted']=bool(row['verifier_acceptance_independently_validated'])
            if row['verifier_pair_accepted']:
                review=information_boundary_review(row,root,output)
                row['information_boundary_review']=review
                row['accepted']=review['approved']
                row['verifier_pair_reason']=row['reason']
                if not row['accepted']:row['reason']=review['reason']
            row['independently_validated_acceptance']=bool(row['verifier_pair_accepted'] and row['accepted'])
        rows.extend(complete)
    if len(rows) != 64 or len({r['state_session'] for r in rows}) != 64:
        raise ValueError('merged experiment is not exactly 64 candidates')
    rows.sort(key=lambda r:r['state_session'])
    accepted = [r for r in rows if r['independently_validated_acceptance'] and r['accepted']]
    train = [r for r in accepted if r['split']=='train']
    tasks = sorted({r['task_id'] for r in accepted})
    gate = {'time': time.time(), 'candidates': 64, 'accepted_pairs': len(accepted),
        'merge_code_sha256': sha(Path(__file__)),
        'accepted_tasks': tasks, 'train_accepted_pairs': len(train),
        'two_seed_verifier_passed_pairs':sum(r['verifier_pair_accepted'] for r in rows),
        'pending_information_boundary_reviews':sum(r.get('information_boundary_review',{}).get('decision')=='pending' for r in rows),
        'data_gate_passed': len(accepted)>=16 and len(tasks)>=8 and bool(train),
        'unresolved_sampler_audits': sum(bool(r.get('sampling_audit_error')) for r in rows),
        'no_additional_candidates_allowed': True, 'optimizer_updates': 0}
    gate['data_gate_passed'] = (gate['data_gate_passed']
        and not gate['pending_information_boundary_reviews'] and not gate['unresolved_sampler_audits'])
    save_json(root/'data-branches.json', rows)
    save_json(root/'data_gate.json', gate)
    print(json.dumps(gate, ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');args=p.parse_args()
    root=args.root.resolve()
    if not root.name.startswith('verified-repair-'):
        raise ValueError('run-owned output required')
    manifest=prepare(root)
    if args.prepare:
        print(json.dumps({'shards': [s['output'] for s in manifest['shards']], 'candidates':64}))
    else:
        merge(root,manifest)


if __name__=='__main__':main()
