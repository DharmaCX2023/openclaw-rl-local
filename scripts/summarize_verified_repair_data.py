#!/usr/bin/env python3
"""Fixed-denominator pilot accounting. No reward changes or model updates."""
import argparse
from collections import Counter
import csv
import datetime
import json
from pathlib import Path
import re
import time

from verified_repair_runtime import save_json, paired_accept
from merge_verified_repair_shards import information_boundary_review


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def require_final_evidence(merged, gate, assignment):
    expected={sid for shard in assignment['shards'] for sid in shard['sessions']}
    actual=[row.get('state_session') for row in merged]
    if len(merged)!=64 or len(set(actual))!=64 or set(actual)!=expected or not gate:
        raise RuntimeError('final report requires all 64 independently merged candidates')
    if gate.get('pending_information_boundary_reviews',0):
        raise RuntimeError('final report still has pending information-boundary reviews')
    if gate.get('unresolved_sampler_audits',0) or any(r.get('sampling_audit_error') for r in merged):
        raise RuntimeError('final report still has unresolved native sampler audits')
    for row in merged:
        if row.get('comparisons') and not row.get('branch_artifacts_independently_validated'):
            raise RuntimeError('final report lacks independent branch evidence')
        if row.get('auxiliary_span_validated') and not row.get('native_artifacts_independently_validated'):
            raise RuntimeError('final report lacks independent full-action evidence')
        if row.get('reason') in (None,'pending'):
            raise RuntimeError('final report still has an unfinished candidate')


def engineering_costs(root):
    """Retain failed setup/diagnostics; do not double-count probes in a stage."""
    roots=[root.with_name('verified-repair-v2-20260907-pilot'),
           root.with_name('verified-repair-v2-20260907-pilot-a2'),root]
    stages=[];unstarted=[];probes=[];teachers=[];branches=[]
    parse=lambda s:datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    for folder in roots:
        for path in sorted(folder.glob('*.log')):
            if not re.fullmatch(r'(?:smoke-collect|smoke-branches|gradient-smoke)-\d{8}-\d{6}\.log',path.name):continue
            starts=[];ends=[]
            for line in path.open(errors='replace'):
                if m:=re.match(r'^stage=(\S+) started_utc=(\S+)',line):starts.append(m.groups())
                if m:=re.match(r'^stage=(\S+) exit_code=(\d+) finished_utc=(\S+)',line):ends.append(m.groups())
            if len(starts)==len(ends)==1:
                start,end=parse(starts[0][1]),parse(ends[0][2])
                stages.append({'log':str(path),'stage':starts[0][0],'start_epoch':start,'end_epoch':end,
                    'wrapper_seconds':end-start,'exit_code':int(ends[0][1]),'allocated_model_gpu_slots':1,
                    'includes_failed_attempt':int(ends[0][1])!=0})
            else:
                unstarted.append({'log':str(path),'stage_start_marker_present':bool(starts),
                    'exact_stage_wall_seconds':None,'reason':'wrapper did not record a complete model-stage interval'})
        teachers += [read(p) for p in (folder/'teacher').glob('*.json') if not p.name.endswith('.request.json')]
        branches += [read(p) for p in (folder/'branches').glob('*.json')]
        for path in (folder/'sampler').glob('native-interface*.jsonl'):
            rows=[json.loads(line) for line in path.open()];ts=[r['timestamp'] for r in rows if 'timestamp' in r]
            if ts:probes.append({'file':str(path),'diagnostic_rows':len(rows),'first_record':min(ts),'last_record':max(ts)})
    for p in probes:
        p['already_charged_in_wrapper']=[s['log'] for s in stages if s['start_epoch']<=p['first_record']<=p['last_record']<=s['end_epoch']]
    return {'stages':stages,'pre_stage_attempts_without_complete_timing':unstarted,
        'wrapper_seconds_sum':sum(s['wrapper_seconds'] for s in stages),
        'allocated_model_gpu_slot_seconds_sum':sum(s['wrapper_seconds']*s['allocated_model_gpu_slots'] for s in stages),
        'teacher_result_count':len(teachers),'teacher_tokens':sum(len(t.get('ids',[])) for t in teachers),
        'teacher_seconds_sum':sum(t.get('seconds',0) for t in teachers),
        'branch_result_count':len(branches),'branch_total_seconds_sum':sum(b.get('total_seconds',0) for b in branches),
        'interface_probes':probes,'optimizer_updates':0,
        'definition':'engineering smoke/debug only; one reserved model GPU per wrapper, not kernel-active time; burn excluded; probe intervals are nested, not added again'}


def summarize(root, final=False):
    selection=read(root/'selection.json')
    assignment=read(root/'data_branch_shards.json')
    locations={sid:Path(shard['output']) for shard in assignment['shards'] for sid in shard['sessions']}
    merged=read(root/'data-branches.json',[])
    gate=read(root/'data_gate.json',{})
    if final:require_final_evidence(merged,gate,assignment)
    merged_by_id={r['state_session']:r for r in merged}
    rows=[];branches=[];teachers=[];sources=[];audits=[]
    for task in selection['candidate_tasks']:
        for sample in range(4):
            sid=f"source-{task['task_id']}-{sample}";out=locations[sid]
            src=read(root/'sources'/(sid+'.json'),{})
            state=read(Path(src['selected_state']),{}) if src.get('selected_state') else {}
            teacher=read(out/'teacher'/(sid+'.json'),{})
            pair=merged_by_id.get(sid) or read(out/'pairs'/(sid+'.json'),{})
            verifier_passed=bool(pair.get('verifier_pair_accepted',pair.get('accepted')) and pair.get('auxiliary_span_validated'))
            grounding=information_boundary_review(pair,root,out) if verifier_passed else {}
            sources.append(src);teachers.append(teacher)
            row={'session':sid,'task_id':task['task_id'],'split':task['split'],
                'source_status':src.get('status','pending'),'source_reward':src.get('verifier',{}).get('reward'),
                'state_turn':state.get('turn'),'prompt_tokens':len(state['step']['prompt_ids']) if state else None,
                'original_action_tokens':len(state['step']['response_ids']) if state else None,
                'source_assistant_tokens':sum(s['response_tokens'] for s in src.get('steps',[])) if src else None,
                'teacher_status':teacher.get('status','pending'),'teacher_reason':teacher.get('reason'),
                'teacher_request_recorded':(out/'teacher'/(sid+'.request.json')).exists(),
                'teacher_tokens':len(teacher.get('ids',[])) if teacher else None,
                'teacher_seconds':teacher.get('seconds'),'pair_reason':pair.get('reason','pending'),
                'verifier_pair_accepted':verifier_passed,
                'information_boundary_decision':grounding.get('decision','not_applicable'),
                'accepted':bool(verifier_passed and grounding.get('approved')),
                'comparisons':len(pair.get('comparisons',[])),
                'native_audit_completed':bool(pair.get('auxiliary_span_validated')),
                'branch_artifact_audit':bool(pair.get('branch_artifacts_independently_validated')),
                'first_comparison_passed':False}
            if verifier_passed and not grounding.get('approved'):row['pair_reason']=grounding['reason']
            for repeat,comp in enumerate(pair.get('comparisons',[])):
                for side,b in comp.items():
                    branches.append(b)
                    if repeat==0:
                        row[side+'_branch_status']=b['status']
                        row[side+'_reward']=b.get('verifier',{}).get('reward')
                        row[side+'_execution_seconds']=b.get('execution_seconds')
                if repeat==0 and set(comp)=={'original','repair'}:
                    a,b=comp['original'],comp['repair']
                    valid=(a['status']==b['status']=='completed' and a.get('source_first_observations_equal'))
                    row['first_comparison_passed']=bool(valid and paired_accept(a['verifier'],b['verifier'])[0])
            for side in ('original','repair'):
                audit=pair.get('sampler_audits',{}).get(side,{})
                if audit:audits.append(audit)
                for field in ('tokens','blocked_tokens','action_logprob_T1','action_logprob_T07','teacher_forcing_seconds'):
                    row[side+'_'+field]=audit.get(field)
            rows.append(row)
    def total(items,key):return sum(x.get(key,0) or 0 for x in items)
    vf=[s['verifier'] for s in sources if s.get('verifier')]+[b['verifier'] for b in branches if b.get('verifier')]
    attempts=[read(p) for shard in assignment['shards'] for p in (Path(shard['output'])/'verifier_attempts').glob('*.json')]
    engines=[]
    for out,stage in [(root,'data-collect')]+[(Path(s['output']),'data-branches') for s in assignment['shards']]:
        rec=read(out/(stage+'-execution.json'),{})
        engine=read(out/(stage+'-engine-command.json'),{})
        engines.append({'output':str(out),'stage':stage,'wall_seconds':rec.get('wall_seconds'),
            'reserved_gpu_seconds':engine.get('reserved_gpu_seconds'),'completed':rec.get('process_stage_completed'),
            'legacy_timing_requires_wrapper_log':bool(rec and 'wall_seconds' not in rec)})
    summary={'generated_at':time.time(),'final':final,'registered_candidates':64,
        'source_completed':sum(bool(s) for s in sources),'restorable_states':sum(bool(s.get('selected_state')) for s in sources),
        'teacher_records':sum(bool(t) for t in teachers),'teacher_model_requests':sum(r['teacher_request_recorded'] for r in rows),
        'teacher_proposed_valid':sum(t.get('status')=='proposed' for t in teachers),
        'source_status_counts':dict(Counter(r['source_status'] for r in rows)),
        'teacher_reasons':dict(Counter(r['teacher_reason'] for r in rows if r['teacher_reason'])),
        'pair_reason_counts':dict(Counter(r['pair_reason'] for r in rows)),
        'first_comparison_passed':sum(r['first_comparison_passed'] for r in rows),
        'repeat_attempted':sum(r['comparisons']==2 for r in rows),
        'two_seed_verifier_passed_pairs':sum(r['verifier_pair_accepted'] for r in rows),
        'pending_information_boundary_reviews':sum(r['information_boundary_decision']=='pending' for r in rows),
        'accepted_pairs':sum(r['accepted'] for r in rows),
        'accepted_task_ids':sorted({r['task_id'] for r in rows if r['accepted']}),
        'completed_branches':sum(b['status']=='completed' for b in branches),
        'branch_attempts':len(branches),'branch_status_counts':dict(Counter(b['status'] for b in branches)),
        'restores_evidenced':sum('restore_manifest_equal' in b for b in branches),
        'restores_matched':sum(bool(b.get('restore_manifest_equal')) for b in branches),
        'original_replays_evidenced':sum('source_first_observations_equal' in b for b in branches),
        'original_replays_matched':sum(bool(b.get('source_first_observations_equal')) for b in branches),
        'native_action_audits':len(audits),'native_target_tokens':total(audits,'tokens'),
        'native_blocked_tokens':total(audits,'blocked_tokens'),'gate':gate,
        'costs':{'teacher_seconds_sum':total(teachers,'seconds'),'teacher_tokens':sum(len(t.get('ids',[])) for t in teachers),
            'source_assistant_tokens':sum(r['source_assistant_tokens'] or 0 for r in rows),
            'branch_generated_assistant_tokens':sum(turn.get('response_tokens',0) for b in branches
                for turn in b.get('conversation',[]) if not turn.get('forced_auxiliary_action')),
            'branch_forced_auxiliary_tokens':sum(len(b.get('forced_action_ids',[])) for b in branches),
            'source_startup_seconds_sum':total(sources,'startup_seconds'),'source_rollout_seconds_sum':total(sources,'rollout_seconds'),
            'branch_total_seconds_sum':total(branches,'total_seconds'),
            'branch_initialize_restore_seconds_sum':total(branches,'initialize_restore_seconds'),
            'branch_execution_seconds_sum':total(branches,'execution_seconds'),
            'successful_verifier_calls':len(vf),'successful_verifier_seconds_sum':total(vf,'seconds'),
            'instrumented_verifier_attempts':len(attempts),'failed_verifier_attempts':sum(a['status']=='failed' for a in attempts),
            'failed_verifier_seconds_sum':sum(a.get('seconds',0) for a in attempts if a['status']=='failed'),
            'native_teacher_forcing_seconds_sum':total(audits,'teacher_forcing_seconds'),
            'optimizer_updates':0,'engine_stages':engines,
            'collection_wrapper_timing':read(root/'collection_timing_from_wrapper.json'),
            'engineering_history':engineering_costs(root),
            'definitions':['per-request sums are not end-to-end wall time',
                'branch total includes execution, restore and verifier; do not add those again',
                'source rollout includes in-flight snapshot capture',
                'one verifier call runs the original reward wrapper plus raw criterion extraction',
                'reserved GPU wall time is not kernel-active GPU time',
                'historical smoke/debug costs are separate and not included in these formal64 sums']}}
    if final and (summary['accepted_pairs']!=gate['accepted_pairs']):
        raise ValueError('independent aggregate disagrees with gate')
    out=root/'analysis';out.mkdir(exist_ok=True)
    save_json(out/('final_summary.json' if final else 'progress_summary.json'),summary)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/('candidate_metrics.csv' if final else 'candidate_metrics_progress.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();s=summarize(a.root.resolve(),a.final)
    print(json.dumps({k:s[k] for k in ('final','source_completed','teacher_records','accepted_pairs','pair_reason_counts')},ensure_ascii=False))


if __name__=='__main__':main()
