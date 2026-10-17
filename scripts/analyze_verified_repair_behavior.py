#!/usr/bin/env python3
"""Describe actual proposed/executed actions; no causal labels or filtering."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import statistics
import time

from verified_repair_runtime import save_json
from summarize_verified_repair_data import read, require_final_evidence


def tool_pattern(calls):
    names=[c['function']['name'] for c in calls]
    if not names:return 'no_tool_calls'
    if 'exec' in names:return 'contains_exec'
    if set(names)&{'write','edit'}:return 'contains_write_or_edit'
    if set(names)=={'read'}:return 'read_only'
    return 'other'


def stats(values):
    values=[v for v in values if v is not None]
    return {'n':len(values),'mean':statistics.mean(values) if values else None,
        'median':statistics.median(values) if values else None,
        'min':min(values) if values else None,'max':max(values) if values else None}


def analyze(root,final=False):
    manifest=read(root/'data_branch_shards.json');merged=read(root/'data-branches.json',[])
    if final:require_final_evidence(merged,read(root/'data_gate.json',{}),manifest)
    lookup={r['state_session']:r for r in merged};rows=[]
    for shard in manifest['shards']:
        out=Path(shard['output'])
        for sid in shard['sessions']:
            teacher=read(out/'teacher'/(sid+'.json'),{})
            if not teacher or 'ids' not in teacher:continue
            source=read(root/'sources'/(sid+'.json'));state=read(Path(source['selected_state']))
            pair=lookup.get(sid) or read(out/'pairs'/(sid+'.json'),{})
            original=state['step'];calls=teacher.get('tool_calls',[])
            row={'session':sid,'task_id':state['task_id'],'split':state['split'],
                'state_turn':state['turn'],'prompt_tokens':len(original['prompt_ids']),
                'teacher_prompt_tokens':len(teacher['prompt_ids']),
                'original_tokens':len(original['response_ids']),'repair_tokens':len(teacher['ids']),
                'original_tools':','.join(c['function']['name'] for c in original['tool_calls']),
                'repair_tools':','.join(c['function']['name'] for c in calls),
                'original_pattern':tool_pattern(original['tool_calls']),'repair_pattern':tool_pattern(calls),
                'teacher_status':teacher['status'],'pair_reason':pair.get('reason','pending'),
                'first_pair_completed':False,'reward_delta':None,'criteria_gained':None,'criteria_lost':None}
            comparisons=pair.get('comparisons',[])
            if comparisons and set(comparisons[0])=={'original','repair'}:
                a,b=comparisons[0]['original'],comparisons[0]['repair']
                row['first_pair_completed']=a['status']==b['status']=='completed'
                if row['first_pair_completed']:
                    ca,cb=a['verifier']['criteria'],b['verifier']['criteria']
                    if set(ca)!=set(cb):raise ValueError('criterion keys differ between paired branches')
                    keys=set(ca)-{'internal_error'}
                    row.update(original_reward=a['verifier']['reward'],repair_reward=b['verifier']['reward'],
                        reward_delta=b['verifier']['reward']-a['verifier']['reward'],
                        criteria_gained=sum(not ca[k] and cb[k] for k in keys),
                        criteria_lost=sum(ca[k] and not cb[k] for k in keys),
                        original_continuation_turns=a['budget']['generated_turns'],
                        repair_continuation_turns=b['budget']['generated_turns'],
                        original_execution_seconds=a['execution_seconds'],repair_execution_seconds=b['execution_seconds'])
            rows.append(row)
    groups=defaultdict(list)
    for r in rows:groups[r['repair_pattern']].append(r)
    grouped=[]
    for pattern,group in sorted(groups.items()):
        paired=[r for r in group if r['first_pair_completed']]
        grouped.append({'repair_pattern':pattern,'proposals':len(group),'completed_first_pairs':len(paired),
            'original_tokens':stats([r['original_tokens'] for r in group]),
            'repair_tokens':stats([r['repair_tokens'] for r in group]),
            'reward_delta':stats([r['reward_delta'] for r in paired]),
            'raw_score_increased':sum(r['reward_delta']>1e-8 for r in paired),
            'raw_score_decreased':sum(r['reward_delta']< -1e-8 for r in paired),
            'any_positive_criterion_regression':sum(r['criteria_lost']>0 for r in paired)})
    result={'generated_at':time.time(),'final':final,'registered_candidates':64,'teacher_outputs_with_ids':len(rows),
        'first_pairs_completed':sum(r['first_pair_completed'] for r in rows),
        'teacher_status':dict(Counter(r['teacher_status'] for r in rows)),
        'tool_pattern_transitions':dict(Counter(r['original_pattern']+' -> '+r['repair_pattern'] for r in rows)),
        'state_turn':stats([r['state_turn'] for r in rows]),'prompt_tokens':stats([r['prompt_tokens'] for r in rows]),
        'original_action_tokens':stats([r['original_tokens'] for r in rows]),
        'repair_action_tokens':stats([r['repair_tokens'] for r in rows]),'by_repair_pattern':grouped,
        'definitions':['tool patterns use existing parser output, not regenerated tokens',
            'contains_exec does not prove a workspace write or a beneficial effect',
            'no_tool_calls is not automatically invalid: text-only final answers can be legitimate',
            'first-pair score deltas are descriptive, not two-seed acceptance or causal attribution',
            'criterion counts exclude internal_error health flag; raw rewards and criterion records unchanged']}
    folder=root/'analysis';folder.mkdir(exist_ok=True);suffix='' if final else '_progress'
    save_json(folder/('behavior_statistics'+suffix+'.json'),result)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (folder/('candidate_behavior'+suffix+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();r=analyze(a.root.resolve(),a.final)
    print(json.dumps({k:r[k] for k in ('final','teacher_outputs_with_ids','first_pairs_completed','tool_pattern_transitions')}))
