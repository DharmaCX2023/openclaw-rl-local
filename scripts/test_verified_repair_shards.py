"""CPU-only independent checks of fixed partition and acceptance merge gates."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from merge_verified_repair_shards import prepare,validate_accepted,audit_branch_artifacts,information_boundary_review


def test_partition_is_complete_disjoint_and_immutable(tmp_path):
    root=tmp_path/'verified-repair-unit';root.mkdir()
    (root/'selection.json').write_text(json.dumps({'candidate_tasks':[{'task_id':f'task_{i:04d}'} for i in range(16)]}))
    m=prepare(root)
    left,right=(set(s['sessions']) for s in m['shards'])
    assert len(left)==len(right)==32 and not left&right
    assert len(left|right)==64
    assert prepare(root)==m
    assert {s['gpu'] for s in m['shards']}=={0,1}
    assert len({s[k] for s in m['shards'] for k in ('backend_port','proxy_port','reverse_port')})==6


def accepted_fixture():
    def branch(seed,reward):
        return {'status':'completed','restore_manifest_equal':True,'source_first_observations_equal':True,
            'sampling_identity':seed,'execution_seconds':10.,
            'budget':{'generated_turns':3,'tokens':100,'error':None},
            'verifier':{'reward':reward,'criteria':{'a':1,'b':int(reward==1)}}}
    return {'accepted':True,'auxiliary_span_validated':True,
        'sampler_audits':{'original':{},'repair':{}},
        'comparisons':[{'original':branch(str(s),.5),'repair':branch(str(s),1)} for s in (314159,271828)]}


def test_merge_recomputes_acceptance_instead_of_trusting_flag():
    good=accepted_fixture(); assert validate_accepted(good)
    for mutation in ('timeout','regression','same_seed','missing_span','verifier_error'):
        r=deepcopy(good)
        if mutation=='timeout':r['comparisons'][0]['repair']['execution_seconds']=300.
        elif mutation=='regression':r['comparisons'][1]['repair']['verifier']['criteria']['a']=0
        elif mutation=='same_seed':
            for b in r['comparisons'][1].values():b['sampling_identity']='314159'
        elif mutation=='missing_span':r['auxiliary_span_validated']=False
        elif mutation=='verifier_error':r['comparisons'][0]['repair']['verifier']['raw_verifier']={'internal_error':True}
        with pytest.raises(ValueError):validate_accepted(r)


def test_rejected_branch_still_requires_registered_seed():
    import hashlib
    sid='source-task_0001-0';identity=sid+':paired-seed-314159'
    seed=int.from_bytes(hashlib.sha256(f'20260907:{identity}:main:1'.encode()).digest()[:4],'little')&0x7fffffff
    state={'session_id':sid,'task_id':'task_0001','source_observations':['same']}
    branch={'side':'original','sampling_identity':identity,'source_session':sid,
        'paired_repeat':0,'forced_action_is_auxiliary_not_rollout':True,
        'status':'invalid_or_censored','first_action_observations':['same'],
        'source_first_observations_equal':True,
        'conversation':[{'step':1,'forced_auxiliary_action':True},
                        {'step':2,'sampling_seed':seed}]}
    row={'accepted':False,'state_session':sid,'task_id':'task_0001',
         'comparisons':[{'original':branch}]}
    assert audit_branch_artifacts(row,state)
    branch['conversation'][1]['sampling_seed']+=1
    with pytest.raises(ValueError,match='nonregistered seed'):audit_branch_artifacts(row,state)
    branch['conversation'][1]['sampling_seed']=seed
    branch['source_first_observations_equal']=False
    with pytest.raises(ValueError,match='success flag'):audit_branch_artifacts(row,state)


def test_grounding_review_requires_evidence_bound_to_unchanged_state_and_target(tmp_path):
    import hashlib
    from verified_repair_runtime import sha
    root=tmp_path/'root';out=tmp_path/'shard';sid='source-unit-0'
    (root/'sources').mkdir(parents=True);(out/'teacher').mkdir(parents=True)
    row={'state_session':sid}
    assert information_boundary_review(row,root,out)['decision']=='pending'
    state=root/'state.json';state.write_text('{}')
    (root/'sources'/(sid+'.json')).write_text(json.dumps({'selected_state':str(state)}))
    teacher=out/'teacher'/(sid+'.json');teacher.write_text(json.dumps({'ids':[1,2]}))
    folder=root/'information_boundary_reviews';folder.mkdir()
    review={'protocol':'visible_state_v2','state_session':sid,'state_file_sha256':sha(state),
        'teacher_ids_sha256':hashlib.sha256(json.dumps([1,2]).encode()).hexdigest(),
        'decision':'rejected','reason':'unit_only_unsupported_claim','evidence':{'unit_fixture':True}}
    (folder/(sid+'.json')).write_text(json.dumps(review))
    assert not information_boundary_review(row,root,out)['approved']
    review['decision']='approved';(folder/(sid+'.json')).write_text(json.dumps(review))
    assert information_boundary_review(row,root,out)['approved']
    teacher.write_text(json.dumps({'ids':[1,3]}))
    with pytest.raises(ValueError,match='stale'):information_boundary_review(row,root,out)
