"""Keep pending candidates and pre-request rejections in honest denominators."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parent))
from merge_verified_repair_shards import prepare
from summarize_verified_repair_data import summarize, require_final_evidence


def test_fixed_denominator_and_context_rejection_is_not_model_call(tmp_path):
    root=tmp_path/'verified-repair-summary';root.mkdir()
    tasks=[{'task_id':f'task_{i:04d}','split':'heldout' if i%4==3 else 'train'} for i in range(16)]
    (root/'selection.json').write_text(json.dumps({'candidate_tasks':tasks}))
    manifest=prepare(root);shard=Path(manifest['shards'][0]['output'])
    (shard/'teacher').mkdir(parents=True)
    sid=manifest['shards'][0]['sessions'][0]
    (shard/'teacher'/(sid+'.json')).write_text(json.dumps({'status':'rejected',
        'reason':'teacher_context_budget','started_at':100}))
    s=summarize(root)
    assert s['registered_candidates']==64 and s['teacher_records']==1
    assert s['teacher_model_requests']==0
    assert s['source_status_counts']=={'pending':64}
    assert s['accepted_pairs']==0 and s['pair_reason_counts']=={'pending':64}
    with pytest.raises(RuntimeError,match='64 independently merged'):summarize(root,True)
    assert not (root/'analysis/final_summary.json').exists()
    (shard/'teacher'/(sid+'.request.json')).write_text('{}')
    assert summarize(root)['teacher_model_requests']==1
    (root/'sources').mkdir()
    (root/'sources'/(sid+'.json')).write_text(json.dumps({'status':'collected',
        'steps':[{'response_tokens':7},{'response_tokens':4}]}))
    assert summarize(root)['costs']['source_assistant_tokens']==11


def test_final_requires_exact_identities_and_completed_independent_evidence():
    from copy import deepcopy
    rows=[{'state_session':str(i),'reason':'no_strict_short_window_improvement',
        'comparisons':[{}],'branch_artifacts_independently_validated':True,
        'auxiliary_span_validated':True,'native_artifacts_independently_validated':True} for i in range(64)]
    assignment={'shards':[{'sessions':[str(i) for i in range(64)]}]}
    gate={'accepted_pairs':0,'pending_information_boundary_reviews':0,'unresolved_sampler_audits':0}
    require_final_evidence(rows,gate,assignment)
    for fault in ('duplicate','branch','native','audit_error','pending','review'):
        r=deepcopy(rows);g=deepcopy(gate)
        if fault=='duplicate':r[0]['state_session']='1'
        elif fault=='branch':r[0]['branch_artifacts_independently_validated']=False
        elif fault=='native':r[0]['native_artifacts_independently_validated']=False
        elif fault=='audit_error':r[0]['sampling_audit_error']='unit failure'
        elif fault=='pending':r[0]['reason']='pending'
        elif fault=='review':g['pending_information_boundary_reviews']=1
        with pytest.raises(RuntimeError):require_final_evidence(r,g,assignment)
