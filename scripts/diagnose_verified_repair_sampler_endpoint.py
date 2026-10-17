#!/usr/bin/env python3
"""Run-owned diagnostic only. Never patches an active SGLang server.

Invert the installed MurmurHash32 permutation to identify endpoint collisions
without a vocabulary-sized hash matrix for every generated position.
"""
import argparse
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch

from verified_repair_runtime import save_json,sha

MASK=np.uint64(0xffffffff)
C1=0xcc9e2d51;C2=0x1b873593;ADD=0xe6546b64

def rot(x,n):return ((x<<np.uint64(n))|(x>>np.uint64(32-n)))&MASK
def mix(h,k):
    k=(k*np.uint64(C1))&MASK;k=rot(k,15);k=(k*np.uint64(C2))&MASK
    h=h^k;h=rot(h,13);return (h*np.uint64(5)+np.uint64(ADD))&MASK
def hash_numpy(seed,positions,columns):
    h=mix(np.uint64(0),np.uint64(seed&0xffffffff));h=mix(h,np.uint64(seed>>32))
    h=mix(h,np.asarray(positions,dtype=np.uint64));h=mix(h,np.asarray(columns,dtype=np.uint64))
    h=h^np.uint64(16);h=h^(h>>np.uint64(16));h=(h*np.uint64(0x85ebca6b))&MASK
    h=h^(h>>np.uint64(13));h=(h*np.uint64(0xc2b2ae35))&MASK;return h^(h>>np.uint64(16))
def undo_xor_right(y,n):
    x=int(y)
    for shift in range(n,32,n):x^=int(y)>>shift
    return x&0xffffffff
def inverse_fmix(y):
    y=undo_xor_right(y,16);y=(y*pow(0xc2b2ae35,-1,2**32))&0xffffffff
    y=undo_xor_right(y,13);y=(y*pow(0x85ebca6b,-1,2**32))&0xffffffff
    return undo_xor_right(y,16)
def endpoint_columns(seed,positions):
    h=mix(np.uint64(0),np.uint64(seed&0xffffffff));h=mix(h,np.uint64(seed>>32))
    h=mix(h,np.asarray(positions,dtype=np.uint64))
    target=(inverse_fmix(0xffffffff)^16)
    target=np.uint64(((target-ADD)*pow(5,-1,2**32))&0xffffffff)
    k=h^rot(target,32-13)
    k=(k*np.uint64(pow(C2,-1,2**32)))&MASK;k=rot(k,32-15)
    return (k*np.uint64(pow(C1,-1,2**32)))&MASK


def scan_request(record,kind,identity,vocab_size):
    if record.get('sampling_seed') is None:return []
    ids=record.get('response_ids')
    n=len(ids) if ids is not None else record['response_tokens']
    base=record['actual_prompt_tokens']-1
    positions=np.arange(base,base+n,dtype=np.uint64)
    columns=endpoint_columns(int(record['sampling_seed']),positions)
    hits=np.flatnonzero(columns<vocab_size);events=[]
    for i in hits:
        col=int(columns[i]);pos=int(positions[i])
        assert int(hash_numpy(int(record['sampling_seed']),pos,col))==0xffffffff
        events.append({'kind':kind,'identity':identity,'response_position':int(i),
            'absolute_sampler_position':pos,'sampling_seed':int(record['sampling_seed']),
            'endpoint_sorted_column':col,'definitely_masked_by_topk20':col>=20,
            'actual_output_id':ids[i] if ids is not None else None,
            'actual_output_id_recorded':ids is not None,
            'saved_output_logprob':record.get('response_logprobs',[None]*n)[i]})
    return events


def scan(root):
    manifest=json.loads((root/'data_branch_shards.json').read_text())
    model=json.loads((root/'selection.json').read_text())['student_model']
    vocab=json.loads((Path(model)/'config.json').read_text())['vocab_size']
    events=[];counts={};tokens={};id_counts={};branch_sources={}
    def add(rec,kind,identity):
        if rec.get('sampling_seed') is None:return
        counts[kind]=counts.get(kind,0)+1
        tokens[kind]=tokens.get(kind,0)+(len(rec['response_ids']) if rec.get('response_ids') is not None else rec['response_tokens'])
        id_counts[kind]=id_counts.get(kind,0)+int(rec.get('response_ids') is not None)
        events.extend(scan_request(rec,kind,identity,vocab))
    lengths={}
    for task in json.loads((root/'selection.json').read_text())['candidate_tasks']:
        for sample in range(4):
            sid=f"source-{task['task_id']}-{sample}"
            src=json.loads((root/'sources'/(sid+'.json')).read_text())
            for step in src['steps']:
                lengths[('source_rollout',sid,step['sampling_seed'])]=(step['prompt_tokens'],step['response_tokens'],step['step'])
    for shard in manifest['shards']:
        for path in (Path(shard['output'])/'branches').glob('*.json'):
            branch=json.loads(path.read_text())
            branch_sources[branch['session_id']]=branch['source_session']
            for step in branch.get('conversation',[]):
                if step.get('sampling_seed') is not None:
                    lengths[('branch_continuation',branch['session_id'],step['sampling_seed'])]=(step['prompt_tokens'],step['response_tokens'],step['step'])
    paths=[(root/'data-collect-proxy-records.jsonl','source_rollout')]
    paths += [(Path(s['output'])/'data-branches-proxy-records.jsonl','branch_continuation') for s in manifest['shards']]
    processed=set()
    for path,kind in paths:
        seen=set()
        for line in path.open():
            rec=json.loads(line);key=(rec['session_id'],rec['step'])
            if key in seen:raise ValueError('duplicate live request record')
            seen.add(key)
            if rec['sampling_params'].get('top_k')!=20:raise ValueError('unexpected sampler top_k')
            # canonical_prompt_tokens can describe an alternative re-rendering;
            # the conversation usage is the actual prompt sent to the backend.
            identity=(kind,rec['session_id'],rec['sampling_seed']);processed.add(identity)
            rec['actual_prompt_tokens'],response_tokens,actual_step=lengths[identity]
            if response_tokens!=len(rec['response_ids']):raise ValueError('response count/IDs disagree')
            add(rec,kind,f'{key[0]}:turn-{actual_step}')
    # Terminal text responses need not have a tool-result record. Their exact
    # seed and usage still suffice for hash endpoint detection; do not invent IDs.
    for (kind,sid,seed),(prompt_tokens,response_tokens,step) in lengths.items():
        if (kind,sid,seed) not in processed:
            add({'sampling_seed':seed,'actual_prompt_tokens':prompt_tokens,
                'response_tokens':response_tokens},kind,f'{sid}:turn-{step}')
    for shard in manifest['shards']:
        out=Path(shard['output'])
        for p in (out/'teacher').glob('*.json'):
            if p.name.endswith('.request.json'):continue
            t=json.loads(p.read_text())
            if not t.get('ids'):continue
            add({'response_ids':t['ids'],'actual_prompt_tokens':len(t['prompt_ids']),
                'sampling_seed':t['sampling_params']['sampling_seed'],'response_logprobs':t['teacher_logprobs']},'teacher',p.stem)
    for event in events:
        if event['kind']!='source_rollout':continue
        sid,turn=event['identity'].split(':turn-')
        path=root/'states'/sid/f'turn-{int(turn):03d}.json'
        if path.exists():
            state=json.loads(path.read_text());step=state['step'];i=event['response_position']
            if (step['sampling_seed']!=event['sampling_seed'] or len(step['prompt_ids'])+i-1!=event['absolute_sampler_position']
                    or (event['actual_output_id'] is not None and step['response_ids'][i]!=event['actual_output_id'])):raise ValueError('snapshot and logged endpoint disagree')
            event['state_exact_ids_verified']=True
            event['actual_output_id']=step['response_ids'][i];event['actual_output_id_recorded']=True
            event['saved_output_logprob']=step['response_logprobs'][i]
    affected=sorted({branch_sources[e['identity'].split(':turn-')[0]] if e['kind']=='branch_continuation'
        else e['identity'].split(':turn-')[0] for e in events})
    merged=json.loads((root/'data-branches.json').read_text()) if (root/'data-branches.json').exists() else []
    clean=[p for p in merged if p['state_session'] not in affected]
    return {'generated_at':time.time(),'request_counts':counts,'token_counts':tokens,'requests_with_ids_in_direct_logs':id_counts,'vocab_size':vocab,
        'endpoint_events':events,'exact_case_reproduced':any(e['identity']=='source-task_1267-2:turn-4' and e['response_position']==253 for e in events),
        'affected_candidates':affected,
        'sensitivity_excluding_affected_candidates':{'candidates':len(clean),'accepted_pairs':sum(p['accepted'] for p in clean),
            'definition':'descriptive sensitivity check only, not a replacement denominator or rerun'},
        'definition':'exact hash endpoint search on ALL real request seeds and usage lengths; raw IDs attached where saved (logs or source snapshots), never reconstructed for missing terminal text IDs; forced actions excluded; no generation or rewards modified'}


def safe_multinomial(logprobs,seed,positions):
    from sglang.srt.layers import sampler
    cols=torch.arange(logprobs.shape[1],device=logprobs.device)
    hashed=sampler.murmur_hash32(seed.to(torch.uint64),positions,cols)
    # Prototype for validation only: midpoint mapping is strictly inside (0,1).
    uniform=(hashed.to(torch.float64)+.5)/(2**32)
    score=-torch.log(-torch.log(uniform))+logprobs.to(torch.float64)
    return score.argmax(dim=1,keepdim=True)


def check_sampler(device):
    from sglang.srt.layers import sampler
    seed=torch.tensor([1795547190],dtype=torch.int64,device=device)
    pos=torch.tensor([20755],dtype=torch.int64,device=device)
    cols=torch.arange(152064,device=device)
    expected=hash_numpy(int(seed.item()),int(pos.item()),np.arange(152064,dtype=np.uint64))
    reference=torch.from_numpy(expected.astype(np.uint32)).reshape(1,-1).to(device)
    def cpu_hash(s,p,c):
        return torch.stack([torch.from_numpy(hash_numpy(int(ss),int(pp),c.numpy().astype(np.uint64)).astype(np.uint32)) for ss,pp in zip(s,p)])
    if device=='cpu':
        context=patch.object(sampler,'murmur_hash32',cpu_hash)
    else:
        from contextlib import nullcontext
        context=nullcontext()
    start=time.monotonic()
    with context:
        actual=sampler.murmur_hash32(seed.to(torch.uint64),pos,cols)
        exact=torch.equal(actual,reference)
        lp=torch.full((1,152064),-float('inf'),dtype=torch.float64,device=device);lp[0,0]=0.
        original=sampler.multinomial_with_seed
        if device=='cpu':original=original._torchdynamo_orig_callable
        old=int(original(lp,seed,pos).item())
        new=int(safe_multinomial(lp,seed,pos).item())
    if not exact or old!=35146 or new!=0:raise AssertionError((exact,old,new))
    return {'device':device,'vocab_columns':152064,'hash_matches_cpu_reference':exact,
        'old_sampled_sorted_column':old,'old_selected_masked_probability':0.,
        'open_interval_sampled_sorted_column':new,'open_interval_selected_probability':1.,
        'seconds':time.monotonic()-start,'sampler_file_sha256':sha(Path(sampler.__file__)),
        'installed_sampler_modified':False,'model_weights_loaded':False,
        'torch_compile_enabled_for_original':device!='cpu',
        'scope':'actual installed deterministic sampler with exact production seed/position on one-positive-category input; CPU run substitutes only hash with source-equivalent arithmetic; CUDA run uses actual Triton hash'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--scan-only',action='store_true')
    a=p.parse_args();root=a.root.resolve();out=root/'analysis'
    save_json(out/'sampler_endpoint_scan.json',scan(root))
    if not a.scan_only:
        result=check_sampler(a.device);save_json(out/('sampler_endpoint_check_'+a.device+'.json'),result)
        print(json.dumps(result))
