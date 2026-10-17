#!/usr/bin/env python3
"""Descriptive token buckets from unchanged native token IDs; never a loss mask."""
from collections import Counter, defaultdict
import argparse
import csv
import json
from pathlib import Path
import re
import time

from verified_repair_runtime import save_json
from merge_verified_repair_shards import information_boundary_review

BUCKETS=['prose','tool_arguments_structure','code_payload','tool_name','control','mixed','unmapped']


def byte_spans(tokenizer, ids):
    """Qwen ByteLevel fragments, including multi-byte characters across tokens."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    inverse={v:k for k,v in bytes_to_unicode().items()}
    decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    chunks=[]
    for token_id in ids:
        if token_id in tokenizer.added_tokens_decoder:
            chunk=tokenizer.added_tokens_decoder[token_id].content.encode('utf-8')
        else:
            token=tokenizer.convert_ids_to_tokens(token_id)
            try:chunk=bytes(inverse[c] for c in token)
            except (KeyError,TypeError):return decoded,None
        chunks.append(chunk)
    if b''.join(chunks)!=decoded.encode('utf-8'):
        return decoded,None  # Do not silently retokenize or invent alignment.
    spans=[];offset=0
    for chunk in chunks:
        spans.append((offset,offset+len(chunk)));offset+=len(chunk)
    return decoded,spans


def classify_bytes(text):
    """Syntactic labels, not a claim about causal credit or action usefulness."""
    offsets=[0]
    for c in text:offsets.append(offsets[-1]+len(c.encode('utf-8')))
    labels=bytearray(offsets[-1])
    def mark(start,end,bucket):labels[offsets[start]:offsets[end]]=bytes([BUCKETS.index(bucket)])*(offsets[end]-offsets[start])
    quoted=r'((?:\\.|[^"\\])*)'
    for block in re.finditer(r'<tool_call>(.*?)</tool_call>',text,re.S):
        start,end=block.span(1);body=block.group(1)
        mark(block.start(),block.end(),'tool_arguments_structure')
        name=re.search(r'"name"\s*:\s*"'+quoted+'"',body)
        tool=name.group(1) if name else ''
        path=re.search(r'"(?:path|file_path)"\s*:\s*"'+quoted+'"',body)
        code_file=bool(path and Path(path.group(1)).suffix.lower() in {'.py','.sh','.js','.ts','.sql','.c','.cpp','.h','.go','.rs'})
        if name:mark(start+name.start(1),start+name.end(1),'tool_name')
        for field in re.finditer(r'"(command|content|oldText|newText|old_string|new_string)"\s*:\s*"'+quoted+'"',body):
            if (tool=='exec' and field.group(1)=='command') or (tool in {'write','edit'} and code_file):
                mark(start+field.start(2),start+field.end(2),'code_payload')
        mark(block.start(),start,'control');mark(end,block.end(),'control')
    for special in re.finditer(r'<\|[^<>]*\|>|</?think>',text):
        mark(*special.span(),'control')
    return labels


def token_buckets(tokenizer, ids):
    text,spans=byte_spans(tokenizer,ids)
    if spans is None:return ['unmapped']*len(ids),False
    labels=classify_bytes(text);result=[]
    for start,end in spans:
        values=set(labels[start:end])
        result.append(BUCKETS[next(iter(values))] if len(values)==1 else 'mixed')
    return result,True


def analyze(root, final=False):
    from transformers import AutoTokenizer
    selection=json.loads((root/'selection.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(selection['student_model'],local_files_only=True)
    if json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())['type']!='ByteLevel':
        raise ValueError('this diagnostic requires the validated ByteLevel decoder')
    manifest=json.loads((root/'data_branch_shards.json').read_text())
    locations={sid:Path(s['output']) for s in manifest['shards'] for sid in s['sessions']}
    if final:
        pairs=json.loads((root/'data-branches.json').read_text())
        if len(pairs)!=64:raise ValueError('final analysis requires the complete fixed candidate set')
    else:
        pairs=[json.loads(p.read_text()) for s in manifest['shards'] for p in (Path(s['output'])/'pairs').glob('*.json')]
    actions=[];token_rows=[]
    for pair in pairs:
        if not pair.get('auxiliary_span_validated'):continue
        sid=pair['state_session'];out=locations[sid]
        source=json.loads((root/'sources'/(sid+'.json')).read_text())
        state=json.loads(Path(source['selected_state']).read_text())
        teacher=json.loads((out/'teacher'/(sid+'.json')).read_text())
        verifier_passed=bool(pair.get('verifier_pair_accepted',pair.get('accepted')))
        review=information_boundary_review(pair,root,out) if verifier_passed else {}
        eligible=bool(verifier_passed and review.get('approved'))
        for side,audit in pair['sampler_audits'].items():
            ids=state['step']['response_ids'] if side=='original' else teacher['ids']
            native=[json.loads(line) for line in Path(audit['raw_statistics_file']).open()]
            if [r['target'] for r in native]!=ids:raise ValueError('unaligned native action')
            buckets,exact=token_buckets(tokenizer,ids)
            action={'session':sid,'task_id':pair['task_id'],'split':pair['split'],'side':side,
                'accepted_pair':eligible,'verifier_pair_passed':verifier_passed,
                'tokens':len(ids),'blocked_tokens':sum(r['blocked'] for r in native),
                'first_blocked_position':next((r['position'] for r in native if r['blocked']),None),
                'blocked_positions':[r['position'] for r in native if r['blocked']],
                'native_entry_mean_hinge':sum(max(0.,.1+r['boundary_sampling_logit']-r['target_sampling_logit']) for r in native if r['blocked'])/max(1,sum(r['blocked'] for r in native)),
                'byte_mapping_exact':exact,'bucket_counts':dict(Counter(buckets)),
                'blocked_bucket_counts':dict(Counter(b for b,r in zip(buckets,native) if r['blocked']))}
            actions.append(action)
            for bucket,r in zip(buckets,native):
                token_rows.append({'session':sid,'task_id':pair['task_id'],'split':pair['split'],'side':side,
                    'accepted_pair':eligible,'verifier_pair_passed':verifier_passed,
                    'position':r['position'],'target_id':r['target'],
                    'bucket':bucket,'blocked':r['blocked'],'logprob_T1':r['target_logprob_T1'],
                    'logprob_T07':r['target_logprob_T07'],
                    'native_hinge':max(0.,.1+r['boundary_sampling_logit']-r['target_sampling_logit']) if r['blocked'] else 0.})
    stats=[]
    for side in ('original','repair'):
        for bucket in BUCKETS:
            rows=[r for r in token_rows if r['side']==side and r['bucket']==bucket]
            n=len(rows);blocked=sum(r['blocked'] for r in rows)
            stats.append({'side':side,'bucket':bucket,'tokens':n,'blocked_tokens':blocked,
                'blocked_fraction':blocked/n if n else None,
                'mean_logprob_T07':sum(r['logprob_T07'] for r in rows)/n if n else None})
    result={'generated_at':time.time(),'final':final,'audited_actions':len(actions),
        'audited_tokens':len(token_rows),'exact_byte_mapped_actions':sum(a['byte_mapping_exact'] for a in actions),
        'buckets':stats,'actions':actions,'definition':'diagnostic syntactic buckets only; no masking, retokenizing or training changes',
        'scope':'only actions with a completed native full-action audit; invalid/incomplete proposals are not treated as verified positive actions'}
    folder=root/'analysis';folder.mkdir(exist_ok=True);suffix='' if final else '_progress'
    save_json(folder/('token_statistics'+suffix+'.json'),result)
    if token_rows:
        with (folder/('native_token_buckets'+suffix+'.csv')).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(token_rows[0]));writer.writeheader();writer.writerows(token_rows)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();s=analyze(a.root.resolve(),a.final)
    print(json.dumps({k:s[k] for k in ('final','audited_actions','audited_tokens','exact_byte_mapped_actions')}))


if __name__=='__main__':main()
