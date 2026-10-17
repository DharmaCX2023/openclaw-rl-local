"""Replay recorded JSON turn identities/token counts; never unpickle checkpoints."""
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R/'slime'),str(R/'clawgym-official')]
from artifact_grpo_bridge import prepare
root=R/'outputs/clawgym-spantrace-sft20-20260921-spantrace-sft20/span_audit'
os.environ['OPENCLAW_SPAN_TRACE_ROOT']=str(root)
os.environ['OPENCLAW_ARTIFACT_SCORE_MANIFEST']=str(R/'outputs/artifact_grpo40_20260922/artifact_score_manifest.json')
samples=[]
for p in (root/'sessions').glob('*/runtime.json'):
 meta=json.load(open(p))
 if meta['rollout_id']!=0:continue
 traj=json.load(open(p.with_name('trajectory.json')));idx=meta['task_index']
 # Separate chunks with different lengths exercise logical-trajectory normalization.
 for step in traj['react_steps']:
  n=step['response_tokens'];turn=step['step']
  samples.append(SimpleNamespace(group_index=idx//8,index=idx,metadata={'session_id':meta['session_id']},opsd_step_ids=[turn]*n,loss_mask=[1]*n,response_length=n))
if __name__=='__main__':
 assert len({(s.group_index,s.index) for s in samples})==64
 values=prepare(samples)
 assert len(values)==len(samples)
 assert all(len(v)==s.response_length for v,s in zip(values,samples))
 varying={}
 for s,v in zip(samples,values):
  if v:varying.setdefault(s.index,set()).add(round(v[0],6))
 summary={'logical_trajectories':64,'turn_chunks':len(samples),'response_tokens':sum(s.response_length for s in samples),'trajectories_with_nonuniform_turn_advantage':sum(len(v)>1 for v in varying.values()),'all_response_shapes_match':True,'source':'recorded JSON turn IDs and response token counts; no checkpoint deserialization'}
 (R/'outputs/artifact_grpo40_20260922/replay_preflight.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
