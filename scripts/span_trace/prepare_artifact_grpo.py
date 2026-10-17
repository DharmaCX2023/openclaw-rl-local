"""Freeze verifier-to-artifact score attribution before a training run."""
import ast,collections,hashlib,json,sys,re
from pathlib import Path
from verifier_dataflow import Analyzer
R=Path(__file__).resolve().parents[2];O=R/'outputs/artifact_grpo40_20260922'
roots=[R/'outputs/clawgym-spantrace-sft20-20260921-spantrace-sft20/span_audit/sessions',Path('/mnt/humf/openclaw-rl-checkpoints/clawgym-spantrace-sft250-20260921/span_audit/sessions')]
known={r['task_id']:r['targets'] for r in json.load(open(R/'outputs/span_file_chain248_20260922/summary.json'))['rows']}
sessions=collections.defaultdict(list)
for root in roots:
 for p in root.glob('*/runtime.json'):
  r=json.load(open(p))
  if 0<=r['rollout_id']<40 and p.with_name('verifier.json').exists():sessions[r['task_id']].append((p.parent,r))
result={};audit=[]
for task,ss in sorted(sessions.items()):
 taskdir=Path(ss[0][1]['task_dir']);code=(taskdir/'reward/check.py').read_text();entry=json.load(open(taskdir/'data_entry.json'));scores={}
 for d,_ in ss:
  v=json.load(open(d/'verifier.json')).get('components')
  if isinstance(v,dict):
   if isinstance(v.get('scores'),dict):v=v['scores']
   for k,x in v.items():
    if isinstance(x,(bool,int,float)) and k not in ['reward','score','total','total_score','overall_score']:scores[k]=x
 prompt=entry['user_query']
 targets=known.get(task)
 if targets is None:
  targets=sorted({p.rstrip('.,;') for p in re.findall(r'(?<![\w.-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,10}(?![\w.-])',prompt) if not p.startswith(('input/','inputs/','reward/','http'))})
 mapping=Analyzer(code,targets,scores).run()
 byartifact={a:{} for a in targets};unmapped=[];shared=[]
 for key,m in mapping.items():
  paths=m['paths']
  if not paths:unmapped.append(key);continue
  if len(paths)>1:shared.append(key)
  for a in paths:byartifact[a][key]=1/len(paths)
 empty=[a for a,v in byartifact.items() if not v]
 # No guessed score or existence score for unresolved targets. Audit explicitly.
 active={a:{'checks':v} for a,v in byartifact.items() if v}
 result[task]={'checker_sha256':hashlib.sha256(code.encode()).hexdigest(),'artifacts':active,'targets':targets,'unmapped_targets':empty,'unmapped_checks':unmapped,'shared_checks':shared,'mapping_evidence':mapping,'task_dir':str(taskdir)}
 audit.append({'task':task,'targets':len(targets),'mapped_A':len(active),'unmapped_targets':empty,'unmapped_checks':len(unmapped),'checks':len(scores),'shared_checks':len(shared)})
 print(task,len(active),'/',len(targets),'A',len(unmapped),'unmapped checks',flush=True)
O.mkdir(exist_ok=True);(O/'artifact_score_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(O/'mapping_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('TOTAL',len(result),'zero mapped tasks',sum(not r['artifacts'] for r in result.values()),'missing targets',sum(len(r['unmapped_targets']) for r in result.values()),flush=True)
