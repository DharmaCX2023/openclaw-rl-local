#!/usr/bin/env python3
"""Derive file spans from per-tool syscall evidence. Never executes archived code."""
import argparse,collections,hashlib,io,json,posixpath,re,statistics,tarfile
from pathlib import Path
from writer_chain import process_events, build_files
WORK='/root/.openclaw/workspace/'
SYSTEM={'AGENTS.md','BOOTSTRAP.md','HEARTBEAT.md','IDENTITY.md','MEMORY.md','SOUL.md','TOOLS.md','USER.md','.openclaw_reward_payload.json'}
SCRIPT_EXT=('.py','.sh','.js','.mjs','.cjs','.ts','.pl','.rb')

def relative(p):
 p=p.removesuffix(' (deleted)')
 if not p.startswith(WORK):return None
 p=posixpath.normpath(p[len(WORK):])
 if p.startswith(('../','reward/','.git/','.openclaw/','.venv/')) or p in SYSTEM or '__pycache__' in p:return None
 return p

def parse_trace(text,key,pid):
 events=[];unparsed=0
 for line in text.splitlines():
  m=re.match(r'^(\d+\.\d+)\s+(\w+)\((.*)\)\s+=\s+(-?\d+)(.*)$',line)
  if not m:
   if '<unfinished ...>' in line or 'resumed>' in line:unparsed+=1
   continue
  when,name,args,ret,rest=m.groups();ret=int(ret)
  if ret<0:continue
  paths=re.findall(r'<([^>]+)>',args);returned=re.findall(r'<([^>]+)>',rest)
  def add(p,kind):
   p=relative(p)
   if p:events.append({'time':float(when),'key':key,'pid':pid,'path':p,'kind':kind,'syscall':name})
  if name in ['open','openat','openat2','creat'] and returned:
   p=returned[-1]
   if 'O_TRUNC' in args or name=='creat':add(p,'write')
   elif 'O_CREAT' in args:add(p,'create_intent')
   # Read-only opens never create dependencies.
  elif name in ['write','writev','pwrite64','pwritev'] and ret>0 and paths:add(paths[0],'write')
  elif name=='ftruncate' and paths:add(paths[0],'write')
  elif name in ['copy_file_range','sendfile','splice'] and ret>0:
   if name=='sendfile' and paths:add(paths[0],'write')
   elif len(paths)>=2:add(paths[1],'write')
  elif name in ['rename','renameat','renameat2']:
   # Absolute paths only; unresolved relative rename is audited, not guessed.
   literals=re.findall(r'"((?:[^"\\]|\\.)*)"',args)
   if literals and literals[-1].startswith('/') :add(literals[-1],'write')
   else:unparsed+=1
 return events,unparsed

def segmentation(spans,T):
 runs=[]
 for t in range(1,T+1):
  active=tuple(p for p,s,e in spans if s<=t<=e)
  if runs and runs[-1][0]==active:runs[-1][1]+=1
  else:runs.append([active,1])
 nz=[n for a,n in runs if a]
 return {'segments':len(runs),'nonzero_segments':len(nz),'turns_per_segment':T/len(runs) if runs else None,'nonzero_length':statistics.mean(nz) if nz else None}

def analyze(directory):
 d=Path(directory)
 signature={'analyzer_sha256':hashlib.sha256((Path(__file__).read_bytes()+Path(__file__).with_name('writer_chain.py').read_bytes())).hexdigest(),'evidence_mtime_ns':(d/'evidence.tar.gz').stat().st_mtime_ns,'trajectory_mtime_ns':(d/'trajectory.json').stat().st_mtime_ns}
 cached=d/'span_analysis_writer_chain.json'
 if cached.exists():
  old=json.loads(cached.read_text())
  if old.get('analysis_signature')==signature:return old
 traj=json.loads((d/'trajectory.json').read_text());T=len(traj['react_steps']);turns={}
 for step in traj['react_steps']:
  for c in step.get('tool_calls',[]):
   if c['id'] in turns:raise ValueError('Duplicate toolcall ID: cannot attribute uniquely')
   turns[c['id']]=step['step']
 blobs={}
 with tarfile.open(d/'evidence.tar.gz') as t:
  for m in t:
   if m.isfile() and m.name.startswith('state/span_trace/'):blobs[m.name.split('/')[-1]]=t.extractfile(m).read()
 actions={};events=[];processes=[];exec_meta={};issues=collections.Counter()
 for name,b in blobs.items():
  if name.endswith('.action.json'):
   a=json.loads(b);a['turn']=turns.get(a['tool_call_id']);actions[a['key']]=a
 for key,a in actions.items():
  if a['turn'] is None:issues['unmapped_toolcall']+=1;continue
  before=json.loads(blobs.get(key+'.before.json',b'{"files":{}}'))['files'];after=json.loads(blobs.get(key+'.after.json',b'{"files":{}}'))['files']
  if a['tool']=='exec':
   exec_meta[key]=json.loads(blobs.get(key+'.exec.json',b'{}'))
   for name,b in blobs.items():
    if name.startswith(key+'.trace.'):
     processes.extend(process_events(b.decode(errors='replace'),key,name.rsplit('.',1)[-1]))
     es,gaps=parse_trace(b.decode(errors='replace'),key,name.rsplit('.',1)[-1]);issues['unparsed_syscall_lines']+=gaps
     for e in es:
      if e['kind']=='create_intent':
       if e['path'] in before:continue
       e['kind']='write';e['creation_candidate']=True
      e['turn']=a['turn'];events.append(e)
  elif a['tool'] in ['write','edit']:
   result=json.loads(blobs.get(key+'.result.json',b'{"error":"missing result"}'))
   p=a['params'].get('path',a['params'].get('file_path'));p=relative(p if p and p.startswith('/') else WORK+(p or ''))
   if p and not result.get('error') and not (result.get('result') or {}).get('isError') and p in after and before.get(p)!=after.get(p):events.append({'time':result['end_ms']/1000,'key':key,'path':p,'kind':'write','turn':a['turn'],'syscall':'direct_'+a['tool']})
   elif p:issues['direct_write_not_confirmed_by_snapshot']+=1
 # Origin toolcall and actual operation turn are separate for background children.
 boundaries=[]
 for f in d.glob('turn_*.json'):
  record=json.loads(f.read_text());boundaries.append((record['timestamp_ns']/1e9,record['turn']))
 boundaries.sort();mapped=[]
 for e in events:
  e['origin_turn']=e['turn']
  if e['syscall'].startswith('direct_'):
   mapped.append(e);continue
  if boundaries:
   hit=next((turn for timestamp,turn in boundaries if timestamp>=e['time']),None)
   if hit is None:issues['events_after_final_turn_snapshot']+=1;continue
   if hit<e['origin_turn']:issues['clock_or_boundary_mismatch']+=1;continue
   e['turn']=hit
  else:issues['missing_turn_time_boundaries']+=1
  mapped.append(e)
 events=mapped
 files=build_files(events,processes,exec_meta,relative,issues)
 prompt=traj.get('raw_prompts',{}).get('user_prompt','');paths={m.group().removeprefix('./') for m in re.finditer(r'(?<![\w.-])(?:[\w.-]+/)+[\w.-]+\.[A-Za-z0-9]{1,10}(?![\w.-])',prompt)};paths={p for p in paths if relative(WORK+p)}
 groups={'requested_artifacts':[v for p,v in files.items() if p in paths],'all_modified_files':list(files.values())};metrics={}
 for group,vs in groups.items():
  metrics[group]={mode:segmentation([(v['path'],v[start],v['last_write']) for v in vs],T) for mode,start in [('direct_write','first_write'),('dependency_backtrack','start')]}
 result={'dependency_method':'executed_file_writer_chain_v1','analysis_signature':signature,'task_id':traj['task_id'],'session_id':traj['session_id'],'turns':T,'reward':traj['reward'],'toolcalls':len(turns),'traced_actions':len(actions),'issues':dict(issues),'files':list(files.values()),'requested_paths':sorted(paths),'metrics':metrics,'actions':actions}
 (d/'span_analysis_writer_chain.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();root=Path(args.root);results=[];errors=[]
 for d in sorted((root/'sessions').glob('*')):
  if not (d/'COMPLETE').exists() or not (d/'trajectory.json').exists():continue
  try:results.append(analyze(d))
  except Exception as e:errors.append({'session':d.name,'error':repr(e)})
 def avg(xs):
  xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
 summary={'trajectories':len(results),'errors':errors,'metrics':{}}
 for g in ['requested_artifacts','all_modified_files']:
  summary['metrics'][g]={m:{k:avg([r['metrics'][g][m][k] for r in results]) for k in ['segments','nonzero_segments','turns_per_segment','nonzero_length']} for m in ['direct_write','dependency_backtrack']}
 (root/'SUMMARY_WRITER_CHAIN.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
