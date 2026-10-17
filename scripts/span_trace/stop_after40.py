#!/usr/bin/env python3
"""Stop this specific training job only after iteration 39 is fully saved."""
import json,pickle,re,time,urllib.request
from pathlib import Path
ROOT=Path('/mnt/humf/openclaw-rl-checkpoints/clawgym-spantrace-sft250-20260921')
OUT=Path('/root/openclaw-rl-local/outputs/span_trace_stop40_20260922');OUT.mkdir(exist_ok=True)
JOB='raysubmit_ScpR22tHmqQTEWVx';API='http://127.0.0.1:8275/api/jobs/'+JOB
LOG=ROOT/'logs/train_20260921-spantrace-sft250.log'
def state(**kw):
 p=OUT/'STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(time=time.time(),target_completed_steps=40,target_iteration=39,job=JOB,**kw),indent=2));tmp.replace(p)
def request(suffix='',post=False):
 r=urllib.request.Request(API+suffix,data=b'{}' if post else None,headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(r,timeout=30))
while True:
 try:
  with LOG.open('rb') as f:
   f.seek(max(0,LOG.stat().st_size-4000000));tail=f.read().decode(errors='replace')
  tracker=ROOT/'latest_checkpointed_iteration.txt';ck=ROOT/'iter_0000039'
  ready=tracker.exists() and tracker.read_text().strip()=='39' and re.search(r'successfully saved checkpoint from iteration\s+39\b',tail)
  if ready:
   assert (ck/'common.pt').stat().st_size>0
   assert (ck/'metadata.json').stat().st_size>0
   with (ck/'.metadata').open('rb') as f:meta=pickle.load(f)
   entries=list(meta.storage_data.values());assert entries
   for e in entries:
    path=ck/e.relative_path
    assert path.stat().st_size>=e.offset+e.length,(str(path),e.offset,e.length)
   files={str(p.relative_to(ck)):p.stat().st_size for p in ck.rglob('*') if p.is_file()}
   (OUT/'CHECKPOINT_VERIFIED.json').write_text(json.dumps(dict(iteration=39,path=str(ck),files=files,storage_entries=len(entries),verified_at=time.time()),indent=2))
   result=request('/stop',True);state(status='stop_requested',checkpoint=str(ck),response=result)
   while True:
    j=request();status=j['status']
    if status in ['STOPPED','SUCCEEDED','FAILED']:
     state(status='stopped_after_verified_checkpoint',ray_status=status,checkpoint=str(ck));break
    time.sleep(5)
   break
  j=request()
  if j['status'] not in ['RUNNING','PENDING']:
   state(status='training_exited_before_checkpoint_verification',ray_status=j['status']);break
  steps=re.findall(r'model\.py:\d+ - step (\d+):',tail)
  state(status='waiting_for_checkpoint',last_logged_step=int(steps[-1]) if steps else None)
 except Exception as e:
  state(status='check_error_will_retry',error=repr(e))
 time.sleep(10)
