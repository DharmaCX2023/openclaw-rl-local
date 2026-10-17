"""Persistently monitor this exact 40-step run; never restart unrelated jobs."""
import json,os,re,subprocess,time
from pathlib import Path
from artifact_training_digest import TrainingDigest
ROOT=Path(os.environ.get('SAVE_CKPT','/mnt/humf/openclaw-rl-checkpoints/clawgym-artifact-grpo40-20260922'))
OUT=Path('/root/openclaw-rl-local/outputs/artifact_grpo40_20260922')
LOG=ROOT/'logs'/('train_'+os.environ.get('RUN_STAMP','20260922-artifact-grpo40')+'.log')
digest=TrainingDigest(ROOT,LOG)
while True:
 try:
  digest.poll()
  text=''
  if LOG.exists():
   with LOG.open('rb') as f:
    f.seek(max(0,LOG.stat().st_size-20000000));text=f.read().decode(errors='replace')
  # A resumed attempt may have lost updates after its last checkpoint.
  # Do not report the abandoned attempt's higher step as current progress.
  progress_text=text.rsplit('Runtime env is setting up.',1)[-1]
  ids=sorted(set(map(int,re.findall(r'model\.py:\d+ - step (\d+):.*?train/loss',progress_text))))
  current_file=OUT/'CURRENT_RUN.json'
  if current_file.exists():
   current=json.loads(current_file.read_text())
   if current.get('run_root')==str(ROOT) and current.get('restored_completed_updates'):
    ids.append(int(current['restored_completed_updates'])-1)
  active=subprocess.run(['tmux','has-session','-t','artifact_grpo40_20260922'],capture_output=True).returncode==0
  ck=ROOT/'iter_0000039';tracker=ROOT/'latest_checkpointed_iteration.txt'
  saved=tracker.exists() and tracker.read_text().strip()=='39' and bool(re.search(r'successfully saved checkpoint from iteration\s+39\b',text))
  verified=False
  if saved:
   shards=list(ck.glob('*.distcp'))
   verified=len(shards)>=8 and all(p.stat().st_size>0 for p in shards) and all((ck/name).exists() and (ck/name).stat().st_size>0 for name in ['common.pt','.metadata','metadata.json'])
   if verified:json.loads((ck/'metadata.json').read_text())
  audits=list((ROOT/'artifact_advantage_audit').glob('*.json'))
  state=dict(updated=time.time(),run_root=str(ROOT),target_updates=40,last_completed_step=max(ids) if ids else None,training_active=active,advantage_audit_batches=len(audits),checkpoint_verified=verified,checkpoint=str(ck) if verified else None,status='complete' if verified and not active else ('running' if active else 'exited_before_verified_finish'))
  tmp=OUT/'STATUS.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(OUT/'STATUS.json')
  if verified and not active:
   (OUT/'COMPLETE.json').write_text(json.dumps(state,indent=2));break
  if not active:break
 except Exception as e:
  (OUT/'watch_error.log').write_text(repr(e))
 time.sleep(30)
