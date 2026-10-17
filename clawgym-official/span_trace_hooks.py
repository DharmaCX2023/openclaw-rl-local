"""Opt-in isolated span experiment; no changes to default rollout behavior."""
import asyncio,hashlib,io,json,os,shlex,shutil,tarfile,time
from pathlib import Path
import clawgym_rl_rollout as R
import clawgym_rl_api_server as A
ROOT=Path(os.environ['OPENCLAW_SPAN_TRACE_ROOT'])
SOURCE=Path(__file__).resolve().parents[1]/'scripts/span_trace'
_INSTALLED=False

def session_dir(sid):
 p=ROOT/'sessions'/hashlib.sha256(sid.encode()).hexdigest()[:24];p.mkdir(parents=True,exist_ok=True);return p

def error(sid,phase,e):
 with (session_dir(sid)/'errors.jsonl').open('a') as f:f.write(json.dumps({'phase':phase,'error':repr(e),'time':time.time()})+'\n')

def capture_snapshot(sid,turn):
 out=session_dir(sid);meta=json.loads((out/'runtime.json').read_text())
 cmd=shlex.join(['docker','exec',meta['container'],'python3','/root/.openclaw/skills/span-trace/snapshot.py'])
 result=R._remote_run(cmd,timeout=150);record=json.loads(result.stdout);record.update(turn=turn,session_id=sid)
 (out/f'turn_{turn:04d}.json').write_text(json.dumps(record,separators=(',',':')))

def install():
 global _INSTALLED
 if _INSTALLED:return
 _INSTALLED=True;ROOT.mkdir(parents=True,exist_ok=True)
 config_orig=R._build_gateway_config
 def config(runtime,port):
  config_orig(runtime,port)
  shutil.copytree(SOURCE,runtime.config_dir/'skills/span-trace',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
  p=runtime.config_dir/'openclaw.json';d=json.loads(p.read_text());d['plugins']={'enabled':True,'load':{'paths':['/root/.openclaw/skills/span-trace']},'entries':{'span-trace':{'enabled':True}}};p.write_text(json.dumps(d))
 R._build_gateway_config=config
 start_orig=R._start_container
 def start(task,rollout_id,task_index,rl_port,session_id,model_id,is_train=False):
  runtime=start_orig(task,rollout_id,task_index,rl_port,session_id,model_id,is_train)
  (session_dir(session_id)/'runtime.json').write_text(json.dumps({'session_id':session_id,'container':runtime.container_name,'remote_root':runtime.remote_root,'task_id':task.task_id,'task_dir':str(task.task_dir),'rollout_id':rollout_id,'task_index':task_index}))
  return runtime
 R._start_container=start
 handle_orig=A.ClawGymRLAPIServer._handle_request
 async def handle(self,body,session_id,turn_type):
  self._assert_session_open(session_id)
  if turn_type=='main':
   turn=len(self._conversation_logs.get(session_id,[]))
   try:await asyncio.to_thread(capture_snapshot,session_id,turn)
   except Exception as e:error(session_id,'turn_snapshot',e)
  return await handle_orig(self,body,session_id,turn_type)
 A.ClawGymRLAPIServer._handle_request=handle
 reward_orig=R._run_reward
 def reward(task,runtime,final):
  value=reward_orig(task,runtime,final)
  out=session_dir(runtime.session_id)
  # Capture raw checker result after official reward, without feeding it to policy.
  try:
   cmd=shlex.join(['docker','exec',runtime.container_name,'python3','/root/.openclaw/workspace/reward/check.py','/root/.openclaw/workspace'])
   p=R._remote_run(cmd,timeout=180,check=False);raw=p.stdout.decode(errors='replace')
   try:components=json.loads(raw)
   except ValueError:components=None
   (out/'verifier.json').write_text(json.dumps({'official_reward':value,'returncode':p.returncode,'components':components,'stdout':raw,'stderr':p.stderr.decode(errors='replace')},ensure_ascii=False))
  except Exception as e:error(runtime.session_id,'verifier',e)
  return value
 R._run_reward=reward
 log_orig=R._write_instance_log
 def log(args,rollout_id,file_name,mode_label,task,result,group_index):
  log_orig(args,rollout_id,file_name,mode_label,task,result,group_index)
  out=session_dir(result.session_id)
  (out/'trajectory.json').write_text(json.dumps({'session_id':result.session_id,'task_id':task.task_id,'rollout_id':rollout_id,'reward':result.reward,'termination_reason':result.termination_reason,'react_steps':result.react_steps,'raw_prompts':result.raw_prompts},ensure_ascii=False))
  try:capture_snapshot(result.session_id,len(result.react_steps))
  except Exception as e:error(result.session_id,'terminal_snapshot',e)
 R._write_instance_log=log
 stop_orig=R._stop_container
 def stop(runtime,log_dir):
  out=session_dir(runtime.session_id);archived=False
  try:
   remote=runtime.remote_root
   if remote:
    R._remote_run(shlex.join(['docker','pause',runtime.container_name]),timeout=30)
    # Only this session's tree; include state traces and terminal workspace once.
    cmd='tar -C '+shlex.quote(remote)+' --exclude=workspace/reward --exclude=workspace/.openclaw_reward_payload.json -czf - state/span_trace state/agents/main/sessions workspace'
    p=R._remote_run(cmd,timeout=300,check=False)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[:1000])
    tmp=out/'evidence.tar.gz.tmp';tmp.write_bytes(p.stdout);tmp.replace(out/'evidence.tar.gz');archived=True
    (out/'COMPLETE').write_text(str(time.time()))
  except Exception as e:error(runtime.session_id,'archive',e)
  # Preserve remote evidence on archive failure instead of deleting it.
  if archived:
   R._remote_run(shlex.join(['docker','unpause',runtime.container_name]),timeout=30,check=False)
   stop_orig(runtime,log_dir)
  else:
   R._remote_run(shlex.join(['docker','unpause',runtime.container_name]),timeout=30,check=False)
   R._remote_run(shlex.join(['docker','stop','-t','5',runtime.container_name]),timeout=30,check=False)
 R._stop_container=stop
