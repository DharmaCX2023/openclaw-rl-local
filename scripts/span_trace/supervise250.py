#!/usr/bin/env python3
"""Continue the authorized run after step 20, monitor through step 250."""
import collections,fcntl,json,re,shlex,shutil,subprocess,time
from pathlib import Path
REPO=Path('/root/openclaw-rl-local')
OLD=REPO/'outputs/clawgym-spantrace-sft20-20260921-spantrace-sft20'
NEW=Path('/mnt/humf/openclaw-rl-checkpoints/clawgym-spantrace-sft250-20260921')
CONTROL=REPO/'outputs/span_trace_250_supervision_20260921'
CONTROL.mkdir(exist_ok=True)
lock=(CONTROL/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
LOG20=REPO/'outputs/span_trace_sft20_launcher_20260921.log'
LOG250=REPO/'outputs/span_trace_sft250_launcher_20260921.log'
TRAIN20='span_trace_sft20_20260921';TRAIN250='span_trace_sft250_20260921';WATCH250='span_trace_watch250_20260921'
def active(name):return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0
def updates(log):
 text=log.read_text(errors='replace') if log.exists() else ''
 return set(map(int,re.findall(r'model\.py:\d+ - step (\d+):.*?train/loss',text)))
def stats(root):
 counts=collections.Counter();errors=[];archives=0;latest=0
 for d in (root/'span_audit/sessions').glob('*'):
  if (d/'errors.jsonl').exists():errors.append(str(d/'errors.jsonl'))
  if (d/'COMPLETE').exists():
   r=json.loads((d/'trajectory.json').read_text());counts[r['rollout_id']]+=1;archives+=(d/'evidence.tar.gz').exists()
  for p in d.glob('turn_*.json'):latest=max(latest,p.stat().st_mtime)
 return counts,errors,archives,latest
def start(name,cmd):subprocess.run(['tmux','new-session','-d','-s',name,cmd],check=True)
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
last_signature=None;last_progress=time.time()
while True:
 try:
  now=time.time();u20=updates(LOG20);u250=updates(LOG250)
  c20,e20,a20,t20=stats(OLD);c250,e250,a250,t250=stats(NEW)
  alerts=[];live20=active(TRAIN20);live250=active(TRAIN250)
  ckpt=OLD/'latest_checkpointed_iteration.txt'
  accepted_path=CONTROL/'ACCEPTED_CAPTURE_GAPS.json'
  accepted=json.loads(accepted_path.read_text()).get('session_ids',[]) if accepted_path.exists() else []
  gap_counts=collections.Counter();accepted_errors=set()
  for sid in accepted:
   session=OLD/'span_audit/sessions'/sid
   if not (session/'COMPLETE').exists() and (session/'runtime.json').exists() and (session/'errors.jsonl').exists():
    gap_counts[json.loads((session/'runtime.json').read_text())['rollout_id']]+=1
    accepted_errors.add(str(session/'errors.jsonl'))
  ready=(not live20 and u20>=set(range(20)) and all(c20[i]+gap_counts[i]==64 for i in range(20)) and a20==sum(c20.values()) and not(set(e20)-accepted_errors) and ckpt.exists() and ckpt.read_text().strip()=='19' and (OLD/'iter_0000019').is_dir())
  marker=CONTROL/'CONTINUATION_STARTED.json'
  if not live20 and not live250 and not marker.exists():
   if ready:
    # No overlapping training launch; use full optimizer/RNG checkpoint, same task schedule.
    if shutil.disk_usage(NEW).free<1024**4:alerts.append('共享盘剩余不足1TiB，续跑尚未启动')
    else:
     cmd='cd '+shlex.quote(str(REPO))+' && bash scripts/run_span_trace_sft250_20260921.sh >> '+shlex.quote(str(LOG250))+' 2>&1'
     start(TRAIN250,cmd);atomic(marker,{'time':now,'checkpoint':str(ckpt.parent),'loaded_rollout':19,'target_rollouts':250});live250=True
   else:alerts.append('20步进程已退出但checkpoint/更新/1280条归档未全部就绪；保留现场，未重跑或覆盖')
  if live250 and not active(WATCH250):
   cmd='cd '+shlex.quote(str(REPO))+' && python -u scripts/span_trace/watch250.py '+shlex.quote(str(NEW/'span_audit'))+' >> outputs/span_trace_watch250_20260921.log 2>&1'
   start(WATCH250,cmd)
  counts=c20+c250;us=u20|u250;signature=(sum(counts.values()),len(us),max(t20,t250))
  if signature!=last_signature:last_signature=signature;last_progress=now
  if now-last_progress>5400 and (live20 or live250):alerts.append('90分钟无新增轨迹、turn快照或训练更新')
  if e20 or e250:alerts.append(f'{len(e20)+len(e250)}个会话存在采集错误')
  free_local=shutil.disk_usage(REPO).free/1024**3;free_shared=shutil.disk_usage(NEW).free/1024**3
  if free_local<80:alerts.append('本地剩余空间不足80GiB')
  if free_shared<1024:alerts.append('共享盘剩余空间不足1TiB')
  finished=(not live20 and not live250 and us>=set(range(250)) and all(counts[i]==64 for i in range(250)) and a20+a250==16000 and not(e20 or e250))
  if marker.exists() and not live250 and not finished:alerts.append('续跑进程退出但未完成250步；保留现场，未从头重跑')
  state={'updated_unix':now,'target_steps':250,'target_tasks':2000,'samples_per_task':8,'target_trajectories':16000,'initial20_active':live20,'continuation_active':live250,'continuation_armed':not marker.exists(),'completed_updates':len(us),'update_ids':sorted(us),'trajectories':sum(counts.values()),'by_rollout':dict(sorted(counts.items())),'archives':a20+a250,'local_free_GiB':round(free_local,1),'shared_free_GiB':round(free_shared,1),'alerts':alerts,'capture_errors':e20+e250,'accepted_capture_gaps':accepted,'verified_complete':finished,'continuation_root':str(NEW)}
  atomic(CONTROL/'STATUS.json',state)
  with (CONTROL/'history.jsonl').open('a') as f:f.write(json.dumps(state,ensure_ascii=False)+'\n')
  lines=['# 250步持续训练巡检','',f"更新UTC：{time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(now))}",f"训练更新：{len(us)}/250；轨迹：{sum(counts.values())}/16000。",f"当前20步任务运行：{live20}；续跑运行：{live250}；续跑触发器待命：{not marker.exists()}。",'', '当前20步不打断，确认iteration 19的完整checkpoint后自动续到总计250步。模型、optimizer、RNG恢复；任务游标由rollout_id决定，8任务×8采样配置不变。每20步保存checkpoint，末步也保存。', '',f'续跑输出：`{NEW}`。','', '每5分钟检查进度、归档和磁盘；统计进程退出会重启。训练故障记录告警并保留现场，不盲目重启覆盖数据。此巡检是后台脚本，不会自动进行模型诊断或发送外部通知。','','## 告警','']
  lines+=['- '+a for a in alerts] or ['无。'];(CONTROL/'STATUS.md').write_text('\n'.join(lines)+'\n')
  if finished:break
 except Exception as e:
  with (CONTROL/'errors.log').open('a') as f:f.write(f'{time.time()} {e!r}\n')
 time.sleep(300)
