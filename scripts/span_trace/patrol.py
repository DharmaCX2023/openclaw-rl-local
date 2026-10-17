#!/usr/bin/env python3
"""Five-minute unattended health checks. Never restarts training blindly."""
import collections,json,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(sys.argv[1]).resolve();REPO=Path(__file__).resolve().parents[2];LOG=REPO/'outputs/span_trace_sft20_launcher_20260921.log'
TRAIN='span_trace_sft20_20260921';WATCH='span_trace_watch_20260921'
def active(name):return subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode==0
previous=None;last_progress=time.time()
while True:
 try:
  now=time.time();live=active(TRAIN)
  if live and not active(WATCH):
   subprocess.run(['tmux','new-session','-d','-s',WATCH,f'cd {REPO} && python scripts/span_trace/watch.py {ROOT} > outputs/span_trace_watch_20260921.log 2>&1'],check=True)
  text=LOG.read_text(errors='replace') if LOG.exists() else ''
  updates=sorted({int(m) for m in re.findall(r'model\.py:\d+ - step (\d+):.*?train/loss',text)})
  sessions=list((ROOT/'sessions').glob('*'));done=[d for d in sessions if (d/'COMPLETE').exists() and (d/'trajectory.json').exists()]
  counts=collections.Counter();verifier_ok=0;trace_ok=0;capture_errors=[]
  for d in done:
   tr=json.loads((d/'trajectory.json').read_text());counts[tr['rollout_id']]+=1
   v=d/'verifier.json'
   if v.exists():
    value=json.loads(v.read_text());verifier_ok+=value.get('returncode')==0 and isinstance(value.get('components'),dict)
   if (d/'evidence.tar.gz').exists():trace_ok+=1
  for d in sessions:
   if (d/'errors.jsonl').exists():capture_errors.append(str(d/'errors.jsonl'))
  snapshots=sum(len(list(d.glob('turn_*.json'))) for d in sessions)
  signature=(len(done),snapshots,len(updates))
  if signature!=previous:previous=signature;last_progress=now
  alerts=[]
  if now-last_progress>5400 and live:alerts.append('90分钟没有新增轨迹、turn快照或训练更新；需要检查')
  if capture_errors:alerts.append(f'{len(capture_errors)}个会话有采集错误记录')
  free=shutil.disk_usage(ROOT).free/1024**3
  if free<80:alerts.append(f'磁盘剩余仅{free:.1f}GiB')
  complete=not live and set(updates)>=set(range(20)) and all(counts.get(i,0)==64 for i in range(20)) and verifier_ok==1280 and trace_ok==1280 and not capture_errors
  if not live and not complete:alerts.append('训练进程已结束但未满足20步/1280条完整记录；保留现场，不从头盲目重跑')
  state={'timestamp_unix':now,'training_active':live,'completed_training_updates':len(updates),'update_ids':updates,'completed_trajectories':len(done),'by_rollout':dict(counts),'turn_snapshots':snapshots,'verifier_records_ok':verifier_ok,'evidence_archives':trace_ok,'disk_free_GiB':round(free,1),'seconds_without_progress':round(now-last_progress),'capture_error_files':capture_errors,'alerts':alerts,'verified_complete':complete}
  (ROOT/'PATROL_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
  with (ROOT/'patrol_history.jsonl').open('a') as f:f.write(json.dumps(state,ensure_ascii=False)+'\n')
  lines=['# 20步RL实验巡检状态','',f"更新时间（UTC）：{time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(now))}",f"训练进程：{'运行中' if live else '已结束'}；训练更新 {len(updates)}/20；完整轨迹 {len(done)}/1280。",f"分项verifier记录 {verifier_ok}/{len(done)}；系统调用证据归档 {trace_ok}/{len(done)}；逐turn快照 {snapshots}。",f"剩余磁盘：{free:.1f} GiB。",'', '## 当前告警','']
  lines += ['- '+x for x in alerts] if alerts else ['无巡检告警。']
  lines+=['','## 结果文件','','- `SUMMARY_WRITER_CHAIN.json`：仅执行文件写入链的持续分段统计。','- `TWO_REVIEWED_TRAJECTORIES.md`：已人工核验的两条轨迹。','- `TWO_TRAJECTORIES.md`：自动生成的轨迹执行摘要。','- `sessions/`：逐轨迹文件元数据、完整轨迹、verifier和系统调用证据。','','巡检每5分钟运行一次；统计进程退出时自动恢复。训练失败不盲目从头重跑，故障及缺失记录会显式保留。']
  (ROOT/'RUN_STATUS.md').write_text('\n'.join(lines)+'\n')
  if not live:
   subprocess.run([sys.executable,str(REPO/'scripts/span_trace/analyze_writer_chain.py'),str(ROOT)],check=False)
   break
 except Exception as e:
  with (ROOT/'patrol_errors.log').open('a') as f:f.write(f'{time.time()} {e!r}\n')
 time.sleep(300)
