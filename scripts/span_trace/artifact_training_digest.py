"""Persist one report per four actual optimizer updates; no LLM/API calls."""
import ast,json,re,statistics
from pathlib import Path

class TrainingDigest:
 def __init__(self,root,log):
  self.root=Path(root);self.log=Path(log);self.out=self.root/'progress_summaries'
  self.out.mkdir(exist_ok=True);self.offset=0;self.pending=b'';self.steps={}
 def poll(self):
  if not self.log.exists():return
  if self.log.stat().st_size<self.offset:self.offset=0;self.pending=b'';self.steps={}
  with self.log.open('rb') as f:
   f.seek(self.offset)
   while True:
    chunk=f.read(1024*1024)
    if not chunk:break
    lines=(self.pending+chunk).split(b'\n');self.pending=lines.pop()
    for raw in lines:
     line=re.sub(r'\x1b\[[0-9;]*m','',raw.decode(errors='replace'))
     match=re.search(r'model\.py:\d+ - step (\d+): (\{.*\})',line)
     if match:
      metrics=ast.literal_eval(match.group(2))
      if 'train/loss' in metrics:self.steps[int(match.group(1))]=metrics
   self.offset=f.tell()
  for completed in range(4,41,4):
   if all(i in self.steps for i in range(completed)) and not (self.out/f'step_{completed:02d}.json').exists():
    self.write(completed)
 def write(self,completed):
  window=set(range(completed-4,completed));audits={}
  for p in (self.root/'artifact_advantage_audit').glob('*.json'):
   d=json.loads(p.read_text());sessions=d.get('sessions',[])
   if not sessions:continue
   m=re.match(r'train-task-(\d+)-',sessions[0])
   if m and int(m.group(1)) in window:
    step=int(m.group(1));attempt=max([int(x) for sid in sessions for x in re.findall(r'-retry(\d+)-',sid)] or [0])
    if step not in audits or attempt>audits[step][0]:audits[step]=(attempt,d)
  rewards=[];sizes=[]
  import hashlib
  for _,d in audits.values():
   sizes.append(len(d['trajectories']))
   for sid in d['sessions']:
    p=self.root/'span_audit/sessions'/hashlib.sha256(sid.encode()).hexdigest()[:24]/'trajectory.json'
    if p.exists():rewards.append(float(json.loads(p.read_text())['reward']))
  retries=[]
  for p in (self.root/'rollout_retries').glob('*.json'):
   d=json.loads(p.read_text())
   if d['rollout_id'] in window:retries.append(d)
  tracker=self.root/'latest_checkpointed_iteration.txt';checkpoint=None
  if tracker.exists():
   value=tracker.read_text().strip()
   if value.isdigit():
    ck=self.root/f'iter_{int(value):07d}'
    if ck.is_dir():checkpoint=str(ck)
  ms=[self.steps[i] for i in sorted(window)]
  result=dict(completed_updates=completed,target_updates=40,window_steps=sorted(window),run_root=str(self.root),trajectory_reward_mean=statistics.mean(rewards) if rewards else None,reward_trajectory_count=len(rewards),valid_trajectories_per_step=sizes,mean_loss=statistics.mean(m['train/loss'] for m in ms),mean_grad_norm=statistics.mean(m['train/grad_norm'] for m in ms),retry_count=len(retries),checkpoint_reported_by_tracker=checkpoint)
  report=f"## 已完成 {completed}/40 步（step {completed-4}–{completed-1}）\n\n"
  reward=f"{result['trajectory_reward_mean']:.4f}" if rewards else '待记录'
  report+=f"- 近 4 步平均 outcome reward：{reward}（{len(rewards)} 条轨迹）。\n- 每步有效轨迹数：{sizes}。\n- 平均 loss：{result['mean_loss']:.6f}；平均梯度范数：{result['mean_grad_norm']:.4f}。\n- 近 4 步整批重试：{len(retries)} 次。\n- Checkpoint：{checkpoint or '尚未保存（计划第 20、40 步保存）'}。\n"
  (self.out/f'step_{completed:02d}.md').write_text(report)
  # JSON is the completion marker; a failed write is retried next poll.
  (self.out/f'step_{completed:02d}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
  (self.out/'LATEST.md').write_text(report)
  reports=sorted(self.out.glob('step_*.md'))
  (self.out/'SUMMARY.md').write_text('# 每 4 次参数更新摘要\n\n'+''.join(p.read_text()+'\n' for p in reports))
  print(f'[TrainingDigest] completed={completed}/40 report={self.out}/step_{completed:02d}.md',flush=True)
