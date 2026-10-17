#!/usr/bin/env python3
"""Durable local aggregation while the independent training tmux session runs."""
import json,subprocess,sys,time
from pathlib import Path
from analyze_writer_chain import main as unused
ROOT=Path(sys.argv[1]);ROOT.mkdir(parents=True,exist_ok=True)
script=Path(__file__).with_name('analyze_writer_chain.py');previous=-1
while True:
 complete=sorted((ROOT/'sessions').glob('*/COMPLETE'));n=len(complete)
 active=subprocess.run(['tmux','has-session','-t','span_trace_sft20_20260921'],capture_output=True).returncode==0
 if n!=previous or not active:
  p=subprocess.run([sys.executable,str(script),str(ROOT)],capture_output=True,text=True)
  (ROOT/'aggregation.log').write_text(p.stdout+'\n'+p.stderr)
  analyses=[]
  for f in (ROOT/'sessions').glob('*/span_analysis_writer_chain.json'):
   try:analyses.append((f,json.loads(f.read_text())))
   except ValueError:pass
  analyses.sort(key=lambda x:(json.loads(x[0].with_name('trajectory.json').read_text())['rollout_id'],json.loads(x[0].with_name('runtime.json').read_text())['task_index']))
  chosen=[];tasks=set()
  for f,r in analyses:
   if r['task_id'] in tasks:continue
   chosen.append((f,r));tasks.add(r['task_id'])
   if len(chosen)==2:break
  if len(chosen)<2:
   chosen += [pair for pair in analyses if pair not in chosen][:2-len(chosen)]
  lines=['# 文件操作依赖实验：两条真实轨迹审计','','来源：从SFT模型开始的20步RL实验。下面只使用已完成并成功归档的真实训练轨迹。起点沿实际写入PID执行的workspace脚本、进程执行祖先及脚本此前写入者递归回溯，普通读取不建依赖边；终点为记录到的最后写入action所属turn。脚本内部各段逻辑不做语义拆分。']
  for f,r in chosen:
   lines += ['',f"## {r['task_id']}（{r['turns']} turn，reward={r['reward']:.4f}）",'',f"完整证据目录：`{f.parent}`。",'', '| 文件 | 首次直接写入turn | 依赖回溯起点 | 最后写入turn |','|---|---:|---:|---:|']
   for v in r['files']:lines.append(f"| `{v['path']}` | {v['first_write']} | {v['start']} | {v['last_write']} |")
   lines+=['','| turn | toolcall / 操作 | 直接记录的目标文件 | 生成依赖脚本 |','|---|---|---|---|']
   for key,a in sorted(r['actions'].items(),key=lambda x:(x[1]['turn'] or 0,x[1]['start_ms'])):
    writes=[(v['path'],w) for v in r['files'] for w in v['writes'] if w['key']==key]
    targets=sorted({p for p,w in writes});deps=sorted({p for v,w in writes for p in w.get('dependencies',{})})
    command=a['params'].get('command',a['params'].get('path',a['params'].get('file_path','')));command=str(command).replace('\n',' ').replace('|','\\|')[:180]
    lines.append(f"| {a['turn']} | `{a['tool']}`：{command} | {', '.join(targets) or '无已确认写入'} | {', '.join(deps) or '—'} |")
   lines+=['',f"解析审计：`{json.dumps(r['issues'],ensure_ascii=False)}`。若存在未解析事件，统计不能声称覆盖了全部实际写入。"]
  if not chosen:lines+=['','尚无完成并归档的轨迹。']
  (ROOT/'TWO_TRAJECTORIES.md').write_text('\n'.join(lines)+'\n');previous=n
 (ROOT/'STATUS.json').write_text(json.dumps({'updated_unix':time.time(),'training_tmux_active':active,'completed_archives':n,'analysis_returncode':p.returncode if 'p' in locals() else None},indent=2))
 if not active:break
 time.sleep(30)
