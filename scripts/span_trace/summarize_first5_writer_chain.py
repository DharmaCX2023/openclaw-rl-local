import json,statistics,collections
from pathlib import Path
ROOT=Path('/root/openclaw-rl-local/outputs/clawgym-spantrace-sft20-20260921-spantrace-sft20/span_audit')
rows=[]
for p in (ROOT/'sessions').glob('*/runtime.json'):
 meta=json.loads(p.read_text())
 if not 0<=meta['rollout_id']<5 or not p.with_name('COMPLETE').exists():continue
 r=json.loads(p.with_name('span_analysis_writer_chain.json').read_text());r['rollout_id']=meta['rollout_id'];r['task_index']=meta['task_index'];r['evidence_dir']=str(p.parent);rows.append(r)
assert len(rows)==320
assert collections.Counter(r['rollout_id'] for r in rows)=={i:64 for i in range(5)}
def mean(xs):
 xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
def summarize(rs):
 out={'n':len(rs),'task_count':len({r['task_id'] for r in rs}),'mean_turns':mean([r['turns'] for r in rs]),'metrics':{},'file_counts':{}}
 for group in ['requested_artifacts','all_modified_files']:
  out['metrics'][group]={mode:{k:mean([r['metrics'][group][mode][k] for r in rs]) for k in ['segments','nonzero_segments','turns_per_segment','nonzero_length']} for mode in ['direct_write','dependency_backtrack']}
  selected=[(r,f) for r in rs for f in r['files'] if group=='all_modified_files' or f['path'] in r['requested_paths']]
  changed=[(r,f) for r,f in selected if f['start']<f['first_write']]
  out['file_counts'][group]={'observed_files':len(selected),'files_with_dependencies':sum(any(w['dependencies'] for w in f['writes']) for _,f in selected),'start_moved_earlier':len(changed),'trajectories_start_moved':len({r['session_id'] for r,f in changed}),'mean_start_advance_affected':mean([f['first_write']-f['start'] for _,f in changed]),'mean_span_length_before':mean([f['last_write']-f['first_write']+1 for _,f in selected]),'mean_span_length_after':mean([f['last_write']-f['start']+1 for _,f in selected]),'trajectories_no_requested_writes':sum(not any(f['path'] in r['requested_paths'] for f in r['files']) for r in rs)}
 return out
s=summarize(rows);s['issues']=dict(sum((collections.Counter(r['issues']) for r in rows),collections.Counter()));s['by_rollout']={str(i):summarize([r for r in rows if r['rollout_id']==i]) for i in range(5)}
s['bins']={f'{a}–{b}':summarize([r for r in rows if a<=r['turns']<=b]) for a,b in [(1,5),(6,10),(11,20),(21,40),(41,60),(61,100)] if any(a<=r['turns']<=b for r in rows)}
examples=[]
for r in rows:
 for f in r['files']:
  if f['path'] not in r['requested_paths'] or f['start']>=f['first_write']:continue
  links=[{'turn':w['turn'],'origin_turn':w['origin_turn'],'syscall':w['syscall'],'writer_pid':w.get('pid'),'writer_execution':w.get('writer_execution'),'dependencies':w['dependencies'],'tool':r['actions'][w['key']]['tool'],'command':r['actions'][w['key']]['params'].get('command')} for w in f['writes'] if w['dependencies']]
  if not any(e['task_id']==r['task_id'] for e in examples):examples.append({'task_id':r['task_id'],'sample':r['task_index'],'rollout_id':r['rollout_id'],'file':f['path'],'first_write':f['first_write'],'start':f['start'],'last_write':f['last_write'],'evidence_dir':r['evidence_dir'],'links':links[:2]})
s['examples']=examples[:5]
(ROOT/'FIRST_5_ROLLOUTS_320_WRITER_CHAIN.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in s.items() if k not in ['by_rollout','bins']},ensure_ascii=False,indent=2))
lines=['# 前5轮320条：实际写入与生成依赖回溯','','固定rollout 0–4，共40个任务×8采样=320条完整轨迹；与持续SUMMARY.json分开，避免分母随训练变化。平均轨迹长度 %.2f turn。'%s['mean_turns'],'','覆盖切分按连续active artifact身份集合变化，包含零覆盖段；非零段长只在有非零段的轨迹上平均。表中均为轨迹级宏平均。','','| 文件范围 | 起点口径 | 总段数/轨迹 | 非零段数/轨迹 | turn/段 | 非零段平均长度 |','|---|---|---:|---:|---:|---:|']
for g,gl in [('requested_artifacts','任务显式路径候选'),('all_modified_files','全部观测修改文件（含中间脚本）')]:
 for m,ml in [('direct_write','首次实际写入'),('dependency_backtrack','沿脚本依赖回溯最早写入')]:
  v=s['metrics'][g][m];lines.append(f"| {gl} | {ml} | {v['segments']:.3f} | {v['nonzero_segments']:.3f} | {v['turns_per_segment']:.3f} | {v['nonzero_length']:.3f} |")
lines+=['','终点均为目标文件最后一次观测到的写入，祖先脚本后续无关修改不会延长目标终点。每次目标写入都使用截至当时已记录的脚本写入依赖，再对这些写入及祖先取最早起点；不只回溯第一次输出写入。','','## 依赖覆盖','','```json',json.dumps(s['file_counts'],ensure_ascii=False,indent=2),'```','','## 分桶：任务显式路径候选','','| turn区间 | 轨迹数 | 直接写入 turn/段 | 回溯 turn/段 | 直接写入非零段长 | 回溯非零段长 |','|---|---:|---:|---:|---:|---:|']
for k,v in s['bins'].items():
 a,b=v['metrics']['requested_artifacts'].values();lines.append(f"| {k} | {v['n']} | {a['turns_per_segment']:.3f} | {b['turns_per_segment']:.3f} | {a['nonzero_length']:.3f} | {b['nonzero_length']:.3f} |")
lines+=['','## 证据口径和限制','','exec的写入归属来自每toolcall的strace及其子进程记录，不是command里找文件名。直接write/edit来自工具钩子目标与前后文件状态。依赖边仅来自实际写入PID的execve入口文件和fork/clone继承的执行祖先，再连接该文件执行前已记录的写入者，递归取最早写入turn。普通read/open/cat不建边，同一toolcall无关兄弟进程不共享依赖。直接write/edit没有Linux写入PID证据，仍单独使用工具钩子与快照确认。动态import/source、stdin脚本、管道数据来源与未解析解释器参数不猜测回溯；缺失记录单独审计。路径级历史取截至执行时最早写入，尚未按删除重建/inode版本拆分。','','任务artifact清单仍由prompt显式路径提取，可能漏掉动态命名文件；全部修改文件统计另列。相对路径rename、未解析系统调用等不会强行推断，以下审计计数不能理解成完整性无缺。','','```json',json.dumps(s['issues'],ensure_ascii=False,indent=2),'```','','## 实际回溯例子','','```json',json.dumps(s['examples'],ensure_ascii=False,indent=2),'```']
(ROOT/'FIRST_5_ROLLOUTS_320_WRITER_CHAIN.md').write_text('\n'.join(lines)+'\n')
