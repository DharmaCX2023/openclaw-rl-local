"""Replay fixed first five rollouts without modifying training or old caches."""
import collections,concurrent.futures,hashlib,json,statistics,tarfile,time,posixpath
from pathlib import Path
from writer_chain import process_events,build_files as old_build
from command_input_chain import input_events,input_directory_writes,build_files
from analyze_writer_chain import relative,segmentation
ROOT=Path('/root/openclaw-rl-local/outputs/clawgym-spantrace-sft20-20260921-spantrace-sft20/span_audit')
OUT=Path('/root/openclaw-rl-local/outputs/span_command_inputs_320_20260921')

def dependency_relative(p):
 if p.startswith('/root/.openclaw/workspace/'):
  value=posixpath.normpath(p[len('/root/.openclaw/workspace/'):].removesuffix(' (deleted)'))
  if value.startswith(('input/','inputs/')) and '__pycache__' not in value:return value
 return relative(p)

def one(path):
 old=json.loads(path.read_text());d=path.parent
 writes=[{k:v for k,v in w.items() if k not in ('dependencies','writer_execution','provenance')} for f in old['files'] for w in f['writes']]
 # Existing caches retain authoritative write events/endpoints from the original
 # capture. Only dependencies change in this experiment.
 ps=[];ins=[];meta={};count=0;extra=[];snapshots={};toolresults={}
 with tarfile.open(d/'evidence.tar.gz') as archive:
  for m in archive:
   if not m.isfile() or not m.name.startswith('state/span_trace/'):continue
   name=m.name.rsplit('/',1)[-1]
   if name.endswith('.exec.json'):
    meta[name[:-10]]=json.load(archive.extractfile(m))
   elif name.endswith(('.before.json','.after.json')):
    snap=json.load(archive.extractfile(m));snap['files']={k:v for k,v in snap['files'].items() if k.startswith(('input/','inputs/'))};snapshots[name]=snap
   elif name.endswith('.result.json'):toolresults[name[:-12]]=json.load(archive.extractfile(m))
   elif '.trace.' in name:
    key,pid=name.rsplit('.trace.',1);text=archive.extractfile(m).read().decode(errors='replace')
    ps.extend(process_events(text,key,pid));ins.extend(input_events(text,key,pid,dependency_relative));extra.extend(input_directory_writes(text,key,pid));count+=1
 missing_meta=sum(k not in meta for k,a in old['actions'].items() if a['tool']=='exec')
 for k,a in old['actions'].items():
  if a['tool']=='exec':meta.setdefault(k,{})
 boundaries=sorted((json.loads(p.read_text())['timestamp_ns']/1e9,json.loads(p.read_text())['turn']) for p in d.glob('turn_*.json'))
 mapped=[]
 for e in extra:
  a=old['actions'].get(e['key'],{});origin=a.get('turn')
  if origin is None:continue
  before=snapshots.get(e['key']+'.before.json',{}).get('files',{})
  if e['creation_candidate'] and e['path'] in before:continue
  turn=next((t for ts,t in boundaries if ts>=e['time']),None) if boundaries else origin
  if turn is None or turn<origin:continue
  e.update(turn=turn,origin_turn=origin);mapped.append(e)
 for key,a in old['actions'].items():
  if a['tool'] not in ('write','edit') or a.get('turn') is None:continue
  p=a['params'].get('path',a['params'].get('file_path',''));p=dependency_relative(p if p.startswith('/') else '/root/.openclaw/workspace/'+p)
  if not p or not p.startswith(('input/','inputs/')):continue
  result=toolresults.get(key,{});before=snapshots.get(key+'.before.json',{}).get('files',{});after=snapshots.get(key+'.after.json',{}).get('files',{})
  if result and not result.get('error') and not (result.get('result') or {}).get('isError') and p in after and before.get(p)!=after[p]:
   mapped.append(dict(time=result['end_ms']/1000,key=key,path=p,kind='write',turn=a['turn'],origin_turn=a['turn'],syscall='direct_'+a['tool']))
 baseline=old_build(writes,ps,meta,relative,collections.Counter())
 assert {p:f['start'] for p,f in baseline.items()}=={f['path']:f['start'] for f in old['files']},'baseline replay mismatch'
 rows={};changes={};counts={};issues={}
 for mode in ['argv','process_inputs']:
  issue=collections.Counter();files=build_files(writes+mapped,ps,ins,meta,dependency_relative,issue,mode);issues[mode]=dict(issue)
  assert set(baseline)<=set(files)
  for p,f in files.items():
   if p not in baseline:continue
   b=baseline[p];assert f['first_write']==b['first_write'] and f['last_write']==b['last_write'] and f['start']<=b['start']
  rows[mode]={};counts[mode]={}
  for g in ['requested_artifacts','all_modified_files']:
   vs=[f for p,f in files.items() if p in baseline and (g=='all_modified_files' or p in old['requested_paths'])]
   rows[mode][g]=segmentation([(f['path'],f['start'],f['last_write']) for f in vs],old['turns'])
   counts[mode][g]={'files':len(vs),'new_input_dependencies':sum(any(any(dep.get('dependency_kind')!='executed_file' for dep in w['dependencies'].values()) for w in f['writes']) for f in vs),'earlier_than_v1':sum(f['start']<baseline[f['path']]['start'] for f in vs),'with_backtrack':sum(f['start']<f['first_write'] for f in vs)}
  changes[mode]=[]
  for p,f in files.items():
   if p not in baseline or f['start']>=baseline[p]['start']:continue
   links=[{'turn':w['turn'],'pid':w.get('pid'),'syscall':w['syscall'],'dependencies':w['dependencies']} for w in f['writes'] if any(v['start']<baseline[p]['start'] for v in w['dependencies'].values())]
   changes[mode].append({'artifact':p,'requested':p in old['requested_paths'],'old_start':baseline[p]['start'],'new_start':f['start'],'end':f['last_write'],'links':links[:3]})
 result={'session_id':old['session_id'],'task_id':old['task_id'],'turns':old['turns'],'baseline_metrics':old['metrics'],'metrics':rows,'counts':counts,'changes':changes,'issues':issues,'trace_files':count,'input_events':len(ins),'recovered_input_directory_writes':len(mapped),'missing_exec_metadata':missing_meta,'evidence_dir':str(d)}
 (OUT/'rows'/f'{d.name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

def main():
 OUT.mkdir(exist_ok=True);(OUT/'rows').mkdir(exist_ok=True)
 paths=[]
 for p in (ROOT/'sessions').glob('*/runtime.json'):
  m=json.loads(p.read_text())
  if 0<=m['rollout_id']<5 and p.with_name('COMPLETE').exists():paths.append(p.with_name('span_analysis_writer_chain.json'))
 assert len(paths)==320
 results=[];errors=[];start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(one,p):p for p in sorted(paths)}
  for future in concurrent.futures.as_completed(jobs):
   try:results.append(future.result())
   except Exception as e:errors.append({'path':str(jobs[future]),'error':repr(e)});print('ERROR',errors[-1],flush=True)
   if (len(results)+len(errors))%20==0:print('PROGRESS',len(results),len(errors),'seconds',round(time.time()-start),flush=True)
 publish(results,errors,start)

def publish(results,errors,start):
 def mean(xs):
  xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
 metrics={};counts={}
 for g in ['requested_artifacts','all_modified_files']:
  metrics[g]={}
  for mode in ['direct_write','dependency_backtrack','argv','process_inputs']:
   vs=[r['baseline_metrics'][g][mode] if mode in ['direct_write','dependency_backtrack'] else r['metrics'][mode][g] for r in results]
   metrics[g][mode]={k:mean([v[k] for v in vs]) for k in ['segments','nonzero_segments','turns_per_segment','nonzero_length']}
  counts[g]={mode:{**{k:sum(r['counts'][mode][g][k] for r in results) for k in ['files','new_input_dependencies','earlier_than_v1','with_backtrack']},'trajectories_earlier':sum(r['counts'][mode][g]['earlier_than_v1']>0 for r in results)} for mode in ['argv','process_inputs']}
 summary={'complete':len(results)==320 and not errors,'n':len(results),'mean_turns':mean([r['turns'] for r in results]),'errors':errors,'metrics':metrics,'counts':counts,'elapsed_seconds':time.time()-start,'baseline_replayed_and_matched':len(results)}
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
 lines=['# 同320条：扩展程序输入文件依赖','','固定rollout 0–4，所有写入事件、目标文件集合和终点沿用旧缓存；只改变依赖边。全部重新回放旧执行链并逐文件校验起点一致。含零覆盖段、非零段均按连续artifact身份集合切分；下面全为轨迹级宏平均，无总turn/总段数。',f'完成 {len(results)}/320；错误 {len(errors)}；平均轨迹长度 {summary["mean_turns"]:.3f} turn。','','| 范围 | 方法 | 总段数/轨迹 | 非零段数/轨迹 | turn/段 | 非零段平均长度 |','|---|---|---:|---:|---:|---:|']
 labels={'direct_write':'首次实际写入','dependency_backtrack':'原执行脚本链','argv':'扩展：命令参数输入＋成功打开证据','process_inputs':'扩展：写入进程实际打开的输入文件'}
 for g,gl in [('requested_artifacts','任务目标'),('all_modified_files','全部修改文件')]:
  for m,v in metrics[g].items():lines.append(f'| {gl} | {labels[m]} | {v["segments"]:.3f} | {v["nonzero_segments"]:.3f} | {v["turns_per_segment"]:.3f} | {v["nonzero_length"]:.3f} |')
 lines+=['','## 新增回溯','','```json',json.dumps(counts,ensure_ascii=False,indent=2),'```','','## 边界','','- 命令参数版要求参数路径匹配且该进程/其后代实际成功打开输入，或成功rename/link消费源文件；不会仅因echo了路径就连接。','- 进程输入版还接代码内部成功打开的输入文件；只追已有本轨迹写入历史的文件，包括input目录内被修改的文件；未修改的原始输入不回溯。','- 同一路径按此前最早写入回溯；不会引用未来写入。fork继承已有依赖，兄弟进程相互隔离。','- 直接read/cat之后由模型发起独立write的语义关系不连接。','- open是实际访问证据，不等于内容真的影响输出；进程内可能有保守过连接。','- 现有trace未包含read/readv等字节读取记录，跨兄弟进程管道、子进程结果返回父进程、mmap细粒度流不保证可恢复。已恢复input/inputs中本轨迹实际写入/修改的文件作为依赖源，未修改的原始输入不产生更早起点。','- 本次复用原写入事件；旧采集遗漏的写入、未解析相对rename终点不会被冒充已补齐。因此不能声称程序文件依赖已100%完善。']
 lines+=['','## 起点变化例子','']
 examples=[{'session':r['session_id'],'task':r['task_id'],**c} for r in results for c in r['changes']['process_inputs'] if c['requested']]
 (OUT/'changed_requested.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2))
 for e in examples[:12]:lines.append(f'- {e["task"]} / {e["artifact"]}: t{e["old_start"]} → t{e["new_start"]}，终点t{e["end"]}。')
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
