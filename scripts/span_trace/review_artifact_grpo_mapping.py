"""Explicit path overrides for dynamic verifier paths, checked against check.py."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[2];P=R/'outputs/artifact_grpo40_20260922/artifact_score_manifest.json';data=json.load(open(P))
def bind(task,path,pattern):
 r=data['task_'+task];keys=[k for k in r['mapping_evidence'] if re.fullmatch(pattern,k)]
 if not keys:raise ValueError((task,path,pattern))
 r['artifacts'][path]={'checks':{k:1.0 for k in keys},'attribution':'explicit verifier path/section review'}
 for k in keys:
  if k in r['unmapped_checks']:r['unmapped_checks'].remove(k)
 if path in r['unmapped_targets']:r['unmapped_targets'].remove(path)
 if path not in r['targets']:r['targets'].append(path)
for kind in ['analysis','guidelines']:bind('0196',f'output/brand-{kind}-AuroraCharge-2025-07-17.md',kind+'_.*')
for path,pat in [('memory_index.csv','memory_index_.*'),('tag_summary.json','tag_summary_.*'),('revised_messages.jsonl','revised_messages_.*|rewritten_.*')]:bind('0224','output/'+path,pat)
bind('0076','outputs/batch_1970s_ciihw/summary.csv','summary_.*|status_and_error_fields_valid|extracted_fields_match_raw|keyword_match_correct')
for name in ['li_wei','sato_hiroshi']:bind('0407','out/emails/email_to_'+name+'.txt','email_'+name.split('_')[0]+'_.*')
for name in ['morning','evening']:bind('0588',f'output/briefings/{name}.md',name+'_.*')
for name,pat in [('weekly_events.csv','weekly_events_.*'),('weekly_status.md','weekly_status_.*'),('email_draft.txt','email_.*'),('meeting_prep.md','meeting_prep_.*'),('validation_report.txt','validation_script_runs_successfully')]:bind('0847','out/2026-05-01/'+name,pat)
for path in list(data['task_0847']['unmapped_targets']):
 if 'YYYY-MM-DD' in path:data['task_0847']['targets'].remove(path);data['task_0847']['unmapped_targets'].remove(path)
for a in list(data['task_0362']['targets']):
 if '2026-09-22' in a:
  template=a.replace('2026-09-22','{rollout_date}')
  bind('0362',template,'raw_file_for_today_exists' if '/raw/' in a else 'processed_.*')
  data['task_0362']['targets'].remove(a);data['task_0362']['unmapped_targets'].remove(a)
for model in ['UserAccount','OrderItem']:
 key=model.lower()
 bind('1140',f'output/src/main/java/com/acme/shop/domain/service/{model}Service.java','service_'+key+'_class')
 bind('1140',f'output/src/main/java/com/acme/shop/domain/repository/service/{model}Repository.java','repository_'+key+'_class')
 bind('1140',f'output/src/main/java/com/acme/shop/domain/entity/{model}.java','po_'+key+'_content')
bind('1274','output/summary.txt','summary_.*')
for p,key in [('editor_pitch.txt','editor_pitch'),('subscriber_note.txt','subscriber_note')]:bind('1413','output/drafts/'+p,key+'_.*')
for path,pattern in [('output/.memory-latch/action_summary.txt','action_summary_.*'),('output/.memory-latch/_manifest.md','manifest_.*'),('output/.memory-latch/tokens.json','token_.*|tokens_.*'),('output/recovery/phase_log.md','recovery_log_.*')]:bind('1614',path,pattern)
for p in data['task_1578']['unmapped_targets'][:]:bind('1578',p,'battlecards_each_.*|battlecards_min_files')
# Directory-backed project checks have explicit per-file checks, not bare prompt suffix paths.
r=data['task_1550'];r['targets']=[];r['unmapped_targets']=[];r['artifacts']={}
bind('1550','output/PORTFOLIO.md','portfolio_md_.*');bind('1550','output/summary.json','summary_json_.*')
for project in ['fast-parser','telemetry-collector','user-api']:
 base='output/projects/'+project+'/'
 for path,pat in [('README.md','readme_.*'),('LICENSE','license_.*'),('.gitignore','gitignore_.*'),('.editorconfig','editorconfig_exists'),('.github/workflows/ci.yml','ci_exists'),('Dockerfile','docker_.*')]:bind('1550',base+path,re.escape(project)+'::'+pat)
 structure={'fast-parser':('rust_structure',['Cargo.toml','src/main.rs','src/lib.rs','tests/integration_test.rs']),'telemetry-collector':('go_structure',['go.mod','cmd/telemetry-collector/main.go','internal/app/app.go','pkg/utils/utils.go']),'user-api':('python_structure',['pyproject.toml','src/user_api/__init__.py','src/user_api/main.py','src/user_api/utils.py','tests/test_main.py'])}
 key,paths=structure[project]
 for path in paths:bind('1550',base+path,re.escape(project)+'::'+key)
for name in ['bottlenecks','proposals','tracking','rejected']:bind('1871','output/autonomy/'+name+'.md',name+'_ok')
# Remaining requested helpers have no separately attributable final verifier score.
# Keep a transparent exclusion list rather than manufacturing an A score.
for task,r in data.items():
 r['unscored_requested_paths']=r.pop('unmapped_targets')
 for a in r['unscored_requested_paths']:
  if a in r['targets']:r['targets'].remove(a)
 r['unmapped_targets']=[]
 # Split each shared check's weight among its destination A values.
 counts={}
 for a,v in r['artifacts'].items():
  for k in v['checks']:counts[k]=counts.get(k,0)+1
 for a,v in r['artifacts'].items():v['checks']={k:1/counts[k] for k in v['checks']}
 assert r['artifacts'],task
P.write_text(json.dumps(data,ensure_ascii=False,indent=2))
print('TASKS',len(data),'A',sum(len(r['artifacts']) for r in data.values()))
print('Unscored requested helpers:',{k:r['unscored_requested_paths'] for k,r in data.items() if r['unscored_requested_paths']})
