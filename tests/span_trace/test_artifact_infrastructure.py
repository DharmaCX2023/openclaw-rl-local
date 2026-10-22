import ast,importlib.util,unittest
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('artifact_grpo',R/'slime/slime/utils/artifact_grpo.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
source=ast.parse((R/'clawgym-official/artifact_grpo_bridge.py').read_text())
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='project_available_groups')
ns={'project_samples':core.project_samples};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<bridge>','exec'),ns)
class InfrastructureGroupsTest(unittest.TestCase):
 def test_damaged_group_excluded_without_changing_other_group_advantage(self):
  records={};samples=[]
  for g in (0,1):
   for i in (0,1):
    records[g,i]=dict(group_id=g,trajectory_id=i,turns=2,outcome=i,artifact_scores={'a':i},artifact_spans={'a':[2,2]}) if g==0 else {'infrastructure_error':'container destroyed'}
    samples.append(SimpleNamespace(group_index=g,index=i,metadata={'session_id':f'{g}-{i}'},opsd_step_ids=[1,2] if g==0 else [98,99],loss_mask=[1,1],response_length=2))
  original_masks=[s.loss_mask for s in samples]
  expected,_=core.project_samples(samples[:2],records)
  actual,audit=ns['project_available_groups'](samples,records,{1})
  self.assertEqual(actual[:2],expected);self.assertEqual(actual[2:],[[0.,0.],[0.,0.]])
  self.assertEqual(original_masks,[[1,1],[1,1],[0,0],[0,0]])
  self.assertEqual(len(audit),2)
 def test_all_groups_missing_fails_instead_of_fake_update(self):
  s=SimpleNamespace(group_index=1)
  with self.assertRaises(RuntimeError):ns['project_available_groups']([s],{}, {1})
 def test_one_unscored_trajectory_is_left_out_and_sibling_keeps_artifact_advantage(self):
  records={};samples=[]
  for i in (0,1):
   records[0,i]=dict(group_id=0,trajectory_id=i,turns=2,outcome=float(i),artifact_scores={'a':float(i)},artifact_spans={'a':[2,2]})
   samples.append(SimpleNamespace(group_index=0,index=i,metadata={'session_id':f'0-{i}'},opsd_step_ids=[1,2],loss_mask=[1,1],response_length=2))
  expected,_=core.project_samples(samples[:1],{ (0,0):records[0,0] })
  actual,audit=ns['project_available_groups'](samples,records,set(),1.0,{(0,1)})
  self.assertEqual(actual[0],expected[0])
  self.assertEqual(actual[1],[0.0,0.0])
  self.assertEqual(samples[1].loss_mask,[0,0])
  self.assertEqual(len(audit),1)
 def test_bad_token_alignment_is_removed_before_sibling_normalization(self):
  records={};samples=[]
  for i in (0,1,2):
   records[0,i]=dict(group_id=0,trajectory_id=i,turns=2,outcome=float(i),artifact_scores={'a':float(i)},artifact_spans={'a':[2,2]})
   samples.append(SimpleNamespace(group_index=0,index=i,metadata={'session_id':f'0-{i}'},opsd_step_ids=[1,3] if i==2 else [1,2],loss_mask=[1,1],response_length=2))
  expected,_=core.project_samples(samples[:2],{key:value for key,value in records.items() if key[1]<2})
  actual,audit=ns['project_available_groups'](samples,records,set())
  self.assertEqual(actual[:2],expected)
  self.assertEqual(actual[2],[0.0,0.0])
  self.assertEqual(samples[2].loss_mask,[0,0])
  self.assertEqual(sum('dropped' not in row for row in audit),2)
 def test_every_invalid_trajectory_refuses_empty_update(self):
  records={(0,0):dict(group_id=0,trajectory_id=0,turns=1,outcome=0.0,artifact_scores={'a':0.0},artifact_spans={})}
  sample=SimpleNamespace(group_index=0,index=0,metadata={'session_id':'bad'},opsd_step_ids=[2],loss_mask=[1],response_length=1)
  with self.assertRaises(RuntimeError):
   ns['project_available_groups']([sample],records,set())
if __name__=='__main__':unittest.main()
