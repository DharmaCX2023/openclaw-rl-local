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
if __name__=='__main__':unittest.main()
