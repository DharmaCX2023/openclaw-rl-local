import ast,collections,unittest
from pathlib import Path
source=ast.parse((Path(__file__).resolve().parents[2]/'clawgym-official/artifact_grpo_bridge.py').read_text())
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='require_complete_batch')
ns={'collections':collections,'IncompleteArtifactBatch':RuntimeError};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<bridge>','exec'),ns)
class StrictBatchTest(unittest.TestCase):
 def records(self):return {(g,i):{'session':f'{g}-{i}'} for g in range(8) for i in range(8)}
 def test_complete_64_passes(self):ns['require_complete_batch'](self.records())
 def test_missing_whole_groups_rejected(self):
  r={k:v for k,v in self.records().items() if k[0]<6}
  with self.assertRaises(RuntimeError):ns['require_complete_batch'](r)
 def test_missing_one_trajectory_rejected(self):
  r=self.records();r.pop((7,7))
  with self.assertRaises(RuntimeError):ns['require_complete_batch'](r)
 def test_infrastructure_and_scoring_failure_rejected(self):
  for failure in ['infrastructure_error','scoring_error']:
   r=self.records();r[7,7][failure]='failed'
   with self.assertRaises(RuntimeError):ns['require_complete_batch'](r)
if __name__=='__main__':unittest.main()
