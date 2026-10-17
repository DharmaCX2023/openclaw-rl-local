import importlib.util,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('retention',Path(__file__).resolve().parents[2]/'scripts/span_trace/retain_latest_artifact_checkpoint.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class RetentionTest(unittest.TestCase):
 def checkpoint(self,root,i):
  d=root/f'iter_{i:07d}';d.mkdir()
  for n in ['common.pt','.metadata']+[f'__{k}_0.distcp' for k in range(8)]:(d/n).write_text('data')
  (d/'metadata.json').write_text('{}')
  return d
 def test_protected_latest_and_inflight_survive(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)
   for i in [19,30,31,32,33]:self.checkpoint(r,i)
   (r/'latest_checkpointed_iteration.txt').write_text('32')
   self.assertEqual(mod.prune(r),['iter_0000031'])
   for i in [19,30,32,33]:self.assertTrue((r/f'iter_{i:07d}').exists())
 def test_incomplete_latest_never_deletes_previous(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t)
   for i in [30,31,32]:self.checkpoint(r,i)
   (r/'iter_0000032/.metadata').unlink()
   (r/'latest_checkpointed_iteration.txt').write_text('32')
   self.assertEqual(mod.prune(r),[]);self.assertTrue((r/'iter_0000031').exists())
if __name__=='__main__':unittest.main()
