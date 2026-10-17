import ast,unittest,json,tarfile
from pathlib import Path
source=ast.parse((Path(__file__).resolve().parents[2]/'clawgym-official/artifact_grpo_bridge.py').read_text())
class IncompleteArtifactBatch(RuntimeError):pass
class MissingArchiveTest(unittest.TestCase):
 def wrapper(self,error):
  def record(*_):raise error
  ns={'_record_complete':record,'IncompleteArtifactBatch':IncompleteArtifactBatch,'json':json,'tarfile':tarfile}
  fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_record')
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-bridge>','exec'),ns)
  return ns['_record']
 def test_missing_verifier_excludes_instead_of_crashing(self):
  r=self.wrapper(FileNotFoundError('verifier.json'))('/tmp/session',{})
  self.assertIn('verifier.json',r['infrastructure_error']);self.assertNotIn('artifact_scores',r)
 def test_incomplete_archive_timeout_is_excluded(self):
  self.assertIn('infrastructure_error',self.wrapper(IncompleteArtifactBatch('timeout'))('/tmp/session',{}))
 def test_corrupt_json_is_excluded(self):
  self.assertIn("infrastructure_error",self.wrapper(json.JSONDecodeError("empty", "", 0))("/tmp/session",{}))
 def test_truncated_tar_is_excluded(self):
  self.assertIn("infrastructure_error",self.wrapper(tarfile.ReadError("truncated"))("/tmp/session",{}))
 def test_verifier_change_still_fails(self):
  with self.assertRaises(RuntimeError):self.wrapper(RuntimeError('verifier hash changed'))('/tmp/session',{})
if __name__=='__main__':unittest.main()
