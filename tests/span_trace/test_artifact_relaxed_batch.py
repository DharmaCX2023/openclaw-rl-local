"""Relaxed runs keep sampling identities and never retry a short batch."""
import ast, os, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
SOURCE=ast.parse((Path(__file__).resolve().parents[2]/'clawgym-official/clawgym_rl_rollout.py').read_text())
class RelaxedBatchTest(unittest.TestCase):
 def test_skip_preserves_original_indices(self):
  fn=next(n for n in SOURCE.body if isinstance(n,ast.FunctionDef) and n.name=='_active_train_submission_indices')
  ns={'os':os,'_train_submission_indices':lambda batch:list(range(len(batch)))}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-rollout>','exec'),ns)
  batch=[(i//8,SimpleNamespace(task_id='task_0674' if i//8==2 else str(i//8))) for i in range(64)]
  with patch.dict(os.environ,{'OPENCLAW_TRAIN_EXCLUDE_TASK_IDS':'task_0674'}): actual=ns['_active_train_submission_indices'](batch)
  self.assertEqual(actual,list(range(16))+list(range(24,64)))
 def test_short_batch_returns_once_without_strict_validation(self):
  fn=next(n for n in SOURCE.body if isinstance(n,ast.FunctionDef) and n.name=='generate_rollout_clawgym_rl')
  calls=[];result=SimpleNamespace(samples=[[object()] for _ in range(56)])
  def once(*a,**kw):calls.append(1);return result
  ns={'os':os,'_generate_rollout_clawgym_rl_once':once}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-rollout>','exec'),ns)
  with patch.dict(os.environ,{'OPENCLAW_ARTIFACT_GRPO_STRICT':'0'}): actual=ns['generate_rollout_clawgym_rl'](SimpleNamespace(),31,object())
  self.assertIs(actual,result);self.assertEqual(calls,[1])
if __name__=='__main__':unittest.main()
