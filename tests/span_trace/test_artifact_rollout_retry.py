import ast,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
src=ast.parse((Path(__file__).resolve().parents[2]/'clawgym-official/clawgym_rl_rollout.py').read_text())
functions=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in ('_train_session_id','generate_rollout_clawgym_rl')]
class IncompleteArtifactBatch(RuntimeError):pass
class RetryTest(unittest.TestCase):
 def test_retry_preserves_step_and_changes_only_session_identity(self):
  with tempfile.TemporaryDirectory() as tmp:
   calls=[];checks=[];server=SimpleNamespace(_finalized_sessions=set());result=SimpleNamespace(samples=[[object()]])
   def once(args,step,buffer,evaluation):
    calls.append((step,ns['_train_session_id'](step,0)));return result
   def prepare(samples):
    checks.append(len(samples))
    if len(checks)==1:raise IncompleteArtifactBatch('missing two groups')
   bridge=types.ModuleType('artifact_grpo_bridge');bridge.prepare=prepare;bridge.IncompleteArtifactBatch=IncompleteArtifactBatch
   ns=dict(os=os,Path=Path,json=json,time=SimpleNamespace(sleep=lambda _:None),_TRAIN_ATTEMPTS={},_generate_rollout_clawgym_rl_once=once,get_global_worker=lambda *_:SimpleNamespace(server=server))
   exec(compile(ast.Module(body=functions,type_ignores=[]),'<actual-rollout>','exec'),ns)
   args=SimpleNamespace(rollout_batch_size=8,n_samples_per_prompt=8)
   with patch.dict(sys.modules,{'artifact_grpo_bridge':bridge}),patch.dict(os.environ,{'OPENCLAW_ARTIFACT_GRPO_STRICT':'1','OPENCLAW_SPAN_TRACE_ROOT':tmp+'/span_audit'}):
    actual=ns['generate_rollout_clawgym_rl'](args,9,object())
   self.assertIs(actual,result)
   self.assertEqual(calls,[(9,'train-task-9-0'),(9,'train-task-9-retry1-0')])
   self.assertEqual(int(calls[0][1].rsplit('-',1)[1]),int(calls[1][1].rsplit('-',1)[1]))
   self.assertEqual(len(server._finalized_sessions),64)
   audit=json.loads((Path(tmp)/'rollout_retries/rollout_9_attempt_0.json').read_text())
   self.assertEqual(audit['rollout_id'],9);self.assertEqual(len(audit['sessions']),64)
if __name__=='__main__':unittest.main()
