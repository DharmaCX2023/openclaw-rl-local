import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/span_trace'))
from artifact_run_supervisor import classify
class SupervisorTest(unittest.TestCase):
 def test_recognized_infrastructure_errors(self):
  self.assertEqual(classify("FileNotFoundError: '/run/verifier.json'"),'archive_missing')
  self.assertEqual(classify('json.decoder.JSONDecodeError: Expecting value'),'archive_corrupt')
  self.assertEqual(classify('RuntimeError: No available agent to submit job'),'job_agent_startup')
 def test_unknown_logic_error_not_restarted_blindly(self):
  self.assertIsNone(classify('AssertionError: baseline replay mismatch'))
  self.assertIsNone(classify('RuntimeError: Verifier changed since mapping review'))
if __name__=='__main__':unittest.main()
