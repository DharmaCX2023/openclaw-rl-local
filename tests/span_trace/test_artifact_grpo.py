import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[2]/'slime/slime/utils/artifact_grpo.py'
spec=importlib.util.spec_from_file_location('artifact_grpo',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(i, scores, spans, outcome=1, turns=4):
    return dict(group_id=0,trajectory_id=i,outcome=outcome,artifact_scores=scores,artifact_spans=spans,turns=turns)

class ArtifactGRPOTest(unittest.TestCase):
    def test_same_artifact_across_trajectories_not_within(self):
        s=m.build_group_signals([row(0,{'a':1,'b':0},{'a':[1,2],'b':[3,4]}),row(1,{'a':0,'b':1},{'a':[1,2],'b':[3,4]})])
        self.assertGreater(s[0,0]['artifact_advantages']['a'],0)
        self.assertLess(s[0,0]['artifact_advantages']['b'],0)
        self.assertEqual(s[0,0]['outcome_advantage'],0)
        values,_=m.project_tokens(s[0,0],[dict(turn_ids=[1,2,3,4],loss_mask=[1]*4)])
        self.assertGreater(values[0][0],0);self.assertLess(values[0][-1],0)
    def test_missing_span_broadcasts_not_drops(self):
        s=m.build_group_signals([row(0,{'a':0},{},outcome=0),row(1,{'a':1},{'a':[2,3]})])[0,0]
        self.assertEqual(s['effective_spans']['a'],[1,4]);self.assertEqual(s['full_trajectory_fallback'],['a'])
        self.assertTrue(all(v<0 for v in s['turn_signals']))
    def test_overlap_identity_segments(self):
        s=m.build_group_signals([row(0,{'a':1,'b':1},{'a':[1,2],'b':[2,3]}),row(1,{'a':0,'b':0},{})])[0,0]
        self.assertEqual([v['artifacts'] for v in s['segments']],[['a'],['a','b'],['b'],[]])
    def test_routing_chunking_does_not_change_advantage(self):
        s=m.build_group_signals([row(0,{'a':1},{'a':[2,3]}),row(1,{'a':0},{},outcome=0)])[0,0]
        ids=[1,1,0,2,3,4];mask=[1,1,0,1,1,1]
        a,d=m.project_tokens(s,[dict(turn_ids=ids,loss_mask=mask)])
        b,e=m.project_tokens(s,[dict(turn_ids=ids[:4],loss_mask=mask[:4]),dict(turn_ids=ids[4:],loss_mask=mask[4:])])
        self.assertEqual(a[0],b[0]+b[1]);self.assertAlmostEqual(d['endpoint_error'],0);self.assertEqual(a[0][2],0)
    def test_missing_check_from_successful_sparse_verifier_is_zero(self):
        self.assertEqual(m.artifact_scores({}, {'a':{'checks':{'valid':1}}}),{'a':0})
        self.assertEqual(m.artifact_scores({'valid':False},{'a':{'checks':{'valid':1}}}),{'a':0})
    def test_task_1195_sparse_failed_content_check_does_not_crash(self):
        manifest={'top5.csv':{'checks':{
            'has_top5_details_csv':1,
            'top_5_details_schema_ok':1,
            'top_5_details_content_ok':1,
        }}}
        components={
            'has_top5_details_csv':True,
            'top_5_details_schema_ok':True,
            # The verifier emits the legacy spelling as false and omits the
            # mapped spelling when content validation fails.
            'top5_details_content_ok':False,
        }
        self.assertEqual(m.artifact_scores(components,manifest),{'top5.csv':2/3})
    def test_duplicate_trajectory_and_invalid_alignment_fail(self):
        r=row(0,{'a':1},{})
        with self.assertRaises(ValueError):m.build_group_signals([r,r])
        s=m.build_group_signals([r])[0,0]
        with self.assertRaises(ValueError):m.project_tokens(s,[dict(turn_ids=[0],loss_mask=[1])])
    def test_artifact_keys_must_align_even_if_missing_file(self):
        with self.assertRaises(ValueError):m.build_group_signals([row(0,{'a':1},{}),row(1,{'b':0},{})])

if __name__=='__main__':unittest.main()
