import unittest
from content_provenance import match_sources
class ContentProvenanceTests(unittest.TestCase):
 def src(self,text,id='o1'):return {'id':id,'turn':1,'kind':'obs','text':text}
 def test_record_reformatted(self):
  r=match_sources('{"name":"Alice", "amount":12.50,"count":7}',[self.src('name,amount,count\nAlice,12.5,7')]);self.assertTrue(r['candidates']);self.assertTrue(r['candidates'][0]['unique_evidence'])
 def test_field_names_not_enough(self):
  self.assertTrue(match_sources('{"name":"Bob", "amount":10,"count":4}',[self.src('name,amount,count\nAlice,12.5,7')])['unmatched'])
 def test_single_number_no_edge(self):
  self.assertTrue(match_sources('Result is 12.5.',[self.src('name,amount,count\nAlice,12.5,7')])['unmatched'])
 def test_repeated_sources_ambiguous(self):
  src='name,amount,count\nAlice,12.5,7';r=match_sources('{"name":"Alice","amount":12.5,"count":7}',[self.src(src),self.src(src,'o2')]);self.assertTrue(all(not x['unique_evidence'] for x in r['candidates']))
 def test_prompt_explains_values(self):
  src='name,amount,count\nAlice,12.5,7';self.assertTrue(match_sources('{"name":"Alice","amount":12.5,"count":7}',[self.src(src)],baseline=src)['unmatched'])
 def test_no_inferred_arithmetic(self):
  self.assertTrue(match_sources('Average: 9.75',[self.src('name,amount,count\nAlice,12.5,7')])['unmatched'])
if __name__=='__main__':unittest.main()
