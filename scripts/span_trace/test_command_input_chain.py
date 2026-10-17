import collections,unittest
from writer_chain import process_events
from command_input_chain import build_files,input_events
from test_writer_chain import W,rel,write
class TestInputs(unittest.TestCase):
 def run_case(self,writes,traces,mode='argv'):
  ps=[];ins=[]
  for key,pid,text in traces:
   ps+=process_events(text,key,str(pid));ins+=input_events(text,key,str(pid),rel)
  return build_files(writes,ps,ins,{key:{'cwd':W} for key,_,_ in traces},rel,collections.Counter(),mode)
 def test_cp_inputs_not_destination_or_future(self):
  trace='\n'.join(['3.0 execve("/bin/cp", ["cp", "stats.csv", "report.csv"], 0x1) = 0',f'3.1 openat(AT_FDCWD, "stats.csv", O_RDONLY) = 3<{W}stats.csv>',f'3.2 openat(AT_FDCWD, "report.csv", O_WRONLY|O_TRUNC) = 4<{W}report.csv>'])
  f=self.run_case([write(1,'old','report.csv'),write(2,'a','stats.csv'),write(4,'b','report.csv',10),write(5,'later','stats.csv')],[('b',10,trace)])
  self.assertEqual(set(f['report.csv']['writes'][-1]['dependencies']),{'stats.csv'})
  self.assertEqual(f['report.csv']['writes'][-1]['dependencies']['stats.csv']['producer_time'],2)
 def test_recursive_data_chain_and_no_model_bridge(self):
  traces=[('a',10,'2.0 execve("/usr/bin/python3", ["python3", "generate.py"], 0x1) = 0'),('b',20,'4.0 execve("/usr/bin/pandoc", ["pandoc", "stats.md", "-o", "report.pdf"], 0x1) = 0\n'+f'4.1 openat(AT_FDCWD, "stats.md", O_RDONLY) = 3<{W}stats.md>')]
  f=self.run_case([write(1,'first','generate.py'),write(3,'a','stats.md',10),write(5,'b','report.pdf',20),write(6,'model','manual.md')],traces)
  self.assertEqual(f['report.pdf']['start'],1);self.assertEqual(f['manual.md']['start'],6)
 def test_siblings_and_internal_inputs(self):
  traces=[('b',10,'3.0 execve("/bin/bash", ["bash", "-c", "..."], 0x1) = 0\n3.1 fork() = 11\n3.2 fork() = 12'),('b',11,'3.3 execve("/bin/cat", ["cat", "stats.csv"], 0x1) = 0\n'+f'3.4 openat(AT_FDCWD, "stats.csv", O_RDONLY) = 3<{W}stats.csv>'),('b',12,'3.5 execve("/usr/bin/python3", ["python3", "-c", "..."], 0x1) = 0\n'+f'3.6 openat(AT_FDCWD, "other.csv", O_RDONLY) = 3<{W}other.csv>')]
  writes=[write(1,'a','stats.csv'),write(2,'a2','other.csv'),write(4,'b','out',12)]
  self.assertEqual(self.run_case(writes,traces)['out']['start'],4)
  self.assertEqual(self.run_case(writes,traces,'process_inputs')['out']['start'],2)
 def test_unopened_argument_not_dependency(self):
  f=self.run_case([write(1,'a','stats.csv'),write(3,'b','out',10)],[('b',10,'2.0 execve("/bin/echo", ["echo", "stats.csv"], 0x1) = 0')])
  self.assertEqual(f['out']['start'],3)
 def test_relative_mv_and_child_read(self):
  traces=[('b',10,'2.0 execve("/bin/mv", ["mv", "a.csv", "b.csv"], 0x1) = 0\n3.0 rename("a.csv", "b.csv") = 0')]
  f=self.run_case([write(1,'a','a.csv'),write(3,'b','b.csv',10)],traces)
  self.assertEqual(f['b.csv']['start'],1)
if __name__=='__main__':unittest.main()
