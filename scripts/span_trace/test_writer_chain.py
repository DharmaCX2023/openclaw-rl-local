import collections
import unittest
from writer_chain import build_files, process_events
W='/root/.openclaw/workspace/'
def rel(p): return p[len(W):] if p.startswith(W) else None
def write(t,key,path,pid=None):
 e=dict(time=float(t),turn=int(t),key=key,path=path,kind='write',syscall='write' if pid else 'direct_write')
 if pid:e['pid']=str(pid)
 return e
def events(key,pid,lines):return process_events('\n'.join(lines),key,str(pid))
class TestChain(unittest.TestCase):
 def test_recursive_and_reads_and_siblings(self):
  writes=[write(1,'a','builder.py'),write(3,'b','generate.py',10),write(5,'c','result.json',20),write(6,'c','unrelated.json',21)]
  ps=events('b',10,['2.0 execve("/usr/bin/python3", ["python3", "builder.py"], 0x1) = 0'])
  ps+=events('c',19,['4.0 execve("/bin/bash", ["bash", "-c", "..."], 0x1) = 0','4.1 clone(child_stack=NULL, flags=SIGCHLD) = 20','4.2 clone(child_stack=NULL, flags=SIGCHLD) = 21'])
  ps+=events('c',20,['4.3 execve("/usr/bin/python3", ["python3", "generate.py"], 0x1) = 0'])
  ps+=events('c',21,['4.4 execve("/usr/bin/cat", ["cat", "generate.py"], 0x1) = 0','4.5 openat(AT_FDCWD, "generate.py", O_RDONLY) = 3<'+W+'generate.py>'])
  issues=collections.Counter();f=build_files(writes,ps,{'b':{'cwd':W},'c':{'cwd':W}},rel,issues)
  self.assertEqual(f['result.json']['start'],1)
  self.assertEqual(f['unrelated.json']['start'],6)
  self.assertEqual(f['result.json']['writes'][0]['dependencies']['generate.py']['producer_pid'],'10')
  self.assertEqual(dict(issues),{})
 def test_fork_cwd_inherit_inline(self):
  ps=events('b',10,['2.0 execve("/bin/bash", ["bash", "driver.sh"], 0x1) = 0','2.1 chdir("sub") = 0','2.2 vfork( <unfinished ...>','2.4 <... vfork resumed>) = 11'])
  ps+=events('b',11,['2.3 execve("/usr/bin/python3", ["python3", "-c", "print(1)"], 0x1) = 0'])
  f=build_files([write(1,'a','driver.sh'),write(3,'b','sub/out',11)],ps,{'b':{'cwd':W}},rel,collections.Counter())
  self.assertEqual(f['sub/out']['start'],1)
 def test_no_future_dependency(self):
  ps=events('b',10,['1.0 execve("/usr/bin/python3", ["python3", "later.py"], 0x1) = 0'])
  f=build_files([write(2,'a','later.py'),write(3,'b','out',10)],ps,{'b':{'cwd':W}},rel,collections.Counter())
  self.assertEqual(f['out']['start'],3)
if __name__=='__main__':unittest.main()
