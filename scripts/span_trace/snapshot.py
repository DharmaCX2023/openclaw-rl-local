#!/usr/bin/env python3
"""Workspace metadata snapshot; birth time via Linux statx, never ctime-as-birth."""
import ctypes,hashlib,json,os,struct,sys,time
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True)
def birth(path):
 buf=ctypes.create_string_buffer(256)
 if not hasattr(libc,'statx') or libc.statx(-100,os.fsencode(path),0x100,0xFFF,buf)!=0:return None
 mask=struct.unpack_from('I',buf.raw,0)[0]
 if not mask&0x800:return None
 sec,nsec=struct.unpack_from('qI',buf.raw,80);return sec*10**9+nsec

def snapshot(root):
 out={};root=Path(root)
 for base,dirs,files in os.walk(root):
  dirs[:]=sorted(d for d in dirs if d not in {'.git','__pycache__','.venv','node_modules','reward'})
  for name in sorted(files):
   p=Path(base)/name;rel=p.relative_to(root).as_posix()
   if p.is_symlink() or name=='.openclaw_reward_payload.json':continue
   try:
    st=p.stat()
    if not p.is_file():continue
    digest=None
    if st.st_size<=32*1024**2:
     h=hashlib.sha256()
     with p.open('rb') as f:
      for b in iter(lambda:f.read(1024**2),b''):h.update(b)
     digest=h.hexdigest()
    end=p.stat()
    out[rel]={'size':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'birth_ns':birth(p),'inode':st.st_ino,'dev':st.st_dev,'sha256':digest,'stable_during_hash':(st.st_size,st.st_mtime_ns,st.st_ino)==(end.st_size,end.st_mtime_ns,end.st_ino)}
   except (OSError,ValueError) as e:out[rel]={'error':str(e)}
 return {'timestamp_ns':time.time_ns(),'root':str(root),'files':out}
if __name__=='__main__':
 result=snapshot(sys.argv[1] if len(sys.argv)>1 else '/root/.openclaw/workspace')
 print(json.dumps(result,separators=(',',':')))
