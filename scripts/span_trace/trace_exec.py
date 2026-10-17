#!/usr/bin/env python3
"""One strace process tree per toolcall; preserve command exit status and stdio."""
import base64,json,os,sys,time
from pathlib import Path
key,encoded=sys.argv[1:3];root=Path('/root/.openclaw/span_trace');root.mkdir(parents=True,exist_ok=True)
command=base64.b64decode(encoded).decode()
(root/f'{key}.exec.json').write_text(json.dumps({'key':key,'pid':os.getpid(),'cwd':os.getcwd(),'timestamp_ns':time.time_ns(),'command':command}))
os.execvp('strace',['strace','-ff','-qq','-ttt','-yy','-s','4096','-e','trace=%file,write,writev,pwrite64,pwritev,ftruncate,copy_file_range,sendfile,splice,close,dup,dup2,dup3,fcntl,fork,vfork,clone,clone3','-o',str(root/f'{key}.trace'),'--','/bin/bash','-c',command])
