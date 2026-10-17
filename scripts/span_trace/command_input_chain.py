"""Optional extension of writer_chain: file inputs consumed by writer processes.

No model-mediated read->write edges. Successful readable opens are evidence of
access, not proof of semantic influence. Historical path starts follow v1.
"""
import collections
import posixpath
import re
from writer_chain import absolute, entrypoint, LINE, strings


def input_events(text, key, pid, relative):
    result=[]; pending=None
    for line in text.splitlines():
        if '<unfinished ...>' in line:
            pending=line.split('<unfinished ...>')[0];continue
        if 'resumed>' in line:
            if pending is None:continue
            line=pending+line.split('resumed>',1)[1].lstrip();pending=None
        m=LINE.match(line)
        if not m:continue
        time,name,args,ret,rest=m.groups()
        if int(ret)<0:continue
        path=None
        if name in ('open','openat','openat2'):
            # A writable/truncated open is not evidence of consuming old content.
            if 'O_TRUNC' in args or not re.search(r'\bO_(?:RDONLY|RDWR)\b',args):continue
            returned=re.findall(r'<([^>]+)>',rest)
            if returned:path=returned[-1]
        elif name in ('rename','renameat','renameat2','link','linkat'):
            # Moves/links can consume a source without reading its bytes.
            vals=strings(args)
            if vals:
                path=vals[0]
                if not path.startswith('/'):
                    fds=re.findall(r'<([^>]+)>',args.split('"',1)[0])
                    if fds:path=absolute(path,fds[0])
        if path:
            result.append(dict(time=float(time),key=key,pid=str(pid),kind='input',
                raw_path=path,syscall=name))
    return result


def argument_paths(event,cwd,relative):
    """Exact individual argv paths, never substring-match a shell/code string."""
    out=set()
    for arg in event['argv'][1:]:
        candidates=[arg]
        if arg.startswith('-') and '=' in arg:candidates=[arg.split('=',1)[1]]
        if arg.startswith('@'):candidates.append(arg[1:])
        for value in candidates:
            if not value or value.startswith('-') or any(c in value for c in '\n\r\t'):
                continue
            # An entire -c program containing spaces will not equal an observed
            # file path. Real filenames containing spaces remain supported.
            p=absolute(value,cwd)
            if p:
                p=relative(p)
                if p:out.add(p)
    return out


def build_files(writes,processes,inputs,exec_meta,relative,issues,mode='argv'):
    assert mode in ('argv','process_inputs')
    files={};states={};roots={}
    children={(e['key'],e['child']) for e in processes if 'child' in e}
    for e in processes:
        if (e['key'],e['pid']) not in children:roots.setdefault(e['key'],set()).add(e['pid'])
    def state(e):
        ident=(e['key'],e['pid'])
        if ident not in states:
            root=e['pid'] in roots.get(e['key'],set())
            states[ident]={'fs':{'cwd':exec_meta.get(e['key'],{}).get('cwd') if root else None},
                          'dependencies':{},'execution':[],'candidates':set()}
            if not root:issues['unresolved_process_parent']+=1
        return states[ident]
    def bind(s,path,e,kind):
        if path not in files:return
        f=files[path]
        dep={'start':f['start'],'producer_key':f['last_key'],'producer_pid':f['last_pid'],
             'producer_time':f['last_time'],'executed_at':e['time'],'executed_pid':e['pid'],
             'dependency_kind':kind}
        # Keep the earliest provenance consumed by this process, not a later
        # overwrite it happens to open after already using the earlier version.
        prev=s['dependencies'].get(path)
        if prev is None or dep['start']<prev['start']:s['dependencies'][path]=dep
    for e in sorted(writes+processes+inputs,key=lambda x:(x['time'],{'process':0,'input':1,'write':2}.get(x['kind'],3))):
        if e['kind']=='process':
            s=state(e)
            if 'child' in e:
                states[(e['key'],e['child'])]={'fs':s['fs'] if e['shared_fs'] else dict(s['fs']),
                    'dependencies':dict(s['dependencies']),'execution':list(s['execution']),
                    'candidates':set(s['candidates'])}
            elif 'cwd' in e:
                s['fs']['cwd']=absolute(e['cwd'],s['fs']['cwd']) if e['cwd'] is not None else None
                if e['cwd'] is None:issues['unresolved_chdir']+=1
            else:
                if e['truncated']:issues['truncated_exec_argv']+=1
                script=entrypoint(e,s['fs']['cwd'],relative,issues)
                s['execution'].append({'time':e['time'],'pid':e['pid'],'executable':e['executable'],'script':script})
                s['candidates'].update(argument_paths(e,s['fs']['cwd'],relative))
                if script:bind(s,script,e,'executed_file')
            continue
        if e['kind']=='input':
            s=state(e);p=absolute(e['raw_path'],s['fs']['cwd']);p=relative(p) if p else None
            if p and (mode=='process_inputs' or p in s['candidates']):
                bind(s,p,e,'command_input' if p in s['candidates'] else 'process_input')
            continue
        if e['kind']!='write':continue
        s=state(e) if 'pid' in e else None
        deps=dict(s['dependencies']) if s else {}
        start=min([e['turn']]+[d['start'] for d in deps.values()]);p=e['path']
        if p not in files:files[p]=dict(path=p,first_write=e['turn'],last_write=e['turn'],start=start,writes=[])
        f=files[p];f.update(first_write=min(f['first_write'],e['turn']),last_write=max(f['last_write'],e['turn']),
            start=min(f['start'],start),last_key=e['key'],last_pid=e.get('pid'),last_time=e['time'])
        f['writes'].append({**e,'dependencies':deps,'writer_execution':list(s['execution']) if s else [],
            'provenance':'syscall_process' if s else 'direct_tool_snapshot'})
    return files


def input_directory_writes(text,key,pid,work='/root/.openclaw/workspace/'):
    """Recover written input/inputs files previously excluded as output artifacts."""
    out=[]
    for line in text.splitlines():
        m=LINE.match(line)
        if not m:continue
        time,name,args,ret,rest=m.groups();ret=int(ret)
        if ret<0:continue
        paths=re.findall(r'<([^>]+)>',args);returned=re.findall(r'<([^>]+)>',rest)
        p=None;creation=False
        if name in ('open','openat','openat2','creat') and returned:
            if 'O_TRUNC' in args or name=='creat':p=returned[-1]
            elif 'O_CREAT' in args:p=returned[-1];creation=True
        elif name in ('write','writev','pwrite64','pwritev','ftruncate') and paths and (ret>0 or name=='ftruncate'):p=paths[0]
        elif name in ('copy_file_range','sendfile','splice') and ret>0:
            if name=='sendfile' and paths:p=paths[0]
            elif len(paths)>=2:p=paths[1]
        if not p:continue
        p=p.removesuffix(' (deleted)')
        if not p.startswith(work):continue
        p=posixpath.normpath(p[len(work):])
        if not p.startswith(('input/','inputs/')):continue
        out.append(dict(time=float(time),key=key,pid=str(pid),kind='write',path=p,syscall=name,creation_candidate=creation))
    return out
