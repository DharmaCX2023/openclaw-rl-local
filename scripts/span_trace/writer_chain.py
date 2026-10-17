"""Write provenance through executed files and process ancestry, never read edges."""
import ast
import collections
import posixpath
import re

LINE = re.compile(r'^(\d+\.\d+)\s+(\w+)\((.*)\)\s+=\s+(-?\d+)(.*)$')
STRING = r'"(?:[^"\\]|\\.)*"'


def strings(text):
    return [ast.literal_eval(s) for s in re.findall(STRING, text)]


def process_events(text, key, pid):
    events = []
    pending = None
    for line in text.splitlines():
        if '<unfinished ...>' in line:
            pending = line.split('<unfinished ...>')[0]
            continue
        if 'resumed>' in line:
            if pending is None:
                continue
            line = pending + line.split('resumed>', 1)[1].lstrip()
            pending = None
        m = LINE.match(line)
        if not m:
            continue
        when, name, args, ret, rest = m.groups()
        if int(ret) < 0:
            continue
        event = dict(time=float(when), key=key, pid=str(pid), kind='process', syscall=name)
        if name in ('clone', 'clone3', 'fork', 'vfork') and int(ret) > 0:
            event.update(child=str(ret), shared_fs='CLONE_FS' in args)
        elif name in ('execve', 'execveat'):
            argv = re.search(r'\[((?:' + STRING + r'|[^\]])*)\]', args)
            if argv is None:
                continue
            head = strings(args[:argv.start()])
            if not head:
                continue
            event.update(executable=head[-1], argv=strings(argv.group(1)), truncated='...' in re.sub(STRING, '', argv.group(1)))
            if name == 'execveat':
                fd = re.search(r'<([^>]+)>', args[:argv.start()])
                event['dirfd'] = fd.group(1) if fd else None
        elif name == 'chdir':
            values = strings(args)
            event['cwd'] = values[0] if values else None
        elif name == 'fchdir':
            fd = re.search(r'<([^>]+)>', args)
            if not fd:
                continue
            event['cwd'] = fd.group(1)
        else:
            continue
        events.append(event)
    return events


def absolute(path, cwd):
    if posixpath.isabs(path):
        return posixpath.normpath(path)
    return posixpath.normpath(posixpath.join(cwd, path)) if cwd else None


def entrypoint(event, cwd, relative, issues):
    """Only executed path or interpreter's entry script; no arguments-as-input edges."""
    exe = event['executable']
    base = posixpath.basename(exe)
    exe_path = absolute(exe, event.get('dirfd') or cwd)
    if exe_path and relative(exe_path):
        return relative(exe_path)
    argv = event['argv'][1:]
    family = None
    if re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', base):
        family = 'python'
    elif base in ('bash', 'sh', 'dash', 'zsh', 'ksh'):
        family = 'shell'
    elif base in ('node', 'nodejs'):
        family = 'node'
    elif base in ('perl', 'ruby'):
        family = base
    if not family:
        return None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--':
            i += 1
            break
        if arg == '-':
            return None
        if not arg.startswith('-'):
            break
        if family == 'python':
            if arg in ('-c', '-m') or arg.startswith(('-c', '-m')):
                return None
            if arg in ('-W', '-X'):
                i += 2
                continue
            if arg in ('-u', '-B', '-E', '-I', '-s', '-S', '-O', '-OO', '-q', '-v', '-b', '-bb') or arg.startswith(('-W', '-X')):
                i += 1
                continue
        elif family == 'shell':
            if not arg.startswith('--') and ('c' in arg[1:] or 's' in arg[1:]):
                return None
            if arg in ('-e', '-u', '-x', '-v', '-n', '-eu', '-eux', '--noprofile', '--norc'):
                i += 1
                continue
            if arg in ('-o', '-O'):
                i += 2
                continue
        elif family == 'node':
            if arg in ('-e', '--eval', '-p', '--print') or arg.startswith(('--eval=', '--print=')):
                return None
        elif family in ('perl', 'ruby'):
            if arg == '-e' or arg.startswith('-e'):
                return None
        issues['unresolved_interpreter_options'] += 1
        return None
    if i >= len(argv):
        return None
    path = absolute(argv[i], cwd)
    if path is None:
        issues['unresolved_script_cwd'] += 1
        return None
    return relative(path)


def build_files(writes, processes, exec_meta, relative, issues):
    files = {}
    states = {}
    roots = {}
    children = {(e['key'], e['child']) for e in processes if 'child' in e}
    for e in processes:
        ident = (e['key'], e['pid'])
        if ident not in children:
            roots.setdefault(e['key'], set()).add(e['pid'])
    def state(e):
        ident = (e['key'], e['pid'])
        if ident not in states:
            root = e['pid'] in roots.get(e['key'], set())
            states[ident] = {'fs': {'cwd': exec_meta.get(e['key'], {}).get('cwd') if root else None}, 'dependencies': {}, 'execution': [], 'known': root}
            if not root:
                issues['unresolved_process_parent'] += 1
        return states[ident]
    for e in sorted(writes + processes, key=lambda x: (x['time'], x['kind'] != 'process')):
        if e['kind'] == 'process':
            s = state(e)
            if 'child' in e:
                states[(e['key'], e['child'])] = {'fs': s['fs'] if e['shared_fs'] else dict(s['fs']), 'dependencies': dict(s['dependencies']), 'execution': list(s['execution']), 'known': s['known']}
            elif 'cwd' in e:
                s['fs']['cwd'] = absolute(e['cwd'], s['fs']['cwd']) if e['cwd'] is not None else None
                if e['cwd'] is None:
                    issues['unresolved_chdir'] += 1
            else:
                if e['truncated']:
                    issues['truncated_exec_argv'] += 1
                script = entrypoint(e, s['fs']['cwd'], relative, issues)
                s['execution'] = s['execution'] + [{'time':e['time'], 'pid':e['pid'], 'executable':e['executable'], 'script':script}]
                if script and script in files:
                    f = files[script]
                    # Bind to the producer known before execution, never future writes.
                    s['dependencies'][script] = {'start':f['start'], 'producer_key':f['last_key'], 'producer_pid':f['last_pid'], 'producer_time':f['last_time'], 'executed_at':e['time'], 'executed_pid':e['pid']}
            continue
        if e['kind'] != 'write':
            continue
        s = state(e) if 'pid' in e else None
        dependencies = dict(s['dependencies']) if s else {}
        start = min([e['turn']] + [v['start'] for v in dependencies.values()])
        p = e['path']
        if p not in files:
            files[p] = dict(path=p, first_write=e['turn'], last_write=e['turn'], start=start, writes=[])
        f = files[p]
        f.update(first_write=min(f['first_write'],e['turn']), last_write=max(f['last_write'],e['turn']), start=min(f['start'],start), last_key=e['key'], last_pid=e.get('pid'), last_time=e['time'])
        f['writes'].append({**e, 'dependencies':dependencies, 'writer_execution':s['execution'] if s else [], 'provenance':'syscall_process' if s else 'direct_tool_snapshot'})
    return files
