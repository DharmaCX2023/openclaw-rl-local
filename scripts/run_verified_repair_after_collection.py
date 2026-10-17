#!/usr/bin/env python3
"""Wait for one validated collector, then exec its frozen branch supervisor."""
import argparse
import json
import os
from pathlib import Path
import time


def process_start(pid):
    try:
        # The command name may contain spaces; fields after ')' start at #3.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collector-pid',type=int,required=True)
    p.add_argument('--collector-log',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve()
    if not str(root).startswith('/root/openclaw-rl-local/outputs/verified-repair-'):
        raise ValueError('run-owned output required')
    log=a.collector_log.resolve()
    if log.parent!=root:raise ValueError('collector log outside run')
    start=process_start(a.collector_pid)
    if start is None:raise RuntimeError('collector already gone; inspect manually before launch')
    command=Path(f'/proc/{a.collector_pid}/cmdline').read_bytes()
    if b'launcher-data-collect.sh' not in command or str(root).encode() not in command:
        raise ValueError('PID is not this run collector launcher')
    launcher=root/'launcher-data-shards.sh'
    if not launcher.is_file():raise ValueError('frozen supervisor missing')
    state={'collector_pid':a.collector_pid,'collector_start_ticks':start,
           'started_at':time.time(),'status':'waiting_for_validated_collector',
           'launcher':str(launcher)}
    def save():
        dest=root/'collection_to_branches.json';tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(dest)
    save();print(json.dumps(state),flush=True)
    while process_start(a.collector_pid)==start:
        time.sleep(5)
    if 'stage=data-collect exit_code=0 ' not in log.read_text()[-8192:]:
        state.update(status='collector_exit_not_successful',finished_at=time.time());save()
        raise RuntimeError('collector did not record successful terminal exit')
    selection=json.loads((root/'selection.json').read_text())
    expected={f"source-{row['task_id']}-{i}" for row in selection['candidate_tasks'] for i in range(4)}
    sources=json.loads((root/'data-collect.json').read_text())
    if len(sources)!=64 or {row['session_id'] for row in sources}!=expected:
        raise RuntimeError('collector result is not the 64 registered candidates')
    state.update(status='executing_frozen_branch_supervisor',collector_finished_at=time.time());save()
    print('collector_terminal_verified; starting fixed paired branches',flush=True)
    os.execv('/bin/bash',['bash',str(launcher),str(root)])


if __name__=='__main__':main()
