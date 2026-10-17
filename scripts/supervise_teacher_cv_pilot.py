#!/usr/bin/env python3
"""Own-process-only supervisor: smoke, tasks, per-idle-GPU burn, full-burn restore."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=str(ROOT/'.venv-openclaw/bin/python')
BURN_SESSION='pypenguin_phase57_full_burn'
children={}
burns={}
STOP=False


def stop_own(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid,signal.SIGTERM)
    try:
        proc.wait(timeout=40)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGKILL)
        proc.wait(timeout=20)


def signal_stop(*_):
    global STOP
    STOP=True


def apps():
    return subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()


def stop_verified_burn():
    name=subprocess.check_output(['tmux','list-panes','-t',BURN_SESSION,'-F','#{pane_start_command}'],text=True)
    if 'synthetic_train_stress.py' not in name:
        raise RuntimeError('Refuse unrecognized session')
    for line in apps().splitlines():
        argv=Path(f'/proc/{int(line.strip())}/cmdline').read_bytes()
        if b'synthetic_train_stress.py' not in argv:
            raise RuntimeError('Non-burn GPU process exists; refuse to interfere')
    subprocess.run(['tmux','send-keys','-t',BURN_SESSION,'C-c'],check=True)
    for _ in range(60):
        if not apps():
            return
        time.sleep(1)
    raise RuntimeError('Existing burn did not release GPUs')


def start_burn(gpu,out):
    if gpu in burns and burns[gpu].poll() is None:
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2')
    with (out/f'burn_gpu{gpu}.log').open('a') as f:
        burns[gpu]=subprocess.Popen(['/usr/local/bin/torchrun','--standalone','--nproc_per_node=1',
            '/root/synthetic_train_stress.py','--steps','0','--hidden-size','4096','--num-layers','72',
            '--expansion','4','--micro-batch-size','1','--seq-len','1024','--grad-accum','1',
            '--precision','bf16','--lr','1e-4','--log-every','20'],env=env,cwd=ROOT,
            stdout=f,stderr=subprocess.STDOUT,start_new_session=True)


def launch(task,gpu,out,smoke=False):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
        FLASH_ATTENTION_DETERMINISTIC='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    args=[PYTHON,str(ROOT/'scripts/run_teacher_cv_pilot.py'),'--output',str(out),'--task',task]
    if smoke:
        args+=['--smoke']
    with (out/f'worker_{task}.log').open('a') as f:
        p=subprocess.Popen(args,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    children[p.pid]=p
    return p


def restore(out):
    for p in list(children.values())+list(burns.values()):
        stop_own(p)
    for _ in range(30):
        if not apps():
            break
        time.sleep(1)
    if apps():
        raise RuntimeError('Other GPU processes remain; will not overlap burn')
    exists=subprocess.run(['tmux','has-session','-t',BURN_SESSION],capture_output=True).returncode==0
    if exists:
        name=subprocess.check_output(['tmux','list-panes','-t',BURN_SESSION,'-F','#{pane_start_command}'],text=True)
        if 'synthetic_train_stress.py' not in name:
            raise RuntimeError('Burn session identity changed')
        subprocess.run(['tmux','kill-session','-t',BURN_SESSION],check=True)
    command=('cd /root && exec env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 '
        '/usr/local/bin/torchrun --standalone --nproc_per_node=8 /root/synthetic_train_stress.py '
        '--steps 0 --hidden-size 4096 --num-layers 72 --expansion 4 --micro-batch-size 1 '
        '--seq-len 1024 --grad-accum 1 --precision bf16 --lr 1e-4 --log-every 5 '
        f'>>{out}/restored_full_burn.log 2>&1')
    subprocess.run(['tmux','new-session','-d','-s',BURN_SESSION,command],check=True)


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((out/'manifest.json').read_text())
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    stopped=False
    try:
        stop_verified_burn()
        stopped=True
        smoke=out/'smoke'
        smoke.mkdir(exist_ok=True)
        shutil.copytree(out/'inputs',smoke/'inputs',dirs_exist_ok=True)
        for gpu in range(1,8):
            start_burn(gpu,out)
        p=launch(manifest['tasks'][0],0,smoke,smoke=True)
        while p.poll() is None and not STOP:
            time.sleep(5)
        if STOP or p.poll()!=0:
            raise RuntimeError(f'Smoke failed/interrupted: {p.poll()}')
        result=json.loads((smoke/f"{manifest['tasks'][0]}.result.json").read_text())
        print(json.dumps(dict(event='smoke_passed',elapsed_s=result['elapsed_s'],repeat=result['repeat'])),flush=True)
        for p in burns.values():
            stop_own(p)
        burns.clear()
        active={}
        for gpu,task in enumerate(manifest['tasks']):
            active[gpu]=(task,launch(task,gpu,out))
        while active and not STOP:
            for gpu,(task,p) in list(active.items()):
                code=p.poll()
                if code is None:
                    continue
                if code!=0:
                    raise RuntimeError(f'Task {task} failed: {code}')
                del active[gpu]
                start_burn(gpu,out)
                print(json.dumps(dict(event='task_finished_burn_started',task=task,gpu=gpu)),flush=True)
            (out/'status.json').write_text(json.dumps(dict(state='RUNNING',active={g:t for g,(t,p) in active.items()},utc=time.strftime('%FT%TZ',time.gmtime()))))
            time.sleep(5)
        if STOP:
            raise RuntimeError('Interrupted')
        subprocess.run([PYTHON,str(ROOT/'scripts/analyze_teacher_cv_pilot.py'),str(out)],check=True)
        (out/'status.json').write_text(json.dumps(dict(state='COMPLETE',utc=time.strftime('%FT%TZ',time.gmtime()))))
    except BaseException as exc:
        (out/'status.json').write_text(json.dumps(dict(state='FAILED',error=str(exc))))
        raise
    finally:
        if stopped:
            restore(out)
            print('Restored verified eight-GPU burn session',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    run(p.parse_args().output)
