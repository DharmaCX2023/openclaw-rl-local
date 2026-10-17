"""Supervise only the authorized artifact-GRPO run; never terminate live jobs.

Restart recognized infrastructure failures from a verified checkpoint. Unknown
errors are preserved for diagnosis, not hidden by unbounded restarts.
"""
import collections,fcntl,json,os,re,shlex,socket,subprocess,time
from pathlib import Path
from retain_latest_artifact_checkpoint import committed_checkpoint,prune
REPO=Path('/root/openclaw-rl-local')
ROOT=Path('/mnt/humf/openclaw-rl-checkpoints/clawgym-artifact-grpo40-20260923-strict2')
OUT=REPO/'outputs/artifact_grpo40_20260922'
SESSION='artifact_grpo40_20260922'
LOG=ROOT/'logs/train_20260923-artifact-grpo40-strict2.log'

def classify(text):
    patterns={
      'archive_missing':r'FileNotFoundError:.*(?:verifier\.json|trajectory\.json|evidence\.tar\.gz|turn_\d+\.json)',
      'archive_corrupt':r'JSONDecodeError:|tarfile.ReadError:|EOFError:',
      'job_agent_startup':r'No available agent to submit job',
      'remote_transport':r'(?:ConnectionResetError|BrokenPipeError|ConnectionRefusedError|SSHException|ChannelException):',
      'ray_worker_lost':r'(?:WorkerCrashedError|ActorDiedError|RaySystemError):',
    }
    return next((name for name,pattern in patterns.items() if re.search(pattern,text)),None)

def active():return subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0

def ports_free():
    for port in (6393,8283,30411):
        with socket.socket() as sock:
            sock.settimeout(.3)
            if sock.connect_ex(('127.0.0.1',port))==0:return False
    return True

def tail(path,size=100000):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-size));return f.read().decode(errors='replace')

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=OUT/'supervisor_restarts.json'
    history=json.loads(ledger.read_text()) if ledger.exists() else []
    while True:
        try:
            latest=int((ROOT/'latest_checkpointed_iteration.txt').read_text().strip())
            live=active();verified=committed_checkpoint(ROOT,latest)
            state={'updated':time.time(),'latest_saved_updates':latest+1,'training_active':live,'latest_checkpoint_verified':verified,'automatic_restarts':len(history)}
            if latest==39 and verified:
                prune(ROOT)
                state['status']='final_checkpoint_saved' if live else 'complete'
            elif live:
                state['status']='running'
                watcher='artifact_grpo40_watch_20260922'
                if subprocess.run(['tmux','has-session','-t',watcher],capture_output=True).returncode!=0:
                    command=f'cd {shlex.quote(str(REPO))} && SAVE_CKPT={shlex.quote(str(ROOT))} RUN_STAMP=20260923-artifact-grpo40-strict2 .venv-openclaw/bin/python scripts/span_trace/watch_artifact_grpo40.py >> outputs/artifact_grpo40_20260922/watch_resume.log 2>&1'
                    subprocess.run(['tmux','new-session','-d','-s',watcher,command],check=True)
            else:
                logs=list(OUT.glob('resume_*.log'))+[LOG]
                recent=max((p for p in logs if p.exists()),key=lambda p:p.stat().st_mtime)
                text=tail(recent).rsplit('Runtime env is setting up.',1)[-1]
                reason=classify(text)
                state.update(status='needs_diagnosis',failure_class=reason,log=str(recent),last_error=text[-8000:])
                repeats=sum(row['checkpoint']==latest and row['reason']==reason for row in history)
                if reason and verified and 30<=latest<39 and repeats<3 and ports_free():
                    current_path=OUT/'CURRENT_RUN.json'
                    current=json.loads(current_path.read_text())
                    attempt=max([current.get('resume_session_attempt_base',300)]+[r['attempt'] for r in history])+100
                    restart_log=OUT/f'resume_auto_{attempt}.log'
                    env=os.environ.copy();env['OPENCLAW_TRAIN_ATTEMPT_BASE']=str(attempt)
                    cmd=f'cd {shlex.quote(str(REPO))} && OPENCLAW_TRAIN_ATTEMPT_BASE={attempt} bash scripts/resume_artifact_grpo40_20260924.sh > {shlex.quote(str(restart_log))} 2>&1'
                    subprocess.run(['tmux','new-session','-d','-s',SESSION,cmd],check=True,env=env)
                    history.append(dict(time=time.time(),checkpoint=latest,attempt=attempt,reason=reason,log=str(restart_log)))
                    ledger.write_text(json.dumps(history,indent=2))
                    current.update(restored_completed_updates=latest+1,resume_session_attempt_base=attempt,resume_status='supervisor_restarted',updated=time.time())
                    current_path.write_text(json.dumps(current,ensure_ascii=False,indent=2))
                    state.update(status='restarting',restart_log=str(restart_log))
            tmp=OUT/'SUPERVISOR.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(OUT/'SUPERVISOR.json')
            if state['status']=='complete':return
        except Exception as exc:
            with (OUT/'supervisor_errors.log').open('a') as f:f.write(f'{time.time()} {exc!r}\n')
        time.sleep(60)
if __name__=='__main__':main()
