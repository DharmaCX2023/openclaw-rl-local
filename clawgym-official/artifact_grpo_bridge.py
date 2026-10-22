"""Opt-in online bridge from saved verifier + write evidence to turn advantages."""
import collections,concurrent.futures,fnmatch,hashlib,json,os,sys,time,tarfile
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts/span_trace'))
from slime.utils.artifact_grpo import artifact_scores,project_samples
from analyze_writer_chain import analyze
from command_input_chain import build_files as input_build_files
from verifier_dataflow import Analyzer
import compare_command_inputs320 as replay


class IncompleteArtifactBatch(RuntimeError):
    """Retry the same rollout; no optimizer update is allowed."""


def _verifier_components(session_dir):
    raw=json.loads((Path(session_dir)/'verifier.json').read_text())
    components=raw.get('components')
    if not isinstance(components,dict) or raw.get('returncode')!=0:
        return None
    if isinstance(components.get('scores'),dict):components=components['scores']
    return {k:v for k,v in components.items()
            if isinstance(v,(bool,int,float))
            and k not in {'reward','score','total','total_score','overall_score'}}


def _concrete_targets(requested,files):
    """Resolve requested templates against files actually written by this group."""
    concrete=set()
    for target in requested:
        pattern=(target.replace('{rollout_date}','*')
                 .replace('YYYY-MM-DD','*'))
        pattern=__import__('re').sub(r'\{[^{}]+\}','*',pattern)
        matches={path for path in files if fnmatch.fnmatch(path,pattern)}
        concrete.update(matches)
    return concrete


def _task_string_bindings(task_dir):
    """Load literal strings that verifier code may obtain from task JSON.

    This does not execute task code. It lets the static dataflow analyzer
    resolve patterns such as cfg.get("output_path") to the concrete path from
    input/config.json.
    """
    bindings=collections.defaultdict(set)
    input_root=Path(task_dir)/'input_files'
    if not input_root.exists():
        return {}
    def collect(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if isinstance(item,str):bindings[str(key)].add(item)
                collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
    for path in input_root.rglob('*.json'):
        try:
            if path.stat().st_size<=1024*1024:
                collect(json.loads(path.read_text()))
        except (OSError,UnicodeDecodeError,json.JSONDecodeError):
            continue
    return dict(bindings)


def _writer_chain(session_dir):
    """Use cached analysis; only unpack evidence when that cache is missing.

    Missing archives are normal: a few sessions never produced span_trace.
    Scoring already drops those groups. Mapping must do the same.
    """
    d=Path(session_dir)
    cached=d/'span_analysis_writer_chain.json'
    if (d/'evidence.tar.gz').exists():
        return analyze(d)
    if cached.exists():
        return json.loads(cached.read_text())
    return None


def _build_task_manifest(task,session_dirs):
    """Freeze one task mapping from its group's verifier and writer evidence."""
    criteria=set();requested=set();written=set();task_dir=None
    for session_dir in session_dirs:
        d=Path(session_dir)
        try:
            runtime=json.loads((d/'runtime.json').read_text())
            if runtime['task_id']!=task:
                raise RuntimeError(f'mixed tasks while mapping {task}: {runtime["task_id"]}')
            task_dir=Path(runtime['task_dir'])
            chain=_writer_chain(d)
            if chain is None:
                continue
            components=_verifier_components(d)
            if components is not None:criteria.update(components)
            requested.update(chain.get('requested_paths') or [])
            written.update(row['path'] for row in chain.get('files',[])
                           if isinstance(row,dict) and isinstance(row.get('path'),str))
        except Exception as exc:
            print(
                f'[ArtifactGRPO] ignoring unusable mapping evidence for {task} at {d}: '
                f'{type(exc).__name__}: {exc}',
                flush=True,
            )
            continue
    if task_dir is None:
        raise IncompleteArtifactBatch(f'{task}: no session metadata')
    if not criteria:
        raise IncompleteArtifactBatch(f'{task}: no successful verifier components')
    code=(task_dir/'reward/check.py').read_text()
    targets=_concrete_targets(requested,written)
    bindings=_task_string_bindings(task_dir)
    mapping=Analyzer(code,sorted(targets),criteria,bindings).run()
    if not any(value['paths'] for value in mapping.values()):
        # Some prompts request a directory rather than naming every deliverable.
        # In that case use files actually written by the group, but still require
        # the verifier dataflow analysis to prove each score-to-file edge.
        targets=written
        mapping=Analyzer(code,sorted(targets),criteria,bindings).run()
    byartifact=collections.defaultdict(dict)
    unmapped=[]
    for key,value in mapping.items():
        paths=value['paths']
        if not paths:
            unmapped.append(key)
            continue
        for artifact in paths:byartifact[artifact][key]=1.0/len(paths)
    if not byartifact:
        raise IncompleteArtifactBatch(f'{task}: no verifier checks map to written artifacts')
    return {
        'checker_sha256':hashlib.sha256(code.encode()).hexdigest(),
        'artifacts':{artifact:{'checks':checks,'attribution':'runtime frozen verifier dataflow'}
                     for artifact,checks in sorted(byartifact.items())},
        'targets':sorted(targets),
        'unmapped_targets':[],
        'unmapped_checks':sorted(unmapped),
        'mapping_evidence':mapping,
        'task_dir':str(task_dir),
    }


def _extend_policy(root,identities,policy,path):
    """Add unseen tasks before worker processes score the current rollout."""
    sessions=collections.defaultdict(list)
    for sid in identities.values():
        d=root/hashlib.sha256(sid.encode()).hexdigest()[:24]
        try:
            runtime=json.loads((d/'runtime.json').read_text())
            sessions[runtime['task_id']].append(d)
        except Exception as exc:
            # _record will turn this session into one excluded trajectory. Other
            # sessions must still be allowed to establish their task mappings.
            print(
                f'[ArtifactGRPO] cannot read mapping identity for {sid}; '
                f'dropping that trajectory later: {type(exc).__name__}: {exc}',
                flush=True,
            )
    missing=sorted(set(sessions)-set(policy))
    if missing and os.environ.get('OPENCLAW_ARTIFACT_GRPO_DYNAMIC_MANIFEST')!='1':
        raise IncompleteArtifactBatch(f'artifact manifest missing tasks: {missing}')
    built=[]
    for task in missing:
        try:
            policy[task]=_build_task_manifest(task,sessions[task]);built.append(task)
        except Exception as exc:
            # A mapping failure invalidates only this task's trajectories in the
            # current step. Do not persist it, so a later batch can retry.
            policy[task]={'unmapped':True,'error':f'{type(exc).__name__}: {exc}'}
            print(
                f'[ArtifactGRPO] task {task} has no artifact mapping; '
                f'dropping its trajectories: {type(exc).__name__}: {exc}',
                flush=True,
            )
    if built:
        persisted={k:v for k,v in policy.items() if not (isinstance(v,dict) and v.get('unmapped'))}
        tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_text(json.dumps(persisted,ensure_ascii=False,indent=2))
        os.replace(tmp,path)
        print(f'[ArtifactGRPO] froze mappings for tasks={built}',flush=True)


def _record(session_dir,policy):
    # Any failure local to one session excludes only that trajectory. The caller
    # may continue with all scored siblings; no artifact score is fabricated.
    try:
        return _record_complete(session_dir,policy)
    except Exception as exc:
        d=Path(session_dir)
        result=dict(
            session_dir=str(d),
            artifact_scores=None,
            artifact_spans={},
            turns=0,
            outcome=0.0,
            scoring_error=f'{type(exc).__name__}: {exc}',
        )
        try:
            runtime=json.loads((d/'runtime.json').read_text())
            result.update(task_id=runtime.get('task_id'),session=runtime.get('session_id'))
        except Exception:
            pass
        print(
            f'[ArtifactGRPO] dropping trajectory with unusable scoring evidence at {d}: '
            f'{type(exc).__name__}: {exc}',
            flush=True,
        )
        return result


def _record_complete(session_dir,policy):
    d=Path(session_dir)
    deadline=time.monotonic()+420
    while not (d/'COMPLETE').exists():
        error_file=d/'errors.jsonl'
        errors=[json.loads(line) for line in error_file.read_text().splitlines()] if error_file.exists() else []
        archive_errors=[e for e in errors if e.get('phase')=='archive']
        if archive_errors:
            runtime=json.loads((d/'runtime.json').read_text())
            return dict(session=runtime['session_id'],infrastructure_error=archive_errors[-1]['error'])
        if time.monotonic()>deadline:raise IncompleteArtifactBatch(f'No complete file-write evidence: {d}')
        time.sleep(1)
    runtime=json.loads((d/'runtime.json').read_text())
    task=runtime['task_id'];manifest=policy[task]
    if manifest.get('unmapped'):
        return dict(task_id=task,session=runtime['session_id'],outcome=0.0,artifact_scores=None,artifact_spans={},turns=0,scoring_error=manifest.get('error') or 'no artifact mapping')
    code=Path(runtime['task_dir'])/'reward/check.py'
    if hashlib.sha256(code.read_bytes()).hexdigest()!=manifest['checker_sha256']:
        raise RuntimeError(f'Verifier changed since mapping review: {task}')
    raw=json.loads((d/'verifier.json').read_text())
    components=raw['components']
    if not isinstance(components,dict) or raw['returncode']!=0:
        trajectory=json.loads((d/'trajectory.json').read_text())
        return dict(task_id=task,session=runtime['session_id'],outcome=float(trajectory['reward']),
                    artifact_scores=None,artifact_spans={},turns=len(trajectory['react_steps']),
                    scoring_error='original verifier failed; no fabricated A scores')
    if isinstance(components.get('scores'),dict):components=components['scores']
    scores=artifact_scores(components,manifest['artifacts'])
    if not scores:raise RuntimeError(f'No mapped artifact scores: {task}')
    old=analyze(d)
    # Replay already captured syscall evidence with existing process-input rules.
    # This runs in a process pool: the capture is isolated from other sessions.
    captured={};original=replay.build_files
    def capture(*args,**kwargs):
        files=original(*args,**kwargs)
        if args[-1]=='process_inputs':captured.update(files)
        return files
    replay.build_files=capture;replay.OUT=d/'artifact_grpo_replay';(replay.OUT/'rows').mkdir(parents=True,exist_ok=True)
    try:replay.one(d/'span_analysis_writer_chain.json')
    finally:replay.build_files=original
    stamp=json.loads((d/f"turn_{old['turns']:04d}.json").read_text())['timestamp_ns']/1e9
    date=datetime.fromtimestamp(stamp,ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    paths={a:a.replace('{rollout_date}',date) for a in scores}
    spans={a:[captured[p]['start'],captured[p]['last_write']] for a,p in paths.items() if p in captured}
    result={'task_id':task,'session':runtime['session_id'],'outcome':float(old['reward']),'artifact_scores':scores,'artifact_spans':spans,'turns':old['turns'],'unmapped_targets':manifest['unmapped_targets'],'unmapped_checks':manifest['unmapped_checks']}
    (d/'artifact_grpo_record.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def prepare(samples):
    policy_path=Path(os.environ['OPENCLAW_ARTIFACT_SCORE_MANIFEST'])
    policy=json.loads(policy_path.read_text())
    root=Path(os.environ['OPENCLAW_SPAN_TRACE_ROOT'])/'sessions'
    identities={}
    for sample in samples:
        metadata=sample.metadata or {};sid=metadata.get('session_id')
        if not sid:raise RuntimeError('artifact GRPO requires session identity')
        key=(sample.group_index,sample.index)
        if key in identities and identities[key]!=sid:raise RuntimeError('logical trajectory identity collision')
        identities[key]=sid
    _extend_policy(root,identities,policy,policy_path)
    if os.environ.get('OPENCLAW_ARTIFACT_GRPO_STRICT')=='1':
        require_complete_batch({key:{} for key in identities})
    records={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures={
            pool.submit(_record,str(root/hashlib.sha256(sid.encode()).hexdigest()[:24]),policy):(key,sid)
            for key,sid in identities.items()
        }
        for future in concurrent.futures.as_completed(futures):
            key,sid=futures[future]
            try:
                r=future.result()
            except Exception as exc:
                # Process-pool transport or an unforeseen worker exception is
                # still local to this trajectory.
                r=dict(
                    session=sid,
                    artifact_scores=None,
                    artifact_spans={},
                    turns=0,
                    outcome=0.0,
                    scoring_error=f'{type(exc).__name__}: {exc}',
                )
                print(
                    f'[ArtifactGRPO] scoring worker failed for {sid}; dropping trajectory: '
                    f'{type(exc).__name__}: {exc}',
                    flush=True,
                )
            r.update(group_id=key[0],trajectory_id=key[1]);records[key]=r
    if os.environ.get('OPENCLAW_ARTIFACT_GRPO_STRICT')=='1':
        try:require_complete_batch(records)
        except RuntimeError as exc:
            out=root.parent.parent/'artifact_advantage_audit';out.mkdir(exist_ok=True)
            (out/'INCOMPLETE_BATCH.json').write_text(json.dumps({'error':str(exc),'records':list(records.values())},ensure_ascii=False,indent=2))
            raise
    expected=int(os.environ.get('N_SAMPLES_PER_PROMPT','8'))
    counts=collections.Counter(k[0] for k in identities)
    if os.environ.get("OPENCLAW_ARTIFACT_GRPO_STRICT")=="1" and any(n!=expected for n in counts.values()):
        raise RuntimeError(f"incomplete A-GRPO groups: {dict(counts)}")
    # A trajectory whose verifier, archive, mapping, or score projection is not
    # usable is left out individually. Scored siblings retain artifact credit.
    unavailable_groups=set()
    unscored=sorted(
        key for key,r in records.items()
        if r.get('scoring_error') or r.get('infrastructure_error')
    )
    if unscored:
        print(f'[ArtifactGRPO] leaving {len(unscored)} unscored trajectories out of the update: {unscored}',flush=True)
    projected,audit=project_available_groups(samples,records,unavailable_groups,
        float(os.environ.get('OPENCLAW_ARTIFACT_GRPO_STRENGTH','1')),set(unscored))
    out=Path(os.environ['OPENCLAW_SPAN_TRACE_ROOT']).parent/'artifact_advantage_audit';out.mkdir(exist_ok=True)
    stamp=hashlib.sha256('\n'.join(sorted(identities.values())).encode()).hexdigest()[:20]
    (out/f'{stamp}.json').write_text(json.dumps({'sessions':list(identities.values()),'trajectories':audit,'sample_count':len(samples),'excluded_unscored_trajectories':[list(key) for key in unscored],'excluded_infrastructure_groups':sorted(unavailable_groups),'infrastructure_errors':[r for r in records.values() if r.get('infrastructure_error')]},ensure_ascii=False,indent=2))
    return projected


def project_available_groups(samples,records,excluded_groups,strength=1.0,excluded_trajectories=()):
    """Drop damaged groups and individual unscored trajectories from the loss.

    Mutate mask lists in place: rollout.py has already attached these exact
    lists to train_data; logical-rollout denominators are computed afterwards.
    Excluded trajectories get no outcome-only advantage.
    """
    excluded_trajectories=set(excluded_trajectories)
    def dropped(sample):
        return sample.group_index in excluded_groups or (sample.group_index,sample.index) in excluded_trajectories
    positions=[i for i,s in enumerate(samples) if not dropped(s)]
    if not positions:raise RuntimeError('No usable artifact-GRPO trajectories remain after scoring')
    usable=[samples[i] for i in positions]
    values,audit=project_samples(usable,records,strength,skip_broken=True)
    projected=[[0.0]*s.response_length for s in samples]
    for i,v in zip(positions,values):projected[i]=v
    for s in samples:
        if dropped(s):
            s.loss_mask[:]=[0]*len(s.loss_mask)
    trained=sum(1 for row in audit if 'dropped' not in row)
    if trained == 0:
        raise RuntimeError('No usable artifact-GRPO trajectories remain after validation')
    if excluded_groups or excluded_trajectories:
        print(f'[ArtifactGRPO] excluded infrastructure-damaged groups={sorted(excluded_groups)}; unscored trajectories={sorted(excluded_trajectories)}; training {trained} complete trajectories',flush=True)
    return projected,audit


def require_complete_batch(records,expected_groups=8,expected_samples=8):
    counts=collections.Counter(key[0] for key in records)
    failures=[r.get('session',str(key)) for key,r in records.items()
              if r.get('infrastructure_error') or r.get('scoring_error')]
    if len(counts)!=expected_groups or any(n!=expected_samples for n in counts.values()) or failures:
        raise IncompleteArtifactBatch(f'Strict A-GRPO: refusing update with incomplete/unscored batch; groups={dict(counts)}, failed_sessions={failures}')
