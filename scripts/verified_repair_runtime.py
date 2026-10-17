"""Run-owned ClawGym snapshots and isolated, unmodified code verifiers."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'clawgym-official'), str(ROOT / 'slime')]
import clawgym_rl_rollout as rollout


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def command(args):
    return ' '.join(shlex.quote(str(x)) for x in args)


def action_scope_rejection(calls):
    """Conservative pilot scope: no live network/package or process state."""
    import re
    for call in calls:
        function = call.get('function', call)
        name = function['name']
        if name not in {'read', 'write', 'edit', 'exec'}:
            return 'out_of_scope_tool:' + name
        arguments = function.get('arguments', {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if name == 'exec':
            cmd = arguments.get('command', '')
            if arguments.get('background') is True:
                return 'background_execution'
            if re.search(r'\b(?:pip3?|uv|apt|apt-get|npm|pnpm)\s+(?:install|add)|https?://|\b(?:curl|wget)\b', cmd):
                return 'external_network_or_package_install'
        if any(p in json.dumps(arguments) for p in ('/tmp/openclaw', '/tmp/jiti')):
            return 'uncaptured_gateway_runtime_dependency'
    return None


def select_observed_boundary(paths, steps):
    """Select by immutable call IDs, never drainable conversation row numbers."""
    rejected = []
    for path in reversed(paths):
        candidate = json.loads(Path(path).read_text())
        calls = candidate['step']['tool_calls']
        if not calls:
            continue
        reason = action_scope_rejection(calls)
        expected = [c['id'] for c in calls]
        matches = [s for s in steps if [c['id'] for c in s.get('tool_calls', [])] == expected]
        observations = matches[0].get('observations') if len(matches) == 1 else None
        if not reason and (observations is None or len(observations) != len(calls)):
            reason = 'source_tool_observations_missing_or_ambiguous'
        if not reason and any('command still running (session' in str(o).lower()
                              or 'process still running.' in str(o).lower() for o in observations):
            reason = 'original_action_background_result'
        if reason:
            rejected.append({'path': str(path), 'reason': reason})
            continue
        return str(path), observations, rejected
    for path in reversed(paths):
        candidate = json.loads(Path(path).read_text())
        if not candidate['step']['tool_calls']:
            return str(path), [], rejected
    return None, None, rejected


def archive_manifest(path):
    result = {}
    with tarfile.open(path) as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name in {'', '.'}:
                continue
            if member.name.startswith('/') or '..' in PurePosixPath(name).parts:
                raise ValueError('unsafe archive path')
            if not (member.isfile() or member.isdir() or member.issym()):
                raise ValueError(f'unsupported snapshot entry: {name}')
            record = {'mode': member.mode, 'type': member.type.decode(), 'size': member.size}
            if member.issym():
                # No external symlink dependencies in the initial pilot.
                target = PurePosixPath(member.linkname)
                if target.is_absolute() or '..' in target.parts:
                    raise ValueError('external symlink dependency')
                record['link'] = member.linkname
            if member.isfile():
                record['sha256'] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            result[name] = record
    return result


def extract_snapshot(path, destination):
    archive_manifest(path)  # validate before writing anything
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        archive.extractall(destination, filter='data')


class RemoteState:
    def __init__(self, output, remote_base, image):
        self.output = Path(output)
        self.base = remote_base.rstrip('/')
        if not self.base.startswith('/data/clawgym_rl_runs/verified-repair-'):
            raise ValueError('run-owned remote base required')
        self.image = image
        self.owned = {}
        self.lock = threading.RLock()
        history = self.output / 'owned_containers.json'
        self.ownership_history = json.loads(history.read_text()) if history.exists() else {}

    def register(self, runtime):
        if not runtime.remote_root.startswith(self.base + '/'):
            raise ValueError('foreign runtime')
        with self.lock:
            self.owned[runtime.container_name] = runtime
            self.ownership_history.update({
                name: {'remote_root': r.remote_root, 'session_id': r.session_id,
                       'port': int(r.gateway_url.rsplit(':', 1)[1])}
                for name, r in self.owned.items()})
            save_json(self.output / 'owned_containers.json', self.ownership_history)

    def check_owned(self, runtime):
        if self.owned.get(runtime.container_name) is not runtime:
            raise ValueError('runtime is not owned by this experiment')

    def snapshot(self, runtime, destination, *, audit_external=True):
        self.check_owned(runtime)
        started = time.perf_counter()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cname = runtime.container_name
        diff = rollout._remote_run(command(['docker', 'diff', cname]), timeout=30).stdout.decode()
        # /root/.openclaw is bind-mounted. No unseen writable-layer dependency
        # may be silently excluded. Parent directory metadata is not file data.
        # Docker also reports bind-mount placeholders and Gateway's own fresh
        # log/lock/JIT directories. These are NOT agent state and must not be
        # restored over a fresh Gateway. All other writable-layer changes,
        # including installed packages and arbitrary /tmp scripts, still reject.
        def gateway_runtime_only(path):
            return (path in {'/root', '/root/.openclaw', '/tmp', '/tmp/jiti',
                             '/tmp/openclaw', '/tmp/openclaw-0'}
                    or path.startswith('/tmp/jiti/')
                    or (path.startswith('/tmp/openclaw/openclaw-') and path.endswith('.log'))
                    or (path.startswith('/tmp/openclaw-0/gateway.') and path.endswith('.lock')))
        outside = [line for line in diff.splitlines()
                   if not gateway_runtime_only(line.split(' ', 1)[-1])]
        if audit_external and outside:
            raise ValueError('outside_workspace_mutation:' + '\n'.join(outside)[:500])
        rollout._remote_run(command(['docker', 'pause', cname]), timeout=30)
        try:
            tar_cmd = command(['tar', '-C', runtime.remote_root + '/workspace',
                               '--exclude=./reward', '--exclude=./.openclaw_reward_payload.json',
                               '-czf', '-', '.'])
            with destination.open('wb') as handle:
                result = subprocess.run(rollout._ssh_base() + [tar_cmd], stdout=handle,
                                        stderr=subprocess.PIPE, timeout=90)
            if result.returncode:
                raise RuntimeError('snapshot tar failed, exit=' + str(result.returncode))
            if destination.stat().st_size > 512 * 2**20:
                raise ValueError('snapshot exceeds 512 MiB pilot limit')
            manifest = archive_manifest(destination)
            return {'archive': str(destination), 'sha256': sha(destination),
                    'bytes': destination.stat().st_size, 'manifest': manifest,
                    'seconds': time.perf_counter() - started,
                    'docker_diff': diff, 'workspace_frozen': True,
                    'outside_workspace_audit': 'strict_except_gateway_runtime_files',
                    'runtime_files_not_restored': True}
        finally:
            rollout._remote_run(command(['docker', 'unpause', cname]), timeout=30, check=False)

    def verify(self, task, archive, label):
        """Account for failed verification attempts as well as successful ones."""
        import re
        if not re.fullmatch('[A-Za-z0-9_-]+', label):
            raise ValueError('unsafe verifier label')
        started = time.time()
        attempt = {'label': label, 'task_id': task.task_id, 'started_at': started,
                   'archive_sha256': sha(archive), 'status': 'started'}
        path = self.output / 'verifier_attempts' / (label + '-' + str(time.time_ns()) + '.json')
        save_json(path, attempt)
        try:
            result = self._verify(task, archive, label)
            attempt.update(status='completed', verifier_result=str(self.output/'verifiers'/(label+'.json')))
            return result
        except Exception as exc:
            attempt.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            attempt.update(finished_at=time.time(), seconds=time.time()-started)
            save_json(path, attempt)

    def _verify(self, task, archive, label):
        """Original reward.sh and check.py in a NEW non-network verifier container.

        The branch workspace is never changed. The exact official script is
        executed, not a reweighted or rewritten criterion surrogate.
        """
        import re
        if not re.fullmatch('[A-Za-z0-9_-]+', label):
            raise ValueError('unsafe verifier label')
        started = time.perf_counter()
        remote_root = self.base + '/verifiers/' + label
        cname = 'verified-repair-v-' + hashlib.sha256(remote_root.encode()).hexdigest()[:20]
        rollout._remote_run(command(['mkdir', '-p', remote_root + '/workspace']), timeout=30)
        with Path(archive).open('rb') as handle:
            upload = subprocess.run(rollout._ssh_base() + [command(['tar', '-xzf', '-', '-C', remote_root + '/workspace'])],
                                    stdin=handle, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
        if upload.returncode:
            raise RuntimeError('verifier snapshot upload failed')
        assets = rollout._tar_bytes([(task.task_dir / 'reward', 'reward')], gzip=True)
        rollout._remote_run(command(['tar', '-xzf', '-', '-C', remote_root + '/workspace']),
                            input_data=assets, timeout=60)
        prefix = ['docker', 'run', '--rm', '--name', cname, '--network', 'none',
                  '--user', 'root', '-v', remote_root + '/workspace:/root/.openclaw/workspace', self.image]
        official = rollout._remote_run(command(prefix + ['bash', '/root/.openclaw/workspace/reward/reward.sh']),
                                        timeout=180, check=False)
        raw = rollout._remote_run(command(prefix + ['python3', '/root/.openclaw/workspace/reward/check.py',
                                                    '/root/.openclaw/workspace']), timeout=180, check=False)
        if official.returncode or raw.returncode:
            raise RuntimeError('original verifier failed: ' + str((official.returncode, raw.returncode)))
        value = json.loads(raw.stdout.decode().strip())
        reward = float(official.stdout.decode().strip().splitlines()[-1])
        if not isinstance(value, dict) or not math.isfinite(reward):
            raise ValueError('verifier did not emit a finite structured result')
        criteria = {k: int(v) for k, v in value.items() if isinstance(v, bool)}
        if len(criteria) < 2:
            raise ValueError('reliable Boolean criterion vector unavailable')
        result = {'reward': reward, 'criteria': criteria, 'raw_verifier': value,
                  'reward_stdout': official.stdout.decode(), 'checker_stdout': raw.stdout.decode(),
                  'reward_sha256': sha(task.task_dir / 'reward/reward.sh'),
                  'checker_sha256': sha(task.task_dir / 'reward/check.py'),
                  'seconds': time.perf_counter() - started, 'network': 'none',
                  'isolated_workspace': remote_root + '/workspace'}
        save_json(self.output / 'verifiers' / (label + '.json'), result)
        return result

    def stop(self, runtime):
        self.check_owned(runtime)
        # Preserve remote workspaces for diagnosis. Remove only this exact
        # registered container, and keep logs locally before removal.
        result = rollout._remote_run(command(['docker', 'logs', runtime.container_name]), timeout=30, check=False)
        logs = self.output / 'gateway_logs'
        logs.mkdir(exist_ok=True)
        (logs / (runtime.session_id + '.log')).write_bytes(result.stdout + result.stderr)
        rollout._remote_run(command(['docker', 'rm', '-f', runtime.container_name]), timeout=60, check=False)
        rollout._release_remote_gateway_port(int(runtime.gateway_url.rsplit(':', 1)[1]))


def paired_accept(original, repaired, tolerance=1e-8):
    if original['criteria'].keys() != repaired['criteria'].keys():
        return False, 'criterion_schema_changed'
    if any(v and not repaired['criteria'][k] for k, v in original['criteria'].items()):
        return False, 'criterion_regression'
    if repaired['reward'] <= original['reward'] + tolerance:
        return False, 'no_strict_short_window_improvement'
    return True, 'strict_improvement'
