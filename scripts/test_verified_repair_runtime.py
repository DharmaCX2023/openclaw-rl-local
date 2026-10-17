from pathlib import Path
import io
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from verified_repair_runtime import archive_manifest, extract_snapshot, paired_accept, select_observed_boundary, action_scope_rejection


def test_pair_requires_strict_improvement_and_no_regression():
    old = {'reward': .5, 'criteria': {'a': 1, 'b': 0}}
    assert paired_accept(old, {'reward': 1, 'criteria': {'a': 1, 'b': 1}})[0]
    assert not paired_accept(old, {'reward': .9, 'criteria': {'a': 0, 'b': 1}})[0]
    assert not paired_accept(old, old)[0]
    assert not paired_accept(old, {'reward': 1, 'criteria': {'a': 1, 'c': 1}})[0]


def test_observation_mapping_uses_call_id_not_drained_step_number(tmp_path):
    import json
    path = tmp_path / 'state.json'
    call = {'id': 'real-call', 'function': {'name': 'exec', 'arguments': '{"command":"pwd"}'}}
    path.write_text(json.dumps({'turn': 27, 'step': {'tool_calls': [call]}}))
    step = {'step': 12, 'tool_calls': [{'id': 'real-call'}], 'observations': ['actual path']}
    selected, observations, _ = select_observed_boundary([path], [step])
    assert selected == str(path) and observations == ['actual path']
    step['observations'] = ['Command still running (session new-process, pid 50)']
    assert select_observed_boundary([path], [step])[0] is None


def test_scope_rejects_package_install_before_forcing():
    call = {'function': {'name': 'exec', 'arguments': '{"command":"pip install pandas -q"}'}}
    assert action_scope_rejection([call]) == 'external_network_or_package_install'


def test_failed_verifier_attempt_retains_cost(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from verified_repair_runtime import RemoteState
    runtime=RemoteState(tmp_path,'/data/clawgym_rl_runs/verified-repair-unit','unit-image')
    archive=tmp_path/'archive';archive.write_bytes(b'unit-test-only')
    def fail(*args):raise TimeoutError('unit test')
    monkeypatch.setattr(runtime,'_verify',fail)
    with pytest.raises(TimeoutError):runtime.verify(SimpleNamespace(task_id='task_1'),archive,'unit')
    records=list((tmp_path/'verifier_attempts').glob('*.json'))
    assert len(records)==1
    result=json.loads(records[0].read_text())
    assert result['status']=='failed' and result['seconds']>=0 and result['error_type']=='TimeoutError'


def test_snapshot_contents_and_modes_roundtrip(tmp_path):
    path = tmp_path / 'state.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo('./output/action.py'); item.mode = 0o755
        data = b'print(42)\n'; item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
    manifest = archive_manifest(path)
    extract_snapshot(path, tmp_path / 'restored')
    restored = tmp_path / 'restored/output/action.py'
    assert restored.read_bytes() == data
    assert restored.stat().st_mode & 0o777 == manifest['output/action.py']['mode']


@pytest.mark.parametrize('name', ['/etc/something', '../escape', 'output/../../escape'])
def test_unsafe_snapshot_paths_rejected(tmp_path, name):
    path = tmp_path / 'bad.tar.gz'
    with tarfile.open(path, 'w:gz') as archive:
        item = tarfile.TarInfo(name); item.size = 1
        archive.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='unsafe'):
        extract_snapshot(path, tmp_path / 'restored')
