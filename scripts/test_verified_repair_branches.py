"""Protocol tests with synthetic verifier results, never experimental samples."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import verified_repair_branches as branches


@pytest.mark.parametrize('fail_repeat,expected_calls,accepted', [(None,4,True),(0,2,False),(1,4,False)])
def test_both_prespecified_repeats_must_pass(tmp_path, monkeypatch, fail_repeat, expected_calls, accepted):
    calls = []
    def fake_run(state, forced, pair_index, side, *args):
        calls.append((pair_index, side))
        value = .5 if side == 'original' or fail_repeat == pair_index else 1.
        return {'status': 'completed', 'source_first_observations_equal': True,
                'verifier': {'reward': value, 'criteria': {'a': 1, 'b': int(value == 1)}}}
    monkeypatch.setattr(branches, 'run_branch', fake_run)
    state = {'session_id': 'unit-only', 'task_id': 'unit-task', 'split': 'train',
             'step': {'response_text': 'unit-original', 'response_ids': [1]}}
    result = branches.verify_candidate(state, {'status': 'proposed'}, 0,
        SimpleNamespace(output_dir=tmp_path), None, None, None)
    assert result['accepted'] is accepted
    assert len(calls) == expected_calls
    assert calls[:2] == [(0, 'original'), (0, 'repair')]
    if expected_calls == 4:
        assert calls[2:] == [(1, 'repair'), (1, 'original')]
