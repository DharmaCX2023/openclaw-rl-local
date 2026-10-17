"""Exact integer endpoint discovery and actual CPU sampler regression."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from diagnose_verified_repair_sampler_endpoint import endpoint_columns,hash_numpy,check_sampler,scan_request


def test_hash_inverse_identifies_actual_endpoint_position():
    seed=1795547190;pos=np.array([20754,20755,20756],dtype=np.uint64)
    col=endpoint_columns(seed,pos)
    assert int(col[1])==35146
    assert all(int(x)==0xffffffff for x in hash_numpy(seed,pos,col))
    assert int(col[0])>=152064 and int(col[2])>=152064


def test_installed_sampler_endpoint_and_open_interval_fix_cpu():
    result=check_sampler('cpu')
    assert result['old_sampled_sorted_column']==35146
    assert result['open_interval_sampled_sorted_column']==0
    assert not result['installed_sampler_modified']


def test_scan_uses_actual_request_length_not_alternative_canonical_rendering():
    ids=[0]*260;ids[253]=115834
    record={'sampling_seed':1795547190,'actual_prompt_tokens':20503,
        'canonical_prompt_tokens':20511,'response_ids':ids}
    events=scan_request(record,'source_rollout','unit',152064)
    assert len(events)==1
    assert events[0]['response_position']==253 and events[0]['actual_output_id']==115834
