"""Behavior categories describe emitted tools, not presumed usefulness."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from analyze_verified_repair_behavior import tool_pattern,stats


def test_structured_tool_patterns_and_unknown_effects():
    calls=lambda *names:[{'function':{'name':name}} for name in names]
    assert tool_pattern([])=='no_tool_calls'
    assert tool_pattern(calls('read','read'))=='read_only'
    assert tool_pattern(calls('read','write'))=='contains_write_or_edit'
    assert tool_pattern(calls('read','exec'))=='contains_exec'
    assert tool_pattern(calls('other'))=='other'
    assert stats([None])['mean'] is None
    assert stats([1,None,3])['mean']==2
