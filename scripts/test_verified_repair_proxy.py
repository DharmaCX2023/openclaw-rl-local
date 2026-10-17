"""Transport contract tests before running real branch experiments."""
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import verified_repair_runtime  # installs the production module search path
from verified_repair_proxy import RepairProxy


def test_branch_stop_is_valid_production_stream():
    proxy = object.__new__(RepairProxy)
    result = proxy._branch_stop({'model': 'isolated-test'}, 'branch-test')

    async def consume():
        return [item async for item in proxy._stream_response(result)]

    chunks = asyncio.run(consume())
    assert chunks[-1] == 'data: [DONE]\n\n'
    first = json.loads(chunks[0][6:])
    assert first['session_id'] == 'branch-test'
    assert first['choices'][0]['delta']['content'] == ''
    assert result['response']['usage']['completion_tokens'] == 0
