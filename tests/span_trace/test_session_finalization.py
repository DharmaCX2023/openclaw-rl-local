"""Exercise the real archive endpoint without loading GPU/model dependencies."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest

source=ast.parse((Path(__file__).resolve().parents[2]/'clawgym-official/clawgym_rl_api_server.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='ClawGymRLAPIServer')
app=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build_app')
archive=next(n for n in app.body if isinstance(n,ast.AsyncFunctionDef) and n.name=='get_conversation_log')
guard=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_assert_session_open')
archive.decorator_list=[]
class HTTPException(Exception):
    def __init__(self,status_code,detail):self.status_code=status_code
ns={'Any':object,'Request':object,'HTTPException':HTTPException,'_require_string':lambda x,_:x}
exec(compile(ast.Module(body=[archive,guard],type_ignores=[]),'<actual-api-functions>','exec'),ns)

class FinalizationTest(unittest.IsolatedAsyncioTestCase):
    async def test_inflight_response_cannot_extend_archived_trajectory(self):
        owner=SimpleNamespace()
        # All per-session maps referenced by the actual archive endpoint.
        for node in ast.walk(archive):
            if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='owner':
                setattr(owner,node.attr,{})
        owner._finalized_sessions=set()
        owner._conversation_logs={'s':[{'step':1}]}
        owner._session_metadata={'s':{'group_index':0}}
        owner._pending_step_data={'s':{1:{'response_ids':[42]}}}
        owner._step_counts={'s':1}
        class Request:
            app=SimpleNamespace(state=SimpleNamespace(owner=owner))
            async def json(self):return {'session_id':'s'}
        started=asyncio.Event();finish=asyncio.Event()
        async def late_response():
            ns['_assert_session_open'](owner,'s')
            started.set();await finish.wait()
            ns['_assert_session_open'](owner,'s')
            owner._pending_step_data['s'][2]={'response_ids':[99]}
        task=asyncio.create_task(late_response());await started.wait()
        archived=await ns['get_conversation_log'](Request())
        finish.set()
        with self.assertRaises(HTTPException) as caught:await task
        self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(archived['steps'],[{'step':1}])
        self.assertEqual(list(owner._pending_step_data['s']),[1])
        with self.assertRaises(HTTPException):ns['_assert_session_open'](owner,'s')
        ns['_assert_session_open'](owner,'other-live-session')

if __name__=='__main__':unittest.main()
