"""Check registered JobAgent, not just the independently ready job-list API."""
import json,sys,socket
from ray._raylet import GcsClient
from ray.dashboard import consts
from ray._private import ray_constants
client=GcsClient(address=sys.argv[1])
keys=client.internal_kv_keys(consts.DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX.encode(),namespace=ray_constants.KV_NAMESPACE_DASHBOARD)
for key in keys:
 value=client.internal_kv_get(key,namespace=ray_constants.KV_NAMESPACE_DASHBOARD)
 if value:
  ip,port,_=json.loads(value)
  if port>0:
   try:
    with socket.create_connection((ip,port),timeout=1):pass
    sys.exit(0)
   except OSError:pass
sys.exit(1)
