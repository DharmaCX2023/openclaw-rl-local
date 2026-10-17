import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
const root='/root/.openclaw/span_trace';
const src='/root/.openclaw/skills/span-trace';
const pending=new Map();
function write(name,value){fs.mkdirSync(root,{recursive:true});fs.writeFileSync(path.join(root,name),JSON.stringify(value));}
function snap(){return JSON.parse(execFileSync('python3',[src+'/snapshot.py'],{maxBuffer:64*1024*1024,timeout:120000}).toString());}
function id(e,c){return String(e.toolCallId||c.toolCallId||'');}
export default {id:'span-trace',name:'Span filesystem provenance',register(api){
 api.on('before_tool_call',(e,c)=>{
  const call=id(e,c);if(!call)throw new Error('span-trace: missing toolCallId');
  const key=crypto.createHash('sha256').update(String(c.runId||e.runId||'')+'\0'+call).digest('hex').slice(0,32);
  const rec={key,tool_call_id:call,run_id:c.runId||e.runId,session_id:c.sessionId,tool:e.toolName,params:e.params,start_ms:Date.now()};
  pending.set(call,rec);write(key+'.before.json',snap());write(key+'.action.json',rec);
  if(e.toolName==='exec' && typeof e.params.command==='string'){
   const encoded=Buffer.from(e.params.command).toString('base64');
   return {params:{...e.params,command:`python3 ${src}/trace_exec.py ${key} ${encoded}`}};
  }
 });
 api.on('after_tool_call',(e,c)=>{
  const call=id(e,c),rec=pending.get(call);if(!rec)return;
  write(rec.key+'.after.json',snap());write(rec.key+'.result.json',{tool_call_id:call,end_ms:Date.now(),error:e.error||null,result:e.result});pending.delete(call);
 });
}};
