import plugin from './index.js';
import fs from 'node:fs';
import {exec} from 'node:child_process';
import {promisify} from 'node:util';
const hooks={};plugin.register({on:(n,f)=>hooks[n]=f});
const root='/root/.openclaw/workspace';fs.mkdirSync(root,{recursive:true});process.chdir(root);
async function action(id,tool,params,run){const e={toolName:tool,toolCallId:id,params},c={toolCallId:id,runId:'smoke',sessionId:'smoke'};const r=hooks.before_tool_call(e,c);const result=await run(r?.params||params);hooks.after_tool_call({...e,result},c);}
await action('write-script','write',{path:root+'/generate.py',content:'from pathlib import Path\nPath("a.json").write_text("a")\n'},p=>fs.writeFileSync(p.path,p.content));
await Promise.all([
 action('exec-a','exec',{command:'python3 generate.py'},p=>promisify(exec)(p.command)),
 action('exec-b','exec',{command:`python3 -c 'from pathlib import Path; Path("b.json").write_text("b")'`},p=>promisify(exec)(p.command))
]);
console.log('SMOKE_OK',fs.readFileSync('a.json','utf8'),fs.readFileSync('b.json','utf8'));
