"""Conservative AST data/control-dependency candidates; not a semantic oracle.
Records source lines and flags unresolved mappings. Never executes verifier code.
"""
import ast,json,collections,copy,fnmatch,itertools
from pathlib import Path
O=Path(__file__).resolve().parent
class V:
 def __init__(self,deps=(),strings=(),items=None):self.d=set(deps);self.s=set(strings);self.items=items
 def __or__(self,o):
  items=None
  if self.items is not None and o.items is not None and len(self.items)==len(o.items):items=[a|b for a,b in zip(self.items,o.items)]
  elif self.items is not None and o.items is None:items=[x|o for x in self.items]
  elif o.items is not None and self.items is None:items=[self|x for x in o.items]
  return V(self.d|o.d,list(self.s|o.s)[:256],items)
def union(vs):
 v=V()
 for x in vs:v=v|x
 return v
class Analyzer:
 def __init__(self,code,targets,criteria,string_bindings=None):
  self.tree=ast.parse(code);self.targets=targets;self.criteria=set(criteria);self.string_bindings={k:set(v) for k,v in (string_bindings or {}).items()};self.funcs={n.name:n for n in ast.walk(self.tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};self.summaries={};self.function_scores={};self.current="";self.scores=collections.defaultdict(set);self.lines=collections.defaultdict(set);self.unknown=[]
 def tag(self,v):
  for s in v.s:
   x=s.replace('\\','/').removeprefix('/root/.openclaw/workspace/').removeprefix('./')
   for p in self.targets:
    if x==p or x.endswith('/'+p) or ('*' in x and fnmatch.fnmatch(p,x.lstrip('/'))):v.d.add(p)
  return v
 def ev(self,n,e):
  if n is None:return V()
  if isinstance(n,ast.JoinedStr):
   ss={''};deps=set()
   for part in n.values:
    v=self.ev(part.value,e) if isinstance(part,ast.FormattedValue) else self.ev(part,e);deps|=v.d
    ss=set(itertools.islice((a+b for a in ss for b in (v.s if isinstance(part,ast.Constant) else {'*'})),256))
   return self.tag(V(deps,ss))
  if isinstance(n,ast.Tuple):
   items=[self.ev(x,e) for x in n.elts];v=union(items);v.items=items;return v
  if isinstance(n,ast.Constant):return self.tag(V(strings=[n.value] if isinstance(n.value,str) else []))
  if isinstance(n,ast.Name):return e.get(n.id,V())
  if isinstance(n,ast.Attribute):return self.ev(n.value,e)
  if isinstance(n,ast.BinOp):
   a,b=self.ev(n.left,e),self.ev(n.right,e);v=a|b
   if isinstance(n.op,(ast.Div,ast.Add)):
    sep='/' if isinstance(n.op,ast.Div) else '';v.s=set(itertools.islice((x+sep+y for x in a.s for y in b.s),256)) if a.s and b.s else v.s
   return self.tag(v)
  if isinstance(n,ast.Call):
   args=[self.ev(x,e) for x in n.args]+[self.ev(k.value,e) for k in n.keywords];v=union(args)
   if isinstance(n.func,ast.Attribute):v=v|self.ev(n.func.value,e)
   # Resolve task configuration lookups supplied by the caller. Verifiers
   # commonly derive output paths via cfg.get("output_path"), so preserving
   # those concrete strings is necessary to follow criterion-to-file flow.
   if isinstance(n.func,ast.Attribute) and n.func.attr=='get' and n.args:
    for key in self.ev(n.args[0],e).s:
     v.s|=self.string_bindings.get(key,set())
   if isinstance(n.func,ast.Name) and n.func.id in self.summaries:
    summary=self.summaries[n.func.id]
    def substitute(vv):
     vv=V(vv.d,vv.s,[substitute(x) for x in vv.items] if vv.items is not None else None)
     for i,a in enumerate(args):
      token='@'+n.func.id+':'+str(i)
      if token in vv.d:vv.d.remove(token);vv.d|=a.d;vv.s|=a.s
     return vv
    v=substitute(summary)
    for c,dep in self.function_scores.get(n.func.id,{}).items():
     self.scores[c]|=substitute(V(dep)).d;self.lines[c].add(n.lineno)
   if isinstance(n.func,ast.Attribute) and n.func.attr in ['join','joinpath']:
    vals=[self.ev(a,e).s for a in n.args]
    if vals and all(vals):
     ss={''}
     for vs in vals:ss=set(itertools.islice((a+'/'+b if a else b for a in ss for b in vs),256))
     v.s|=set(list(ss)[:256])
   return self.tag(v)
  if isinstance(n,(ast.ListComp,ast.SetComp,ast.GeneratorExp,ast.DictComp)):
   ee=dict(e)
   for g in n.generators:self.assign(g.target,self.ev(g.iter,ee),ee,V())
   return union(self.ev(x,ee) for x in ast.iter_child_nodes(n) if isinstance(x,ast.expr))
  if isinstance(n,ast.Subscript):
   rawkeys=self.ev(n.slice,e).s
   keys=rawkeys & self.criteria
   if keys and isinstance(n.value,ast.Name) and n.value.id in ['checks','scores','results','result']:return union(V(self.scores[k]) for k in keys)
   v=self.ev(n.value,e)|self.ev(n.slice,e)
   for key in rawkeys:v.s|=self.string_bindings.get(key,set())
   return self.tag(v)
  return union(self.ev(x,e) for x in ast.iter_child_nodes(n) if isinstance(x,ast.expr))
 def assign(self,t,v,e,ctrl):
  val=v|V(ctrl.d)
  if isinstance(t,ast.Name):e[t.id]=val
  elif isinstance(t,(ast.Tuple,ast.List)):
   for i,x in enumerate(t.elts):self.assign(x,v.items[i] if v.items is not None and len(v.items)==len(t.elts) else val,e,ctrl)
  elif isinstance(t,ast.Subscript):
   keys=self.ev(t.slice,e).s
   for k in keys&self.criteria:self.scores[k]|=val.d;self.lines[k].add(t.lineno)
   if isinstance(t.value,ast.Name):e[t.value.id]=e.get(t.value.id,V())|val
 def block(self,body,e,ctrl=V()):
  ret=V()
  for n in body:
   if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)):
    val=self.ev(n.value,e)
    if isinstance(n,ast.AugAssign):val=val|self.ev(n.target,e)
    for t in (n.targets if isinstance(n,ast.Assign) else [n.target]):self.assign(t,val,e,ctrl)
    if isinstance(n.value,ast.Dict):
     for k,v in zip(n.value.keys,n.value.values):
      for key in self.ev(k,e).s&self.criteria:self.scores[key]|=(self.ev(v,e)|ctrl).d;self.lines[key].add(n.lineno)
   elif isinstance(n,ast.If):
    ce=ctrl|self.ev(n.test,e);a=dict(e);b=dict(e);ret=ret|self.block(n.body,a,ce)|self.block(n.orelse,b,ce)
    for k in a.keys()|b.keys():e[k]=a.get(k,V())|b.get(k,V())
   elif isinstance(n,(ast.For,ast.While)):
    ce=ctrl|(self.ev(n.iter,e) if isinstance(n,ast.For) else self.ev(n.test,e));before=dict(e)
    if isinstance(n,ast.For):self.assign(n.target,self.ev(n.iter,e),e,ctrl)
    for _ in range(2):ret=ret|self.block(n.body,e,ce)
    for k,v in before.items():e[k]=e.get(k,V())|v
    ret=ret|self.block(n.orelse,e,ctrl)
   elif isinstance(n,ast.Try):
    before=dict(e);branches=[];normal=dict(e);ret=ret|self.block(n.body,normal,ctrl)|self.block(n.orelse,normal,ctrl);branches.append(normal)
    for h in n.handlers:
     branch=dict(before);ret=ret|self.block(h.body,branch,ctrl);branches.append(branch)
    for k in set().union(*(x.keys() for x in branches)):e[k]=union(x.get(k,V()) for x in branches)
    ret=ret|self.block(n.finalbody,e,ctrl)
   elif isinstance(n,(ast.With,ast.AsyncWith)):
    ce=ctrl
    for it in n.items:
     v=self.ev(it.context_expr,e);ce=ce|v
     if it.optional_vars:self.assign(it.optional_vars,v,e,ctrl)
    ret=ret|self.block(n.body,e,ce)
   elif isinstance(n,ast.Return):
    v=self.ev(n.value,e)
    if v.items is not None:v.items=[x|ctrl for x in v.items]
    ret=ret|v|ctrl
   elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
    c=n.value
    if isinstance(c.func,ast.Attribute) and isinstance(c.func.value,ast.Name) and c.func.attr in ['append','extend','add','update','setdefault']:
     name=c.func.value.id;e[name]=e.get(name,V())|union(self.ev(a,e) for a in c.args)|ctrl
  return ret
 def run(self):
  globalenv={};self.block([n for n in self.tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],globalenv)
  for _ in range(4):
   self.scores=collections.defaultdict(set);self.lines=collections.defaultdict(set)
   for name,f in self.funcs.items():
    saved=self.scores;self.scores=collections.defaultdict(set);self.current=name
    env=dict(globalenv)
    for i,a in enumerate(f.args.args):env[a.arg]=V([] if 'workspace' in a.arg else ['@'+name+':'+str(i)],[''] if 'workspace' in a.arg else [])
    self.summaries[name]=self.block(f.body,env);self.function_scores[name]=dict(self.scores)
    for c,ds in self.scores.items():saved[c]|=ds
    self.scores=saved
  self.block([n for n in self.tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],dict(globalenv))
  return {c:{'paths':sorted(self.scores[c]&set(self.targets)),'lines':sorted(self.lines[c]),'unresolved_parameter_dependencies':sorted(x for x in self.scores[c] if x.startswith('@'))} for c in sorted(self.criteria)}
if __name__=='__main__':
 s=json.loads((O/'summary.json').read_text());source={r['session']:r for r in json.loads((O.parent/'verifier_targets160_20260922/summary.json').read_text())['rows']};out=[]
 for r in s['rows']:
  if r['status']!='ok':continue
  rec=json.loads((O/'recovery'/f"{r['session']}.json").read_text());p=Path(rec['task_dir'])/'reward/check.py'
  a=Analyzer(p.read_text(),list(source[r['session']]['scored_targets']),list(r['history'][-1]));m=a.run()
  out.append({'session':r['session'],'task':r['task_id'],'checker':str(p),'mapping':m})
 (O/'structure_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
 print('tasks',len(out),'criteria',sum(len(r['mapping']) for r in out),'mapped',sum(bool(m['paths']) for r in out for m in r['mapping'].values()))
 for r in out[:3]:print(r['task'],r['mapping'])
