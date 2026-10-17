"""Conservative content-overlap candidates, not proof of semantic causality.
Input sources: {id, turn, kind: obs|reasoning|action, text}. Caller supplies
only sources available BEFORE the target action (including known intra-turn order).
"""
import ast,csv,io,json,re,unicodedata
from collections import defaultdict
from decimal import Decimal,InvalidOperation

def norm(text):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',str(text))).strip()
def value(x):
 if x is None or isinstance(x,bool):return None
 s=norm(x)
 if not s:return None
 if re.fullmatch(r'-?\d+(?:\.\d+)?',s):
  try:return 'n:'+str(Decimal(s).normalize())
  except InvalidOperation:pass
 if len(s)<3 or s.lower() in {'true','false','null','none','unknown','success','error','passed','failed'}:return None
 return 's:'+s

def structured(text):
 """Recover self-contained JSON dicts/lists and CSV records; no execution."""
 trees=[];decoder=json.JSONDecoder();i=0
 while i<len(text):
  if text[i] in '{[':
   try:v,n=decoder.raw_decode(text[i:]);trees.append(v);i+=n;continue
   except ValueError:pass
  i+=1
 rows=[]
 def walk(v):
  if isinstance(v,dict):
   row={norm(k):value(x) for k,x in v.items() if not isinstance(x,(dict,list))};row={k:x for k,x in row.items() if x}
   if row:rows.append(row)
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 for tree in trees:walk(tree)
 lines=text.splitlines()
 for sep in [',','\t']:
  for i,line in enumerate(lines):
   if line.count(sep)<1:continue
   try:
    header=next(csv.reader([line],delimiter=sep))
    if not 2<=len(header)<=40 or any(not re.fullmatch(r'[A-Za-z_][\w ./%()-]{0,60}',x.strip()) for x in header):continue
    for row in csv.reader(lines[i+1:i+201],delimiter=sep):
     if len(row)!=len(header):break
     item={norm(k):value(v) for k,v in zip(header,row) if value(v)}
     if item:rows.append(item)
   except csv.Error:pass
 return rows

def tokens(text):return re.findall(r'[\w]+(?:[.:-][\w]+)*',norm(text),re.UNICODE)
def chunks(text):
 ts=tokens(text);out={' '.join(ts[i:i+8]) for i in range(len(ts)-7)}
 # Chinese text has no whitespace word boundaries.
 for part in re.findall(r'[\u4e00-\u9fff]{16,}',text):out.update(part[i:i+16] for i in range(len(part)-15))
 return {s for s in out if len(s)>=40 or re.fullmatch(r'[\u4e00-\u9fff]{16}',s)}

def target_text(content):
 parts=[content]
 try:
  tree=ast.parse(content)
  parts.extend(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
 except (SyntaxError,ValueError):pass
 return '\n'.join(parts)

def has_value(text,v):
 kind,s=v.split(':',1)
 if kind=='s':return bool(re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text))
 for x in re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])',text):
  if value(x)==v:return True
 return False

def match_sources(content,sources,baseline=''):
 target=norm(target_text(content));target_rows=structured(content);target_chunks=chunks(content);base_chunks=chunks(baseline)
 found=[];support=defaultdict(list)
 for source in sources:
  evidence=[];src=source['text'];seen=set()
  for row in structured(src):
   vals=set(row.values());present={v for v in vals if has_value(target,v)}
   # Require record context: >=3 distinct values, including one nonnumeric value.
   # Values already all supplied by initial prompt give no attribution evidence.
   if len(present)<3 or not any(v.startswith('s:') for v in present):continue
   if all(has_value(norm(baseline),v) for v in present):continue
   paired=[r for r in target_rows if len({k for k,v in row.items() if r.get(k)==v})>=3]
   # For free prose/code require same-line co-occurrence, not scattered literals.
   local=any(sum(has_value(norm(line),v) for v in present)>=3 for line in target_text(content).splitlines())
   if not paired and not local:continue
   key=('record',tuple(sorted(present)))
   if key not in seen:seen.add(key);evidence.append({'type':'record_values','values':sorted(present),'fields':sorted(row),'paired_fields':bool(paired)})
  common=sorted((chunks(src)&target_chunks)-base_chunks)
  if common:evidence.append({'type':'exact_text','examples':common[:3],'chunk_count':len(common)})
  if evidence:
   candidate={'source_id':source['id'],'turn':source['turn'],'kind':source['kind'],'evidence':evidence}
   found.append(candidate)
   for e in evidence:
    if e['type']=='record_values':support[('record',tuple(e['values']))].append(source['id'])
    else:
     for c in common:support[('text',c)].append(source['id'])
 for c in found:
  unique=any(c['source_id'] in ids and len(set(ids))==1 for ids in support.values())
  c['unique_evidence']=unique;c['decision']='content_supported_candidate' if unique else 'ambiguous_repeated_source'
 return {'candidates':found,'unmatched':not found,'policy':'No causal guarantee; no fuzzy paraphrase/arithmetic inference; duplicated sources are not resolved by earliest/nearest turn.'}
