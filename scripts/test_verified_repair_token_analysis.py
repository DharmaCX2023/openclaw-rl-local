"""Byte-exact diagnostic mapping; no model inference or changed training spans."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).parent))
from analyze_verified_repair_tokens import byte_spans,token_buckets


class ByteTokenizer:
    added_tokens_decoder={256:SimpleNamespace(content='<|im_end|>')}
    def convert_ids_to_tokens(self,i):
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
        return bytes_to_unicode()[i]
    def decode(self,ids,**kwargs):
        return b''.join(b'<|im_end|>' if i==256 else bytes([i]) for i in ids).decode('utf-8',errors='replace')


def test_split_utf8_and_added_token_preserve_every_position():
    t=ByteTokenizer();text='说明\n<tool_call>\n{"name":"exec","command":"print(\\"你好\\")"}\n</tool_call>'
    ids=list(text.encode('utf-8'))+[256]
    decoded,spans=byte_spans(t,ids)
    assert decoded==text+'<|im_end|>' and len(spans)==len(ids)
    labels,exact=token_buckets(t,ids)
    assert exact and len(labels)==len(ids)
    assert labels[-1]=='control'
    assert labels.count('tool_name')==4
    for start,word in [('说明','prose'),('你好','code_payload')]:
        offset=text.encode('utf-8').index(start.encode('utf-8'))
        assert labels[offset:offset+len(start.encode('utf-8'))]==[word]*len(start.encode('utf-8'))


def test_invalid_utf8_is_unmapped_not_silently_retokenized():
    labels,exact=token_buckets(ByteTokenizer(),[0xff])
    assert not exact and labels==['unmapped']
