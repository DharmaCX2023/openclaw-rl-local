#!/usr/bin/env python3
"""Real frozen-initialization 8B forward/backward diagnostics; NO optimizer step."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('FLASH_ATTENTION_DETERMINISTIC', '1')
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from verified_repair_runtime import ROOT, save_json
from verified_repair_losses import all_blocked_margin, action_dpo, sampling_support

MODEL = ROOT / 'outputs/clawgym-8b-sft-formal-20260902-165551/final_model'


def action_forward(model, prefix, response, *, gold=None):
    ids = torch.tensor([prefix + response[:-1]], device='cuda', dtype=torch.long)
    targets = torch.tensor(response, device='cuda', dtype=torch.long)
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, len(prefix)-1:]
    assert hidden.shape[0] == len(response)
    pieces, entry_sums, counts, diagnostics = [], [], [], []
    # Project response rows only. Never allocate context_length × vocab logits.
    for begin in range(0, len(response), 128):
        end = min(begin + 128, len(response))
        logits = model.lm_head(hidden[begin:end]).float()
        selected = targets[begin:end]
        lp = -F.cross_entropy(logits, selected, reduction='none')
        native_members = [r['candidate_ids'] for r in gold[begin:end]] if gold is not None else None
        entry, stats = all_blocked_margin(logits, selected, native_candidate_ids=native_members)
        training_support, _ = sampling_support(logits)
        train_blocked = (~training_support.gather(-1, selected[:, None]).squeeze(-1)).tolist()
        pieces.append(lp.sum())
        entry_sums.append(entry * stats['blocked_count'])
        counts.append(stats['blocked_count'])
        for offset, blocked in enumerate(stats['blocked_mask'].tolist()):
            i = begin + offset
            row = {'position': i, 'target': response[i], 'train_blocked': train_blocked[offset],
                   'entry_blocked': blocked, 'entry_mask_source': stats['mask_source'],
                   'train_logprob_T1': float(lp[offset].detach()),
                   'train_target_sampling_logit': float(stats['target_sampling_logit'][offset]),
                   'train_boundary_sampling_logit': float(stats['boundary'][offset])}
            if gold is not None:
                ref = gold[i]
                assert ref['position'] == i and ref['target'] == response[i]
                row.update(sglang_blocked=ref['blocked'],
                           native_boundary_sampling_logit=ref['boundary_sampling_logit'],
                           native_target_sampling_logit=ref['target_sampling_logit'],
                           entry_hinge_training=float(stats['hinge'][offset]),
                           entry_hinge_native=max(0, .1+ref['boundary_sampling_logit']-ref['target_sampling_logit']),
                           logprob_T1_abs_diff=abs(row['train_logprob_T1']-ref['target_logprob_T1']))
            diagnostics.append(row)
    total_lp = torch.stack(pieces).sum()
    entry_loss = torch.stack(entry_sums).sum() / torch.stack(counts).sum().clamp_min(1)
    return total_lp, entry_loss, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    torch.manual_seed(20260907)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='cuda')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    assert model.config.attention_dropout == 0
    parameters = list(model.parameters())
    versions = [p._version for p in parameters]
    summary = []
    records = [json.loads(p.read_text()) for p in sorted((out / 'sources').glob('*.json'))]
    records = [r for r in records if r.get('split') == 'engineering_smoke']
    if len(records) != 5:
        raise ValueError('exactly five engineering smoke sources required')
    for record in records:
        if not record.get('selected_state'):
            raise ValueError('missing real restorable state: ' + record['session_id'])
        source = json.loads(Path(record['selected_state']).read_text())
        sid = source['session_id']
        original = source['step']['response_ids']
        teacher_file = out / 'teacher' / (sid + '.json')
        teacher = json.loads(teacher_file.read_text()) if teacher_file.exists() else {}
        positive = teacher.get('ids') if teacher.get('status') == 'proposed' else original
        label = 'repair' if teacher.get('status') == 'proposed' else 'original'
        gold_path = out / 'sampler' / (sid + '-' + label + '.jsonl')
        gold = [json.loads(line) for line in gold_path.open()]
        if len(gold) != len(positive):
            raise ValueError('full-action native sampler audit missing')
        origin_hash = hashlib.sha256(json.dumps(source['step']['prompt_ids']).encode()).hexdigest()
        assert all(r['origin_ids_sha256'] == origin_hash and
                   r['prefix_length'] == len(source['step']['prompt_ids'])+i
                   and r['generated_prefix_matches_teacher'] for i, r in enumerate(gold))
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        pos, entry, details = action_forward(model, source['step']['prompt_ids'], positive, gold=gold)
        neg, _, _ = action_forward(model, source['step']['prompt_ids'], original)
        # At the common untouched SFT initialization, reference equals Student.
        # Detaching these identical-weight scores is valid only for this zero-
        # update diagnostic, NEVER a replacement for the fixed ref in training.
        dpo = action_dpo(pos, neg, pos.detach(), neg.detach(), beta=.1)
        nll = -pos / len(positive)
        diagnostic_loss = dpo + entry + nll
        diagnostic_loss.backward()
        norm_squared = 0.0
        finite = True
        grad_parameters = 0
        for p in parameters:
            if p.grad is None:
                continue
            grad_parameters += p.numel()
            finite &= bool(torch.isfinite(p.grad).all())
            norm_squared += float(p.grad.float().square().sum())
        assert finite and grad_parameters > 0
        assert all(p._version == v for p, v in zip(parameters, versions)), 'model parameter mutation'
        mismatch = sum(d['train_blocked'] != d['sglang_blocked'] for d in details)
        result = {'session_id': sid, 'task_id': source['task_id'],
            'prompt_ids_sha256': hashlib.sha256(json.dumps(source['step']['prompt_ids']).encode()).hexdigest(),
            'positive_ids_sha256': hashlib.sha256(json.dumps(positive).encode()).hexdigest(),
            'negative_ids_sha256': hashlib.sha256(json.dumps(original).encode()).hexdigest(),
            'training_backend': 'transformers_flash_attention_2_bf16',
            'sampling_temperature': .7, 'sampling_top_p': .8, 'sampling_top_k': 20,
            'positive_tokens': len(positive), 'negative_tokens': len(original),
            'real_teacher_proposal_for_diagnostic': label == 'repair',
            'positive_logprob_sum_T1': float(pos.detach()), 'negative_logprob_sum_T1': float(neg.detach()),
            'dpo_loss': float(dpo.detach()), 'all_blocked_margin_loss': float(entry.detach()),
            'repair_nll': float(nll.detach()), 'diagnostic_loss_weights': [1,1,1],
            'diagnostic_coefficients_are_not_training_calibration': True,
            'finite_gradients': finite, 'parameters_with_gradients': grad_parameters,
            'gradient_norm': norm_squared ** .5, 'model_updated': False,
            'optimizer_steps': 0, 'seconds': time.monotonic() - started,
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'native_train_membership_mismatches': mismatch,
            'native_train_membership_agreement': 1-mismatch/len(details),
            'actual_native_margin_mask_verified': all(d['entry_blocked'] == d['sglang_blocked']
                and d['entry_mask_source'] == 'native_sglang_candidate_ids' for d in details),
            'entry_boundary_definition': 'detached min training-temperature-logit over actual SGLang candidate IDs',
            'native_candidates_freshness': 'same immutable SFT weights; zero optimizer updates in either backend',
            'token_details': details}
        save_json(out / 'gradient_smoke_native_support' / (sid + '.json'), result)
        summary.append({k: v for k, v in result.items() if k != 'token_details'})
        print(json.dumps(summary[-1]), flush=True)
        del pos, neg, entry, dpo, nll, diagnostic_loss
    save_json(out / 'gradient_smoke_native_support.json', summary)


if __name__ == '__main__':
    main()
