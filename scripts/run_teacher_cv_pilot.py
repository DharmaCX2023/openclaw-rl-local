#!/usr/bin/env python3
"""Frozen-model, real-rollout Teacher-CV screening. No optimizer or environment writes.

This is deliberately NOT a full-RL efficacy or unbiasedness claim: historical
sampling support is reconstructed with HF, and only a prespecified early window
is differentiated. Primary moments use ALL parameter gradients, not logit or
last-layer gradients. Controls additionally use fixed coordinate probes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/clawgym-8b-sft-formal-20260902-165551/final_model"
SOURCE = ROOT / "outputs/qwen3-8b-centered-formal-seed0-smoke-20260906-0618"
ANCHORS = ROOT / "outputs/opsd-anchor-base-seed0-800-20260831/full800/anchors.jsonl"
SEED = 20260907
KINDS = ("grpo", "real", "shuffled", "constant", "unhinted")


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, utc=time.strftime("%FT%TZ", time.gmtime()), **kwargs)), flush=True)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def runs(mask):
    padded = torch.nn.functional.pad(torch.as_tensor(mask).bool().to(torch.int8), (1, 1))
    delta = padded[1:] - padded[:-1]
    return list(zip(torch.where(delta == 1)[0].tolist(), torch.where(delta == -1)[0].tolist()))


def static_hint(anchors):
    # No outcome, incorrect-answer cue, future tool result, or final answer.
    lines = ["Teacher-only static task evidence:"]
    for i, a in enumerate(anchors, 1):
        lines.append(f"{i}. Condition: {a['condition']}")
        for key, label in (("artifact", "Artifact"), ("verified_fact", "Evidence"), ("explanation", "Explanation")):
            if a.get(key):
                lines.append(f"   {label}: {a[key]}")
    lines += ["Use this evidence when solving the original task.",
              "Do not quote or discuss this teacher-only information."]
    return "\n\n" + "\n".join(lines)


def inject(tokenizer, prefix, hint):
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for p in range(len(prefix) - len(marker) + 1):
        if prefix[p:p + len(marker)] == marker:
            pos = prefix.index(end, p + len(marker))
            h = tokenizer.encode(hint, add_special_tokens=False)
            return prefix[:pos] + h + prefix[pos:]
    raise ValueError("Original user message not found; refuse guessed insertion")


def prepare(out):
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    records = json.loads((SOURCE / "opsd_step_summaries/0.json").read_text())["records"]
    d = torch.load(SOURCE / "debug_train/0_0.pt", map_location="cpu", weights_only=True)["rollout_data"]
    anchor_map = {x["task_id"]: x["anchors"] for x in map(json.loads, ANCHORS.open())}
    tasks = sorted({r["task_id"] for r in records})
    calibration = sorted(tasks, key=lambda t: hashlib.sha256(f"{SEED}:{t}".encode()).hexdigest())[:4]
    proxy = {}
    for line in (SOURCE / "proxy_records.jsonl").open():
        row = json.loads(line)
        if "messages" in row:
            proxy[(row["session_id"], row["step"])] = row
    prepared = {task: [] for task in tasks}
    for i, r in enumerate(records):
        if r["segment_index"] != 0:
            continue
        assert r["group_index"] == d["group_indices"][i]
        assert r["trajectory_index"] == d["sample_indices"][i]
        ids = d["tokens"][i].tolist()
        offset = len(ids) - d["response_lengths"][i]
        shuffled_task = tasks[(tasks.index(r["task_id"]) + 1) % len(tasks)]
        item = {k: r[k] for k in ("task_id", "session_id", "trajectory_index", "advantage", "raw_reward")}
        item["logical_tokens"] = int(d["logical_rollout_token_counts"][i])
        item["turns"] = []
        for step, (start, end) in enumerate(runs(d["loss_masks"][i])[:4], 1):
            pr = proxy[(r["session_id"], step)]
            prefix = ids[:offset + start]
            response = ids[offset + start:offset + min(end, start + 256)]
            # Exact saved training prefix must agree with saved proxy text.
            assert tok.decode(prefix, clean_up_tokenization_spaces=False) == pr["prompt_text"]
            # Keep saved IDs: decode->encode can merge BPE boundaries in prior turns.
            sp = pr["sampling_params"]
            assert sp["temperature"] == .7 and sp["top_k"] == 20 and sp["top_p"] == .8
            assert sp.get("repetition_penalty", 1) == 1 and sp.get("min_p", 0) == 0
            old = d["log_probs"][i][start:start + len(response)].float()
            rollout = d["rollout_log_probs"][i][start:start + len(response)].float()
            teacher = inject(tok, prefix, static_hint(anchor_map[r["task_id"]]))
            shuffled = inject(tok, prefix, static_hint(anchor_map[shuffled_task]))
            assert max(len(teacher), len(shuffled)) + len(response) <= 65536
            item["turns"].append(dict(step=step, prefix=prefix, response=response,
                teacher_prefix=teacher, shuffled_prefix=shuffled,
                rollout_logp=rollout.tolist(), old_train_logp=old.tolist(),
                tis=(old - rollout).exp().clamp(0, 2).tolist()))
        prepared[r["task_id"]].append(item)
    for task, items in prepared.items():
        assert len(items) == 8 and len({i["session_id"] for i in items}) == 8
        items.sort(key=lambda i: i["trajectory_index"])
        write_json(out / "inputs" / f"{task}.json", items)
    manifest = dict(model=str(MODEL), source=str(SOURCE), tasks=tasks,
        calibration_tasks=calibration, validation_tasks=[t for t in tasks if t not in calibration],
        seed=SEED, logical_trajectories=64, max_turns=4, max_tokens_per_turn=256,
        turns=sum(len(i["turns"]) for x in prepared.values() for i in x),
        tokens=sum(len(t["response"]) for x in prepared.values() for i in x for t in i["turns"]),
        cv_normalizer=1024, student_temperature=.7, teacher_temperature=.7,
        reconstructed_sampling="HF logits, SGLang native torch joint top-k=20/top-p=.8", score_clip=5,
        source_sha256=hashlib.sha256((SOURCE / "debug_train/0_0.pt").read_bytes()).hexdigest(),
        primary="All-parameter exact within-task gradient dispersion, early-window frozen surrogate",
        limitations=["Not full trajectories or an optimizer update", "HF, not Megatron, backward",
          "Historical candidate support not saved; reconstructed mu is NOT certified actual behavior mu",
          "Group-relative advantages couple sibling samples; dispersion is not independent whole-batch variance",
          "Only four held-out tasks; not final paper evidence"],
        go_gate="Held-out primary dispersion reduction >=10%, task-bootstrap CI above zero, controls and mismatch audit also pass")
    write_json(out / "manifest.json", manifest)
    example = prepared[tasks[0]][0]["turns"][0]
    (out / "teacher_prompt_example.txt").write_text(tok.decode(example["teacher_prefix"]), encoding="utf-8")
    emit("prepared", **{k: manifest[k] for k in ("tasks", "calibration_tasks", "validation_tasks", "turns", "tokens")})


def behavior(logp):
    vals, ids = logp.topk(20, dim=-1)
    probs = vals.exp()
    keep = probs.cumsum(-1) - probs <= .8
    raw = probs * keep
    return ids, raw / raw.sum(-1, keepdim=True), vals


def coefficients(logp, q_real, q_shuffle, actual):
    ids, mu, vals = behavior(logp)
    logmu = mu.clamp_min(1e-30).log()
    real = (q_real.gather(-1, ids) - logmu).clamp(-5, 5)
    shuffled = (q_shuffle.gather(-1, ids) - logmu).clamp(-5, 5)
    unhinted = (vals - logmu).clamp(-5, 5)
    match = ids == actual[:, None]
    supported = (match & (mu > 0)).any(-1)
    # c=0 off the reconstructed support; still evaluate sampled term there.
    c = {"real": real, "shuffled": shuffled, "constant": torch.ones_like(real), "unhinted": unhinted}
    payload = {}
    for kind, scores in c.items():
        mean_weights = mu * scores
        sampled = (scores * match * (mu > 0)).sum(-1)
        payload[kind] = (sampled.detach(), mean_weights.detach())
    return ids, mu, payload, supported


def logprobs(model, prefix, response):
    ids = torch.tensor([prefix + response[:-1]], device="cuda")
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    chosen = hidden[:, len(prefix) - 1:, :]
    return torch.log_softmax(model.lm_head(chosen).float()[0] / .7, dim=-1)


def zeros(params):
    return [torch.zeros(p.shape, dtype=torch.float32, device="cpu") for _, p in params]


def dot(left, right):
    return math.fsum(float(torch.dot(a.reshape(-1), b.reshape(-1))) for a, b in zip(left, right))


def capture(params, indices, full_buffer=None):
    small = []
    for (name, p), ix in zip(params, indices):
        if p.grad is None:
            raise RuntimeError(f"Missing full-parameter gradient: {name}")
        grad = p.grad.detach()
        small.append((grad.flatten()[ix].float() * math.sqrt(p.numel() / len(ix))).cpu())
    if full_buffer is not None:
        for (_, p), target in zip(params, full_buffer):
            target.add_(p.grad.detach().to(device="cpu", dtype=torch.float32))
    return torch.cat(small)


def worker(out, task, smoke=False):
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.monotonic()
    items = json.loads((out / "inputs" / f"{task}.json").read_text())
    if smoke:
        items = items[:1]
        items[0]["turns"] = items[0]["turns"][:1]
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # Needed by HF checkpointing; Qwen3 attention dropout is zero.
    assert model.config.attention_dropout == 0
    params = list(model.named_parameters())
    indices = []
    for name, p in params:
        rng = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % (2**63))
        indices.append(torch.randint(p.numel(), (min(1024, p.numel()),), generator=rng).cuda())
    emit("model_loaded", task=task, seconds=time.monotonic() - started, parameters=sum(p.numel() for _, p in params))
    # Four FP32 CPU vectors: exact sum and per-trajectory accumulation for G/Z.
    sum_g, sum_z, grad_g, grad_z = [zeros(params) for _ in range(4)]
    moments = dict(sum_g2=0., sum_z2=0., sum_gz=0.)
    projected = []
    diagnostics = []
    repeat = None
    for n, item in enumerate(items):
        for buf in (grad_g, grad_z):
            for t in buf:
                t.zero_()
        probe = {k: None for k in KINDS}
        for turn_no, turn in enumerate(item["turns"]):
            t0 = time.monotonic()
            resp = turn["response"]
            actual = torch.tensor(resp, device="cuda")
            with torch.no_grad():
                q_real = logprobs(model, turn["teacher_prefix"], resp)
                q_shuffle = logprobs(model, turn["shuffled_prefix"], resp)
            torch.cuda.synchronize()
            teacher_s = time.monotonic() - t0
            t_student = time.monotonic()
            lp = logprobs(model, turn["prefix"], resp)
            torch.cuda.synchronize()
            student_forward_s = time.monotonic() - t_student
            actual_lp = lp.gather(-1, actual[:, None]).squeeze(-1)
            with torch.no_grad():
                candidate_ids, mu, cs, supported = coefficients(lp.detach(), q_real, q_shuffle, actual)
                old = torch.tensor(turn["old_train_logp"], device="cuda")
                roll = torch.tensor(turn["rollout_logp"], device="cuda")
                diff = actual_lp.detach() - roll
                # Exact expectation of sampled coefficient minus mean coefficient.
                enumerated = torch.zeros_like(mu)
                for j in range(mu.shape[-1]):
                    enumerated[:, j] = mu[:, j] * (q_real.gather(-1, candidate_ids)[:, j] - mu[:, j].clamp_min(1e-30).log()).clamp(-5, 5)
                zero_check = float((enumerated - cs["real"][1]).abs().max())
            tis = torch.tensor(turn["tis"], device="cuda")
            losses = {"grpo": -(actual_lp * tis * item["advantage"]).sum() / item["logical_tokens"]}
            cand_lp = lp.gather(-1, candidate_ids)
            for kind in KINDS[1:]:
                ca, mean_weights = cs[kind]
                losses[kind] = (ca * actual_lp - (mean_weights * cand_lp).sum(-1)).sum() / 1024.
            if n == 0 and turn_no == 0:
                model.zero_grad(set_to_none=True)
                losses["grpo"].backward(retain_graph=True)
                capture(params, indices, grad_g)
                model.zero_grad(set_to_none=True)
                losses["grpo"].backward(retain_graph=True)
                # Full-parameter comparison, not a probe-only repeat check.
                norm1 = dot(grad_g, grad_g)
                diff2, norm2, cross = 0., 0., 0.
                for (_, p), saved in zip(params, grad_g):
                    now = p.grad.detach().float().cpu()
                    dv = now - saved
                    diff2 += float(torch.sum(dv * dv))
                    norm2 += float(torch.dot(now.reshape(-1), now.reshape(-1)))
                    cross += float(torch.dot(now.reshape(-1), saved.reshape(-1)))
                repeat = dict(relative_l2=math.sqrt(diff2 / max(norm1, 1e-30)), cosine=cross / math.sqrt(max(norm1 * norm2, 1e-60)))
                for t in grad_g:
                    t.zero_()
                emit("repeat_backward", task=task, **repeat)
                if repeat["relative_l2"] > .01:
                    raise RuntimeError("Repeated backward noise >1%; stop variance inference")
            for k_idx, kind in enumerate(KINDS):
                model.zero_grad(set_to_none=True)
                losses[kind].backward(retain_graph=k_idx < len(KINDS) - 1)
                vec = capture(params, indices, grad_g if kind == "grpo" else grad_z if kind == "real" else None)
                probe[kind] = vec if probe[kind] is None else probe[kind] + vec
            row = dict(task=task, session=item["session_id"], step=turn["step"], tokens=len(resp),
                prefix_tokens=len(turn["prefix"]), reward=item["raw_reward"], advantage=item["advantage"],
                elapsed_s=time.monotonic() - t0, teacher_s=teacher_s,
                student_forward_s=student_forward_s, support_misses=int((~supported).sum()),
                support_size_mean=float((mu > 0).float().sum(-1).mean()),
                logprob_rollout_mae=float(diff.abs().mean()), logprob_rollout_max=float(diff.abs().max()),
                logprob_train_mae=float((actual_lp.detach() - old).abs().mean()),
                zero_enumeration_max=zero_check, gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9)
            diagnostics.append(row)
            emit("turn_done", **row)
            del losses, lp, actual_lp, q_real, q_shuffle, cand_lp, cs, candidate_ids, mu
            model.zero_grad(set_to_none=True)
            gc.collect()
        g2, z2, gz = dot(grad_g, grad_g), dot(grad_z, grad_z), dot(grad_g, grad_z)
        moments["sum_g2"] += g2
        moments["sum_z2"] += z2
        moments["sum_gz"] += gz
        for dest, src in zip(sum_g, grad_g):
            dest.add_(src)
        for dest, src in zip(sum_z, grad_z):
            dest.add_(src)
        projected.append(dict(session=item["session_id"], reward=item["raw_reward"], advantage=item["advantage"],
                              vectors=probe, exact_g2=g2, exact_z2=z2, exact_gz=gz))
        torch.save(projected, out / f"{task}.probes.pt")
        write_json(out / f"{task}.progress.json", dict(completed=n + 1, total=len(items), diagnostics=diagnostics, repeat=repeat))
        emit("trajectory_done", task=task, index=n + 1, total=len(items), g2=g2, z2=z2, gz=gz)
    size = len(items)
    moments.update(n=size, norm_sum_g2=dot(sum_g, sum_g), norm_sum_z2=dot(sum_z, sum_z), sum_g_dot_sum_z=dot(sum_g, sum_z))
    if size > 1:
        moments.update(var_g=(moments["sum_g2"] - moments["norm_sum_g2"] / size) / (size - 1),
                       var_z=(moments["sum_z2"] - moments["norm_sum_z2"] / size) / (size - 1),
                       cov_gz=(moments["sum_gz"] - moments["sum_g_dot_sum_z"] / size) / (size - 1))
    write_json(out / f"{task}.result.json", dict(task=task, smoke=smoke, exact_moments=moments,
        diagnostics=diagnostics, repeat=repeat, elapsed_s=time.monotonic() - started,
        selected_tokens=sum(d["tokens"] for d in diagnostics)))
    emit("task_done", task=task, seconds=time.monotonic() - started, **moments)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--task")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare(args.output)
    else:
        worker(args.output, args.task, args.smoke)


if __name__ == "__main__":
    main()
