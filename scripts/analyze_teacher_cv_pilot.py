#!/usr/bin/env python3
"""Held-out analysis: full parameter moments primary; coordinate probes controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def fit(rows):
    vg = sum(x['var_g'] for x in rows)
    vz = sum(x['var_z'] for x in rows)
    cov = sum(x['cov_gz'] for x in rows)
    raw = -cov / max(vz, 1e-30)
    return dict(alpha=.5 * float(np.clip(raw, -10, 10)), alpha_raw=raw,
                var_g=vg, var_z=vz, cov_gz=cov,
                correlation=cov / max((vg*vz)**.5, 1e-30))


def reduction(rows, alpha):
    vg = sum(x['var_g'] for x in rows)
    vz = sum(x['var_z'] for x in rows)
    cov = sum(x['cov_gz'] for x in rows)
    return -(2*alpha*cov + alpha*alpha*vz) / max(vg, 1e-30)


def run(out):
    torch.set_num_threads(4)
    m = json.loads((out/'manifest.json').read_text())
    results = {t: json.loads((out/f'{t}.result.json').read_text()) for t in m['tasks']}
    exact = {t: r['exact_moments'] for t,r in results.items()}
    cal = [exact[t] for t in m['calibration_tasks']]
    valid = [exact[t] for t in m['validation_tasks']]
    primary = fit(cal)
    primary['validation_reduction'] = reduction(valid, primary['alpha'])
    primary['calibration_reduction'] = reduction(cal, primary['alpha'])
    primary['validation_correlation'] = fit(valid)['correlation']
    rng = np.random.default_rng(m['seed'])
    boot = [reduction([valid[j] for j in rng.integers(0,len(valid),len(valid))],primary['alpha']) for _ in range(5000)]
    primary['validation_task_bootstrap_ci95'] = np.quantile(boot,[.025,.975]).tolist()
    probes = {}
    for task in m['tasks']:
        data = torch.load(out/f'{task}.probes.pt', weights_only=True, map_location='cpu')
        g = torch.stack([r['vectors']['grpo'] for r in data]).double()
        g -= g.mean(0)
        probes[task] = {}
        for kind in ('real','shuffled','constant','unhinted'):
            z = torch.stack([r['vectors'][kind] for r in data]).double()
            z -= z.mean(0)
            n = len(data)-1
            probes[task][kind] = dict(var_g=float(g.square().sum())/n,
                var_z=float(z.square().sum())/n, cov_gz=float((g*z).sum())/n)
    controls = {}
    for kind in ('real','shuffled','constant','unhinted'):
        c = [probes[t][kind] for t in m['calibration_tasks']]
        v = [probes[t][kind] for t in m['validation_tasks']]
        controls[kind] = fit(c)
        controls[kind]['validation_reduction'] = reduction(v,controls[kind]['alpha'])
    ds = [d for x in results.values() for d in x['diagnostics']]
    nt = sum(d['tokens'] for d in ds)
    audit = dict(tokens=nt, support_misses=sum(d['support_misses'] for d in ds),
        support_miss_fraction=sum(d['support_misses'] for d in ds)/nt,
        hf_rollout_logprob_mae=sum(d['tokens']*d['logprob_rollout_mae'] for d in ds)/nt,
        hf_train_logprob_mae=sum(d['tokens']*d['logprob_train_mae'] for d in ds)/nt,
        zero_enumeration_max=max(d['zero_enumeration_max'] for d in ds),
        repeat_relative_l2_max=max(x['repeat']['relative_l2'] for x in results.values()),
        repeat_cosine_min=min(1.0 if x['repeat']['relative_l2']==0 else min(1.,max(-1.,x['repeat']['cosine'])) for x in results.values()),
        repeat_cosine_raw_max=max(x['repeat']['cosine'] for x in results.values()),
        max_gpu_peak_gb=max(d['gpu_peak_gb'] for d in ds))
    decision = 'NO_GO_IN_THIS_PILOT' if primary['validation_reduction']<.1 or primary['validation_task_bootstrap_ci95'][0]<=0 else 'PROMISING_NOT_CONFIRMED'
    report = dict(manifest=m, primary_exact=primary, coordinate_probe_controls=controls,
        audit=audit, exact_task_moments=exact, decision=decision,
        warning='Early-window offline screening, not certified behavior mu, no full-RL efficacy claim.')
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    text = f'''# Teacher-CV 冻结模型梯度预实验

结论：**{decision}**。这是离线早期 turn 筛选，不是完整 RL 效果实验。

## 配置

- 相同 8B SFT；8 题 × 8 条真实轨迹，4 题校准、4 题验证。
- 正负 advantage 均使用；每条前 4 个 turn，每 turn 前 256 个实际生成 token。
- 保留完整历史；Teacher 不注入未来答案、reward 或失败标签；无 optimizer update。
- 主比较对完整参数梯度计算精确范数、内积和组内离散程度，不是仅 logits/LM head。
- CV 固定除以 1024；基础 GRPO 使用真实 A、原 logical token 分母和保存的 TIS。
- α 只在校准任务拟合，固定使用 50% 收缩；验证任务不调参。

## 主结果（完整参数）

- 校准 α：{primary['alpha']:.8g}
- 校准 / 验证梯度相关性：{primary['correlation']:.6f} / {primary['validation_correlation']:.6f}
- 校准离散程度下降：{100*primary['calibration_reduction']:.3f}%
- 验证离散程度下降：{100*primary['validation_reduction']:.3f}%（负数表示恶化）
- 验证任务 bootstrap 95% 区间：[{100*primary['validation_task_bootstrap_ci95'][0]:.3f}%, {100*primary['validation_task_bootstrap_ci95'][1]:.3f}%]

## 对照（固定梯度坐标抽样，仅辅助指标）

| 控制变量 | 校准 α | 验证离散程度下降 |
|---|---:|---:|
'''
    for k,x in controls.items():
        text += f"| {k} | {x['alpha']:.6g} | {100*x['validation_reduction']:.3f}% |\n"
    text += f'''
## 审计与限制

- token 数：{nt}；重建支持集外：{audit['support_misses']}（{100*audit['support_miss_fraction']:.4f}%）。
- HF 与保存 rollout 的 logprob MAE：{audit['hf_rollout_logprob_mae']:.6f}。
- 同图重复完整 backward 最大相对 L2：{audit['repeat_relative_l2_max']:.8g}。
- 原始 cosine 使用两种 FP32 归约，可能略超出 1；原始最大值 {audit['repeat_cosine_raw_max']:.8f} 保留审计。相对 L2 为 0 时报告 cosine=1，未改变梯度。
- 枚举均值校验最大误差：{audit['zero_enumeration_max']:.8g}。
- 旧候选采样分布未保存，不能认证 HF 重建 μ 等于历史 μ；尚不能认证 live rollout 下的零均值。
- HF/FlashAttention2 不是正式 Megatron；正式训练代码未改。
- GRPO 组内 A 相互耦合；组内离散程度不是完整 RL batch 的独立样本方差。
- 只覆盖早期窗口；仅 4 个验证任务，区间不稳定，不推广到全部长轨迹。
- 对照使用固定坐标抽样，只有主 real 对照的矩是完整参数精确统计。

权重保持不变。未操作远端容器，未删除既有权重、数据或结果。
详细结果：`{out/'analysis.json'}`。
'''
    (out/'REPORT_ZH.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(decision=decision,primary=primary,audit=audit),indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path)
    run(p.parse_args().output)
