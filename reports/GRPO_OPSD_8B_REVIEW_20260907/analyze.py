#!/usr/bin/env python3
"""CPU-only audit of the two completed 8B formal-v3 runs. No model or environment writes."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = Path('/root/openclaw-rl-local')
ARMS = ['grpo', 'opsd']
COLORS = {'grpo': '#2864A8', 'opsd': '#D06D2F'}
ROOTS = {a: PROJECT / 'outputs' / f'qwen3-8b-{a}-formal-seed0-formal-v3-20260904' for a in ARMS}


def dump(name, obj):
    (HERE / 'data' / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def stats(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {'n': 0}
    return dict(n=len(x), mean=float(x.mean()), p50=float(np.quantile(x, .5)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()), min=float(x.min()))


def read_events():
    rows = []
    for arm, root in ROOTS.items():
        for path in sorted((root / 'tensorboard').glob('events*')):
            acc = EventAccumulator(str(path), size_guidance={'scalars': 0})
            acc.Reload()
            for tag in acc.Tags()['scalars']:
                # Deliberately keep stream identity: rollout/truncated has two meanings.
                for e in acc.Scalars(tag):
                    rows.append(dict(arm=arm, source=path.name, tag=tag,
                                     step=e.step, value=e.value, wall_time=e.wall_time))
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'data/tensorboard_scalars.csv', index=False)
    train = df[(df.step < 100) & ~df.tag.str.startswith('eval/') & (df.tag != 'rollout/truncated')]
    assert not train.duplicated(['arm', 'tag', 'step']).any(), 'Unexpected duplicate scalar tag'
    wide = train.pivot(index=['arm', 'step'], columns='tag', values='value').reset_index()
    wide.to_csv(HERE / 'data/train_by_step.csv', index=False)
    return wide


def read_evals():
    rows, checks, sources = [], [], []
    all_params, seed_paths = set(), defaultdict(dict)
    for arm, root in ROOTS.items():
        for step in (20, 40, 60, 80, 100):
            directory = root / 'checkpoints/eval_artifacts' / f'train_step_{step:06}'
            manifest = json.loads((directory / 'manifest.json').read_text())
            records = [json.loads(x) for x in (directory / 'results.jsonl').read_text().splitlines()]
            assert len(records) == 200 and len({r['task_id'] for r in records}) == 200
            summary = manifest['summaries']['clawgym_eval']
            assert abs(np.mean([r['reward'] for r in records]) - summary['average_reward']) < 1e-10
            assert sum(r['reward'] > .5 for r in records) == summary['correct_count']
            checks.append(dict(arm=arm, train_step=step, **summary,
                               configuration=manifest['configuration'], dataset=manifest['datasets'][0]['task_and_verifier_sha256']))
            for filename in ('manifest.json', 'results.jsonl'):
                p = directory / filename
                sources.append(dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            for r in records:
                requests = r['model_requests']
                for q in requests:
                    all_params.add(json.dumps(q['sampling_params'], sort_keys=True))
                    seed_paths[(arm, step)][(r['task_id'], q['step'])] = q['sampling_seed']
                row = {k: r.get(k) for k in ['task_id', 'reward', 'correct', 'agent_steps', 'duration_seconds',
                        'termination_reason', 'status', 'compaction_count', 'compaction_model_call_count',
                        'tool_result_pruning_events', 'context_overflow_count', 'sampling_identity']}
                row.update(arm=arm, train_step=step, main_requests=len(requests),
                           assistant_tokens=sum(q['response_tokens'] for q in requests),
                           max_prompt_tokens=max((q['prompt_tokens'] for q in requests), default=0),
                           turns_hit_limit=sum(q['response_tokens'] >= 8192 or q.get('finish_reason') == 'length' for q in requests),
                           instance_log=r.get('instance_log', ''))
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'data/eval_task_metrics.csv', index=False)
    aggs = []
    for (arm, step), g in df.groupby(['arm', 'train_step']):
        aggs.append(dict(arm=arm, train_step=int(step), n=len(g), reward=g.reward.mean(), correct_rate=g.correct.mean(),
                         agent_steps_mean=g.agent_steps.mean(), agent_steps_p90=g.agent_steps.quantile(.9),
                         assistant_tokens_mean=g.assistant_tokens.mean(), assistant_tokens_p90=g.assistant_tokens.quantile(.9),
                         at_96_steps=int((g.agent_steps >= 96).sum()), max_steps_termination=int((g.termination_reason == 'max_agent_steps').sum()),
                         response_limit_trajectories=int((g.turns_hit_limit > 0).sum()),
                         pruning_trajectories=int((g.tool_result_pruning_events > 0).sum()),
                         duration_mean=g.duration_seconds.mean()))
    agg = pd.DataFrame(aggs)
    agg.to_csv(HERE / 'data/eval_by_checkpoint.csv', index=False)
    common_seeds = mismatched_seeds = 0
    ref = seed_paths[('grpo', 20)]
    # All available shared task/turn keys, without claiming histories are identical.
    for key, values in seed_paths.items():
        for task_turn in ref.keys() & values.keys():
            common_seeds += 1
            mismatched_seeds += ref[task_turn] != values[task_turn]
    dump('eval_protocol_audit.json', dict(manifests=checks, sampling_params=[json.loads(x) for x in sorted(all_params)],
        shared_task_turn_seed_checks=common_seeds, seed_mismatches=mismatched_seeds,
        configurations_identical=len({json.dumps(c['configuration'], sort_keys=True) for c in checks}) == 1,
        datasets_identical=len({c['dataset'] for c in checks}) == 1))
    dump('source_manifest.json', dict(project=str(PROJECT), runs={a: str(r) for a, r in ROOTS.items()}, eval_sources=sources))
    return df, agg


def read_train_headers():
    rows = []
    for arm, root in ROOTS.items():
        for rollout_id in range(100):
            directory = root / 'checkpoints/openclaw_logs' / f'rollout_{rollout_id}'
            for path in sorted(directory.glob('instance-train-*.log')):
                # Stop before any tool output; no trajectory contents are exported.
                chunks = []
                with path.open(errors='replace') as f:
                    while True:
                        chunk = f.read(8192)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        if '--- Step 1 ---' in ''.join(chunks) or sum(map(len, chunks)) > 262144:
                            break
                head = ''.join(chunks).split('--- Step 1 ---', 1)[0]
                task = re.search(r'^=== TRAIN Task (\S+) ===', head)
                if not task:
                    continue
                row = dict(arm=arm, rollout_id=rollout_id, local_index=int(path.stem.rsplit('-', 1)[-1]), task_id=task[1])
                for field, label, cast in [('agent_steps', 'Steps', int), ('reward', 'Reward', float),
                                          ('group_index', 'Group Index', int), ('pruning', 'Tool-result pruning events', int),
                                          ('compactions', 'Compactions', int), ('termination_reason', 'Termination Reason', str)]:
                    matches = re.findall(r'^' + re.escape(label) + r': (.+)$', head, re.M)
                    row[field] = cast(matches[-1]) if matches else None
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / 'data/train_trajectory_headers.csv', index=False)
    keys = ['rollout_id', 'local_index']
    pair = df[df.arm == 'grpo'].merge(df[df.arm == 'opsd'], on=keys, suffixes=('_grpo', '_opsd'))
    dump('task_order_audit.json', dict(rows=len(df), paired_records=len(pair), task_mismatches=int((pair.task_id_grpo != pair.task_id_opsd).sum()),
                                     rows_by_arm=df.groupby('arm').size().to_dict()))
    return df


def read_step0():
    """Prior launches evaluated the same SFT checkpoint before any update."""
    rows, summaries, audits = [], [], []
    reference_dir = ROOTS['grpo'] / 'checkpoints/eval_artifacts/train_step_000020'
    reference = json.loads((reference_dir / 'manifest.json').read_text())
    ref_seeds = {(r['task_id'], q['step']): q['sampling_seed']
                 for r in map(json.loads, (reference_dir / 'results.jsonl').read_text().splitlines())
                 for q in r['model_requests']}
    for version in ('v1', 'v2'):
        for arm in ARMS:
            root = ROOTS[arm].with_name(ROOTS[arm].name.replace('formal-v3-', f'formal-{version}-'))
            directory = root / 'checkpoints/eval_artifacts/train_step_000000'
            meta = dict(line.split('=', 1) for line in (root / 'run.env').read_text().splitlines() if '=' in line)
            m = json.loads((directory / 'manifest.json').read_text())
            records = list(map(json.loads, (directory / 'results.jsonl').read_text().splitlines()))
            assert len(records) == 200 and len({r['task_id'] for r in records}) == 200
            assert meta['load_ckpt'] == ''
            assert meta['hf_ckpt'] == str(PROJECT / 'outputs/clawgym-8b-sft-formal-20260902-165551/final_model')
            assert m['configuration'] == reference['configuration']
            assert m['datasets'][0]['task_and_verifier_sha256'] == reference['datasets'][0]['task_and_verifier_sha256']
            assert m['summaries']['clawgym_eval']['n_skipped'] == 0
            checked = mismatched = 0
            for r in records:
                requests = r['model_requests']
                for q in requests:
                    key = (r['task_id'], q['step'])
                    if key in ref_seeds:
                        checked += 1
                        mismatched += q['sampling_seed'] != ref_seeds[key]
                    params = q['sampling_params']
                    assert (params['temperature'], params['top_p'], params['top_k']) == (.7, .8, 20)
                rows.append(dict(arm=arm, version=version, train_step=0, task_id=r['task_id'], reward=r['reward'],
                                 correct=r['correct'], agent_steps=r['agent_steps'],
                                 assistant_tokens=sum(q['response_tokens'] for q in requests),
                                 termination_reason=r['termination_reason']))
            summary = m['summaries']['clawgym_eval']
            summaries.append(dict(arm=arm, version=version, train_step=0, n=200, reward=summary['average_reward'],
                                  correct_rate=summary['correct_rate'],
                                  agent_steps_mean=float(np.mean([r['agent_steps'] for r in records])),
                                  assistant_tokens_mean=float(np.mean([sum(q['response_tokens'] for q in r['model_requests']) for r in records]))))
            audits.append(dict(arm=arm, version=version, source=str(directory), repository=m['repository'],
                               model_path=meta['hf_ckpt'], configuration=m['configuration'],
                               shared_seed_checks=checked, seed_mismatches=mismatched,
                               file_hashes={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                            for name in ('manifest.json', 'results.jsonl')}))
            assert mismatched == 0
    pd.DataFrame(rows).to_csv(HERE / 'data/step0_task_metrics.csv', index=False)
    df = pd.DataFrame(summaries)
    df.to_csv(HERE / 'data/step0_summary.csv', index=False)
    dump('step0_protocol_audit.json', audits)
    return df


def read_opsd():
    segments, steps = [], []
    for path in sorted((ROOTS['opsd'] / 'opsd_step_summaries').glob('*.json'), key=lambda p: int(p.stem)):
        payload = json.loads(path.read_text())
        for r in payload['records']:
            common = dict(rollout_id=payload['rollout_id'], task_id=r['task_id'], group_index=r['group_index'],
                          trajectory_index=r['trajectory_index'], segment_index=r['segment_index'],
                          advantage=r['advantage'], raw_reward=r['raw_reward'])
            segments.append(dict(**common, loss_tokens=r['loss_token_count'], response_space_tokens=r['response_token_count'],
                                 applied=r['opsd_applied'], requested=r['opsd_requested'], fallback_reason=r['fallback_reason'].split(':', 1)[0]))
            for t in r['steps']:
                steps.append(dict(**common, **t))
    sd, td = pd.DataFrame(segments), pd.DataFrame(steps)
    sd.to_csv(HERE / 'data/opsd_segments.csv', index=False)
    td.to_csv(HERE / 'data/opsd_turn_weights.csv', index=False)
    assert (td.advantage < 0).all()
    assert np.max(np.abs(td.weight - np.clip(np.exp(-td.teacher_minus_rollout_old_mean), 0, 1.2))) < 1e-5
    assert np.max(np.abs(td.weighted_advantage - td.advantage * td.weight)) < 1e-5
    keys = ['rollout_id', 'group_index', 'trajectory_index']
    logical = []
    for key, g in sd.groupby(keys):
        logical.append(dict(zip(keys, key), task_id=g.task_id.iloc[0], advantage=g.advantage.iloc[0],
                            loss_tokens=int(g.loss_tokens.sum()), segments=len(g), fully_applied=bool(g.applied.all()),
                            any_applied=bool(g.applied.any())))
    ld = pd.DataFrame(logical)
    ld.to_csv(HERE / 'data/opsd_logical_rollouts.csv', index=False)
    per_traj = []
    for key, g in td.groupby(keys):
        per_traj.append(dict(zip(keys, key), n_turns=len(g), token_weight_mean=float(np.average(g.weight, weights=g.loss_token_count)),
                             turn_weight_std=float(g.weight.std(ddof=0)), weight_range=float(g.weight.max() - g.weight.min()),
                             all_weights_gt_1=bool((g.weight > 1).all())))
    pd.DataFrame(per_traj).to_csv(HERE / 'data/opsd_logical_weight_stats.csv', index=False)
    w = td.weight.to_numpy()
    summary = dict(scored_turns=len(td), scored_segments=int(sd.applied.sum()), scored_logical_rollouts=len(per_traj),
                   total_segments=len(sd), total_logical_rollouts=len(ld), negative_logical_rollouts=int((ld.advantage < 0).sum()),
                   requested_segments=int(sd.requested.sum()), fallback_counts=sd.fallback_reason.value_counts().to_dict(),
                   duplicate_logical_turn_records=int(td.duplicated(keys + ['step_id']).sum()),
                   weight=stats(w), delta=stats(td.teacher_minus_rollout_old_mean),
                   fraction_gt_1=float((w > 1).mean()), fraction_lt_1=float((w < 1).mean()), fraction_at_1=float((w == 1).mean()),
                   fraction_ge_upper=float((w >= 1.2 - 1e-6).mean()),
                   histogram_below_0p9_count=int((w < .9).sum()),
                   token_weight_mean=float(np.average(w, weights=td.loss_token_count)),
                   logical_equal_weight_mean=float(np.mean([r['token_weight_mean'] for r in per_traj])),
                   all_turns_penalized_trajectory_fraction=float(np.mean([r['all_weights_gt_1'] for r in per_traj])),
                   within_trajectory_std=stats([r['turn_weight_std'] for r in per_traj]),
                   within_trajectory_range=stats([r['weight_range'] for r in per_traj]),
                   delta_length_spearman=float(spearmanr(td.teacher_minus_rollout_old_mean, td.loss_token_count).statistic))
    bins = pd.cut(td.loss_token_count, [0, 128, 512, 2048, np.inf], labels=['1-128', '129-512', '513-2048', '>2048'])
    bylength = td.assign(length_bin=bins).groupby('length_bin', observed=True).agg(
        count=('weight', 'size'), mean_weight=('weight', 'mean'), mean_delta=('teacher_minus_rollout_old_mean', 'mean'))
    bylength.to_csv(HERE / 'data/opsd_by_turn_length.csv')
    dump('opsd_summary.json', summary)
    return td, summary


def paired_analysis(evals):
    rng = np.random.default_rng(20260907)
    summaries = []
    for step in (20, 40, 60, 80, 100):
        g = evals[(evals.arm == 'grpo') & (evals.train_step == step)].set_index('task_id')
        o = evals[(evals.arm == 'opsd') & (evals.train_step == step)].set_index('task_id').loc[g.index]
        d = o.reward.to_numpy() - g.reward.to_numpy()
        bs = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
        b = int((g.correct & ~o.correct).sum()); c = int((~g.correct & o.correct).sum())
        summaries.append(dict(train_step=step, reward_diff_opsd_minus_grpo=float(d.mean()),
                              bootstrap_95ci=np.quantile(bs, [.025, .975]).tolist(),
                              reward_wins=int((d > 1e-8).sum()), reward_losses=int((d < -1e-8).sum()),
                              reward_ties=int((abs(d) <= 1e-8).sum()), correct_to_incorrect=b, incorrect_to_correct=c,
                              mcnemar_exact_p=float(binomtest(min(b, c), b + c, .5).pvalue) if b + c else 1.0))
        if step == 100:
            joined = g[['reward', 'correct', 'agent_steps', 'assistant_tokens', 'termination_reason', 'instance_log']].join(
                o[['reward', 'correct', 'agent_steps', 'assistant_tokens', 'termination_reason', 'instance_log']], lsuffix='_grpo', rsuffix='_opsd')
            joined['reward_diff_opsd_minus_grpo'] = d
            joined.to_csv(HERE / 'data/final_eval_paired.csv')
    dump('paired_eval_statistics.json', summaries)


def summarize_train(train, headers):
    result = {}
    for arm in ARMS:
        t = train[train.arm == arm].set_index('step')
        h = headers[headers.arm == arm]
        arm_result = {'trajectory_count': len(h), 'steps': stats(h.agent_steps.dropna()),
                      'termination_counts': h.termination_reason.value_counts().to_dict()}
        for label, ids in [('all', range(100)), ('first20', range(20)), ('last20', range(80, 100))]:
            block = t.loc[list(ids)]
            means = block.mean(numeric_only=True).dropna().to_dict()
            sums = {tag: float(block[tag].sum()) for tag in block.columns if any(s in tag for s in
                    ['submitted_trajectories', 'skipped_', 'timed_out_trajectories', 'hit_response_limit', 'budget_terminated', 'termination/', 'incomplete_tool_calls', 'repetition_degenerated'])}
            arm_result[label] = dict(means=means, selected_sums=sums)
        result[arm] = arm_result
    dump('train_summary.json', result)


def draw(evals, agg, train, td, baseline):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelcolor': '#263446',
                         'text.color': '#263446', 'axes.edgecolor': '#A0A9B5', 'savefig.facecolor': 'white'})
    def add_baseline(ax, arm, metric, first_value, scale):
        b = baseline[baseline.arm == arm].set_index('version')
        ax.plot([0, 20], [b.loc['v2', metric] * scale, first_value * scale], ls='--', lw=1.5, color=COLORS[arm], alpha=.7)
        ax.scatter([0], [b.loc['v1', metric] * scale], marker='s', s=55, facecolors='white', edgecolors=COLORS[arm], linewidths=1.6, zorder=5)
        ax.scatter([0], [b.loc['v2', metric] * scale], marker='D', s=46, color=COLORS[arm], zorder=6)

    baseline_note = 'Step 0: filled diamond = v2; hollow square = v1. Dashed line links the prior-launch SFT reference.'
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6), layout='constrained')
    for ax, metric, title in zip(axs, ['reward', 'correct_rate'], ['Mean verifier reward', 'Correct rate (reward > 0.5)']):
        for arm in ARMS:
            d = agg[agg.arm == arm].sort_values('train_step')
            ax.plot(d.train_step, d[metric] * 100, marker='o', lw=2.2, color=COLORS[arm], label=arm.upper())
            add_baseline(ax, arm, metric, d[metric].iloc[0], 100)
        ax.set(title=title, xlabel='Optimizer updates', ylabel='Percent', xticks=[0, 20, 40, 60, 80, 100], xlim=(-4, 104), ylim=(48, 61))
        ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, loc='upper left')
    fig.suptitle('8B GRPO vs. OPSD | Same 200 evaluation tasks, n = 1', fontsize=15)
    fig.supxlabel(baseline_note, fontsize=9)
    fig.savefig(HERE / 'figures/01_eval.png', dpi=190); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6), layout='constrained')
    for arm in ARMS:
        d = agg[agg.arm == arm].sort_values('train_step')
        axs[0].plot(d.train_step, d.agent_steps_mean, marker='o', lw=2.2, color=COLORS[arm], label=arm.upper())
        axs[1].plot(d.train_step, d.assistant_tokens_mean / 1000, marker='o', lw=2.2, color=COLORS[arm], label=arm.upper())
        add_baseline(axs[0], arm, 'agent_steps_mean', d.agent_steps_mean.iloc[0], 1)
        add_baseline(axs[1], arm, 'assistant_tokens_mean', d.assistant_tokens_mean.iloc[0], .001)
    for ax in axs:
        ax.set(xlabel='Optimizer updates', xticks=[0, 20, 40, 60, 80, 100], xlim=(-4, 104)); ax.grid(axis='y', alpha=.2); ax.legend(frameon=False)
    axs[0].set(title='Mean agent turns per task', ylabel='Turns')
    axs[1].set(title='Mean generated assistant tokens per task', ylabel='Thousand tokens')
    fig.suptitle('Evaluation behavior | Whole trajectories, not segments', fontsize=15)
    fig.supxlabel(baseline_note, fontsize=9)
    fig.savefig(HERE / 'figures/02_eval_behavior.png', dpi=190); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.3), layout='constrained')
    ax = axs[0]
    ax.hist(td.weight, bins=np.linspace(.90, 1.20001, 70), weights=np.ones(len(td)) * 100 / len(td), color=COLORS['opsd'], alpha=.9)
    ax.axvline(1, color='#263446', ls='--', lw=1.3)
    ax.set(title='Applied negative-advantage turns', xlabel='OPSD multiplier w', ylabel='Percent of all scored turns', xlim=(.90, 1.205))
    ax.text(.97, .94, f'n = {len(td):,}\nw > 1: {(td.weight > 1).mean():.2%}', ha='right', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=.4', facecolor='white', edgecolor='#D6DDE4'))
    ax = axs[1]
    for arm in ARMS:
        d = train[train.arm == arm].sort_values('step')
        blocks = d.assign(block=d.step // 20).groupby('block')['openclaw/avg_reward_submitted'].mean()
        ax.plot(np.arange(20, 101, 20), blocks.to_numpy() * 100, marker='o', lw=2.2, color=COLORS[arm], label=arm.upper())
    ax.set(title='Training submitted reward', xlabel='End of non-overlapping 20-update window', ylabel='Percent', xticks=[20, 40, 60, 80, 100])
    ax.grid(axis='y', alpha=.2); ax.legend(frameon=False)
    fig.suptitle('OPSD reweighting and training reward', fontsize=15)
    fig.savefig(HERE / 'figures/03_weights_training.png', dpi=190); plt.close(fig)


def main():
    for name in ('data', 'figures'):
        (HERE / name).mkdir(parents=True, exist_ok=True)
    train = read_events()
    evals, agg = read_evals()
    baseline = read_step0()
    pd.concat([baseline[baseline.version == 'v2'], agg.assign(version='v3')], ignore_index=True).sort_values(
        ['arm', 'train_step']).to_csv(HERE / 'data/eval_with_step0.csv', index=False)
    headers = read_train_headers()
    td, weight_summary = read_opsd()
    paired_analysis(evals)
    summarize_train(train, headers)
    draw(evals, agg, train, td, baseline)
    print(agg.to_string(index=False))
    print('WEIGHTS', json.dumps(weight_summary, ensure_ascii=False))
    print('Finished. Artifacts:', HERE)


if __name__ == '__main__':
    main()
