#!/usr/bin/env python3
"""Readable evidence plots; incomplete runs are explicitly marked provisional."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot(root, progress=False):
    folder=root/'analysis';suffix='_progress' if progress else ''
    summary=json.loads((folder/('progress_summary.json' if progress else 'final_summary.json')).read_text())
    tokens=json.loads((folder/('token_statistics'+suffix+'.json')).read_text())
    if not progress and (not summary['final'] or not tokens['final']):
        raise ValueError('final plots require complete final aggregates')
    with (folder/('candidate_metrics_progress.csv' if progress else 'candidate_metrics.csv')).open() as handle:
        rows=list(csv.DictReader(handle))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    watermark='PROVISIONAL - fixed 64 candidates' if progress else 'Fixed 64 candidates - short-window verification'
    labels=['Registered candidates','Restorable states','Teacher requests','Format/scope-valid proposals',
            'First paired test passed','Two-seed verifier passed','Training-eligible pairs']
    values=[64,summary['restorable_states'],summary['teacher_model_requests'],summary['teacher_proposed_valid'],
            summary['first_comparison_passed'],summary['two_seed_verifier_passed_pairs'],summary['accepted_pairs']]
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    y=np.arange(len(labels));axes[0].barh(y,values,color=['#607d8b']*4+['#4a78c2']*2+['#21825b'],height=.62)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,72)
    axes[0].set_xlabel('Candidates');axes[0].set_title('Data-production gate',pad=14)
    for yy,v in zip(y,values):axes[0].text(v+1,yy,str(v),va='center')
    axes[0].axvline(16,color='#7b7b7b',linestyle=':',linewidth=1)
    axes[0].text(17,6.45,'Gate also requires >=8 tasks',fontsize=9,color='#555')
    grouped=Counter((float(r['original_reward']),float(r['repair_reward']),r['accepted']=='True') for r in rows
        if r.get('original_branch_status')==r.get('repair_branch_status')=='completed'
        and r.get('original_reward') and r.get('repair_reward'))
    for (x,y,accepted),n in grouped.items():
        axes[1].scatter(x,y,s=40+25*n,c='#21825b' if accepted else '#ba5d32',alpha=.75,edgecolors='white')
    axes[1].legend(handles=[Line2D([],[],color=c,marker='o',linestyle='',label=label)
        for c,label in [('#21825b','Training-eligible'),('#ba5d32','Not accepted / pending' if progress else 'Not training-eligible')]],
        loc='upper left',frameon=False,fontsize=9)
    axes[1].plot([0,1],[0,1],color='#888',linestyle='--',linewidth=1)
    axes[1].set(xlim=(-.035,1.07),ylim=(-.035,1.09),xlabel='Original branch reward',ylabel='Repair branch reward')
    axes[1].set_title(f'First paired comparison: n={sum(grouped.values())}',pad=14)
    axes[1].grid(alpha=.15)
    fig.suptitle(watermark,fontsize=14,fontweight='bold')
    fig.supxlabel('Same state; <=3 continuation turns, <=300 s per branch. Marker size shows multiplicity. Reward is not final task success.',fontsize=9)
    fig.savefig(folder/('verification_outcomes'+suffix+'.png'));plt.close(fig)

    data=[b for b in tokens['buckets'] if b['side']=='repair' and b['tokens']]
    names={'prose':'Prose','tool_arguments_structure':'Arguments /\nstructure','code_payload':'Code payload',
           'tool_name':'Tool name','control':'Control tokens','mixed':'Mixed boundary','unmapped':'Unmapped'}
    rates=[100*d['blocked_fraction'] for d in data]
    fig,ax=plt.subplots(figsize=(11.5,5.3),layout='constrained')
    x=np.arange(len(data));ax.bar(x,rates,width=.6,color='#527bb3')
    ax.set_xticks(x,[names[d['bucket']] for d in data]);ax.set_ylabel('Blocked target positions (%)')
    ax.set_ylim(0,max([5]+rates)*1.35+1)
    for i,(rate,d) in enumerate(zip(rates,data)):
        ax.annotate(f"{d['blocked_tokens']:,} / {d['tokens']:,}",(i,rate),xytext=(0,7),textcoords='offset points',ha='center',fontsize=10)
    n=sum(a['side']=='repair' for a in tokens['actions']);b=sum(d['blocked_tokens'] for d in data);total=sum(d['tokens'] for d in data)
    ax.set_title(f'Unhinted repair-prefix sampler audit: {n} actions; {b:,}/{total:,} targets blocked',pad=18)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.supxlabel(('PROVISIONAL. ' if progress else '')+'Syntactic buckets; includes rejected proposals. Counts label each bar. This is not a learning result.',fontsize=9)
    fig.savefig(folder/('repair_token_blocking'+suffix+'.png'));plt.close(fig)
    return [str(folder/(name+suffix+'.png')) for name in ('verification_outcomes','repair_token_blocking')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--progress',action='store_true')
    a=p.parse_args();print(json.dumps(plot(a.root.resolve(),a.progress)))


if __name__=='__main__':main()
