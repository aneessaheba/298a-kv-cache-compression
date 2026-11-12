from pathlib import Path
import json,csv,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
parser=argparse.ArgumentParser(description='Regenerate all five Workbook 2 figures from audited data and saved experiments.')
parser.add_argument('--dev',type=Path,required=True)
ARGS=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]; B=Path(__file__).resolve().parent; R=ROOT; F=B/'figures';F.mkdir(exist_ok=True)
a=json.loads((B/'data_audit.json').read_text());plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):fig.savefig(F/(name+'.png'),dpi=190,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(9.5,3));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3))
for x,y,w,text in [(0.1,1.7,2.1,'LongBench raw files\n3,750 English records'),(2.65,1.7,2.1,'Cleaning and checks\n3,731 unique pairs'),(5.2,1.7,2.1,'Official prompt templates\nLlama cap 32,000 tokens'),(7.75,1.7,2.1,'JSONL artifact warehouse\nFull + 800-record dev')]:
 ax.add_patch(FancyBboxPatch((x,y),w,.8,boxstyle='round,pad=.04',facecolor='#eef3f6',edgecolor='#405b70'));ax.text(x+w/2,y+.4,text,ha='center',va='center',fontsize=9)
 if x<7:ax.annotate('',xy=(x+2.5,y+.4),xytext=(x+w+.05,y+.4),arrowprops={'arrowstyle':'->'})
ax.text(5,.95,'Manifests and saved IDs  →  shared BF16 method runs  →  JSONL predictions  →  scorer and paired analysis',ha='center',fontsize=10)
ax.text(5,.25,'Monitoring: cleaning counts · hash and schema assertions · OOM flags · run manifests · resume · backup',ha='center',fontsize=9)
save(fig,'data_lineage')
fig,(ax,ay)=plt.subplots(1,2,figsize=(12,2.8),gridspec_kw={'width_ratios':[1.6,1]})
tasks=list(a['task_stats']);x=np.arange(len(tasks))
ax.bar(x,[a['task_stats'][t]['raw'] for t in tasks],color='#607d8b',label='Raw');ax.bar(x,[a['task_stats'][t]['dev'] for t in tasks],color='#ca783b',label='Development');ax.set_xticks(x,[t.replace('_en','') for t in tasks],rotation=65,ha='right',fontsize=8);ax.set_ylabel('Records');ax.legend(fontsize=8);ax.set_title('Task balance and fixed sample sizes')
tr={t:v['truncated_full'] for t,v in a['task_stats'].items() if v['truncated_full']};ay.barh(list(tr),list(tr.values()),color='#ca783b');ay.set(xlabel='Records truncated',title='Exact Llama-tokenizer truncation');ay.invert_yaxis()
save(fig,'balance_and_truncation')
# Raw/prepared prompt correlation is a data statistic, not model correlation.
rows=[json.loads(l) for l in ARGS.dev.open()]
x=np.array([r['context_words'] for r in rows]);y=np.array([r['prompt_tokens_after'] for r in rows]);rho=float(np.corrcoef(x,y)[0,1])
fig,ax=plt.subplots(figsize=(10,2.7));ax.scatter(x,y,s=11,c='#405b70',alpha=.45);ax.axhline(32000,ls='--',c='#ca783b',lw=1);ax.set(xlabel='Context words',ylabel='Prepared prompt tokens',title=f'800 development records · Pearson r = {rho:.3f}');save(fig,'length_correlation')
# Scope-matched real results; TriAttention is deliberately separate.
tq=json.loads((ROOT/'members/vedant/results/sem1/analysis/results.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,4))
for mse,color,label in [(False,'#ad4938','TurboQuant QJL keys'),(True,'#216f86','TurboQuant MSE keys')]:
 group=sorted([r for r in tq['runs'] if r['method'].startswith('turboquant_') and 'adaptive' not in r['method'] and ('mse' in r['method'])==mse],key=lambda r:r['mean_computed_cache_gb'])
 ax.plot([r['mean_computed_cache_gb'] for r in group],[r['score'] for r in group],'-o',label=label,color=color)
restore=[]
for p in (R/'results/restorekv/longbench').glob('*.summary.json'):
 s=json.loads(p.read_text());pred=[json.loads(l) for l in p.with_suffix('').with_suffix('.jsonl').open()];restore.append((np.mean([r['kv_cache_gb'] for r in pred]),s['scores']['average']))
restore.sort();ax.plot([r[0] for r in restore],[r[1] for r in restore],'-s',color='#637639',label='RestoreKV')
ax.axhline(47.72,c='gray',ls='--',label='Full cache 47.72');ax.set(xlabel='Mean computed KV-cache size in GB',ylabel='LongBench score /100',title='Same 800-record development set');ax.legend(fontsize=8,loc='lower right');ax.grid(alpha=.12)
save(fig,'team_comparison')
fig,ax=plt.subplots(figsize=(9.5,3.4));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3))
items=[(.1,1.7,2.1,'Versioned data and IDs\nHF sources / team exports'),(2.65,1.7,2.1,'Docker + Torch + BF16\nRTX 5090, Llama 8B'),(5.2,1.7,2.1,'Shared kvpress harness\nFive method adapters'),(7.75,1.7,2.1,'Artifact store + analysis\nQuality / size / failures')]
for x,y,w,text in items:
 ax.add_patch(FancyBboxPatch((x,y),w,.8,boxstyle='round,pad=.04',facecolor='#eef3f6',edgecolor='#405b70'));ax.text(x+w/2,y+.4,text,ha='center',va='center',fontsize=9)
 if x<7:ax.annotate('',xy=(x+2.5,y+.4),xytext=(x+w+.05,y+.4),arrowprops={'arrowstyle':'->'})
ax.text(5,.9,'Control plane: GitHub commits and PRs · saved configurations · Linear issue links · environment snapshots',ha='center',fontsize=9)
ax.text(5,.25,'Validation gates: source/data fingerprints · protected question tokens · canonical scorer · matched IDs and budgets',ha='center',fontsize=9)
save(fig,'model_platform')
a['dev_context_word_token_pearson']=rho;(B/'data_audit.json').write_text(json.dumps(a,indent=2)+'\n')
print('Created five source-derived workbook figures.')
