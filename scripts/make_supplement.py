"""Regenerate reported-value figures and executed fixture summary. Never merge scopes."""
import csv,json,sys,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
root=Path(__file__).resolve().parents[1];out=root/'supplementary';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def read(name):return list(csv.DictReader((root/'reported'/name).open()))
def save(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=200);fig.savefig(out/(name+'.svg'));plt.close(fig)
r=read('reported_development_metrics.csv');fig,ax=plt.subplots(figsize=(8,4));x=np.arange(len(r));ax.bar(x-.18,[float(a['top1_accuracy_pct']) for a in r],.36,label='Reported accuracy');ax.bar(x+.18,[float(a['macro_f1_pct']) for a in r],.36,label='Reported macro F1');ax.set_xticks(x,[a['model'] for a in r]);ax.set_ylabel('Percent');ax.set_ylim(0,100);ax.set_title('Supplied development summary — not re-executed');ax.legend();save(fig,'reported_classification')
r=read('ablation_latency.csv');fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(len(r));axes[0].plot(x,[float(a['accuracy_pct']) for a in r],'o-',label='Accuracy');axes[0].plot(x,[float(a['macro_f1_pct']) for a in r],'s-',label='Macro F1');axes[0].set_ylabel('Percent');axes[0].legend();axes[1].plot(x,[float(a['latency_ms']) for a in r],'o-');axes[1].set_ylabel('Reported latency (ms)')
for ax in axes:ax.set_xticks(x,[str(i+1) for i in x]);ax.set_xlabel('Ablation stage (source CSV order)')
fig.suptitle('Supplied ablation aggregates — not re-executed');save(fig,'reported_ablation')
r=read('robustness.csv');fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(r));ax.bar(x-.18,[float(a['cnn_accuracy_pct']) for a in r],.36,label='CNN');ax.bar(x+.18,[float(a['tritrust_accuracy_pct']) for a in r],.36,label='TriTrust');ax.set_xticks(x,[a['condition'] for a in r],rotation=12);ax.set_ylim(0,100);ax.set_ylabel('Reported accuracy (%)');ax.set_title('Supplied robustness summary — not re-executed');ax.legend();save(fig,'reported_robustness')
r=read('explanation_audit.csv');fig,ax=plt.subplots(figsize=(7,4));ax.bar([a['method'] for a in r],[float(a['predicted_score_drop']) for a in r]);ax.set_ylabel('Reported score drop');ax.set_ylim(0,1);ax.set_title('Supplied explanation summary — not re-executed');save(fig,'reported_explanation')
runrows=[]
for f in sorted((root/'validation/executed/runs').glob('seed_*/*/metrics.json')):
 m=json.loads(f.read_text());runrows.append(dict(seed=int(f.parent.parent.name.split('_')[1]),model=f.parent.name,rank1=m['rank1_macro_cells'],macro_f1=m['macro_f1'],scope='synthetic_pipeline_validation'))
with (out/'executed_fixture_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=runrows[0]);w.writeheader();w.writerows(runrows)
summary=[]
for model in sorted({r['model'] for r in runrows}):
 vals=np.array([r['rank1'] for r in runrows if r['model']==model]);mean=vals.mean();sd=vals.std(ddof=1);margin=t.ppf(.975,len(vals)-1)*sd/np.sqrt(len(vals));summary.append(dict(model=model,n=len(vals),mean_rank1=float(mean),sd=float(sd),ci_low=float(mean-margin),ci_high=float(mean+margin),scope='synthetic_fixture_only'))
with (out/'fixture_seed_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
sections=['<h1>TriTrust-Gait supplementary evidence guide</h1><p><b>Software implementation and synthetic validation. Original paper results have not been reproduced.</b></p>']
for title,file in [('Classification','reported_classification'),('Ablation','reported_ablation'),('Robustness','reported_robustness'),('Explanation','reported_explanation')]:sections.append(f'<h2>{title}: reported values</h2><img src="{file}.svg" style="max-width:100%"><p>Source: earlier supplied aggregate CSV; no new experimental support implied.</p>')
sections.append('<h2>Executed synthetic seed summary</h2><table><tr><th>Model</th><th>Runs</th><th>Mean Rank-1</th><th>SD</th></tr>')
for r in summary:sections.append(f'<tr><td>{html.escape(r["model"])}</td><td>{r["n"]}</td><td>{r["mean_rank1"]:.2f}</td><td>{r["sd"]:.2f}</td></tr>')
sections.append('</table><p>Small synthetic sample. These are software checks, not dataset benchmark scores.</p>')
(out/'SUPPLEMENT.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>TriTrust supplementary guide</title><style>body{max-width:1000px;margin:40px auto;font:17px Georgia;line-height:1.5;color:#172235}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style></head><body>'+''.join(sections)+'</body></html>')
