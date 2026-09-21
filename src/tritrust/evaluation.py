"""Gallery/probe evaluation and non-fabricated statistics."""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score
from scipy.stats import t,ttest_rel

def identify(emb,rows,split='test',exclude_same_view=True):
    emb=np.asarray(emb,float);emb=emb/np.maximum(np.linalg.norm(emb,axis=1,keepdims=True),1e-12)
    g=[i for i,r in enumerate(rows) if r['split']==split and r['role']=='gallery'];p=[i for i,r in enumerate(rows) if r['split']==split and r['role']=='probe']
    out=[]
    # Evaluate each gallery view separately, then macro-average view/condition cells.
    for i in p:
        for view in sorted({rows[j]['view'] for j in g}):
            if exclude_same_view and rows[i]['view']==view:continue
            ids=[j for j in g if rows[j]['view']==view]
            if rows[i]['subject_id'] not in {rows[j]['subject_id'] for j in ids}:raise ValueError('Incomplete gallery coverage')
            scores=emb[ids]@emb[i];best=ids[int(np.argmax(scores))]
            out.append(dict(sample_id=rows[i]['sample_id'],subject_id=rows[i]['subject_id'],predicted_id=rows[best]['subject_id'],gallery_sample_id=rows[best]['sample_id'],probe_view=rows[i]['view'],gallery_view=view,condition=rows[i]['condition'],score=float(scores.max()),correct=int(rows[i]['subject_id']==rows[best]['subject_id'])))
    if not out:raise ValueError('No eligible comparisons')
    return out

def metrics(pred):
    cells={}
    for r in pred:cells.setdefault((r['condition'],r['probe_view'],r['gallery_view']),[]).append(r['correct'])
    y=[r['subject_id'] for r in pred];h=[r['predicted_id'] for r in pred];p,r,f,_=precision_recall_fscore_support(y,h,average='macro',zero_division=0)
    return dict(rank1_macro_cells=100*float(np.mean([np.mean(v) for v in cells.values()])),accuracy_micro_comparisons=100*float(np.mean([r['correct'] for r in pred])),macro_precision=100*p,macro_recall=100*r,macro_f1=100*f,n_comparisons=len(pred),n_unique_probes=len(set(r['sample_id'] for r in pred)),roc_auc=None,roc_auc_reason='Not derived from hard top-1 labels; needs specified class probability contract',cells=[dict(condition=k[0],probe_view=k[1],gallery_view=k[2],n=len(v),rank1=100*float(np.mean(v))) for k,v in sorted(cells.items())])

def paired_summary(a,b):
    if set(a)!=set(b) or len(a)<2:raise ValueError('Need identical seed sets with at least two independent runs')
    seeds=sorted(a);d=np.array([a[s]-b[s] for s in seeds],float);sd=d.std(ddof=1);mean=d.mean();margin=t.ppf(.975,len(d)-1)*sd/np.sqrt(len(d))
    return dict(seeds=seeds,mean_difference=float(mean),ci95=[float(mean-margin),float(mean+margin)],p_value=float(ttest_rel([a[s] for s in seeds],[b[s] for s in seeds]).pvalue) if sd>0 else None,constant_difference=bool(sd==0))
