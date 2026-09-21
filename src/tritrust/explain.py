"""Matched eight-rule-coordinate audit: exact Shapley, IG and single-rule removal.
These are post-fusion audits, not pixel-level SHAP/IG or calibrated probabilities.
"""
import itertools,math,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .data import read_manifest,prepare,sha256
from .models import create_model
from .runner import write_json,write_csv

def audit(manifest,run_dir,out_dir):
    torch.set_num_threads(2);run_dir=Path(run_dir);out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    ck=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=True)
    if ck['model']!='tritrust':raise ValueError('Rule audit requires TriTrust')
    if sha256(manifest)!=ck['config']['manifest_sha256']:raise ValueError('Checkpoint manifest mismatch')
    model=create_model('tritrust',len(ck['classes']));model.load_state_dict(ck['state_dict']);model.eval()
    rows=read_manifest(manifest);samples,a,q,_=prepare(rows);cache=[]
    handle=model.final_fuse.register_forward_pre_hook(lambda mod,inputs:cache.append(inputs[0].detach()))
    with torch.no_grad():
        for i in range(0,len(rows),8):model(torch.tensor(np.stack([s[1] for s in samples[i:i+8]])),torch.tensor(np.stack([s[0] for s in samples[i:i+8]])),torch.tensor(a[i:i+8]),torch.tensor(q[i:i+8]))
    handle.remove();latent=torch.cat(cache);gallery_emb=F.normalize(model.final_fuse(latent).detach(),dim=1);records=[];summaries=[];attribs=[]
    for i,r in enumerate(rows):
        if r['split']!='test' or r['role']!='probe':continue
        for view in sorted({x['view'] for x in rows if x['split']=='test' and x['role']=='gallery'}):
            if view==r['view']:continue
            ids=[j for j,x in enumerate(rows) if x['split']=='test' and x['role']=='gallery' and x['view']==view];g=gallery_emb[ids];classes=sorted({rows[j]['subject_id'] for j in ids})
            def scores(x):
                if x.ndim==1:x=x[None]
                z=latent[i:i+1,:256].expand(len(x),-1);emb=F.normalize(model.final_fuse(torch.cat([z,x],1)),dim=1);sim=emb@g.T
                class_sim=torch.stack([sim[:,[k for k,j in enumerate(ids) if rows[j]['subject_id']==c]].max(1).values for c in classes],1)
                return torch.softmax(class_sim/.1,1)
            x=latent[i,256:].detach();base=scores(x).detach()[0];target=int(base.argmax());v0=float(base[target]);allmask=torch.tensor([[(mask>>k)&1 for k in range(8)] for mask in range(256)],dtype=torch.float32)
            with torch.no_grad():coalition=scores(allmask*x)[:,target].numpy()
            shap=np.zeros(8)
            for k in range(8):
                for mask in range(256):
                    if mask&(1<<k):continue
                    size=mask.bit_count();weight=math.factorial(size)*math.factorial(7-size)/math.factorial(8);shap[k]+=weight*(coalition[mask|(1<<k)]-coalition[mask])
            grads=[]
            for alpha in torch.linspace(0,1,33):
                point=(x*alpha).detach().requires_grad_(True);grad=torch.autograd.grad(scores(point)[0,target],point)[0];grads.append(grad.detach())
            ig=(x*torch.trapezoid(torch.stack(grads),dx=1/32,dim=0)).numpy();drops=[];flips=[]
            for k in range(8):
                removed=x.clone();removed[k]=0
                with torch.no_grad():post=scores(removed)[0]
                drop=v0-float(post[target]);flip=int(int(post.argmax())!=target);drops.append(drop);flips.append(flip)
                records.append(dict(sample_id=r['sample_id'],gallery_view=view,rule=f'R{k+1}',weighted_activation=float(x[k]),original_predicted_id=classes[target],original_score=v0,intervened_score=float(post[target]),score_drop=drop,decision_flip=flip,scope=ck['scope']))
            for method,values in [('rule_removal',np.array(drops)),('integrated_gradients',ig),('exact_shapley',shap)]:
                chosen=int(np.argmax(values));summaries.append(dict(sample_id=r['sample_id'],gallery_view=view,method=method,chosen_rule=f'R{chosen+1}',score_drop=drops[chosen],decision_flip=flips[chosen],positive_effect=int(drops[chosen]>0),selection_budget=1,scope=ck['scope']))
                for k,v in enumerate(values):attribs.append(dict(sample_id=r['sample_id'],gallery_view=view,method=method,rule=f'R{k+1}',attribution=float(v)))
    write_csv(out_dir/'rule_interventions.csv',records);write_csv(out_dir/'matched_top1_audit.csv',summaries);write_csv(out_dir/'attributions.csv',attribs)
    write_json(out_dir/'audit_contract.json',dict(scope=ck['scope'],checkpoint_sha256=sha256(run_dir/'best.pt'),manifest_sha256=sha256(manifest),input_space='8 reliability-weighted rule coordinates; neural coordinates and gallery fixed',neutral_value=0,temperature=.1,temperature_status='diagnostic fixed value, not validation-calibrated',ig_intervals=32,shapley_coalitions=256,perturbation_budget=1,attribution_compute_budget='different across methods; all use identical removal evaluation budget',intervention_precision='not claimed; positive_effect is only positive score-drop indicator',selection_bias='rule_removal selects best observed removal; comparison is descriptive, not unbiased superiority evidence'))
