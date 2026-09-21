"""Train and retain auditable checkpoints. No target metrics are used in optimization."""
import csv,json,random,time,platform,sys,subprocess
from pathlib import Path
import numpy as np
from .data import read_manifest,prepare,sha256
from .evaluation import identify,metrics

def write_json(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def write_csv(p,rows):
    if not rows:raise ValueError('No rows to write')
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(manifest,out,model,seed,epochs=100,batch_size=32,patience=15,device='cpu',fixture=False):
    import torch
    import torch.nn.functional as F
    from .models import create_model
    from .losses import batch_hard_triplet
    torch.set_num_threads(2);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True);rows=read_manifest(manifest)
    kinds={r['source_kind'] for r in rows}
    if fixture and kinds!={'synthetic_fixture'}:raise ValueError('Fixture runner accepts only labelled synthetic inputs')
    if not fixture and kinds!={'research'}:raise ValueError('Research runner requires source_kind=research for every sample')
    out=Path(out)
    if out.exists():raise ValueError('Refusing to overwrite existing run '+str(out))
    out.mkdir(parents=True);samples,a,q,scaler=prepare(rows)
    train=[i for i,r in enumerate(rows) if r['split']=='train'];classes=sorted({rows[i]['subject_id'] for i in train});labels={v:i for i,v in enumerate(classes)}
    tensors=[torch.tensor(np.stack([s[1] for s in samples])),torch.tensor(np.stack([s[0] for s in samples])),torch.tensor(a),torch.tensor(q)]
    config=dict(model=model,seed=seed,epochs=epochs,batch_size=batch_size,patience=patience,learning_rate=1e-4,weight_decay=1e-5,triplet_margin=.3,triplet_weight=1.,symbolic_weight=.1,device=device,scope='synthetic_pipeline_validation' if fixture else 'new_research_rerun',manifest_sha256=sha256(manifest),classes=classes,rule_loss_status='fixed rules have no trainable path; diagnostic constant only')
    code_root=Path(__file__).parent
    config['source_hashes']={p.name:sha256(p) for p in sorted(code_root.glob('*.py'))}
    write_json(out/'run_config.json',config);write_json(out/'scaler.json',scaler)
    write_csv(out/'input_manifest.csv',[{k:v for k,v in r.items() if k!='_path'} for r in rows])
    model_obj=create_model(model,len(classes)).to(device);opt=torch.optim.Adam(model_obj.parameters(),lr=1e-4,weight_decay=1e-5);history=[];best=-1;stale=0
    def infer():
        model_obj.eval();parts=[]
        with torch.no_grad():
            for i in range(0,len(rows),batch_size):parts.append(model_obj(*[x[i:i+batch_size].to(device) for x in tensors])['embedding'].cpu().numpy())
        return np.concatenate(parts)
    started=time.time()
    for epoch in range(epochs):
        model_obj.train();ids=np.array(train);np.random.shuffle(ids);losses=[];rule_losses=[]
        for start in range(0,len(ids),batch_size):
            idx=ids[start:start+batch_size];ys=torch.tensor([labels[rows[i]['subject_id']] for i in idx],device=device)
            pred=model_obj(*[x[idx].to(device) for x in tensors]);ce=F.cross_entropy(pred['logits'],ys);trip=batch_hard_triplet(pred['embedding'],ys)
            rule=F.relu(.5-pred['rules']).mean() if 'rules' in pred else ce.new_tensor(0.)
            loss=ce+trip+.1*rule
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model_obj.parameters(),5.);opt.step();losses.append(loss.item());rule_losses.append(rule.item())
        emb=infer();val=metrics(identify(emb,rows,'val'))['rank1_macro_cells'];selected=val>best
        history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),rule_loss=float(np.mean(rule_losses)),validation_rank1=val,selected=int(selected)))
        if selected:
            best=val;stale=0
            torch.save({'state_dict':model_obj.state_dict(),'model':model,'classes':classes,'epoch':epoch+1,'seed':seed,'scope':config['scope'],'config':config},out/'best.pt')
        else:stale+=1
        if stale>=patience:break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=True);model_obj.load_state_dict(checkpoint['state_dict']);emb=infer();pred=identify(emb,rows)
    digest=sha256(out/'best.pt')
    for r in pred:r.update(seed=seed,model=model,checkpoint_sha256=digest,scope=config['scope'])
    write_csv(out/'predictions.csv',pred);write_csv(out/'training_history.csv',history);write_json(out/'metrics.json',metrics(pred));np.savez_compressed(out/'embeddings.npz',embedding=emb,sample_ids=np.array([r['sample_id'] for r in rows]))
    write_json(out/'provenance.json',dict(checkpoint_sha256=digest,manifest_sha256=sha256(manifest),python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),device=device,duration_seconds=time.time()-started,selected_epoch=checkpoint['epoch'],scope=config['scope']))
    return out

def classical(manifest,out,name,seed,fixture=False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    rows=read_manifest(manifest)
    if {r['source_kind'] for r in rows}!=({'synthetic_fixture'} if fixture else {'research'}):raise ValueError('Source scope mismatch')
    out=Path(out)
    if out.exists():raise ValueError('Run already exists')
    out.mkdir(parents=True);samples,a,q,scale=prepare(rows);idx=[i for i,r in enumerate(rows) if r['split']=='train'];y=[rows[i]['subject_id'] for i in idx]
    estimator=make_pipeline(StandardScaler(),SVC(C=10,kernel='rbf',gamma='scale',class_weight='balanced',random_state=seed)) if name=='svm' else RandomForestClassifier(n_estimators=300,class_weight='balanced_subsample',random_state=seed,n_jobs=1)
    estimator.fit(a[idx],y);joblib.dump(estimator,out/'model.joblib');write_json(out/'scaler.json',scale)
    # Inductive disjoint-ID adaptation: training-class decision scores are features, not test identity labels.
    emb=estimator.decision_function(a) if name=='svm' else estimator.predict_proba(a)
    if emb.ndim==1:emb=np.column_stack([-emb,emb])
    pred=identify(emb,rows);write_csv(out/'predictions.csv',pred);write_json(out/'metrics.json',metrics(pred))
    write_json(out/'provenance.json',dict(seed=seed,scope='synthetic_pipeline_validation' if fixture else 'new_research_rerun',adaptation='training-class score embedding plus cosine gallery matching; not original closed-set Table 5 classifier protocol',manifest_sha256=sha256(manifest),checkpoint_sha256=sha256(out/'model.joblib')))
    return out
