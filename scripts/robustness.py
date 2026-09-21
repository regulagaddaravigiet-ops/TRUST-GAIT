"""Execute defined probe perturbations on a trained model; not manuscript stress definitions."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from tritrust.data import read_manifest,prepare,sha256
from tritrust.models import create_model
from tritrust.evaluation import identify,metrics
from tritrust.runner import write_json,write_csv
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2);run=Path(a.run);c=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
if sha256(a.manifest)!=c['config']['manifest_sha256']:raise ValueError('Manifest mismatch')
model=create_model(c['model'],len(c['classes']));model.load_state_dict(c['state_dict']);model.eval();rows=read_manifest(a.manifest);samples,giat,rel,scale=prepare(rows);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);allmetrics=[]
for condition in ['clean','occlusion_20_percent','pixel_flip_1_percent','missing_frames_20_percent','speed_subsample_2x']:
 rng=np.random.default_rng(c['seed']);emb=[]
 with torch.no_grad():
  for i,r in enumerate(rows):
   frames=samples[i][0].copy();gei=samples[i][1].copy();q=rel[i].copy()
   if r['split']=='test' and r['role']=='probe' and condition!='clean':
    if condition=='occlusion_20_percent':frames[:,:,26:39,:]=0
    elif condition=='pixel_flip_1_percent':mask=rng.random(frames.shape)<.01;frames[mask]=1-frames[mask]
    elif condition=='missing_frames_20_percent':frames[np.sort(rng.choice(30,6,replace=False))]=0
    elif condition=='speed_subsample_2x':frames=frames[::2];frames=np.repeat(frames,2,axis=0)
    gei=frames.mean(0);q[:]=0 # Pose is not re-estimated; no clean pose leaks into corrupt condition.
   e=model(torch.tensor(gei[None]),torch.tensor(frames[None]),torch.tensor(giat[i:i+1]),torch.tensor(q[None]))['embedding'];emb.append(e.numpy()[0])
 pred=identify(emb,rows);m=metrics(pred);write_csv(out/(condition+'_predictions.csv'),pred);allmetrics.append(dict(condition=condition,rank1=m['rank1_macro_cells'],scope=c['scope']))
write_csv(out/'summary.csv',allmetrics);write_json(out/'contract.json',dict(checkpoint_sha256=sha256(run/'best.pt'),scope=c['scope'],pose_policy='all descriptor reliability zero on perturbed probes; isolates availability as well as perturbation',angle_variation='not simulated; use real view labels',definitions='new deterministic diagnostic definitions; not original Table 9 protocol'))
