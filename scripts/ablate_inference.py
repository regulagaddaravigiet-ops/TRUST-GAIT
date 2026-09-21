"""Post-training branch knockout diagnostics, NOT a retrained ablation study."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from tritrust.data import read_manifest,prepare,sha256
from tritrust.models import create_model
from tritrust.evaluation import identify,metrics
from tritrust.runner import write_csv,write_json
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(2);run=Path(a.run);c=torch.load(run/'best.pt',weights_only=True,map_location='cpu')
if c['model']!='tritrust' or c['config']['manifest_sha256']!=sha256(a.manifest):raise ValueError('TriTrust checkpoint/manifest mismatch')
m=create_model('tritrust',len(c['classes']));m.load_state_dict(c['state_dict']);m.eval();rows=read_manifest(a.manifest);s,g,q,_=prepare(rows);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);res=[]
for mode in ['full','zero_symbolic','zero_gei_input','zero_temporal_input']:
 emb=[]
 with torch.no_grad():
  for i in range(len(rows)):
   gei=torch.tensor(s[i][1][None]);frames=torch.tensor(s[i][0][None]);rel=torch.tensor(q[i:i+1])
   if mode=='zero_symbolic':rel.zero_()
   if mode=='zero_gei_input':gei.zero_()
   if mode=='zero_temporal_input':frames.zero_()
   emb.append(m(gei,frames,torch.tensor(g[i:i+1]),rel)['embedding'].numpy()[0])
 pred=identify(emb,rows);write_csv(out/(mode+'_predictions.csv'),pred);res.append(dict(mode=mode,rank1=metrics(pred)['rank1_macro_cells'],scope=c['scope']))
write_csv(out/'summary.csv',res);write_json(out/'contract.json',dict(checkpoint_sha256=sha256(run/'best.pt'),scope=c['scope'],type='fixed-checkpoint input/branch knockout on gallery and probe; not retraining; not paper Table8',rule_loss_ablation='not meaningful for fixed non-learned rules'))
