"""New experimental runs; cannot recover missing historical experiments."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.runner import run,classical
from tritrust.data import sha256,read_manifest
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--contract',required=True);p.add_argument('--out',required=True);p.add_argument('--models',nargs='+',default=['cnn','bilstm','tritrust','svm','random_forest']);p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53,71]);p.add_argument('--device',default='cpu');a=p.parse_args()
c=json.loads(Path(a.contract).read_text())
required=['dataset','protocol_id','manifest_sha256','split_source','preprocessing_description','giat_measurement_convention','license_access_confirmed','evaluator_status']
if any(not c.get(k) for k in required):raise ValueError('Complete the protocol contract with real values')
if c['manifest_sha256']!=sha256(a.manifest):raise ValueError('Contract manifest hash mismatch')
if c['dataset']=='CCGR':raise ValueError('CCGR hard/easy requires the pinned official evaluator; generic evaluator cannot certify it. See docs/PROTOCOLS.md')
rows=read_manifest(a.manifest)
for seed in a.seeds:
 for model in a.models:
  out=Path(a.out)/f'seed_{seed}'/model
  if model in ['svm','random_forest']:classical(a.manifest,out,model,seed)
  else:run(a.manifest,out,model,seed,device=a.device)
  (out/'protocol_contract.json').write_text(json.dumps(c,indent=2))
