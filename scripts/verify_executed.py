"""Verify saved embeddings, checkpoint and sample provenance against all run metrics."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from tritrust.data import read_manifest,sha256
from tritrust.evaluation import identify,metrics
from tritrust.runner import write_json,write_csv
root=Path('validation/executed');manifest=root/'fixture/samples.csv';rows=read_manifest(manifest);report=[];ledger=[]
for d in sorted((root/'runs').glob('seed_*/*')):
 prov=json.loads((d/'provenance.json').read_text());m=json.loads((d/'metrics.json').read_text());pred=list(csv.DictReader((d/'predictions.csv').open()));p=[{**r,'correct':int(r['correct'])} for r in pred]
 if prov['manifest_sha256']!=sha256(manifest):raise ValueError('Manifest hash mismatch')
 checkpoint=d/('best.pt' if (d/'best.pt').exists() else 'model.joblib')
 if prov['checkpoint_sha256']!=sha256(checkpoint):raise ValueError('Checkpoint hash mismatch')
 if abs(metrics(p)['rank1_macro_cells']-m['rank1_macro_cells'])>1e-9:raise ValueError('CSV metric mismatch')
 if (d/'embeddings.npz').exists():
  with np.load(d/'embeddings.npz',allow_pickle=False) as z:
   if list(z['sample_ids'])!=[r['sample_id'] for r in rows]:raise ValueError('Row-order mismatch')
   pred2=identify(z['embedding'],rows)
  if [r['predicted_id'] for r in pred2]!=[r['predicted_id'] for r in pred]:raise ValueError('Embedding/prediction mismatch')
  config=json.loads((d/'run_config.json').read_text())
  for name,h in config['source_hashes'].items():
   if sha256(Path('src/tritrust')/name)!=h:raise ValueError('Executed source changed: '+name)
 report.append(dict(run=str(d),status='passed',comparisons=len(p)))
 ledger.append(dict(run=str(d),scope=prov['scope'],manifest_sha256=prov['manifest_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),predictions_sha256=sha256(d/'predictions.csv'),metrics_sha256=sha256(d/'metrics.json')))
if len(report)!=25:raise ValueError('Expected all 25 runs')
write_json(root/'artifact_verification.json',dict(runs_checked=len(report),status='passed',checks=report));write_csv(root/'evidence_ledger.csv',ledger);print('25 runs verified')
