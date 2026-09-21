"""Paired stats from executed manifests and seed metrics, never reconstructed CSV tuples."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.evaluation import paired_summary
from tritrust.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--model-a',default='tritrust');p.add_argument('--model-b',default='bilstm');p.add_argument('--out',required=True);a=p.parse_args();values=[{},{}];hashes=set();scopes=set()
for idx,model in enumerate([a.model_a,a.model_b]):
 for f in Path(a.runs).glob(f'seed_*/{model}/metrics.json'):
  seed=int(f.parent.parent.name.split('_')[1]);prov=json.loads((f.parent/'provenance.json').read_text());hashes.add(prov['manifest_sha256']);scopes.add(prov['scope']);values[idx][seed]=json.loads(f.read_text())['rank1_macro_cells']
if len(hashes)!=1 or len(scopes)!=1:raise ValueError('Protocol or evidence-scope mismatch')
r=paired_summary(*values);r.update(scope=scopes.pop(),manifest_sha256=hashes.pop(),metric='rank1_macro_cells',model_a=a.model_a,model_b=a.model_b,interpretation='For fixture scope this describes software test outcomes only; no scientific significance claim');write_json(a.out,r)
