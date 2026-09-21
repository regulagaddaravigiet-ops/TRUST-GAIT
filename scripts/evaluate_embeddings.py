import sys,argparse,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.data import read_manifest
from tritrust.evaluation import identify,metrics
from tritrust.runner import write_json,write_csv
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--embeddings',required=True);p.add_argument('--out',required=True);a=p.parse_args();rows=read_manifest(a.manifest)
with np.load(a.embeddings,allow_pickle=False) as z:
 if list(z['sample_ids'])!=[r['sample_id'] for r in rows]:raise ValueError('Embedding row IDs/order mismatch')
 e=z['embedding']
 if e.ndim!=2 or len(e)!=len(rows) or not np.isfinite(e).all():raise ValueError('Invalid embeddings')
out=Path(a.out);out.mkdir(parents=True,exist_ok=True);pred=identify(e,rows);write_csv(out/'predictions.csv',pred);write_json(out/'metrics.json',metrics(pred))
