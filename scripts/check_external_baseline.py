import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.data import sha256
p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--manifest',required=True);a=p.parse_args();path=Path(a.record);r=json.loads(path.read_text())
for k in ['repository_url','commit','architecture','input_adapter_description','training_config_path','training_log_path','checkpoint_path','embeddings_path','manifest_sha256','seed','scope']:
 if k not in r or r[k] in [None,'']:raise ValueError('Missing provenance: '+k)
if not re.fullmatch('[0-9a-f]{40}',r['commit']):raise ValueError('Need full upstream git commit')
if r['scope']!='research':raise ValueError('Matched research baseline requires genuine research run')
if r['manifest_sha256']!=sha256(a.manifest):raise ValueError('Manifest mismatch')
for k in ['training_config','training_log','checkpoint','embeddings']:
 f=path.parent/r[k+'_path']
 if sha256(f)!=r.get(k+'_sha256'):raise ValueError('Missing/mismatched hash for '+k)
print('External baseline artifact consistency passed; this does not independently establish training authenticity.')
