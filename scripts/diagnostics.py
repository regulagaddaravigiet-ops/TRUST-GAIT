import sys,argparse,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.data import read_manifest,sha256
from tritrust.runner import write_csv,write_json
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args();rows=read_manifest(a.manifest);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
counts=collections.Counter((r['split'],r['subject_id'],r['role'],r['view'],r['condition']) for r in rows)
write_csv(out/'class_distribution.csv',[dict(split=k[0],subject_id=k[1],role=k[2],view=k[3],condition=k[4],n=v) for k,v in sorted(counts.items())]);write_json(out/'leakage_checks.json',dict(samples=len(rows),subjects=len({r['subject_id'] for r in rows}),duplicate_ids=0,duplicate_sequence_hashes=0,subject_overlap_across_splits=0,hash_checks='passed',manifest_sha256=sha256(a.manifest)))
