"""Convert authorized CASIA-B silhouettes, subject/condition-sequence/view/*.png.
Explicit split JSON required; this script never silently invents train/val subjects.
"""
import sys,argparse,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from PIL import Image
from tritrust.data import FIELDS,sha256
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--splits',required=True);p.add_argument('--out',required=True);a=p.parse_args();splits=json.loads(Path(a.splits).read_text());mapping={}
for split,subjects in splits.items():
 if split not in ['train','val','test']:raise ValueError('Invalid split name')
 for subject in subjects:
  s=f'{int(subject):03d}'
  if s in mapping:raise ValueError('Overlapping split subjects')
  mapping[s]=split
out=Path(a.out);(out/'sequences').mkdir(parents=True,exist_ok=True);rows=[]
for directory in sorted(Path(a.root).glob('*/*/*')):
 if not directory.is_dir():continue
 subject,condition_seq,view=directory.parts[-3:];subject=f'{int(subject):03d}'
 if subject not in mapping:continue
 cond,seq=condition_seq.lower().split('-');seq=int(seq);split=mapping[subject]
 if cond not in ['nm','bg','cl']:raise ValueError('Unknown condition')
 role='train' if split=='train' else ('gallery' if cond=='nm' and seq<=4 else 'probe')
 files=sorted(directory.glob('*.png'))
 if not files:continue
 frames=np.stack([np.asarray(Image.open(f).convert('L')) for f in files]);sid=f'{subject}_{cond}{seq:02d}_{view}';dest=out/'sequences'/f'{sid}.npz'
 # Pose must be separately supplied in the same coordinate system. No pose is invented.
 np.savez_compressed(dest,silhouettes=frames);rows.append(dict(sample_id=sid,subject_id=subject,split=split,role=role,view=view,condition=cond.upper(),sequence=f'{seq:02d}',path=f'sequences/{sid}.npz',sha256=sha256(dest),source_kind='research'))
if not rows:raise ValueError('No matching sequences')
with (out/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
print('Converted',len(rows),'sequences. Pose absent: all GIAT reliabilities will be zero until measured pose is supplied.')
