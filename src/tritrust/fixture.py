"""Synthetic moving stick-person silhouettes; ONLY pipeline verification."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .data import FIELDS,sha256

def generate(root):
    root=Path(root);(root/'sequences').mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(90417);rows=[]
    for subject in range(8):
        split='train' if subject<4 else ('val' if subject<6 else 'test')
        for view in [0,90]:
            for seq in range(2):
                role='train' if split=='train' else ('gallery' if seq==0 else 'probe');sid=f's{subject:03d}_v{view:03d}_{seq}'
                frames=[];poses=[]
                for frame in range(30):
                    phase=2*np.pi*frame/(10+subject%3)+seq*.3;cx=22+rng.normal(0,.15);hip=36.;swing=(4+subject*.2)*np.sin(phase)
                    p=np.zeros((17,3),float);p[:,2]=.95;p[:,:2]=[cx,15]
                    p[5,:2]=[cx-5,20];p[6,:2]=[cx+5,20];p[11,:2]=[cx-3,hip];p[12,:2]=[cx+3,hip]
                    p[13,:2]=[cx-3+swing*.5,46];p[14,:2]=[cx+3-swing*.5,46];p[15,:2]=[cx-3+swing,59];p[16,:2]=[cx+3-swing,59]
                    p[:,:2]+=rng.normal(0,.1,(17,2));im=Image.new('L',(44,64));d=ImageDraw.Draw(im)
                    d.ellipse((cx-4,4,cx+4,13),fill=255);d.polygon([(cx-5-subject*.2,17),(cx+5+subject*.2,17),(cx+4,38),(cx-4,38)],fill=255)
                    for joints in [[11,13,15],[12,14,16]]:d.line([tuple(p[j,:2]) for j in joints],fill=255,width=4)
                    frames.append(np.array(im));poses.append(p)
                path=root/'sequences'/f'{sid}.npz';np.savez_compressed(path,silhouettes=np.stack(frames),pose=np.stack(poses).astype('float32'))
                rows.append(dict(sample_id=sid,subject_id=f'{subject:03d}',split=split,role=role,view=str(view),condition='NM',sequence=str(seq),path=f'sequences/{sid}.npz',sha256=sha256(path),source_kind='synthetic_fixture'))
    with (root/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    return root/'samples.csv'
