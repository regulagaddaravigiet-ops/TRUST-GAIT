"""Explicit sequence manifests; train-only calibration; strict subject isolation."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image

FIELDS = ['sample_id','subject_id','split','role','view','condition','sequence','path','sha256','source_kind']

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read_manifest(path, verify=True):
    path=Path(path); rows=list(csv.DictReader(path.open()))
    if not rows or not set(FIELDS).issubset(rows[0]):raise ValueError('Empty manifest or missing fields')
    seen=set(); hashes={}; subjects={}; counts={}
    for r in rows:
        if r['sample_id'] in seen:raise ValueError('Duplicate sample ID')
        seen.add(r['sample_id'])
        if r['split'] not in ['train','val','test']:raise ValueError('Invalid split')
        if r['role'] not in ['train','gallery','probe']:raise ValueError('Invalid role')
        if (r['split']=='train') != (r['role']=='train'):raise ValueError('Train role mismatch')
        old=subjects.setdefault(r['subject_id'],r['split'])
        if old!=r['split']:raise ValueError('Subject leakage across splits')
        p=(path.parent/r['path']).resolve()
        if verify and sha256(p)!=r['sha256']:raise ValueError('Sample hash mismatch: '+r['sample_id'])
        if r['sha256'] in hashes:raise ValueError('Duplicate sequence bytes: '+r['sample_id'])
        hashes[r['sha256']]=r['sample_id']
        counts[r['split']]=counts.get(r['split'],0)+1
        r['_path']=str(p)
    for split in ['val','test']:
        sub=[r for r in rows if r['split']==split]
        g={r['subject_id'] for r in sub if r['role']=='gallery'}
        q={r['subject_id'] for r in sub if r['role']=='probe'}
        if not q or not q<=g:raise ValueError('Missing gallery identities for '+split)
    if not counts.get('train'):raise ValueError('Missing training data')
    return rows

def preprocess(frames, n=30):
    frames=np.asarray(frames)
    if frames.ndim!=3 or len(frames)<3:raise ValueError('Need T,H,W silhouettes with T>=3')
    if not np.isfinite(frames).all():raise ValueError('Nonfinite silhouettes')
    out=[]
    for x in frames:
        x=x>0; y,c=np.where(x)
        if not len(y):raise ValueError('Empty frame: remove/repair explicitly in source manifest')
        crop=x[y.min():y.max()+1,c.min():c.max()+1]
        scale=min(64/crop.shape[0],44/crop.shape[1]);h=max(1,round(crop.shape[0]*scale));w=max(1,round(crop.shape[1]*scale))
        im=np.asarray(Image.fromarray(crop.astype('uint8')*255).resize((w,h),Image.Resampling.NEAREST))>0
        padded=np.zeros((64,44),dtype=np.float32);padded[(64-h)//2:(64-h)//2+h,(44-w)//2:(44-w)//2+w]=im
        out.append(padded)
    arr=np.stack(out);idx=np.linspace(0,len(arr)-1,n).round().astype(int)
    return arr[idx,None],arr.mean(0)[None]

def extract_pose(pose):
    """COCO-17 x,y,confidence; explicit reconstruction choices in docs/METHOD_MAPPING.md."""
    from scipy.signal import find_peaks
    p=np.asarray(pose,float).copy()
    if p.ndim!=3 or p.shape[1:]!=(17,3):raise ValueError('pose must be T,17,3')
    # Interpolate only internal short gaps. Interpolated confidence uses endpoint minimum.
    for j in range(17):
        valid=np.isfinite(p[:,j]).all(1)&(p[:,j,2]>=.5)
        ids=np.flatnonzero(valid)
        for a,b in zip(ids[:-1],ids[1:]):
            if 1<b-a<=4:
                for t in range(a+1,b):
                    p[t,j,:2]=p[a,j,:2]+(p[b,j,:2]-p[a,j,:2])*(t-a)/(b-a)
                    p[t,j,2]=min(p[a,j,2],p[b,j,2])
    groups=[[15,16],[11,12,13,14,15,16],[5,6,11,12],[15,16],[13,14,15,16],[5,6,11,12,13,14,15,16]]
    vals=np.full(6,np.nan);rel=np.zeros(6)
    for k,ids in enumerate(groups):
        height_ids=[5,6,15,16]
        needed=sorted(set(ids+height_ids))
        valid=np.isfinite(p[:,needed,:]).all((1,2))&(p[:,needed,2]>=.5).all(1)
        coverage=valid.mean()
        if coverage<.6:continue
        q=p[valid];xy=q[:,:,:2];height=np.median(np.ptp(xy[:,height_ids,1],axis=1))
        if height<=1e-8:continue
        rel[k]=coverage*q[:,ids,2].mean();hips=xy[:,[11,12]].mean(1);shoulders=xy[:,[5,6]].mean(1)
        sep=np.linalg.norm(xy[:,15]-xy[:,16],axis=1)/height
        if k in [0,3]:
            # Only contiguous valid frames qualify for cycle measures.
            if not valid.all():rel[k]=0;continue
            smooth=np.convolve(sep,np.ones(3)/3,mode='same');peaks,_=find_peaks(smooth,distance=3,prominence=.01)
            a=sep[peaks]
            if len(a)<2:rel[k]=0;continue
            vals[k]=1-a.std()/(a.mean()+1e-8)
        elif k==1:
            left=xy[:,[11,13,15]]-hips[:,None];right=xy[:,[12,14,16]]-hips[:,None];right[:,:,0]*=-1
            vals[k]=1-np.linalg.norm(left-right,axis=-1).mean()/height
        elif k==2:
            torso=shoulders-hips;angle=np.arctan2(torso[:,0],-torso[:,1]);cent=(shoulders+hips)/2
            vals[k]=np.exp(-(np.var(cent/height,axis=0).sum()+np.var(angle)))
        elif k==4:
            # Do not bridge long missing intervals in second differences.
            good=valid[:-2]&valid[1:-1]&valid[2:]
            d=np.diff(p[:,[13,14,15,16],:2],n=2,axis=0)[good]
            if not len(d):rel[k]=0;continue
            vals[k]=np.exp(-np.linalg.norm(d,axis=-1).mean()/height)
        else:
            leg=(np.linalg.norm(xy[:,11]-xy[:,13],axis=1)+np.linalg.norm(xy[:,13]-xy[:,15],axis=1)+np.linalg.norm(xy[:,12]-xy[:,14],axis=1)+np.linalg.norm(xy[:,14]-xy[:,16],axis=1))/2
            torso=np.linalg.norm(shoulders-hips,axis=1);vals[k]=np.exp(-np.std(leg/(torso+1e-8)))
    return vals,np.clip(rel,0,1)

def load_sample(row):
    with np.load(row['_path'],allow_pickle=False) as z:
        frames,gei=preprocess(z['silhouettes'])
        if 'pose' in z:
            if len(z['pose'])!=len(z['silhouettes']):raise ValueError('Pose/silhouette frame count mismatch')
            raw,q=extract_pose(z['pose'])
        else:raw,q=np.full(6,np.nan),np.zeros(6)
    return frames,gei,raw,q

def prepare(rows):
    samples=[load_sample(r) for r in rows];raw=np.stack([x[2] for x in samples]);q=np.stack([x[3] for x in samples]);train=np.array([r['split']=='train' for r in rows])
    lo=[];hi=[];med=[]
    for k in range(6):
        x=raw[train,k];x=x[np.isfinite(x)]
        lo.append(float(np.quantile(x,.01)) if len(x) else 0.);hi.append(float(np.quantile(x,.99)) if len(x) else 1.);med.append(float(np.median(x)) if len(x) else .5)
    raw=np.where(np.isfinite(raw),raw,np.array(med));scaled=np.clip((raw-lo)/np.maximum(np.array(hi)-lo,1e-6),0,1);scaled[q==0]=.5
    return samples,scaled.astype('float32'),q.astype('float32'),dict(lower=lo,upper=hi,median=med,fit_split='train')
