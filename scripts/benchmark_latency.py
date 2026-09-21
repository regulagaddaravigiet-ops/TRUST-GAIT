import sys,argparse,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from tritrust.models import create_model
from tritrust.data import sha256
from tritrust.runner import write_json
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True);p.add_argument('--warmup',type=int,default=50);p.add_argument('--repeats',type=int,default=1000);p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(2)
c=torch.load(a.checkpoint,map_location='cpu',weights_only=True);m=create_model(c['model'],len(c['classes'])).to(a.device).eval();m.load_state_dict(c['state_dict']);x=[torch.zeros(1,1,64,44,device=a.device),torch.zeros(1,30,1,64,44,device=a.device),torch.full((1,6),.5,device=a.device),torch.ones(1,6,device=a.device)]
def sync():
 if a.device.startswith('cuda'):torch.cuda.synchronize()
times=[]
with torch.inference_mode():
 for i in range(a.warmup+a.repeats):
  sync();t=time.perf_counter_ns();m(*x);sync();elapsed=(time.perf_counter_ns()-t)/1e6
  if i>=a.warmup:times.append(elapsed)
write_json(a.out,dict(mean_ms=float(np.mean(times)),median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),warmup=a.warmup,repetitions=a.repeats,device=a.device,threads=2,checkpoint_sha256=sha256(a.checkpoint),scope='model-only latency on zero tensor; excludes IO and GIAT extraction; not manuscript end-to-end latency',raw_ms=times))
