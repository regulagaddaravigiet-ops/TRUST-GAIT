import sys,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tritrust.explain import audit
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();audit(a.manifest,a.run,a.out)
