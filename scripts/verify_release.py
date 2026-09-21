import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();root=Path(a.root);bad=[];count=0
for line in (root/'release/SHA256SUMS.txt').read_text().splitlines():
 h,name=line.split('  ',1);f=root/name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:bad.append(name)
 count+=1
print(json.dumps(dict(files_checked=count,failures=bad),indent=2));raise SystemExit(bool(bad))
