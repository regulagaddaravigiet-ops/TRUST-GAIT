import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
from tritrust.fixture import generate
from tritrust.runner import run,classical,write_json
p=argparse.ArgumentParser();p.add_argument('--out',default='validation/executed');p.add_argument('--epochs',type=int,default=2);p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37,53,71]);a=p.parse_args()
root=Path(a.out);manifest=generate(root/'fixture');results=[]
for seed in a.seeds:
 for model in ['cnn','bilstm','tritrust','svm','random_forest']:
  dest=root/'runs'/f'seed_{seed}'/model
  print(f'RUN {seed} {model}',flush=True)
  if model in ['svm','random_forest']:classical(manifest,dest,model,seed,fixture=True)
  else:run(manifest,dest,model,seed,epochs=a.epochs,batch_size=8,fixture=True)
  results.append(dict(seed=seed,model=model,status='executed',path=str(dest)))
write_json(root/'execution_matrix.json',dict(scope='synthetic_pipeline_validation_only',runs=results))
