"""One command for the complete synthetic software-verification pipeline."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',default='validation/fresh');a=p.parse_args();root=Path(a.out)
if root.exists():raise ValueError('Choose an unused output directory')
def command(script,*args):subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)
command('run_fixture.py','--out',root)
m=root/'fixture/samples.csv'
command('diagnostics.py','--manifest',m,'--out',root/'diagnostics')
for seed in [11,23,37,53,71]:command('audit_rules.py','--manifest',m,'--run',root/'runs'/f'seed_{seed}'/'tritrust','--out',root/'audits'/f'seed_{seed}')
r=root/'runs/seed_11/tritrust'
command('robustness.py','--manifest',m,'--run',r,'--out',root/'robustness_seed11')
command('ablate_inference.py','--manifest',m,'--run',r,'--out',root/'ablation_seed11')
command('benchmark_latency.py','--checkpoint',r/'best.pt','--out',root/'latency_seed11.json')
command('compare_seeds.py','--runs',root/'runs','--out',root/'paired_fixture_statistics.json')
print('Completed synthetic verification only:',root)
