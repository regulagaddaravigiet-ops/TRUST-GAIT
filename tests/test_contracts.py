import unittest,tempfile,csv,json
from pathlib import Path
import numpy as np
from tritrust.data import preprocess,read_manifest,FIELDS
from tritrust.fixture import generate
from tritrust.evaluation import identify,metrics,paired_summary

class Contracts(unittest.TestCase):
 def test_preprocess(self):
  x=np.zeros((5,80,20));x[:,5:75,5:15]=1;f,g=preprocess(x);self.assertEqual(f.shape,(30,1,64,44));self.assertEqual(g.shape,(1,64,44));self.assertLessEqual(g.max(),1)
 def test_empty_frame(self):
  with self.assertRaises(ValueError):preprocess(np.zeros((5,64,44)))
 def test_manifest_leakage(self):
  with tempfile.TemporaryDirectory() as d:
   p=generate(d);r=list(csv.DictReader(p.open()));r[-1]['subject_id']=r[0]['subject_id']
   with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(r)
   with self.assertRaisesRegex(ValueError,'leakage'):read_manifest(p)
 def test_hash_mismatch(self):
  with tempfile.TemporaryDirectory() as d:
   p=generate(d);q=next((Path(d)/'sequences').glob('*.npz'));q.write_bytes(b'changed')
   with self.assertRaisesRegex(ValueError,'hash'):read_manifest(p)
 def test_perfect_crossview(self):
  r=[];e=[]
  for subject in ['a','b']:
   for view in ['0','90']:
    for role in ['gallery','probe']:
     r.append(dict(sample_id=subject+view+role,subject_id=subject,split='test',role=role,view=view,condition='NM'));e.append([1,0] if subject=='a' else [0,1])
  p=identify(e,r);self.assertEqual(len(p),4);self.assertEqual(metrics(p)['rank1_macro_cells'],100)
 def test_seed_mismatch(self):
  with self.assertRaises(ValueError):paired_summary({11:1,23:2},{11:1,37:2})
 def test_zero_reliability(self):
  import torch
  from tritrust.rules import soft_predicates,rule_activations
  _,p=soft_predicates(torch.ones(2,6),torch.zeros(2,6));r,g=rule_activations(p,torch.zeros(2,6));self.assertTrue(torch.all(p==.5));self.assertTrue(torch.all(g*r==0))
 def test_triplet_no_pairs(self):
  import torch
  from tritrust.losses import batch_hard_triplet
  x=torch.randn(3,8,requires_grad=True);loss=batch_hard_triplet(x,torch.arange(3));self.assertEqual(loss.item(),0);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
 def test_no_rule_gradient(self):
  import torch
  from tritrust.rules import soft_predicates,rule_activations
  _,p=soft_predicates(torch.rand(2,6),torch.ones(2,6));r,g=rule_activations(p,torch.ones(2,6));self.assertFalse(r.requires_grad)
 def test_model_shapes(self):
  import torch
  from tritrust.models import create_model
  torch.set_num_threads(2)
  for name in ['cnn','bilstm','tritrust']:
   m=create_model(name,4).eval()
   with torch.no_grad():o=m(torch.rand(2,1,64,44),torch.rand(2,30,1,64,44),torch.rand(2,6),torch.rand(2,6))
   self.assertEqual(o['embedding'].shape,(2,256));self.assertEqual(o['logits'].shape,(2,4));self.assertTrue(torch.isfinite(o['embedding']).all())
if __name__=='__main__':unittest.main()
