import torch
import torch.nn.functional as F

def batch_hard_triplet(emb,y,margin=.3):
    emb=F.normalize(emb,dim=1);d=torch.cdist(emb,emb);same=y[:,None]==y[None,:];eye=torch.eye(len(y),dtype=torch.bool,device=y.device)
    pos=d.masked_fill(~(same&~eye),-torch.inf).max(1).values;neg=d.masked_fill(same,torch.inf).min(1).values;valid=torch.isfinite(pos)&torch.isfinite(neg)
    return F.relu(pos[valid]-neg[valid]+margin).mean() if valid.any() else emb.sum()*0
