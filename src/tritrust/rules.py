import torch

GIAT_ORDER = ["SR","BS","TS","SWC","MS","LRC"]
RULES = {
    "R1": ["SR","BS"],
    "R2": ["TS","SWC"],
    "R3": ["MS","LRC"],
    "R4": ["SR","TS","MS"],
    "R5": ["BS","SWC","LRC"],
    "R6": ["SR","BS","MS"],
    "R7": ["TS","SWC","MS"],
    "R8": ["SR","BS","TS","SWC","MS","LRC"],
}

def soft_predicates(giat, reliability, slope=10.0, midpoint=0.5, neutral=0.5):
    p = torch.sigmoid(slope * (giat - midpoint))
    p_tilde = reliability * p + (1.0 - reliability) * neutral
    return p, p_tilde

def rule_activations(p_tilde, reliability):
    acts=[]; rels=[]
    index={k:i for i,k in enumerate(GIAT_ORDER)}
    for members in RULES.values():
        ids=[index[m] for m in members]
        acts.append(torch.prod(p_tilde[..., ids], dim=-1))
        rels.append(torch.min(reliability[..., ids], dim=-1).values)
    return torch.stack(acts, dim=-1), torch.stack(rels, dim=-1)
