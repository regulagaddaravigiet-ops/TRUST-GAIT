import torch
import torch.nn as nn
from .rules import soft_predicates, rule_activations

class GEIEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        ch=[1,32,64,128,256]
        blocks=[]
        for i in range(4):
            blocks += [nn.Conv2d(ch[i],ch[i+1],3,padding=1), nn.BatchNorm2d(ch[i+1]), nn.ReLU(inplace=True)]
            if i<3: blocks += [nn.MaxPool2d(2)]
        self.net=nn.Sequential(*blocks)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.proj=nn.Linear(256,out_dim)
    def forward(self,x):
        x=self.pool(self.net(x)).flatten(1)
        return self.proj(x)

class FrameEncoder(nn.Module):
    def __init__(self,out_dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.proj=nn.Linear(64,out_dim)
    def forward(self,x): return self.proj(self.net(x).flatten(1))

class TemporalBiLSTM(nn.Module):
    def __init__(self,input_dim=128,hidden=256,layers=2,dropout=0.30,out_dim=256):
        super().__init__()
        self.rnn=nn.LSTM(input_dim,hidden,num_layers=layers,batch_first=True,bidirectional=True,dropout=dropout)
        self.att=nn.Linear(hidden*2,1)
        self.proj=nn.Linear(hidden*2,out_dim)
    def forward(self,x):
        h,_=self.rnn(x)
        a=torch.softmax(self.att(h).squeeze(-1),dim=1)
        pooled=(h*a.unsqueeze(-1)).sum(dim=1)
        return self.proj(pooled)

class TriTrustGait(nn.Module):
    def __init__(self,num_train_ids,embedding_dim=256):
        super().__init__()
        self.gei=GEIEncoder(embedding_dim)
        self.frame=FrameEncoder(128)
        self.temporal=TemporalBiLSTM(128,256,2,0.30,embedding_dim)
        self.neural_fuse=nn.Linear(embedding_dim*2,embedding_dim)
        self.final_fuse=nn.Linear(embedding_dim+8,embedding_dim)
        self.classifier=nn.Linear(embedding_dim,num_train_ids)
    def forward(self,gei,frames,giat,reliability):
        b,t,c,h,w=frames.shape
        f=self.frame(frames.reshape(b*t,c,h,w)).reshape(b,t,-1)
        z=torch.relu(self.neural_fuse(torch.cat([self.gei(gei), self.temporal(f)],dim=-1)))
        _,pt=soft_predicates(giat,reliability)
        r,g=rule_activations(pt,reliability)
        hvec=self.final_fuse(torch.cat([z,g*r],dim=-1))
        logits=self.classifier(hvec)
        return {"embedding":hvec,"logits":logits,"rules":r,"rule_reliability":g,"predicates":pt}

class CNN(nn.Module):
    def __init__(self,classes):
        super().__init__();self.encoder=GEIEncoder();self.classifier=nn.Linear(256,classes)
    def forward(self,gei,frames,giat,reliability):
        h=self.encoder(gei);return {'embedding':h,'logits':self.classifier(h)}

class BiLSTM(nn.Module):
    def __init__(self,classes):
        super().__init__();self.frame=FrameEncoder();self.temporal=TemporalBiLSTM();self.classifier=nn.Linear(256,classes)
    def forward(self,gei,frames,giat,reliability):
        b,t,c,h,w=frames.shape;f=self.frame(frames.reshape(b*t,c,h,w)).reshape(b,t,-1);z=self.temporal(f)
        return {'embedding':z,'logits':self.classifier(z)}

def create_model(name,classes):
    return {'cnn':CNN,'bilstm':BiLSTM,'tritrust':TriTrustGait}[name](classes)
