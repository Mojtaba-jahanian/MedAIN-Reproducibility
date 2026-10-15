"""CPU PyTorch smoke test of ResNet18-U-Net with the frozen MedAIN operator.

The deliberately small cohort/resolution makes this an execution gate, not a
benchmark. No torchvision dependency or pretrained weights are used.
"""
from __future__ import annotations

import csv, json, random, sys, time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1=nn.Conv2d(cin,cout,3,stride,1,bias=False); self.n1=nn.BatchNorm2d(cout)
        self.c2=nn.Conv2d(cout,cout,3,1,1,bias=False); self.n2=nn.BatchNorm2d(cout)
        self.skip=nn.Identity() if cin==cout and stride==1 else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.n2(self.c2(F.relu(self.n1(self.c1(x)))))+self.skip(x))


def stage(cin, cout, stride): return nn.Sequential(BasicBlock(cin,cout,stride),BasicBlock(cout,cout))


class MedAIN(nn.Module):
    def __init__(self, channels=64, rank=4, alpha=.5):
        super().__init__(); self.alpha=alpha
        self.register_buffer("basis", torch.zeros(channels,rank)); self.active=False
    def forward(self,z):
        if not self.active: return z
        proj=torch.einsum("bchw,cr->brhw",z,self.basis)
        return z-self.alpha*torch.einsum("brhw,cr->bchw",proj,self.basis)


class ResNet18UNet(nn.Module):
    def __init__(self, medain=False):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),nn.BatchNorm2d(64),nn.ReLU())
        self.l1=stage(64,64,1); self.medain=MedAIN(64,4,.5); self.use_medain=medain
        self.l2=stage(64,128,2); self.l3=stage(128,256,2); self.l4=stage(256,512,2)
        self.d3=nn.Conv2d(512+256,256,3,1,1); self.d2=nn.Conv2d(256+128,128,3,1,1)
        self.d1=nn.Conv2d(128+64,64,3,1,1); self.head=nn.Conv2d(64,3,1)
    def encode_l1(self,x): return self.l1(self.stem(x))
    def forward(self,x,return_feature=False):
        z1=self.encode_l1(x); z1.retain_grad() if z1.requires_grad else None
        a1=self.medain(z1) if self.use_medain else z1
        z2=self.l2(a1); z3=self.l3(z2); z4=self.l4(z3)
        y=F.interpolate(z4,z3.shape[-2:],mode="bilinear",align_corners=False); y=F.relu(self.d3(torch.cat([y,z3],1)))
        y=F.interpolate(y,z2.shape[-2:],mode="bilinear",align_corners=False); y=F.relu(self.d2(torch.cat([y,z2],1)))
        y=F.interpolate(y,z1.shape[-2:],mode="bilinear",align_corners=False); y=F.relu(self.d1(torch.cat([y,z1],1)))
        y=F.interpolate(self.head(y),x.shape[-2:],mode="bilinear",align_corners=False)
        return (y,z1) if return_feature else y


def load_case(row, root, size=64):
    x=np.asarray(Image.open(root/row["image_relpath"]).convert("RGB").resize((size,size)),dtype=np.float32)/255
    masks=[np.asarray(Image.open(root/p).resize((size,size),Image.Resampling.NEAREST)) for p in row["mask_relpaths"].split(";")]
    s=np.stack(masks); y=np.stack([(s==v).sum(0) for v in (0,128,255)]).argmax(0).astype(np.int64)
    return torch.from_numpy(x.transpose(2,0,1)),torch.from_numpy(y)


def batches(rows,root,batch=2):
    for i in range(0,len(rows),batch):
        items=[load_case(r,root) for r in rows[i:i+batch]]
        yield torch.stack([a for a,_ in items]),torch.stack([b for _,b in items])


def explicit_sgd(model,lr):
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None: p.add_(p.grad,alpha=-lr); p.grad=None


def train_epoch(model,rows,root,lr=.01):
    model.train(); losses=[]
    for x,y in batches(rows,root):
        loss=F.cross_entropy(model(x),y); loss.backward(); explicit_sgd(model,lr); losses.append(loss.item())
    return float(np.mean(losses))


def estimate_basis(model,rows,root,beta=1e-3,rank=4):
    model.eval(); av=[]; tv=[]
    for x,y in list(batches(rows[:8],root)):
        logits,z=model(x,True); loss=F.cross_entropy(logits,y)
        g=torch.autograd.grad(loss,z)[0].mean((2,3)); tv.append(g.detach())
        with torch.no_grad():
            hp=model.encode_l1(x.clamp_min(1e-6).pow(.8)).mean((2,3))
            hm=model.encode_l1(x.clamp_min(1e-6).pow(1.2)).mean((2,3))
            av.append((hp-hm)/.4)
    a=torch.cat(av); t=torch.cat(tv)
    ca=a.T@a/len(a); ct=t.T@t/len(t)
    ca=(ca+ca.T)/2/(torch.trace(ca)+1e-12); ct=(ct+ct.T)/2/(torch.trace(ct)+1e-12)
    l=torch.linalg.cholesky(ct+beta*torch.eye(ct.shape[0])); li=torch.linalg.inv(l)
    w,v=torch.linalg.eigh(li@ca@li.T); vg=torch.linalg.solve(l.T,v[:,-rank:]); q,_=torch.linalg.qr(vg)
    return q, w[-rank:].flip(0)


@torch.no_grad()
def evaluate(model,rows,root):
    model.eval(); losses=[]
    for x,y in batches(rows,root): losses.append(F.cross_entropy(model(x),y).item())
    return float(np.mean(losses))


def main(manifest,dataset_root,outfile):
    torch.manual_seed(1103); np.random.seed(1103); random.seed(1103); torch.set_num_threads(4)
    rows=list(csv.DictReader(open(manifest,encoding="utf-8")))
    train=[]
    for d in ("Magrabia","MESSIDOR_Base1","MESSIDOR_Base2","MESSIDOR_Base3"): train += [r for r in rows if r["domain"]==d][:4]
    test=[r for r in rows if r["domain"]=="BinRushed"][:4]; root=Path(dataset_root)
    base=ResNet18UNet(False); state={k:v.clone() for k,v in base.state_dict().items()}
    q,eig=estimate_basis(base,train,root)
    erm=ResNet18UNet(False); erm.load_state_dict(state)
    med=ResNet18UNet(True); med.load_state_dict(state); med.medain.basis.copy_(q); med.medain.active=True
    t0=time.time(); le=train_epoch(erm,train,root); te=time.time()-t0
    t0=time.time(); lm=train_epoch(med,train,root); tm=time.time()-t0
    ve=evaluate(erm,test,root); vm=evaluate(med,test,root)
    checks={"torch_cpu":not torch.cuda.is_available(),"finite_erm":np.isfinite([le,ve]).all().item(),"finite_medain":np.isfinite([lm,vm]).all().item(),"orthonormal_basis":torch.linalg.norm(q.T@q-torch.eye(4)).item()<1e-5,"basis_nonzero":torch.linalg.norm(q).item()>0}
    report={"status":"PASS" if all(checks.values()) else "FAIL","scope":"PyTorch execution smoke; not benchmark evidence","torch":torch.__version__,"seed":1103,"model":"ResNet18-U-Net","resolution":64,"train_images":len(train),"heldout_images":len(test),"erm_epoch_loss":le,"medain_epoch_loss":lm,"erm_heldout_loss":ve,"medain_heldout_loss":vm,"erm_seconds":te,"medain_seconds":tm,"basis_norm":torch.linalg.norm(q).item(),"basis_orthogonality_error":torch.linalg.norm(q.T@q-torch.eye(4)).item(),"top_generalized_eigenvalues":eig.tolist(),"checks":checks}
    Path(outfile).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))
    if report["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main(sys.argv[1],sys.argv[2],sys.argv[3])
