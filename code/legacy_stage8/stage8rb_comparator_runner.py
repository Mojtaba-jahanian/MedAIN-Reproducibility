from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

ROOT10 = Path.home() / "MedAIN_V10"
ROOT8 = Path.home() / "MedAIN_V15_STAGE8"
REPAIR_ROOT = Path.home() / "MedAIN_V13_IDENTITY_REPAIR"
CACHE_MANIFEST = REPAIR_ROOT / "manifests/riga_cache256_manifest_identitysafe.csv"
SPLIT_DIR = REPAIR_ROOT / "manifests"
N_CLASSES = 3
IGNORE_INDEX = 255
DOMAINS = ["BinRushed","Magrabia","MESSIDOR_Base1","MESSIDOR_Base2","MESSIDOR_Base3"]
METHODS = ["ERM_stage8","StrongAug","IBN","MixStyle","CORAL","Fishr"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def stable_u01(text: str) -> float:
    h = hashlib.sha256(text.encode()).hexdigest()
    return int(h[:16], 16) / 2**64


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def build_native_domain_lookup():
    # Native source domain is derived only from the identity-safe LODO test
    # assignment, using the collision-free sample_uid key.
    lookup = {}
    for d in DOMAINS:
        p = SPLIT_DIR / f"riga_lodo_{d}_identitysafe.csv"
        if not p.exists():
            raise RuntimeError(f"Missing identity-safe LODO manifest: {p}")
        with p.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 744 or len({r["sample_uid"] for r in rows}) != 744:
            raise RuntimeError(f"Bad identity-safe LODO manifest: {p}")
        for r in rows:
            if r["split"] == "test":
                uid = r["sample_uid"]
                if uid in lookup and lookup[uid] != d:
                    raise RuntimeError(f"Native-domain collision for {uid}")
                lookup[uid] = d
    if len(lookup) != 744:
        raise RuntimeError(f"Expected 744 native-domain assignments, got {len(lookup)}")
    return lookup


def load_rows(target_domain):
    with CACHE_MANIFEST.open(encoding="utf-8") as f:
        cache_rows = list(csv.DictReader(f))
    if len(cache_rows) != 744:
        raise RuntimeError(f"Expected 744 corrected cache rows, got {len(cache_rows)}")
    cache = {r["sample_uid"]: r for r in cache_rows}
    if len(cache) != 744:
        raise RuntimeError(f"sample_uid cache mapping not one-to-one: {len(cache)}/744")
    native = build_native_domain_lookup()

    split_file = SPLIT_DIR / f"riga_lodo_{target_domain}_identitysafe.csv"
    with split_file.open(encoding="utf-8") as f:
        split_rows = list(csv.DictReader(f))
    if len(split_rows) != 744 or len({r["sample_uid"] for r in split_rows}) != 744:
        raise RuntimeError(f"Bad identity-safe split manifest: {split_file}")

    merged = []
    for s in split_rows:
        uid = s["sample_uid"]
        if uid not in cache:
            raise RuntimeError(f"Missing corrected cached sample: {uid}")
        if uid not in native:
            raise RuntimeError(f"Missing native domain: {uid}")
        if s.get("domain") and s["domain"] != native[uid]:
            raise RuntimeError(f"Native-domain mismatch for {uid}: {s['domain']} vs {native[uid]}")
        merged.append({
            **cache[uid],
            "split": s["split"],
            "outer_target": s.get("outer_target", target_domain),
            "source_domain": native[uid],
        })
    if len(merged) != 744 or len({r["sample_uid"] for r in merged}) != 744:
        raise RuntimeError("Corrected merged rows are not 744/744 unique")
    out = {k:[r for r in merged if r["split"]==k] for k in ("train","val","test")}
    sets={k:{r["sample_uid"] for r in out[k]} for k in ("train","val","test")}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("Identity-safe split overlap detected")
    # Source partitions must not contain the held-out target domain; test must be pure target.
    for k in ("train","val"):
        bad=[r["sample_uid"] for r in out[k] if r["source_domain"]==target_domain]
        if bad:
            raise RuntimeError(f"Target-domain leakage in {k}: {bad[:5]}")
    bad_test=[r["sample_uid"] for r in out["test"] if r["source_domain"]!=target_domain]
    if bad_test:
        raise RuntimeError(f"Held-out test-domain impurity: {bad_test[:5]}")
    return out


class RigaDataset(Dataset):
    def __init__(self, rows, augmentation="baseline", seed=1103, epoch=0):
        self.rows = rows
        self.augmentation = augmentation
        self.seed = seed
        self.epoch = epoch
    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.rows)

    def _baseline(self, x, y, sid):
        key=f"{self.seed}:{self.epoch}:{sid}:baseline"
        flip=stable_u01(key+":flip") < 0.5
        angle=stable_u01(key+":angle")*20.0-10.0
        if flip:
            x=TF.hflip(x); y=TF.hflip(y)
        x=TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        y=TF.rotate(y.unsqueeze(0).float(), angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=float(IGNORE_INDEX)).squeeze(0).long()
        return x,y

    def _strong(self, x, y, sid):
        key=f"{self.seed}:{self.epoch}:{sid}:strong"
        flip=stable_u01(key+":flip") < 0.5
        angle=stable_u01(key+":angle")*30.0-15.0
        if flip:
            x=TF.hflip(x); y=TF.hflip(y)
        x=TF.rotate(x, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
        y=TF.rotate(y.unsqueeze(0).float(), angle,
                    interpolation=InterpolationMode.NEAREST,
                    fill=float(IGNORE_INDEX)).squeeze(0).long()
        bf=0.8+0.4*stable_u01(key+":brightness")
        cf=0.8+0.4*stable_u01(key+":contrast")
        sf=0.8+0.4*stable_u01(key+":saturation")
        gf=0.85+0.30*stable_u01(key+":gamma")
        x=TF.adjust_brightness(x,bf)
        x=TF.adjust_contrast(x,cf)
        x=TF.adjust_saturation(x,sf)
        x=TF.adjust_gamma(torch.clamp(x,0,1),gf)
        return torch.clamp(x,0,1),y

    def __getitem__(self, idx):
        r=self.rows[idx]
        with Image.open(ROOT10 / r["cached_image"]) as im:
            x=TF.to_tensor(im.convert("RGB"))
        with Image.open(ROOT10 / r["cached_hard_mask"]) as im:
            y=torch.from_numpy(np.array(im,dtype=np.uint8,copy=True)).long()
        if self.augmentation=="baseline":
            x,y=self._baseline(x,y,r["sample_uid"])
        elif self.augmentation=="strong":
            x,y=self._strong(x,y,r["sample_uid"])
        elif self.augmentation!="none":
            raise ValueError(self.augmentation)
        return x,y,r["sample_uid"],r["source_domain"]


class DomainBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=8, seed=1103, epoch=0):
        if batch_size != 8:
            raise ValueError("Frozen CORAL/Fishr batching requires batch_size=8")
        self.dataset=dataset; self.batch_size=batch_size; self.seed=seed; self.epoch=epoch
        groups=defaultdict(list)
        for i,r in enumerate(dataset.rows): groups[r["source_domain"]].append(i)
        self.domains=sorted(groups)
        if len(self.domains)!=4:
            raise RuntimeError(f"Expected 4 source domains, got {self.domains}")
        self.groups=dict(groups)
        self.steps=math.ceil(len(dataset)/batch_size)
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed + 100003*self.epoch)
        pools={d:rng.permutation(v).tolist() for d,v in self.groups.items()}
        ptr={d:0 for d in self.domains}
        for _ in range(self.steps):
            batch=[]
            for d in self.domains:
                for _ in range(2):
                    if ptr[d] >= len(pools[d]):
                        pools[d]=rng.permutation(self.groups[d]).tolist(); ptr[d]=0
                    batch.append(pools[d][ptr[d]]); ptr[d]+=1
            rng.shuffle(batch)
            yield batch


def collate(batch):
    xs,ys,sids,doms=zip(*batch)
    return torch.stack(xs),torch.stack(ys),list(sids),list(doms)


def make_loader(dataset,batch_size,shuffle,seed,balanced=False,epoch=0):
    if balanced:
        sampler=DomainBalancedBatchSampler(dataset,batch_size=batch_size,seed=seed,epoch=epoch)
        return DataLoader(dataset,batch_sampler=sampler,num_workers=0,pin_memory=False,collate_fn=collate)
    gen=torch.Generator(); gen.manual_seed(seed)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,generator=gen,
                      num_workers=0,pin_memory=False,drop_last=False,collate_fn=collate)


class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self,cin,skip,cout): super().__init__(); self.conv=ConvBlock(cin+skip,cout)
    def forward(self,x,s):
        x=F.interpolate(x,size=s.shape[-2:],mode="bilinear",align_corners=False)
        return self.conv(torch.cat([x,s],dim=1))

class IBN(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d):
        super().__init__(); c=bn.num_features; self.half=c//2
        self.IN=nn.InstanceNorm2d(self.half,affine=True,track_running_stats=False)
        self.BN=nn.BatchNorm2d(c-self.half,eps=bn.eps,momentum=bn.momentum,
                              affine=True,track_running_stats=True)
        with torch.no_grad():
            if bn.affine:
                self.IN.weight.copy_(bn.weight[:self.half]); self.IN.bias.copy_(bn.bias[:self.half])
                self.BN.weight.copy_(bn.weight[self.half:]); self.BN.bias.copy_(bn.bias[self.half:])
            if bn.track_running_stats:
                self.BN.running_mean.copy_(bn.running_mean[self.half:])
                self.BN.running_var.copy_(bn.running_var[self.half:])
                self.BN.num_batches_tracked.copy_(bn.num_batches_tracked)
    def forward(self,x):
        a,b=torch.split(x,[self.half,x.shape[1]-self.half],dim=1)
        return torch.cat([self.IN(a.contiguous()),self.BN(b.contiguous())],dim=1)

class MixStyle(nn.Module):
    def __init__(self,p=0.5,alpha=0.1,eps=1e-6):
        super().__init__(); self.p=p; self.alpha=alpha; self.eps=eps
    def forward(self,x):
        if not self.training or random.random()>self.p: return x
        B=x.size(0)
        mu=x.mean(dim=[2,3],keepdim=True)
        var=x.var(dim=[2,3],keepdim=True)
        sig=(var+self.eps).sqrt(); mu,sig=mu.detach(),sig.detach()
        xn=(x-mu)/sig
        beta=torch.distributions.Beta(self.alpha,self.alpha)
        lmda=beta.sample((B,1,1,1)).to(x.device)
        perm=torch.randperm(B,device=x.device)
        return xn*(sig*lmda+sig[perm]*(1-lmda)) + (mu*lmda+mu[perm]*(1-lmda))


class ResNet18UNetComparator(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        b=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if method=="IBN":
            for layer in (b.layer1,b.layer2,b.layer3):
                for block in layer:
                    block.bn1=IBN(block.bn1)
        self.conv1=b.conv1; self.bn1=b.bn1; self.relu=b.relu; self.maxpool=b.maxpool
        self.layer1=b.layer1; self.layer2=b.layer2; self.layer3=b.layer3; self.layer4=b.layer4
        self.mixstyle=MixStyle(0.5,0.1) if method=="MixStyle" else None
        self.up3=UpBlock(512,256,256); self.up2=UpBlock(256,128,128)
        self.up1=UpBlock(128,64,64); self.up0=UpBlock(64,64,64)
        self.final_conv=ConvBlock(64,32); self.head=nn.Conv2d(32,N_CLASSES,1)
        self.register_buffer("img_mean",torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer("img_std",torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def normalize(self,x): return (x-self.img_mean)/self.img_std
    def forward(self,x,return_layer2=False,return_head_input=False):
        input_hw=x.shape[-2:]
        x=self.normalize(x); s0=self.relu(self.bn1(self.conv1(x))); z=self.maxpool(s0)
        f1=self.layer1(z)
        if self.mixstyle is not None: f1=self.mixstyle(f1)
        f2=self.layer2(f1)
        if self.mixstyle is not None: f2=self.mixstyle(f2)
        f3=self.layer3(f2); f4=self.layer4(f3)
        d3=self.up3(f4,f3); d2=self.up2(d3,f2); d1=self.up1(d2,f1); d0=self.up0(d1,s0)
        d=F.interpolate(d0,scale_factor=2,mode="bilinear",align_corners=False)
        h=self.final_conv(d); logits=self.head(h)
        if logits.shape[-2:]!=input_hw:
            logits=F.interpolate(logits,size=input_hw,mode="bilinear",align_corners=False)
        if return_layer2 or return_head_input:
            out=[logits]
            if return_layer2: out.append(F.adaptive_avg_pool2d(f2,1).flatten(1))
            if return_head_input: out.append(h)
            return tuple(out)
        return logits


def foreground_dice_loss(logits,target):
    valid=target!=IGNORE_INDEX; safe=target.clone(); safe[~valid]=0
    probs=torch.softmax(logits,dim=1)
    onehot=F.one_hot(safe,num_classes=N_CLASSES).permute(0,3,1,2).float()
    valid4=valid.unsqueeze(1).float(); probs=probs*valid4; onehot=onehot*valid4
    ds=[]
    for c in range(1,N_CLASSES):
        p=probs[:,c]; y=onehot[:,c]; inter=(p*y).sum(); den=p.sum()+y.sum()
        ds.append((2*inter+1)/(den+1))
    return 1-torch.stack(ds).mean()

def loss_fn(logits,target,class_weights):
    ce=F.cross_entropy(logits,target,weight=class_weights,ignore_index=IGNORE_INDEX)
    dl=foreground_dice_loss(logits,target)
    return ce+dl,ce.detach(),dl.detach()

def compute_train_weights(rows):
    counts=np.zeros(N_CLASSES,dtype=np.float64)
    for r in rows:
        with Image.open(ROOT10/r["cached_hard_mask"]) as im: a=np.asarray(im)
        for c in range(N_CLASSES): counts[c]+=np.sum(a==c)
    freq=counts/counts.sum(); w=1/np.sqrt(np.maximum(freq,1e-10)); w=w/w.mean(); w=np.clip(w,0.2,3.0)
    return torch.tensor(w,dtype=torch.float32),counts,freq

class MetricAccumulator:
    def __init__(self):
        self.inter=np.zeros(N_CLASSES); self.pred=np.zeros(N_CLASSES); self.true=np.zeros(N_CLASSES)
    def update(self,logits,target):
        pred=logits.argmax(1); valid=target!=IGNORE_INDEX
        for c in range(N_CLASSES):
            p=(pred==c)&valid; y=(target==c)&valid
            self.inter[c]+=(p&y).sum().item(); self.pred[c]+=p.sum().item(); self.true[c]+=y.sum().item()
    def compute(self):
        r={}; ds=[]; js=[]
        for c in range(N_CLASSES):
            inter=self.inter[c]; dice=(2*inter+1e-8)/(self.pred[c]+self.true[c]+1e-8)
            union=self.pred[c]+self.true[c]-inter; iou=(inter+1e-8)/(union+1e-8)
            r[f"dice_c{c}"]=float(dice); r[f"iou_c{c}"]=float(iou)
            if c>0: ds.append(dice); js.append(iou)
        r["mean_fg_dice"]=float(np.mean(ds)); r["mean_fg_iou"]=float(np.mean(js)); return r

@torch.no_grad()
def evaluate(model,dataset,class_weights,device,batch_size=8):
    loader=make_loader(dataset,batch_size,False,0,balanced=False); model.eval(); acc=MetricAccumulator()
    ls=cs=ds=0.; n=0
    for x,y,_,_ in loader:
        x=x.to(device); y=y.to(device); logits=model(x); loss,ce,dl=loss_fn(logits,y,class_weights)
        bs=x.shape[0]; ls+=loss.item()*bs; cs+=ce.item()*bs; ds+=dl.item()*bs; n+=bs; acc.update(logits,y)
    m=acc.compute(); m.update(loss=ls/n,ce_loss=cs/n,dice_loss=ds/n,n_samples=n); return m


def coral_penalty(features,domains):
    uniq=sorted(set(domains)); stats={}
    for d in uniq:
        idx=torch.tensor([i for i,x in enumerate(domains) if x==d],device=features.device)
        z=features.index_select(0,idx); mu=z.mean(0)
        zc=z-mu; cov=(zc.T@zc)/max(z.shape[0]-1,1); stats[d]=(mu,cov)
    ps=[]
    for i in range(len(uniq)):
        for j in range(i+1,len(uniq)):
            m1,c1=stats[uniq[i]]; m2,c2=stats[uniq[j]]
            ps.append((m1-m2).pow(2).mean()+(c1-c2).pow(2).mean())
    return torch.stack(ps).mean()


def fishr_per_image_head_grads(model,logits,target,class_weights):
    grads=[]
    for i in range(logits.shape[0]):
        li,_,_=loss_fn(logits[i:i+1],target[i:i+1],class_weights)
        gw,gb=torch.autograd.grad(li,[model.head.weight,model.head.bias],retain_graph=True,create_graph=True)
        grads.append(torch.cat([gw.reshape(-1),gb.reshape(-1)]))
    return torch.stack(grads,0)


def fishr_penalty(model,logits,target,domains,class_weights,ema_state,ema_updates,ema=0.95):
    G=fishr_per_image_head_grads(model,logits,target,class_weights)
    current={}; corrected={}
    for d in sorted(set(domains)):
        idx=torch.tensor([i for i,x in enumerate(domains) if x==d],device=G.device)
        gd=G.index_select(0,idx); v=((gd-gd.mean(0,keepdim=True))**2).mean(0)
        current[d]=v
        old=ema_state.get(d)
        if old is None: old=torch.zeros_like(v).detach()
        e=ema*old+(1-ema)*v
        t=ema_updates.get(d,0)+1
        corrected[d]=e/(1-ema**t)
    meanv=torch.stack([corrected[d] for d in sorted(corrected)]).mean(0)
    penalty=torch.stack([(corrected[d]-meanv).pow(2).mean() for d in sorted(corrected)]).mean()
    new_state={d:(ema*ema_state.get(d,torch.zeros_like(current[d])).detach()+(1-ema)*current[d]).detach() for d in current}
    new_updates={d:ema_updates.get(d,0)+1 for d in current}
    return penalty,new_state,new_updates


def optimizer_new(model):
    return torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--method",choices=METHODS,required=True)
    ap.add_argument("--target-domain",choices=DOMAINS,required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--patience",type=int,default=10)
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1); seed_everything(args.seed)
    device=torch.device("cpu"); rows=load_rows(args.target_domain)
    aug="strong" if args.method=="StrongAug" else "baseline"
    train_ds=RigaDataset(rows["train"],augmentation=aug,seed=args.seed)
    val_ds=RigaDataset(rows["val"],augmentation="none",seed=args.seed)
    class_weights,counts,freq=compute_train_weights(rows["train"]); class_weights=class_weights.to(device)
    model=ResNet18UNetComparator(args.method).to(device)
    balanced=args.method in ("CORAL","Fishr")

    # Non-analytic one-batch backward smoke.
    if args.smoke:
        train_ds.set_epoch(0)
        loader=make_loader(train_ds,args.batch_size,True,args.seed,balanced=balanced,epoch=0)
        x,y,_,doms=next(iter(loader)); x=x.to(device); y=y.to(device); model.train()
        if args.method=="CORAL":
            logits,feat=model(x,return_layer2=True); base,_,_=loss_fn(logits,y,class_weights); pen=coral_penalty(feat,doms); total=base+pen
        elif args.method=="Fishr":
            logits=model(x); base,_,_=loss_fn(logits,y,class_weights); pen,_,_=fishr_penalty(model,logits,y,doms,class_weights,{},{}); total=base+pen
        else:
            logits=model(x); base,_,_=loss_fn(logits,y,class_weights); pen=torch.tensor(0.); total=base
        if logits.shape!=(args.batch_size,N_CLASSES,256,256): raise RuntimeError(f"bad logits {tuple(logits.shape)}")
        if not torch.isfinite(total): raise RuntimeError("non-finite smoke loss")
        total.backward()
        print(json.dumps({"SMOKE":"PASS","method":args.method,"base_loss":float(base.detach()),"penalty":float(pen.detach()),"domains":sorted(set(doms))}))
        return

    run_name=f"{args.method}_{args.target_domain}_seed{args.seed}"
    out_dir=ROOT8/"results/stage8rb"/run_name; ckpt_dir=ROOT8/"checkpoints/stage8rb"/run_name
    out_dir.mkdir(parents=True,exist_ok=True); ckpt_dir.mkdir(parents=True,exist_ok=True)
    config={**vars(args),"run_name":run_name,"train_n":len(train_ds),"val_n":len(val_ds),"test_n_locked_not_evaluated":len(rows["test"]),
            "source_domains":sorted(set(r["source_domain"] for r in rows["train"])),"class_counts":counts.tolist(),"class_freq":freq.tolist(),
            "class_weights":class_weights.cpu().tolist(),"external_inference":False,"target_test_evaluated":False,
            "torch_version":torch.__version__,
            "identity_key":"sample_uid","identity_rows":744,
            "cache_manifest":str(CACHE_MANIFEST),"split_manifest":str(SPLIT_DIR / f"riga_lodo_{args.target_domain}_identitysafe.csv"),
            "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(out_dir/"config.json",config)

    optimizer=optimizer_new(model); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    best_score=-1.; best_epoch=-1; bad_epochs=0; history=[]; start_epoch=0; update_count=0; fishr_ema_state={}; fishr_ema_updates={}; fishr_active_updates=0; fishr_selection_unlocked=(args.method != "Fishr"); fishr_activation_epoch=None
    last=ckpt_dir/"last_state.pt"
    if last.exists():
        st=torch.load(last,map_location=device,weights_only=False)
        st_cfg=st.get("config",{})
        current_runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if st_cfg.get("identity_key") != "sample_uid" or st_cfg.get("identity_rows") != 744 or st_cfg.get("runner_sha256") != current_runner_sha:
            raise RuntimeError("Refusing resume: checkpoint provenance does not match corrected Stage8R-B runner/data contract")
        model.load_state_dict(st["model_state"]); optimizer.load_state_dict(st["optimizer_state"]); scheduler.load_state_dict(st["scheduler_state"])
        best_score=float(st["best_score"]); best_epoch=int(st["best_epoch"]); bad_epochs=int(st["bad_epochs"]); history=st.get("history",[]); start_epoch=int(st["epoch"])+1
        update_count=int(st.get("update_count",0)); fishr_ema_state={k:v.to(device) for k,v in st.get("fishr_ema_state",{}).items()}; fishr_ema_updates=st.get("fishr_ema_updates",{}); fishr_active_updates=int(st.get("fishr_active_updates",0)); fishr_selection_unlocked=bool(st.get("fishr_selection_unlocked", args.method != "Fishr")); fishr_activation_epoch=st.get("fishr_activation_epoch")
        print(f"RESUME epoch={start_epoch} best={best_epoch} val={best_score:.6f} updates={update_count} fishr_selection_unlocked={fishr_selection_unlocked}")

    start=time.time()
    for epoch in range(start_epoch,args.epochs):
        epoch_start=time.time(); train_ds.set_epoch(epoch)
        # deterministic model-side stochasticity, including MixStyle
        random.seed(args.seed+700001*epoch); np.random.seed((args.seed+700001*epoch)%(2**32-1)); torch.manual_seed(args.seed+700001*epoch)
        loader=make_loader(train_ds,args.batch_size,True,args.seed+epoch,balanced=balanced,epoch=epoch)
        model.train(); train_loss=0.; base_loss_sum=0.; penalty_sum=0.; train_n=0; tracc=MetricAccumulator(); updates_epoch=0
        for x,y,_,doms in loader:
            x=x.to(device); y=y.to(device)
            # Reset AdamW at exact Fishr penalty activation boundary.
            if args.method=="Fishr" and update_count==1500:
                # Reset AdamW moment state while preserving the exact current LR and
                # the already-running frozen cosine scheduler.
                optimizer.state.clear()
                print("FISHR PENALTY ACTIVATED; ADAMW MOMENT STATE RESET",flush=True)
            optimizer.zero_grad(set_to_none=True)
            if args.method=="CORAL":
                logits,feat=model(x,return_layer2=True); base,_,_=loss_fn(logits,y,class_weights); pen=coral_penalty(feat,doms); total=base+1.0*pen
            elif args.method=="Fishr":
                logits=model(x); base,_,_=loss_fn(logits,y,class_weights)
                pen,new_ema,new_updates=fishr_penalty(model,logits,y,doms,class_weights,fishr_ema_state,fishr_ema_updates,ema=0.95)
                weight=1000.0 if update_count>=1500 else 0.0
                total=base+weight*pen
                fishr_ema_state=new_ema; fishr_ema_updates=new_updates
                if weight>0: fishr_active_updates+=1
            else:
                logits=model(x); base,_,_=loss_fn(logits,y,class_weights); pen=torch.tensor(0.,device=device); total=base
            if not torch.isfinite(total): raise RuntimeError(f"Non-finite loss at epoch {epoch} update {update_count}")
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.0); optimizer.step(); update_count+=1; updates_epoch+=1
            bs=x.shape[0]; train_loss+=float(total.detach())*bs; base_loss_sum+=float(base.detach())*bs; penalty_sum+=float(pen.detach())*bs; train_n+=bs; tracc.update(logits.detach(),y)
        scheduler.step(); trm=tracc.compute(); valm=evaluate(model,val_ds,class_weights,device,args.batch_size)
        row={"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],"train_loss":train_loss/train_n,"train_base_loss":base_loss_sum/train_n,"train_penalty_raw":penalty_sum/train_n,
             "train_mean_fg_dice":trm["mean_fg_dice"],"train_mean_fg_iou":trm["mean_fg_iou"],"val_loss":valm["loss"],"val_mean_fg_dice":valm["mean_fg_dice"],"val_mean_fg_iou":valm["mean_fg_iou"],
             "val_dice_c1":valm["dice_c1"],"val_dice_c2":valm["dice_c2"],"update_count":update_count,"fishr_active_updates":fishr_active_updates,"seconds":time.time()-epoch_start}
        history.append(row); print(json.dumps(row),flush=True)
        with (out_dir/"training_history.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        score=valm["mean_fg_dice"]
        # Fishr identity guard: its frozen penalty is inactive before update 1500.
        # Pre-anneal epochs are deterministic warm-up only and cannot become the
        # formal Fishr checkpoint or consume the post-activation patience budget.
        if args.method=="Fishr" and not fishr_selection_unlocked and fishr_active_updates>0:
            fishr_selection_unlocked=True; fishr_activation_epoch=epoch
            best_score=-1.; best_epoch=-1; bad_epochs=0
            print(f"FISHR CHECKPOINT SELECTION UNLOCKED epoch={epoch} active_updates={fishr_active_updates}",flush=True)

        if args.method!="Fishr" or fishr_selection_unlocked:
            if score>best_score+1e-5:
                best_score=score; best_epoch=epoch; bad_epochs=0
                torch.save({"model_state":model.state_dict(),"epoch":epoch,"best_val_mean_fg_dice":best_score,"config":config,
                            "fishr_selection_unlocked":fishr_selection_unlocked,"fishr_activation_epoch":fishr_activation_epoch},ckpt_dir/"best_model.pt")
            else:
                bad_epochs+=1
        else:
            bad_epochs=0
            print(f"FISHR WARMUP epoch={epoch} updates={update_count}; checkpoint ineligible before penalty activation",flush=True)

        tmp=ckpt_dir/"last_state.tmp"
        torch.save({"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"epoch":epoch,"best_score":best_score,"best_epoch":best_epoch,"bad_epochs":bad_epochs,
                    "history":history,"config":config,"update_count":update_count,"fishr_ema_state":{k:v.cpu() for k,v in fishr_ema_state.items()},"fishr_ema_updates":fishr_ema_updates,"fishr_active_updates":fishr_active_updates,
                    "fishr_selection_unlocked":fishr_selection_unlocked,"fishr_activation_epoch":fishr_activation_epoch},tmp)
        os.replace(tmp,last)
        if bad_epochs>=args.patience:
            print(f"Early stopping epoch={epoch} best={best_epoch}",flush=True); break

    if args.method=="Fishr" and (not fishr_selection_unlocked or fishr_active_updates<=0 or best_epoch<0):
        raise RuntimeError("Fishr never produced a post-anneal eligible checkpoint; refusing formal PASS")
    best=torch.load(ckpt_dir/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(best["model_state"])
    val_best=evaluate(model,val_ds,class_weights,device,args.batch_size)
    # CRITICAL: held-out RIGA test is not evaluated in Stage8B.
    cksha=hashlib.sha256((ckpt_dir/"best_model.pt").read_bytes()).hexdigest()
    summary={"status":"PASS","stage":"MedAIN_V15_Stage8R_B","run_name":run_name,"method":args.method,"target_domain":args.target_domain,"seed":args.seed,
             "best_epoch":best_epoch,"best_val_mean_fg_dice":best_score,"val_best":val_best,"epochs_completed":len(history),"total_updates":update_count,"fishr_active_updates":fishr_active_updates,
             "target_test_evaluated":False,"external_inference":False,"best_checkpoint_sha256":cksha,"total_seconds":time.time()-start,
             "identity_key":"sample_uid","identity_rows":744,
             "fishr_selection_unlocked":fishr_selection_unlocked,"fishr_activation_epoch":fishr_activation_epoch,
             "fishr_post_anneal_best_checkpoint":(args.method!="Fishr" or (fishr_selection_unlocked and best_epoch>=fishr_activation_epoch)),
             "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save_json(out_dir/"summary.json",summary); (out_dir/"PASS").write_text("PASS\n")
    print(json.dumps(summary),flush=True)

if __name__=="__main__": main()
