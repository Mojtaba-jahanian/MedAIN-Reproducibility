from __future__ import annotations

import argparse, csv, json, math, time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

N_CLASSES = 3
DOMAINS = ['REFUGE','Drishti_GS','ORIGA','PAPILA']
PERTURBATIONS = [
    ('brightness_085', 'brightness', 0.85),
    ('brightness_115', 'brightness', 1.15),
    ('contrast_085', 'contrast', 0.85),
    ('contrast_115', 'contrast', 1.15),
    ('gamma_085', 'gamma', 0.85),
    ('gamma_115', 'gamma', 1.15),
]

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__(); self.conv = ConvBlock(cin + skip, cout)
    def forward(self, x, s):
        x = F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, s], dim=1))

class MedAINLayer(nn.Module):
    def __init__(self, channels, rank=2, alpha=0.5):
        super().__init__(); self.channels=channels; self.rank=rank; self.alpha=float(alpha)
        self.register_buffer('U', torch.zeros(channels, rank))
        self.register_buffer('initialized', torch.tensor(False))
    def project(self, z):
        n,c,h,w = z.shape
        flat = z.reshape(n,c,-1)
        coeff = torch.einsum('cr,ncs->nrs', self.U, flat)
        proj = torch.einsum('cr,nrs->ncs', self.U, coeff).reshape_as(z)
        return proj
    def forward(self, z):
        if not bool(self.initialized): return z
        return z - self.alpha * self.project(z)

class ResNet18UNet(nn.Module):
    def __init__(self, method='erm', rank=2, alpha=0.5):
        super().__init__(); backbone=resnet18(weights=None); self.method=method
        self.conv1=backbone.conv1; self.bn1=backbone.bn1; self.relu=backbone.relu; self.maxpool=backbone.maxpool
        self.layer1=backbone.layer1; self.layer2=backbone.layer2; self.layer3=backbone.layer3; self.layer4=backbone.layer4
        self.med1=MedAINLayer(64,rank,alpha); self.med2=MedAINLayer(128,rank,alpha)
        self.up3=UpBlock(512,256,256); self.up2=UpBlock(256,128,128); self.up1=UpBlock(128,64,64); self.up0=UpBlock(64,64,64)
        self.final_conv=ConvBlock(64,32); self.head=nn.Conv2d(32,N_CLASSES,1)
        self.register_buffer('img_mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('img_std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def normalize(self,x): return (x-self.img_mean)/self.img_std
    def forward_mechanism(self,x,apply_medain=True):
        input_hw=x.shape[-2:]; x=self.normalize(x)
        s0=self.relu(self.bn1(self.conv1(x))); z=self.maxpool(s0)
        f1_pre=self.layer1(z); f1=f1_pre; p1=None
        if self.method=='medain' and apply_medain:
            p1=self.med1.project(f1_pre); f1=f1_pre-self.med1.alpha*p1
        f2_pre=self.layer2(f1); f2=f2_pre; p2=None
        if self.method=='medain' and apply_medain:
            p2=self.med2.project(f2_pre); f2=f2_pre-self.med2.alpha*p2
        f3=self.layer3(f2); f4=self.layer4(f3)
        d3=self.up3(f4,f3); d2=self.up2(d3,f2); d1=self.up1(d2,f1); d0=self.up0(d1,s0)
        d=F.interpolate(d0,scale_factor=2,mode='bilinear',align_corners=False)
        logits=self.head(self.final_conv(d))
        if logits.shape[-2:] != input_hw: logits=F.interpolate(logits,size=input_hw,mode='bilinear',align_corners=False)
        gap1=F.adaptive_avg_pool2d(f1,1).flatten(1)
        gap2=F.adaptive_avg_pool2d(f2,1).flatten(1)
        rel1=rel2=None
        if p1 is not None:
            rel1=(self.med1.alpha*p1).flatten(1).norm(dim=1)/(f1_pre.flatten(1).norm(dim=1)+1e-12)
        if p2 is not None:
            rel2=(self.med2.alpha*p2).flatten(1).norm(dim=1)/(f2_pre.flatten(1).norm(dim=1)+1e-12)
        return logits,gap1,gap2,rel1,rel2

class ExternalDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['cached_image256']) as im: x=TF.to_tensor(im.convert('RGB'))
        return x,i

def load_mask(path):
    with Image.open(path) as im: return np.asarray(im.convert('L'),dtype=np.uint8)

def binary_dice(a,b):
    a=np.asarray(a,dtype=bool); b=np.asarray(b,dtype=bool); den=int(a.sum()+b.sum())
    return 1.0 if den==0 else float(2*np.logical_and(a,b).sum()/den)

def task_mean_dice(pred,row):
    vals=[]
    paths=[row['cached_mask256_1']]
    p2=(row.get('cached_mask256_2') or '').strip()
    if p2 and p2.lower()!='nan': paths.append(p2)
    for mp in paths:
        gt=load_mask(mp)
        od=binary_dice(pred>0,gt>0); oc=binary_dice(pred==1,gt==1); vals.append((od+oc)/2)
    return float(np.mean(vals))

def perturb(x,kind,value):
    if kind=='brightness': return torch.clamp(x*value,0,1)
    if kind=='contrast':
        mean=x.mean(dim=(-2,-1),keepdim=True); return torch.clamp((x-mean)*value+mean,0,1)
    if kind=='gamma': return torch.clamp(x,0,1).pow(value)
    raise ValueError(kind)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['erm','medain'],required=True); ap.add_argument('--source-fold',required=True)
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args(); torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device=torch.device('cpu'); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); sp=out/'summary.json'
    if sp.exists():
        try:
            if json.loads(sp.read_text()).get('status')=='PASS': print('ALREADY PASS:',out.name); return
        except Exception: pass
    with open(args.manifest,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=769: raise RuntimeError(f'manifest rows={len(rows)}')
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ckpt.get('config',{}) or {}
    rank=int(cfg.get('rank',2 if args.method=='medain' else 4)); alpha=float(cfg.get('alpha',0.5 if args.method=='medain' else 0.75))
    if args.method=='medain' and (rank!=2 or abs(alpha-0.5)>1e-12): raise RuntimeError(f'Unexpected MedAIN config r={rank} a={alpha}')
    model=ResNet18UNet(args.method,rank,alpha).to(device); missing,unexpected=model.load_state_dict(ckpt['model_state'],strict=False)
    if missing or unexpected: raise RuntimeError(f'state mismatch missing={missing} unexpected={unexpected}')
    if args.method=='medain' and (not bool(model.med1.initialized) or not bool(model.med2.initialized)): raise RuntimeError('basis not initialized')
    model.eval(); ds=ExternalDataset(rows); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    domain_to_int={d:i for i,d in enumerate(DOMAINS)}; f1s=[]; f2s=[]; labels=[]; sample_ids=[]; proj1=[]; proj2=[]
    clean_pred={}; clean_task={}; off_rows=[]; pert_rows=[]; t0=time.time()
    with torch.inference_mode():
        for xb,idxb in loader:
            xb=xb.to(device); logits,g1,g2,r1,r2=model.forward_mechanism(xb,apply_medain=True); pred=logits.argmax(1).cpu().numpy().astype(np.uint8)
            f1s.append(g1.cpu().numpy().astype(np.float32)); f2s.append(g2.cpu().numpy().astype(np.float32))
            if r1 is not None: proj1.extend(r1.cpu().numpy().tolist())
            if r2 is not None: proj2.extend(r2.cpu().numpy().tolist())
            pred_off=None
            if args.method=='medain':
                logits_off,_,_,_,_=model.forward_mechanism(xb,apply_medain=False); pred_off=logits_off.argmax(1).cpu().numpy().astype(np.uint8)
            for j,idx in enumerate(idxb.tolist()):
                r=rows[idx]; sid=r['sample_id']; labels.append(domain_to_int[r['domain']]); sample_ids.append(sid); clean_pred[idx]=pred[j]
                ctask=task_mean_dice(pred[j],r); clean_task[idx]=ctask
                if pred_off is not None:
                    off_task=task_mean_dice(pred_off[j],r)
                    off_rows.append({'external_domain':r['domain'],'sample_id':sid,
                        'od_prediction_agreement':binary_dice(pred[j]>0,pred_off[j]>0),
                        'oc_prediction_agreement':binary_dice(pred[j]==1,pred_off[j]==1),
                        'mean_prediction_agreement':(binary_dice(pred[j]>0,pred_off[j]>0)+binary_dice(pred[j]==1,pred_off[j]==1))/2,
                        'medain_on_task_mean_dice':ctask,'medain_off_task_mean_dice':off_task,'task_delta_on_minus_off':ctask-off_task})
        # fixed symmetric photometric perturbation suite
        for pname,pkind,pval in PERTURBATIONS:
            for xb,idxb in loader:
                xb=perturb(xb.to(device),pkind,pval); logits,_,_,_,_=model.forward_mechanism(xb,apply_medain=True); pp=logits.argmax(1).cpu().numpy().astype(np.uint8)
                for j,idx in enumerate(idxb.tolist()):
                    r=rows[idx]; cp=clean_pred[idx]; pd=pp[j]
                    odc=binary_dice(cp>0,pd>0); occ=binary_dice(cp==1,pd==1)
                    pert_rows.append({'external_domain':r['domain'],'sample_id':r['sample_id'],'perturbation':pname,
                        'od_clean_perturbed_agreement':odc,'oc_clean_perturbed_agreement':occ,'mean_clean_perturbed_agreement':(odc+occ)/2,
                        'clean_task_mean_dice':clean_task[idx],'perturbed_task_mean_dice':task_mean_dice(pd,r),
                        'task_delta_perturbed_minus_clean':task_mean_dice(pd,r)-clean_task[idx]})
    elapsed=time.time()-t0
    f1=np.concatenate(f1s,axis=0); f2=np.concatenate(f2s,axis=0); y=np.asarray(labels,dtype=np.int64)
    if f1.shape!=(769,64) or f2.shape!=(769,128): raise RuntimeError(f'feature shapes {f1.shape} {f2.shape}')
    np.savez_compressed(out/'features.npz',f1=f1,f2=f2,y=y,sample_ids=np.asarray(sample_ids,dtype='U64'))
    if off_rows:
        with (out/'medain_on_off_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(off_rows[0].keys())); w.writeheader(); w.writerows(off_rows)
    with (out/'perturbation_image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(pert_rows[0].keys())); w.writeheader(); w.writerows(pert_rows)
    ortho={}
    if args.method=='medain':
        for name,layer in [('stage1',model.med1),('stage2',model.med2)]:
            U=layer.U.detach().cpu().double(); I=torch.eye(U.shape[1],dtype=torch.double); ortho[name]=float(torch.linalg.norm(U.T@U-I).item())
    summary={'status':'PASS','method':args.method,'source_fold':args.source_fold,'seed':args.seed,'checkpoint':args.checkpoint,
        'external_images':769,'feature_dims':{'f1':64,'f2':128},'perturbations':[x[0] for x in PERTURBATIONS],
        'mechanism_seconds':elapsed,'projection_relative_energy_mean':{'stage1':float(np.mean(proj1)) if proj1 else None,'stage2':float(np.mean(proj2)) if proj2 else None},
        'basis_orthogonality_error':ortho,'medain_on_off_evaluated':bool(off_rows)}
    sp.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
