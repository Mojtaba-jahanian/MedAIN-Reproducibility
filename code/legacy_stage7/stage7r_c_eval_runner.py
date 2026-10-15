from __future__ import annotations

import argparse, csv, hashlib, json, math, os, time
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

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

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
    def forward(self, z):
        if not bool(self.initialized):
            return z
        n,c,h,w = z.shape
        flat = z.reshape(n,c,-1)
        coeff = torch.einsum('cr,ncs->nrs', self.U, flat)
        proj = torch.einsum('cr,nrs->ncs', self.U, coeff).reshape_as(z)
        return z - self.alpha * proj

class ResNet18UNet(nn.Module):
    def __init__(self, method='erm', rank=2, alpha=0.5):
        super().__init__()
        # weights=None avoids any network access; checkpoint supplies all weights.
        backbone = resnet18(weights=None)
        self.method = method
        self.conv1=backbone.conv1; self.bn1=backbone.bn1; self.relu=backbone.relu; self.maxpool=backbone.maxpool
        self.layer1=backbone.layer1; self.layer2=backbone.layer2; self.layer3=backbone.layer3; self.layer4=backbone.layer4
        self.med1=MedAINLayer(64,rank,alpha); self.med2=MedAINLayer(128,rank,alpha)
        self.up3=UpBlock(512,256,256); self.up2=UpBlock(256,128,128); self.up1=UpBlock(128,64,64); self.up0=UpBlock(64,64,64)
        self.final_conv=ConvBlock(64,32); self.head=nn.Conv2d(32,N_CLASSES,1)
        self.register_buffer('img_mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('img_std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def normalize(self,x):
        return (x-self.img_mean)/self.img_std
    def forward(self,x,apply_medain=True):
        input_hw=x.shape[-2:]
        x=self.normalize(x)
        s0=self.relu(self.bn1(self.conv1(x))); z=self.maxpool(s0)
        f1=self.layer1(z)
        if self.method=='medain' and apply_medain: f1=self.med1(f1)
        f2=self.layer2(f1)
        if self.method=='medain' and apply_medain: f2=self.med2(f2)
        f3=self.layer3(f2); f4=self.layer4(f3)
        d3=self.up3(f4,f3); d2=self.up2(d3,f2); d1=self.up1(d2,f1); d0=self.up0(d1,s0)
        d=F.interpolate(d0,scale_factor=2,mode='bilinear',align_corners=False)
        logits=self.head(self.final_conv(d))
        if logits.shape[-2:] != input_hw:
            logits=F.interpolate(logits,size=input_hw,mode='bilinear',align_corners=False)
        return logits

class ExternalDataset(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r=self.rows[i]
        with Image.open(r['cached_image256']) as im:
            x=TF.to_tensor(im.convert('RGB'))
        return x, i

def binary_metrics(pred, gt):
    pred=np.asarray(pred,dtype=bool); gt=np.asarray(gt,dtype=bool)
    inter=int(np.logical_and(pred,gt).sum()); ps=int(pred.sum()); gs=int(gt.sum())
    den=ps+gs; union=ps+gs-inter
    dice=1.0 if den==0 else 2.0*inter/den
    iou=1.0 if union==0 else inter/union
    return float(dice), float(iou)

def load_mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'),dtype=np.uint8)

def mean_dict(rows, keys):
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',choices=['erm','medain'],required=True)
    ap.add_argument('--source-fold',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out-dir',required=True)
    ap.add_argument('--qualitative-lock',required=True)
    ap.add_argument('--qualitative-dir',required=True)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args()

    torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device=torch.device('cpu')

    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary_path=out/'summary.json'
    if summary_path.exists():
        try:
            old=json.loads(summary_path.read_text())
            if old.get('status')=='PASS':
                print('ALREADY PASS:', out.name); return
        except Exception: pass

    with open(args.manifest,newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=769: raise RuntimeError(f'manifest rows={len(rows)}')

    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    ckcfg=ckpt.get('config',{}) or {}
    if ckcfg.get('method') not in (None,args.method):
        raise RuntimeError(f'Checkpoint method mismatch: {ckcfg.get("method")} != {args.method}')
    if ckcfg.get('target_domain') not in (None,args.source_fold):
        raise RuntimeError(f'Checkpoint fold mismatch: {ckcfg.get("target_domain")} != {args.source_fold}')
    if ckcfg.get('seed') not in (None,args.seed):
        raise RuntimeError(f'Checkpoint seed mismatch: {ckcfg.get("seed")} != {args.seed}')
    ck_rank=int(ckcfg.get('rank', 2 if args.method=='medain' else 4))
    ck_alpha=float(ckcfg.get('alpha', 0.5 if args.method=='medain' else 0.75))
    if args.method=='medain':
        if ck_rank != 2 or abs(ck_alpha-0.5) > 1e-12:
            raise RuntimeError(f'Unexpected MedAIN checkpoint config rank={ck_rank} alpha={ck_alpha}')
    model=ResNet18UNet(method=args.method,rank=ck_rank,alpha=ck_alpha).to(device)
    missing,unexpected=model.load_state_dict(ckpt['model_state'],strict=False)
    if missing or unexpected:
        raise RuntimeError(f'state mismatch missing={missing} unexpected={unexpected}')
    if args.method=='medain':
        if not bool(model.med1.initialized) or not bool(model.med2.initialized):
            raise RuntimeError('MedAIN basis buffers are not initialized')
        if float(model.med1.U.abs().sum())<=0 or float(model.med2.U.abs().sum())<=0:
            raise RuntimeError('MedAIN basis is zero')
    model.eval()

    ds=ExternalDataset(rows)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=False)

    q=json.loads(Path(args.qualitative_lock).read_text())
    qids=set(x['sample_id'] for x in q['samples'])
    save_qual=(args.source_fold=='BinRushed' and args.seed==1103)
    qdir=Path(args.qualitative_dir)/args.method
    if save_qual: qdir.mkdir(parents=True,exist_ok=True)

    annotation_rows=[]; image_rows=[]
    metric_keys=['od_dice','oc_dice','mean_dice','od_iou','oc_iou','mean_iou']
    t0=time.time(); seen=0

    with torch.inference_mode():
        for xb, idxb in loader:
            logits=model(xb.to(device))
            pred=logits.argmax(1).cpu().numpy().astype(np.uint8)
            for j, idx in enumerate(idxb.tolist()):
                r=rows[idx]; p=pred[j]
                mask_paths=[('gt1',r['cached_mask256_1'])]
                p2=(r.get('cached_mask256_2') or '').strip()
                if p2 and p2.lower()!='nan': mask_paths.append(('gt2',p2))
                ann=[]
                for ann_id, mp in mask_paths:
                    gt=load_mask(mp)
                    if gt.shape != p.shape: raise RuntimeError(f'shape mismatch {r["sample_id"]}')
                    od_d,od_i=binary_metrics(p>0,gt>0)
                    oc_d,oc_i=binary_metrics(p==1,gt==1)
                    rr={
                        'method':args.method,'source_fold':args.source_fold,'seed':args.seed,
                        'external_domain':r['domain'],'sample_id':r['sample_id'],'annotation_id':ann_id,
                        'od_dice':od_d,'oc_dice':oc_d,'mean_dice':(od_d+oc_d)/2,
                        'od_iou':od_i,'oc_iou':oc_i,'mean_iou':(od_i+oc_i)/2,
                    }
                    annotation_rows.append(rr); ann.append(rr)
                ir={
                    'method':args.method,'source_fold':args.source_fold,'seed':args.seed,
                    'external_domain':r['domain'],'sample_id':r['sample_id'],'n_annotations':len(ann),
                }
                ir.update(mean_dict(ann,metric_keys)); image_rows.append(ir); seen+=1
                if save_qual and r['sample_id'] in qids:
                    safe=r['sample_id'].replace('/','__')
                    Image.fromarray(p,mode='L').save(qdir/f'{safe}__pred.png')

    if seen!=769 or len(image_rows)!=769:
        raise RuntimeError(f'image count seen={seen}')
    if len(annotation_rows)!=1257:
        raise RuntimeError(f'annotation comparisons={len(annotation_rows)}')

    with (out/'annotation_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(annotation_rows[0].keys())); w.writeheader(); w.writerows(annotation_rows)
    with (out/'image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(image_rows[0].keys())); w.writeheader(); w.writerows(image_rows)

    domain_summary={}
    for domain in ['REFUGE','Drishti_GS','ORIGA','PAPILA']:
        sub=[r for r in image_rows if r['external_domain']==domain]
        domain_summary[domain]={'n_images':len(sub),**mean_dict(sub,metric_keys)}
    overall=mean_dict(image_rows,metric_keys)
    macro={k: float(np.mean([domain_summary[d][k] for d in ['REFUGE','Drishti_GS','ORIGA','PAPILA']])) for k in metric_keys}

    summary={
        'status':'PASS','method':args.method,'source_fold':args.source_fold,'seed':args.seed,
        'checkpoint':str(args.checkpoint),'checkpoint_epoch':int(ckpt.get('epoch',-1)),
        'external_images':769,'annotation_comparisons':1257,
        'domain_summary':domain_summary,'overall_domain_macro':macro,'overall_image_weighted':overall,
        'inference_seconds':time.time()-t0,
        'external_tuning_performed':False,
    }
    summary_path.write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
