from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

N_CLASSES=3
METHODS=['ERM_stage8','StrongAug','IBN','MixStyle','CORAL','Fishr']
DOMAINS=['REFUGE','Drishti_GS','ORIGA','PAPILA']

class ConvBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__(); self.net=nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.BatchNorm2d(cout),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UpBlock(nn.Module):
    def __init__(self,cin,skip,cout): super().__init__(); self.conv=ConvBlock(cin+skip,cout)
    def forward(self,x,s):
        x=F.interpolate(x,size=s.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x,s],dim=1))
class IBN(nn.Module):
    def __init__(self,bn:nn.BatchNorm2d):
        super().__init__(); c=bn.num_features; self.half=c//2
        self.IN=nn.InstanceNorm2d(self.half,affine=True,track_running_stats=False)
        self.BN=nn.BatchNorm2d(c-self.half,eps=bn.eps,momentum=bn.momentum,affine=True,track_running_stats=True)
    def forward(self,x):
        a,b=torch.split(x,[self.half,x.shape[1]-self.half],dim=1)
        return torch.cat([self.IN(a.contiguous()),self.BN(b.contiguous())],dim=1)
class MixStyle(nn.Module):
    def __init__(self,p=0.5,alpha=0.1,eps=1e-6): super().__init__(); self.p=p; self.alpha=alpha; self.eps=eps
    def forward(self,x):
        if not self.training: return x
        return x
class Model(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        b=resnet18(weights=None)
        if method=='IBN':
            for layer in (b.layer1,b.layer2,b.layer3):
                for block in layer: block.bn1=IBN(block.bn1)
        self.conv1=b.conv1; self.bn1=b.bn1; self.relu=b.relu; self.maxpool=b.maxpool
        self.layer1=b.layer1; self.layer2=b.layer2; self.layer3=b.layer3; self.layer4=b.layer4
        self.mixstyle=MixStyle(0.5,0.1) if method=='MixStyle' else None
        self.up3=UpBlock(512,256,256); self.up2=UpBlock(256,128,128); self.up1=UpBlock(128,64,64); self.up0=UpBlock(64,64,64)
        self.final_conv=ConvBlock(64,32); self.head=nn.Conv2d(32,N_CLASSES,1)
        self.register_buffer('img_mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('img_std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(self,x):
        hw=x.shape[-2:]; x=(x-self.img_mean)/self.img_std
        s0=self.relu(self.bn1(self.conv1(x))); z=self.maxpool(s0)
        f1=self.layer1(z)
        if self.mixstyle is not None: f1=self.mixstyle(f1)
        f2=self.layer2(f1)
        if self.mixstyle is not None: f2=self.mixstyle(f2)
        f3=self.layer3(f2); f4=self.layer4(f3)
        d3=self.up3(f4,f3); d2=self.up2(d3,f2); d1=self.up1(d2,f1); d0=self.up0(d1,s0)
        d=F.interpolate(d0,scale_factor=2,mode='bilinear',align_corners=False)
        y=self.head(self.final_conv(d))
        if y.shape[-2:]!=hw: y=F.interpolate(y,size=hw,mode='bilinear',align_corners=False)
        return y
class ExternalDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['cached_image256']) as im: x=TF.to_tensor(im.convert('RGB'))
        return x,i

def binary_metrics(pred,gt):
    pred=np.asarray(pred,dtype=bool); gt=np.asarray(gt,dtype=bool)
    inter=int(np.logical_and(pred,gt).sum()); ps=int(pred.sum()); gs=int(gt.sum()); den=ps+gs; union=ps+gs-inter
    return (1.0 if den==0 else 2.0*inter/den),(1.0 if union==0 else inter/union)
def load_mask(p):
    with Image.open(p) as im: return np.asarray(im.convert('L'),dtype=np.uint8)
def mean_dict(rows,keys): return {k:float(np.mean([float(r[k]) for r in rows])) for k in keys}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=METHODS,required=True); ap.add_argument('--source-fold',required=True)
    ap.add_argument('--seed',type=int,required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--expected-checkpoint-sha',required=True)
    ap.add_argument('--manifest',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--qualitative-lock',required=True)
    ap.add_argument('--qualitative-dir',required=True); ap.add_argument('--threads',type=int,default=4); ap.add_argument('--batch-size',type=int,default=8)
    args=ap.parse_args(); torch.set_num_threads(args.threads)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device=torch.device('cpu'); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); sp=out/'summary.json'
    runner_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); cksha=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    if cksha!=args.expected_checkpoint_sha: raise RuntimeError('checkpoint SHA mismatch')
    if sp.exists():
        try:
            old=json.loads(sp.read_text())
            if old.get('status')=='PASS' and old.get('runner_sha256')==runner_sha and old.get('checkpoint_sha256')==cksha:
                print('ALREADY PASS '+out.name); return
        except Exception: pass
    with open(args.manifest,newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    if len(rows)!=769: raise RuntimeError(f'external rows={len(rows)}')
    ck=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=ck.get('config',{}) or {}
    if cfg.get('method')!=args.method or cfg.get('target_domain')!=args.source_fold or int(cfg.get('seed'))!=args.seed: raise RuntimeError('checkpoint config mismatch')
    if cfg.get('identity_key')!='sample_uid' or cfg.get('identity_rows')!=744: raise RuntimeError('checkpoint identity provenance mismatch')
    if cfg.get('target_test_evaluated') is not False or cfg.get('external_inference') is not False: raise RuntimeError('checkpoint training provenance contaminated')
    model=Model(args.method).to(device); model.load_state_dict(ck['model_state'],strict=True); model.eval()
    ds=ExternalDataset(rows); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=False)
    q=json.loads(Path(args.qualitative_lock).read_text()); qids={x['sample_id'] for x in q['samples']}; saveq=(args.source_fold=='BinRushed' and args.seed==1103)
    qdir=Path(args.qualitative_dir)/args.method
    if saveq: qdir.mkdir(parents=True,exist_ok=True)
    metric_keys=['od_dice','oc_dice','mean_dice','od_iou','oc_iou','mean_iou']; annrows=[]; imrows=[]; seen=0; t0=time.time()
    with torch.inference_mode():
        for xb,idxb in loader:
            pred=model(xb.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
            for j,idx in enumerate(idxb.tolist()):
                r=rows[idx]; p=pred[j]; masks=[('gt1',r['cached_mask256_1'])]; p2=(r.get('cached_mask256_2') or '').strip()
                if p2 and p2.lower()!='nan': masks.append(('gt2',p2))
                aa=[]
                for aid,mp in masks:
                    gt=load_mask(mp)
                    if gt.shape!=p.shape: raise RuntimeError('shape mismatch '+r['sample_id'])
                    odd,odi=binary_metrics(p>0,gt>0); ocd,oci=binary_metrics(p==1,gt==1)
                    rr={'method':args.method,'source_fold':args.source_fold,'seed':args.seed,'external_domain':r['domain'],'sample_id':r['sample_id'],'annotation_id':aid,
                        'od_dice':odd,'oc_dice':ocd,'mean_dice':(odd+ocd)/2,'od_iou':odi,'oc_iou':oci,'mean_iou':(odi+oci)/2}
                    annrows.append(rr); aa.append(rr)
                ir={'method':args.method,'source_fold':args.source_fold,'seed':args.seed,'external_domain':r['domain'],'sample_id':r['sample_id'],'n_annotations':len(aa)}
                ir.update(mean_dict(aa,metric_keys)); imrows.append(ir); seen+=1
                if saveq and r['sample_id'] in qids:
                    safe=r['sample_id'].replace('/','__'); Image.fromarray(p,mode='L').save(qdir/f'{safe}__pred.png')
    if seen!=769 or len(imrows)!=769 or len(annrows)!=1257: raise RuntimeError(f'count mismatch images={seen} annotations={len(annrows)}')
    with (out/'annotation_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(annrows[0])); w.writeheader(); w.writerows(annrows)
    with (out/'image_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(imrows[0])); w.writeheader(); w.writerows(imrows)
    domsum={}
    for d in DOMAINS:
        sub=[r for r in imrows if r['external_domain']==d]; domsum[d]={'n_images':len(sub),**mean_dict(sub,metric_keys)}
    overall=mean_dict(imrows,metric_keys); macro={k:float(np.mean([domsum[d][k] for d in DOMAINS])) for k in metric_keys}
    summary={'status':'PASS','stage':'MedAIN_V16_Stage8R_C','method':args.method,'source_fold':args.source_fold,'seed':args.seed,
             'checkpoint_epoch':int(ck.get('epoch',-1)),'checkpoint_sha256':cksha,'external_images':769,'annotation_comparisons':1257,
             'domain_summary':domsum,'overall_domain_macro':macro,'overall_image_weighted':overall,'inference_seconds':time.time()-t0,
             'external_training':False,'external_tuning_performed':False,'checkpoint_reselection':False,'threshold_selection':False,
             'one_shot_external_inference':True,'runner_sha256':runner_sha}
    sp.write_text(json.dumps(summary,indent=2)+'\n'); print('PASS '+out.name)
if __name__=='__main__': main()
