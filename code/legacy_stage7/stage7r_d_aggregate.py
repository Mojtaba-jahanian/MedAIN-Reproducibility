from __future__ import annotations
import csv,json,math,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HOME=Path.home(); ROOT=HOME/'MedAIN_V14_STAGE7R_D'; R=ROOT/'results'; A=ROOT/'audit'
S7C=HOME/'MedAIN_V13_STAGE7R_C'; V10=HOME/'MedAIN_V10'
DOMAINS=['REFUGE','Drishti_GS','ORIGA','PAPILA']; SEED=20260906; LAMBDA=1.0
prov=pd.read_csv(S7C/'audit/checkpoint_provenance.csv')
if len(prov)!=30: raise RuntimeError('checkpoint provenance !=30')

def folds_stratified(y,k=5,seed=SEED):
    rng=np.random.default_rng(seed); fold=np.empty(len(y),dtype=int)
    for c in sorted(np.unique(y)):
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        for j,i in enumerate(idx): fold[i]=j%k
    return fold

def metrics(y,p):
    classes=sorted(np.unique(y)); recalls=[]; f1s=[]
    for c in classes:
        tp=np.sum((y==c)&(p==c)); fn=np.sum((y==c)&(p!=c)); fp=np.sum((y!=c)&(p==c))
        rec=tp/max(tp+fn,1); pre=tp/max(tp+fp,1); f1=0.0 if pre+rec==0 else 2*pre*rec/(pre+rec)
        recalls.append(rec); f1s.append(f1)
    return float(np.mean(recalls)),float(np.mean(f1s))

def ridge_probe(X,y):
    fd=folds_stratified(y); preds=np.empty_like(y)
    for f in range(5):
        tr=fd!=f; te=fd==f; mu=X[tr].mean(0); sd=X[tr].std(0); sd[sd<1e-8]=1
        Xt=(X[tr]-mu)/sd; Xv=(X[te]-mu)/sd
        Xt=np.c_[Xt,np.ones(Xt.shape[0])]; Xv=np.c_[Xv,np.ones(Xv.shape[0])]
        Y=np.eye(4)[y[tr]]; reg=np.eye(Xt.shape[1])*LAMBDA; reg[-1,-1]=0
        W=np.linalg.solve(Xt.T@Xt+reg,Xt.T@Y); preds[te]=(Xv@W).argmax(1)
    return metrics(y,preds)

def boot_ci(delta,n=100000,seed=SEED):
    d=np.asarray(delta,float); rng=np.random.default_rng(seed); means=np.empty(n,float)
    # memory-safe chunking
    step=10000
    for s in range(0,n,step):
        e=min(s+step,n); idx=rng.integers(0,len(d),size=(e-s,len(d))); means[s:e]=d[idx].mean(1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))

def paired_summary(df,value,metric_name):
    piv=df.pivot(index=['source_fold','seed'],columns='method',values=value).dropna(); d=(piv['medain']-piv['erm']).to_numpy(float)
    try: W,p=wilcoxon(d,alternative='two-sided',zero_method='wilcox')
    except ValueError: W,p=0.0,1.0
    lo,hi=boot_ci(d)
    return {'metric':metric_name,'n_pairs':len(d),'erm_mean':float(piv['erm'].mean()),'medain_mean':float(piv['medain'].mean()),
            'mean_delta_medain_minus_erm':float(d.mean()),'median_delta':float(np.median(d)),'bootstrap95_low':lo,'bootstrap95_high':hi,
            'medain_wins':int((d>0).sum()),'erm_wins':int((d<0).sum()),'ties':int((d==0).sum()),'wilcoxon_W':float(W),'wilcoxon_p':float(p)}

# 1) Linear domain leakage probes
probe=[]
for _,r in prov.iterrows():
    name=f"{r.method}_{r.source_fold}_seed{r.seed}"; z=np.load(R/name/'features.npz')
    y=z['y'].astype(int)
    for layer in ['f1','f2']:
        ba,mf1=ridge_probe(z[layer].astype(np.float64),y)
        probe.append({'method':r.method,'source_fold':r.source_fold,'seed':int(r.seed),'layer':layer,'balanced_accuracy':ba,'macro_f1':mf1})
probe=pd.DataFrame(probe); probe.to_csv(R/'domain_leakage_probe_results.csv',index=False)
probe_stats=[]
for layer in ['f1','f2']:
    sub=probe[probe.layer==layer]
    probe_stats.append(paired_summary(sub,'balanced_accuracy',f'{layer}_domain_probe_balanced_accuracy'))
    probe_stats.append(paired_summary(sub,'macro_f1',f'{layer}_domain_probe_macro_f1'))
pd.DataFrame(probe_stats).to_csv(R/'domain_leakage_paired_statistics.csv',index=False)

# 2) Perturbation robustness: equal-weight macro over domains and perturbations per model
rob=[]
for _,r in prov.iterrows():
    name=f"{r.method}_{r.source_fold}_seed{r.seed}"; df=pd.read_csv(R/name/'perturbation_image_metrics.csv')
    # domain x perturbation means -> equal weight across domains and six perturbations
    gp=df.groupby(['external_domain','perturbation']).agg(agreement=('mean_clean_perturbed_agreement','mean'),task_delta=('task_delta_perturbed_minus_clean','mean')).reset_index()
    rob.append({'method':r.method,'source_fold':r.source_fold,'seed':int(r.seed),
                'macro_prediction_consistency':float(gp.agreement.mean()),'macro_task_delta_perturbed_minus_clean':float(gp.task_delta.mean()),
                'worst_domain_perturbation_consistency':float(gp.agreement.min()),'worst_domain_perturbation_task_delta':float(gp.task_delta.min())})
rob=pd.DataFrame(rob); rob.to_csv(R/'perturbation_robustness_model_summary.csv',index=False)
rob_stats=[paired_summary(rob,'macro_prediction_consistency','photometric_prediction_consistency'),
           paired_summary(rob,'macro_task_delta_perturbed_minus_clean','photometric_task_delta'),
           paired_summary(rob,'worst_domain_perturbation_consistency','worst_domain_perturbation_consistency'),
           paired_summary(rob,'worst_domain_perturbation_task_delta','worst_domain_perturbation_task_delta')]
pd.DataFrame(rob_stats).to_csv(R/'perturbation_robustness_paired_statistics.csv',index=False)

# 3) MedAIN on/off task preservation + projection diagnostics
onoff=[]; proj=[]
for _,r in prov[prov.method=='medain'].iterrows():
    name=f"medain_{r.source_fold}_seed{r.seed}"; df=pd.read_csv(R/name/'medain_on_off_image_metrics.csv')
    # equal domain macro
    gp=df.groupby('external_domain').agg(agreement=('mean_prediction_agreement','mean'),delta=('task_delta_on_minus_off','mean'),on=('medain_on_task_mean_dice','mean'),off=('medain_off_task_mean_dice','mean')).reset_index()
    onoff.append({'source_fold':r.source_fold,'seed':int(r.seed),'macro_prediction_agreement_on_vs_off':float(gp.agreement.mean()),
                  'macro_task_delta_on_minus_off':float(gp.delta.mean()),'macro_task_dice_on':float(gp.on.mean()),'macro_task_dice_off':float(gp.off.mean())})
    s=json.loads((R/name/'summary.json').read_text()); proj.append({'source_fold':r.source_fold,'seed':int(r.seed),
        'stage1_projection_relative_energy':s['projection_relative_energy_mean']['stage1'],'stage2_projection_relative_energy':s['projection_relative_energy_mean']['stage2'],
        'stage1_orthogonality_error':s['basis_orthogonality_error']['stage1'],'stage2_orthogonality_error':s['basis_orthogonality_error']['stage2']})
onoff=pd.DataFrame(onoff); onoff.to_csv(R/'medain_on_off_task_preservation.csv',index=False)
proj=pd.DataFrame(proj); proj.to_csv(R/'medain_projection_diagnostics_external.csv',index=False)

# 4) Basis-history diagnostics from selected config (last basis update)
basis=[]
for _,r in prov[prov.method=='medain'].iterrows():
    seed=int(r.seed); d=r.source_fold
    base=V10/'results/v13_stage6r_b'/f'v13r_medain_{d}_seed{seed}_r2_a0p5'/'basis_history.jsonl'
    if not base.exists(): raise RuntimeError(f'Missing corrected basis history: {base}')
    lines=[json.loads(x) for x in base.read_text().splitlines() if x.strip()]; last=lines[-1]
    for stage in ['stage1','stage2']:
        x=last[stage]; basis.append({'source_fold':d,'seed':seed,'stage':stage,'epoch':last['epoch'],'acquisition_capture_ratio':x['acquisition_capture_ratio'],
            'task_leakage_ratio':x['task_leakage_ratio'],'orthogonality_error':x['orthogonality_error']})
pd.DataFrame(basis).to_csv(R/'training_basis_diagnostics_selected_config.csv',index=False)

# 5) Inference efficiency from confirmatory Stage7C summaries
eff=[]
for _,r in prov.iterrows():
    name=f"{r.method}_{r.source_fold}_seed{r.seed}"; s=json.loads((S7C/'results'/name/'summary.json').read_text())
    eff.append({'method':r.method,'source_fold':r.source_fold,'seed':int(r.seed),'inference_seconds_769':s['inference_seconds'],'milliseconds_per_image':1000*s['inference_seconds']/769})
eff=pd.DataFrame(eff); eff.to_csv(R/'inference_efficiency.csv',index=False)
eff_stats=[paired_summary(eff,'milliseconds_per_image','inference_ms_per_image')]
pd.DataFrame(eff_stats).to_csv(R/'inference_efficiency_paired_statistics.csv',index=False)

# 6) Training efficiency per epoch from matched RIGA runs
train=[]
for _,r in prov.iterrows():
    method=r.method; d=r.source_fold; seed=int(r.seed)
    base=V10/'results/v13_stage6r_b'/f'v13r_{method}_{d}_seed{seed}_r2_a0p5'
    if not (base/'summary.json').exists(): raise RuntimeError(f'Missing corrected training summary: {base}')
    s=json.loads((base/'summary.json').read_text()); h=pd.read_csv(base/'training_history.csv')
    train.append({'method':method,'source_fold':d,'seed':seed,'epochs_executed':len(h),'training_seconds':s['training_seconds'],
                  'mean_seconds_per_epoch':float(h.seconds.mean()),'median_seconds_per_epoch':float(h.seconds.median())})
train=pd.DataFrame(train); train.to_csv(R/'training_efficiency.csv',index=False)
train_stats=[paired_summary(train,'mean_seconds_per_epoch','training_seconds_per_epoch')]
pd.DataFrame(train_stats).to_csv(R/'training_efficiency_paired_statistics.csv',index=False)

# 7) Analytical overhead and model size
analytical={'medain_rank':2,'medain_alpha':0.5,'projection_buffers_floats':384,'projection_buffer_bytes_float32':1536,
            'learnable_parameters_added':0,'stage1_projection_MAC_equivalents':2*64*2*64*64,
            'stage2_projection_MAC_equivalents':2*128*2*32*32}
analytical['total_projection_MAC_equivalents_per_image']=analytical['stage1_projection_MAC_equivalents']+analytical['stage2_projection_MAC_equivalents']
(A/'analytical_efficiency_overhead.json').write_text(json.dumps(analytical,indent=2))

# 8) Combined summary + audit
pstat=pd.DataFrame(probe_stats); rstat=pd.DataFrame(rob_stats); estat=pd.DataFrame(eff_stats); tstat=pd.DataFrame(train_stats)
summary={
 'stage':'MedAIN_V14_Stage7R_D','analysis_type':'identity-corrected secondary mechanism and efficiency revalidation after Stage7R-C; no MedAIN retuning',
 'domain_leakage':probe_stats,'perturbation_robustness':rob_stats,
 'medain_task_preservation':{'n_models':len(onoff),'prediction_agreement_mean':float(onoff.macro_prediction_agreement_on_vs_off.mean()),
      'task_delta_on_minus_off_mean':float(onoff.macro_task_delta_on_minus_off.mean()),'task_delta_on_minus_off_std':float(onoff.macro_task_delta_on_minus_off.std(ddof=1))},
 'projection_diagnostics':{'stage1_relative_energy_mean':float(proj.stage1_projection_relative_energy.mean()),'stage2_relative_energy_mean':float(proj.stage2_projection_relative_energy.mean()),
      'stage1_orthogonality_max':float(proj.stage1_orthogonality_error.max()),'stage2_orthogonality_max':float(proj.stage2_orthogonality_error.max())},
 'inference_efficiency':eff_stats[0],'training_efficiency':train_stats[0],'analytical_overhead':analytical,
 'medain_configuration_unchanged':{'rank':2,'alpha':0.5,'beta':0.001},'status':'PASS'}
(A/'stage7d_summary.json').write_text(json.dumps(summary,indent=2))
audit={'stage':'MedAIN_V14_Stage7R_D','models_analyzed':30,'matched_pairs':15,'external_images':769,'fixed_domain_probe':{'type':'5-fold stratified ridge probe','lambda':1.0,'seed':SEED},
       'fixed_perturbations':['brightness_085','brightness_115','contrast_085','contrast_115','gamma_085','gamma_115'],
       'medain_retuned':False,'external_training_of_main_model':False,'errors':[],'status':'PASS'}
(A/'stage7d_final_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(summary,indent=2))
