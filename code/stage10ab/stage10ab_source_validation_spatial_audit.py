#!/usr/bin/env python3
"""MedAIN Stage 10-AB: source-validation-only spatial false-positive audit.
Frozen analysis threshold 32 px, sensitivity thresholds 16/48. No training/test.
Exact Stage 10-V checkpoint and mechanism provenance is required for all 10 runs.
"""
from pathlib import Path
import ast, csv, hashlib, importlib.util, json, time, traceback, zipfile, os
import numpy as np
import scipy
import scipy.linalg
from scipy import ndimage
import torch
from torch.utils.data import DataLoader
from torch.serialization import safe_globals
from torch.torch_version import TorchVersion

ROOT = Path(os.environ.get('MEDAIN_ROOT', str(Path.home() / 'MedAIN_Stage9B_Reproduction')))
AUDIT = ROOT/'stage9b/erm_final_audit'
OUT = AUDIT/'stage10ab_spatial_fp'
OUT.mkdir(parents=True, exist_ok=True)
TRAINER = ROOT/'src/stage9b_anchor_train.py'
EVALUATOR = ROOT/'artifacts/stage9a_original/extracted/stage9a_mechanistic_eval.py'
DOMAINS = ('BinRushed','Magrabia','MESSIDOR_Base1','MESSIDOR_Base2','MESSIDOR_Base3')
SEEDS = (2207,3301)
METHODS = ('OFF','TP','AO','RAND')
THRESHOLDS = (16,32,48)
PRIMARY_DISTANCE = 32
EIGHT = np.ones((3,3),dtype=np.uint8)
VERSION = 'Stage10AB_v1_sourceval_preanalysis_20261015'
torch.set_num_threads(4)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def uid_hash(rows): return hashlib.sha256(('\n'.join(r['sample_uid'] for r in rows)+'\n').encode()).hexdigest()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

print('===== MEDAIN 10-AB: SPATIAL FALSE-POSITIVE AUDIT =====',flush=True)
A=load_module('medain_stage10ab_trainer',TRAINER)
tree=ast.parse(EVALUATOR.read_text(encoding='utf8'))
needed={'stable_hash','select_basis_rows','covnorm','basis_metrics','solve_tp','solve_ao','solve_rand','estimate_covariances'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in needed]
assert {n.name for n in nodes}==needed
module=ast.Module(body=nodes,type_ignores=[])
ast.fix_missing_locations(module)
env={'np':np,'scipy':scipy,'torch':torch,'hashlib':hashlib,'DataLoader':DataLoader,'A':A,
     'SEED':2207,'RANK':2,'BETA':0.001,'BASIS_N':24}
exec(compile(module,str(EVALUATOR),'exec'),env)

def ckpt_path(domain,seed):
    return ROOT/'stage9b/anchors/checkpoints'/f'stage9a_anchor_erm_{domain}_seed{seed}_r2_a0p5'/'best_model.pt'
def val_path(domain,seed): return AUDIT/'stage10v_source_validation'/f'{domain}_seed{seed}.json'
def mechanism_path(domain,seed):
    folder='stage10t_exploratory' if seed==2207 else 'stage10u_seed3301'
    return AUDIT/folder/f'{domain}_seed{seed}.json'

def create_model(path):
    with safe_globals([TorchVersion]):
        state=torch.load(path,map_location='cpu',weights_only=True)
    original=A.resnet18
    A.resnet18=lambda *args,**kwargs: original(*args,**{**kwargs,'weights':None})
    try: model=A.ResNet18UNet(method='erm',rank=2,alpha=0.5)
    finally: A.resnet18=original
    model.load_state_dict(state['model_state'],strict=True)
    model.eval()
    return model

def install(model,method,bases):
    if method=='OFF': model.method='erm'; return
    model.med1.set_basis(torch.as_tensor(bases[('stage1',method)],dtype=model.med1.U.dtype))
    model.med2.set_basis(torch.as_tensor(bases[('stage2',method)],dtype=model.med2.U.dtype))
    model.method='medain'

def metric(pred,gt):
    valid=(gt!=255)
    gtfg=(gt==1)|(gt==2)
    assert valid.any() and gtfg.any()
    assert np.isin(np.unique(gt),[0,1,2,255]).all()
    # The fixed distance is from ANY reference foreground, not a model-dependent ROI.
    distance=ndimage.distance_transform_edt(~gtfg)
    valid_bg=valid & (gt==0)
    pred_fg=(pred==1)|(pred==2)
    fgfp=pred_fg & valid_bg
    metrics={
      'total_fp_foreground_pixels':int(fgfp.sum()),
      'total_gt_foreground_pixels':int(gtfg.sum()),
      'gt_class1_pixels':int((gt==1).sum()),
      'gt_class2_pixels':int((gt==2).sum()),
      'pred_class1_pixels':int(((pred==1)&valid).sum()),
      'pred_class2_pixels':int(((pred==2)&valid).sum()),
      'valid_pixels':int(valid.sum())
    }
    for t in THRESHOLDS:
        remote=fgfp & (distance>t)
        count, n=ndimage.label(remote,structure=EIGHT)
        areas=np.bincount(count.ravel())[1:]
        metrics[f'remote_fp_gt{t}px']=int(remote.sum())
        metrics[f'remote_fp_components_gt{t}px']=int(n)
        metrics[f'largest_remote_fp_component_gt{t}px']=int(areas.max()) if len(areas) else 0
        metrics[f'components_ge64_gt{t}px']=int(np.count_nonzero(areas>=64))
        for cls in (1,2):
            metrics[f'cls{cls}_remote_fp_gt{t}px']=int(((pred==cls)&remote).sum())
    metrics['remote_fp_rate_primary']=metrics['remote_fp_gt32px']/metrics['valid_pixels']
    return metrics

# Preflight all runs before inference: no partial scientific evidence on a missing provenance.
cache={}
for domain in DOMAINS:
    rows=A.load_rows(domain)
    train,val=rows['train'],rows['val']
    assert train and val
    assert all(r['domain']!=domain and r['split']=='train' for r in train)
    assert all(r['domain']!=domain and r['split']=='val' for r in val)
    assert len(set(r['sample_uid'] for r in train))==len(train)
    assert len(set(r['sample_uid'] for r in val))==len(val)
    assert set(r['sample_uid'] for r in train).isdisjoint(r['sample_uid'] for r in val)
    selected=env['select_basis_rows'](train,domain)
    assert len(selected)==24 and len(set(r['sample_uid'] for r in selected))==24
    cache[domain]={'rows':rows,'selected':selected,'selected_hash':uid_hash(selected),'val_hash':uid_hash(val)}
    for seed in SEEDS:
        cp=ckpt_path(domain,seed);vp=val_path(domain,seed);mp=mechanism_path(domain,seed)
        assert cp.is_file() and vp.is_file() and mp.is_file(),f'MISSING PROVENANCE: {domain} {seed}'
        v=json.loads(vp.read_text());m=json.loads(mp.read_text())
        assert v['status']=='PASS' and v['domain']==domain and v['seed']==seed
        assert v['checkpoint_sha256']==sha(cp) and v['trainer_sha256']==sha(TRAINER)
        assert v['evaluator_sha256']==sha(EVALUATOR)
        assert v['basis_uid_sha256']==cache[domain]['selected_hash']
        assert v['source_val_uid_sha256']==cache[domain]['val_hash']
        assert v['source_val_count']==len(val)
        assert v['training_performed'] is False and v['target_test_evaluated'] is False
        assert m['checkpoint_sha256']==sha(cp) and m['basis_uid_sha256']==cache[domain]['selected_hash']
        assert m['target_test_evaluated'] is False
        print('PREFLIGHT_PASS',domain,seed,flush=True)
print('PREFLIGHT = 10/10 VERIFIED',flush=True)

def execute(domain,seed):
    t0=time.time(); torch.manual_seed(seed);env['SEED']=seed
    cp=ckpt_path(domain,seed);v=json.loads(val_path(domain,seed).read_text())
    original=json.loads(mechanism_path(domain,seed).read_text())
    train=cache[domain]['rows']['train'];val=cache[domain]['rows']['val']
    model=create_model(cp)
    weights=A.compute_train_weights(train).to('cpu')
    ca1,ct1,ca2,ct2=env['estimate_covariances'](model,cache[domain]['selected'],weights,torch.device('cpu'))
    bases={}
    for stage,ca,ct in (('stage1',ca1,ct1),('stage2',ca2,ct2)):
        u_tp,_=env['solve_tp'](ca,ct)
        u_ao,_=env['solve_ao'](ca)
        u_rand=env['solve_rand'](ca.shape[0],domain,stage)
        for method,U in (('TP',u_tp),('AO',u_ao),('RAND',u_rand)):
            assert U.shape==(ca.shape[0],2) and np.isfinite(U).all()
            assert np.linalg.norm(U.T@U-np.eye(2))<1e-5
            actual=env['basis_metrics'](U,ca,ct)
            expected=original['results'][stage][method]
            for key in ('acquisition_capture_ratio','task_leakage_ratio','protection_efficiency'):
                x=float(actual[key]);y=float(expected[key]);assert abs(x-y)<=1e-7+1e-4*abs(y),(domain,seed,stage,method,key)
            bases[(stage,method)]=U
    print('MECHANISM_PASS',domain,seed,flush=True)
    ds=A.RigaDataset(val,augment=False,seed=seed)
    gt={};preds={}
    for method in METHODS:
        install(model,method,bases)
        # Check the full official metric, not a substitute image-wise mean.
        official=A.evaluate(model,ds,weights,torch.device('cpu'),batch_size=8)
        observed=float(official['mean_fg_dice']);expected=float(v['mean_fg_dice'][method])
        assert abs(observed-expected)<1e-7,('OFFICIAL_DICE_REPRO_FAIL',domain,seed,method,observed,expected)
        preds[method]={}
        with torch.inference_mode():
            for x,y,uids in DataLoader(ds,batch_size=8,shuffle=False,num_workers=0):
                prediction=model(x).argmax(dim=1).cpu().numpy()
                yy=y.cpu().numpy()
                for j,uid in enumerate(uids):
                    uid=str(uid);assert uid not in preds[method]
                    preds[method][uid]=prediction[j]
                    if method=='OFF':gt[uid]=yy[j]
                    else:assert np.array_equal(gt[uid],yy[j])
        assert set(preds[method])==set(r['sample_uid'] for r in val)
        print('DICE_REPRO_PASS',domain,seed,method,f'{observed:.8f}',flush=True)
    records=[]
    for row in val:
        uid=row['sample_uid'];target=gt[uid]
        entry={'sample_uid':uid,'source_domain':row['domain'],'outer_target':domain,'seed':seed}
        off_corr=(preds['OFF'][uid]==target)&(target!=255)
        measures={m:metric(preds[m][uid],target) for m in METHODS}
        for method in METHODS:
            for key,value in measures[method].items():entry[f'{method}_{key}']=value
        for method in ('TP','AO','RAND'):
            corr=(preds[method][uid]==target)&(target!=255)
            entry[f'{method}_harmed_vs_OFF']=int((off_corr&~corr).sum())
            entry[f'{method}_corrected_vs_OFF']=int((~off_corr&corr).sum())
            entry[f'{method}_minus_OFF_remote_fp_gt32px']=(
                measures[method]['remote_fp_gt32px']-measures['OFF']['remote_fp_gt32px'])
        records.append(entry)
    # For BinRushed, independently reconcile exact pixel-correctness counts against Stage 10-X.
    if domain=='BinRushed':
        xp=AUDIT/'stage10x_image_diagnostics'/f'{domain}_seed{seed}.json'
        assert xp.is_file()
        x=json.loads(xp.read_text());lookup={r['sample_uid']:r for r in x['image_records']}
        assert len(lookup)==len(records)
        for r in records:
            old=lookup[r['sample_uid']]
            for m in ('TP','AO','RAND'):
                assert r[f'{m}_harmed_vs_OFF']==int(old[f'{m}_harmed_pixels'])
                assert r[f'{m}_corrected_vs_OFF']==int(old[f'{m}_corrected_pixels'])
        print('STAGE10X_PIXEL_RECONCILIATION_PASS',seed,flush=True)
    report={
      'stage':'10-AB','version':VERSION,'classification':'EXPLORATORY_SOURCE_VALIDATION_SPATIAL_ERROR_AUDIT',
      'domain':domain,'seed':seed,'distance_thresholds_pixels':list(THRESHOLDS),'primary_distance_pixels':PRIMARY_DISTANCE,
      'component_connectivity':8,'severe_component_min_pixels':64,
      'checkpoint_sha256':sha(cp),'trainer_sha256':sha(TRAINER),'evaluator_sha256':sha(EVALUATOR),
      'stage10v_sha256':sha(val_path(domain,seed)),'stage10mechanism_sha256':sha(mechanism_path(domain,seed)),
      'selected_uid_sha256':cache[domain]['selected_hash'],'val_uid_sha256':cache[domain]['val_hash'],
      'validated_aggregate_dice':v['mean_fg_dice'],'rows':records,'n_source_validation':len(records),
      'training_performed':False,'target_test_evaluated':False,'parameters_tuned':False,
      'causal_claim_authorized':False,'historical_stage9a_go':'NOT_VERIFIED',
      'elapsed_seconds':time.time()-t0,'status':'PASS'
    }
    dest=OUT/f'{domain}_seed{seed}.json'
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print('RUN_PASS',domain,seed,'n=',len(records),'seconds=',round(time.time()-t0,1),flush=True)

def skip_ok(domain,seed):
    p=OUT/f'{domain}_seed{seed}.json'
    if not p.is_file():return False
    try:
        v=json.loads(p.read_text());return all([
           v.get('version')==VERSION,v.get('status')=='PASS',v.get('domain')==domain,v.get('seed')==seed,
           v.get('checkpoint_sha256')==sha(ckpt_path(domain,seed)),
           v.get('trainer_sha256')==sha(TRAINER),v.get('evaluator_sha256')==sha(EVALUATOR),
           v.get('stage10v_sha256')==sha(val_path(domain,seed)),
           v.get('stage10mechanism_sha256')==sha(mechanism_path(domain,seed)),
           v.get('selected_uid_sha256')==cache[domain]['selected_hash'],
           v.get('val_uid_sha256')==cache[domain]['val_hash'],
           v.get('n_source_validation')==len(cache[domain]['rows']['val']),
           len(v.get('rows',[]))==len(cache[domain]['rows']['val']),
           v.get('training_performed') is False,v.get('target_test_evaluated') is False
        ])
    except Exception:return False

for domain in DOMAINS:
    for seed in SEEDS:
        if skip_ok(domain,seed):print('SKIP_VERIFIED',domain,seed,flush=True)
        else:
            print('RUNNING',domain,seed,flush=True)
            execute(domain,seed)

reports=[json.loads((OUT/f'{d}_seed{s}.json').read_text()) for d in DOMAINS for s in SEEDS]
assert len(reports)==10 and all(r['status']=='PASS' for r in reports)
allrows=[row for report in reports for row in report['rows']]
assert len(allrows)==sum(r['n_source_validation'] for r in reports)
fields=list(allrows[0])
assert all(list(row)==fields for row in allrows)
with (OUT/'stage10ab_all_images.csv').open('w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(allrows)

summary={}
for method in METHODS:
    counts=np.array([r[f'{method}_remote_fp_gt32px'] for r in allrows])
    maxima=np.array([r[f'{method}_largest_remote_fp_component_gt32px'] for r in allrows])
    summary[method]={
      'n_image_seed_pairs':len(counts),'mean_remote_fp_pixels':float(counts.mean()),
      'median_remote_fp_pixels':float(np.median(counts)),
      'fraction_any_remote_fp':float(np.mean(counts>0)),
      'fraction_component_ge64px':float(np.mean(maxima>=64)),
      'mean_largest_component_px':float(maxima.mean()),
      'total_remote_fp_pixels':int(counts.sum())
    }
folds={}
for domain in DOMAINS:
    rs=[r for r in allrows if r['outer_target']==domain]
    folds[domain]={method:{
      'mean_remote_fp_pixels':float(np.mean([r[f'{method}_remote_fp_gt32px'] for r in rs])),
      'large_component_images':int(sum(r[f'{method}_largest_remote_fp_component_gt32px']>=64 for r in rs)),
      'n_image_seed_pairs':len(rs)} for method in METHODS}
per_run={}
for r in reports:
    k=f"{r['domain']}_seed{r['seed']}"
    per_run[k]={m:{'mean_remote_fp_pixels':float(np.mean([x[f'{m}_remote_fp_gt32px'] for x in r['rows']])),
                   'n_large_component':int(sum(x[f'{m}_largest_remote_fp_component_gt32px']>=64 for x in r['rows']))}
                for m in METHODS}

aggregate={
  'stage':'10-AB','version':VERSION,'classification':'EXPLORATORY_SOURCE_VALIDATION_ONLY',
  'primary_definition':'Predicted foreground (classes 1/2) at GT background pixels farther than 32 px from all reference foreground; 8-connected remote error components.',
  'sensitivity_thresholds_px':[16,48],'summary':summary,'fold_summary':folds,'per_run':per_run,
  'n_reports':10,'n_image_seed_pairs':len(allrows),'distinct_independent_images_claimed':False,
  'subgroup_fold_dependence':'The five LODO training domains overlap; images are reused across source-validation splits and seeds.',
  'transferred_test_assessment':False,'training_performed':False,
  'prospective_go_confirmed':False,'submission_decision':'SCIENTIFIC_REVIEW_REQUIRED',
  'report_sha256':{f"{d}_seed{s}":sha(OUT/f'{d}_seed{s}.json') for d in DOMAINS for s in SEEDS}
}
(OUT/'stage10ab_summary.json').write_text(json.dumps(aggregate,indent=2)+'\n')

# A provisional fragment: not yet directly injected into the manuscript until audited.
tex=['% Stage10AB provisional fragment. Integrate only after independent verification.','\\begin{table}[t]',
     '\\centering\\small',
     '\\caption{Exploratory remote false-positive segmentation errors on source validation. A remote false positive is a predicted foreground pixel more than 32 pixels from the ground-truth foreground, with eight-connected components. Counts pool repeated image--seed observations and are descriptive, not independent samples.}',
     '\\label{tab:stage10ab-spatial}',
     '\\begin{tabular}{lrrrr}\\toprule',
     'Method & Mean remote FP px & Median remote FP px & Large-island pairs & Total remote FP px \\\\ \\midrule']
for m in METHODS:
    s=summary[m];large=sum(r[f'{m}_largest_remote_fp_component_gt32px']>=64 for r in allrows)
    tex.append(f"{m} & {s['mean_remote_fp_pixels']:.2f} & {s['median_remote_fp_pixels']:.2f} & {large}/{len(allrows)} & {s['total_remote_fp_pixels']} \\\\")
tex+=['\\bottomrule\\end{tabular}','\\end{table}']
(OUT/'stage10ab_table_DRAFT_DO_NOT_AUTO_INCLUDE.tex').write_text('\n'.join(tex)+'\n')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    means=[summary[m]['mean_remote_fp_pixels'] for m in METHODS]
    fig,ax=plt.subplots(figsize=(6.7,3.8))
    ax.bar(list(METHODS),means)
    ax.set_ylabel('Mean remote false-positive pixels / image-seed pair')
    ax.set_title('Exploratory source-validation spatial errors (>32 px)')
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'stage10ab_spatial_error_bar.png',dpi=200)
    plt.close(fig)
    print('FIGURE = CREATED',flush=True)
except Exception as e:print('FIGURE_UNAVAILABLE',type(e).__name__,str(e),flush=True)

zip_path=OUT/'MedAIN_Stage10AB_Audited_Results.zip'
with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.glob('*.json'))+sorted(OUT.glob('*.csv'))+sorted(OUT.glob('*.tex'))+sorted(OUT.glob('*.png')):
        z.write(p,p.name)
print('\n===== STAGE 10-AB FINAL =====',flush=True)
print('RUN_REPORTS = 10/10',flush=True)
print('IMAGE_SEED_OBSERVATIONS =',len(allrows),flush=True)
for method in METHODS:
    print(method,'MEAN_REMOTE_FP =',round(summary[method]['mean_remote_fp_pixels'],4),
          'LARGE_ISLAND_RATE =',round(summary[method]['fraction_component_ge64px'],4),flush=True)
print('TARGET_TEST_EVALUATED = False',flush=True)
print('TRAINING_PERFORMED = False',flush=True)
print('CONFIRMATORY_CLAIM_AUTHORIZED = False',flush=True)
print('ZIP =',zip_path,flush=True)
print('STAGE10AB_STATUS = ANALYSIS_COMPLETE_AWAITING_INDEPENDENT_REVIEW',flush=True)
