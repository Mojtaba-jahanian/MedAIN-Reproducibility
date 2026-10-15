#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate medain-v10
H="$HOME"; ROOT="$H/MedAIN_V16_STAGE8R_C"; S7C="$H/MedAIN_V13_STAGE7R_C"; AUD="$ROOT/audit"; RES="$ROOT/results"; LOG="$ROOT/logs"; SRC="$ROOT/src"
RUNNER="$SRC/stage8rc_external_comparator_runner.py"; MANIFEST="$H/MedAIN_V11_STAGE7B/audit/external_evaluation_manifest.csv"; QLOCK="$S7C/audit/QUALITATIVE_SELECTION_LOCK.json"; QDIR="$ROOT/qualitative"
mkdir -p "$LOG/jobs" "$LOG/slots" "$RES" "$QDIR"
python - <<'PY'
from pathlib import Path
import csv
root=Path.home()/'MedAIN_V16_STAGE8R_C'; rows=list(csv.DictReader((root/'audit/STAGE8RC_JOBS.csv').open()))
for slot in range(6):
 p=root/f'audit/stage8rc_slot_{slot}.tsv'
 with p.open('w') as f:
  for i,r in enumerate(rows):
   if i%6==slot: f.write('\t'.join([r['method'],r['source_fold'],r['seed'],r['checkpoint'],r['checkpoint_sha256'],r['out_dir']])+'\n')
PY
worker () {
 local slot="$1" lo=$((slot*4)) hi=$((slot*4+3))
 while IFS=$'\t' read -r method fold seed ck sha outdir; do
  [ -n "$method" ] || continue
  summary="$outdir/summary.json"; run="${method}_${fold}_seed${seed}"
  if [ -f "$summary" ] && python - "$summary" "$RUNNER" "$sha" <<'PY'
import json,sys,hashlib
from pathlib import Path
try:
 d=json.loads(Path(sys.argv[1]).read_text()); rh=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
 ok=(d.get('status')=='PASS' and d.get('stage')=='MedAIN_V16_Stage8R_C' and d.get('runner_sha256')==rh and d.get('checkpoint_sha256')==sys.argv[3] and d.get('external_tuning_performed') is False and d.get('one_shot_external_inference') is True)
except Exception: ok=False
raise SystemExit(0 if ok else 1)
PY
  then echo "$(date) SKIP-PASS $run" | tee -a "$LOG/master.log"; continue; fi
  ok=0
  for attempt in 1 2; do
   echo "$(date) START slot=$slot cpus=$lo-$hi $run attempt=$attempt" | tee -a "$LOG/master.log"
   if OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 taskset -c "$lo-$hi" python -u "$RUNNER" --method "$method" --source-fold "$fold" --seed "$seed" --checkpoint "$ck" --expected-checkpoint-sha "$sha" --manifest "$MANIFEST" --out-dir "$outdir" --qualitative-lock "$QLOCK" --qualitative-dir "$QDIR" --threads 4 --batch-size 8 > "$LOG/jobs/${run}.log" 2>&1; then
    if [ -f "$summary" ]; then echo "$(date) PASS $run" | tee -a "$LOG/master.log"; ok=1; break; fi
   fi
   echo "$(date) RETRY $run attempt=$attempt" | tee -a "$LOG/master.log"; sleep 3
  done
  if [ "$ok" -ne 1 ]; then echo "$run" >> "$LOG/FAILED_RUNS.txt"; return 1; fi
 done < "$AUD/stage8rc_slot_${slot}.tsv"
}
: > "$LOG/master.log"; rm -f "$LOG/FAILED_RUNS.txt"; pids=()
for slot in 0 1 2 3 4 5; do worker "$slot" > "$LOG/slots/slot${slot}.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done
[ "$status" -eq 0 ] || { echo "$(date) STAGE8R-C WORKER FAILURE" | tee -a "$LOG/master.log"; exit 90; }
echo "$(date) ALL 90 COMPARATOR EXTERNAL RUNS PASS; AGGREGATING" | tee -a "$LOG/master.log"

python - <<'PY'
from pathlib import Path
import csv,json,hashlib,statistics
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon
H=Path.home(); root=H/'MedAIN_V16_STAGE8R_C'; res=root/'results'; aud=root/'audit'; s7=H/'MedAIN_V13_STAGE7R_C'
methods=['ERM_stage8','StrongAug','IBN','MixStyle','CORAL','Fishr']; folds=['BinRushed','Magrabia','MESSIDOR_Base1','MESSIDOR_Base2','MESSIDOR_Base3']; seeds=[1103,2207,3301]; domains=['REFUGE','Drishti_GS','ORIGA','PAPILA']; metrics=['od_dice','oc_dice','mean_dice','od_iou','oc_iou','mean_iou']
# Comparator summaries
summ=[]; errors=[]
for p in sorted(res.glob('*/summary.json')):
 d=json.loads(p.read_text()); summ.append(d)
 if d.get('status')!='PASS' or d.get('external_images')!=769 or d.get('annotation_comparisons')!=1257 or d.get('external_tuning_performed') is not False or d.get('one_shot_external_inference') is not True: errors.append('bad '+str(p))
if len(summ)!=90: errors.append(f'comparator summaries={len(summ)} expected=90')
# Canonical MedAIN summaries from prior corrected external stage
med=[]
for p in sorted((s7/'results').glob('medain_*/summary.json')):
 d=json.loads(p.read_text()); med.append(d)
if len(med)!=15: errors.append(f'MedAIN summaries={len(med)} expected=15')
if errors: raise SystemExit('\n'.join(errors))
# Unified model-domain table
rows=[]
for d in summ:
 for dom in domains:
  rows.append({'method':d['method'],'source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':dom,**{k:float(d['domain_summary'][dom][k]) for k in metrics}})
 rows.append({'method':d['method'],'source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':'ALL_DOMAIN_MACRO',**{k:float(d['overall_domain_macro'][k]) for k in metrics}})
 rows.append({'method':d['method'],'source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':'ALL_IMAGE_WEIGHTED',**{k:float(d['overall_image_weighted'][k]) for k in metrics}})
for d in med:
 for dom in domains:
  rows.append({'method':'MedAIN','source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':dom,**{k:float(d['domain_summary'][dom][k]) for k in metrics}})
 rows.append({'method':'MedAIN','source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':'ALL_DOMAIN_MACRO',**{k:float(d['overall_domain_macro'][k]) for k in metrics}})
 rows.append({'method':'MedAIN','source_fold':d['source_fold'],'seed':int(d['seed']),'external_domain':'ALL_IMAGE_WEIGHTED',**{k:float(d['overall_image_weighted'][k]) for k in metrics}})
with (res/'stage8rc_model_domain_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# Descriptive method summary across 15 matched source-fold/seed replicates
ms=[]
for m in ['MedAIN']+methods:
 for dom in domains+['ALL_DOMAIN_MACRO','ALL_IMAGE_WEIGHTED']:
  sub=[r for r in rows if r['method']==m and r['external_domain']==dom]
  rr={'method':m,'external_domain':dom,'n_replicates':len(sub)}
  for k in metrics:
   v=[r[k] for r in sub]; rr[k+'_mean']=float(np.mean(v)); rr[k+'_sd']=float(np.std(v,ddof=1))
  ms.append(rr)
with (res/'stage8rc_method_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(ms[0])); w.writeheader(); w.writerows(ms)
# Matched MedAIN-comparator paired table
idx={(r['method'],r['source_fold'],int(r['seed']),r['external_domain']):r for r in rows}
paired=[]
for comp in methods:
 for fold in folds:
  for seed in seeds:
   for dom in domains+['ALL_DOMAIN_MACRO','ALL_IMAGE_WEIGHTED']:
    a=idx[('MedAIN',fold,seed,dom)]; b=idx[(comp,fold,seed,dom)]; rr={'comparator':comp,'source_fold':fold,'seed':seed,'external_domain':dom}
    for k in metrics:
     rr['medain_'+k]=a[k]; rr['comparator_'+k]=b[k]; rr['delta_'+k]=a[k]-b[k]
    paired.append(rr)
with (res/'stage8rc_medain_comparator_paired_deltas.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)
def holm(pvals):
 n=len(pvals); order=np.argsort(pvals); out=np.empty(n,float); prev=0.0
 for rank,idx0 in enumerate(order):
  val=min(1.0,(n-rank)*pvals[idx0]); prev=max(prev,val); out[idx0]=prev
 return out.tolist()
rng=np.random.default_rng(20260911)
# Primary family: domain-macro mean Dice, six paired tests
primary=[]
for comp in methods:
 sub=[r for r in paired if r['comparator']==comp and r['external_domain']=='ALL_DOMAIN_MACRO']; d=np.array([r['delta_mean_dice'] for r in sub],float)
 if np.allclose(d,0): W=0.0; p=1.0
 else:
  z=wilcoxon(d,zero_method='wilcox',alternative='two-sided',mode='auto'); W=float(z.statistic); p=float(z.pvalue)
 boot=rng.choice(d,size=(100000,len(d)),replace=True).mean(axis=1); lo,hi=np.percentile(boot,[2.5,97.5])
 primary.append({'comparator':comp,'n_pairs':len(d),'medain_mean':float(np.mean([r['medain_mean_dice'] for r in sub])),'comparator_mean':float(np.mean([r['comparator_mean_dice'] for r in sub])),'mean_delta_medain_minus_comparator':float(d.mean()),'median_delta':float(np.median(d)),'bootstrap95_low':float(lo),'bootstrap95_high':float(hi),'medain_wins':int((d>0).sum()),'comparator_wins':int((d<0).sum()),'ties':int((d==0).sum()),'wilcoxon_W':W,'wilcoxon_p_raw':p})
adj=holm([r['wilcoxon_p_raw'] for r in primary])
for r,a in zip(primary,adj): r['wilcoxon_p_holm6']=a
with (res/'stage8rc_primary_medain_vs_comparators.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(primary[0])); w.writeheader(); w.writerows(primary)
# Secondary matched statistics with Holm across six comparators for each domain/metric family
secondary=[]
for dom in domains+['ALL_DOMAIN_MACRO','ALL_IMAGE_WEIGHTED']:
 for metric in metrics:
  fam=[]
  for comp in methods:
   sub=[r for r in paired if r['comparator']==comp and r['external_domain']==dom]; d=np.array([r['delta_'+metric] for r in sub],float)
   if np.allclose(d,0): W=0.0; p=1.0
   else:
    z=wilcoxon(d,zero_method='wilcox',alternative='two-sided',mode='auto'); W=float(z.statistic); p=float(z.pvalue)
   boot=rng.choice(d,size=(100000,len(d)),replace=True).mean(axis=1); lo,hi=np.percentile(boot,[2.5,97.5])
   fam.append({'external_domain':dom,'metric':metric,'comparator':comp,'n_pairs':len(d),'mean_delta_medain_minus_comparator':float(d.mean()),'median_delta':float(np.median(d)),'bootstrap95_low':float(lo),'bootstrap95_high':float(hi),'medain_wins':int((d>0).sum()),'comparator_wins':int((d<0).sum()),'ties':int((d==0).sum()),'wilcoxon_W':W,'wilcoxon_p_raw':p})
  aa=holm([r['wilcoxon_p_raw'] for r in fam])
  for r,a in zip(fam,aa): r['wilcoxon_p_holm6']=a
  secondary.extend(fam)
with (res/'stage8rc_secondary_statistics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(secondary[0])); w.writeheader(); w.writerows(secondary)
# Ranking on frozen primary endpoint (descriptive only)
pr={r['method']:r for r in ms if r['external_domain']=='ALL_DOMAIN_MACRO'}
ranking=sorted([{'method':m,'mean_domain_macro_mean_dice':pr[m]['mean_dice_mean'],'sd_across15':pr[m]['mean_dice_sd']} for m in pr],key=lambda x:x['mean_domain_macro_mean_dice'],reverse=True)
for i,r in enumerate(ranking,1): r['rank']=i
with (res/'stage8rc_primary_ranking.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['rank','method','mean_domain_macro_mean_dice','sd_across15']); w.writeheader(); w.writerows(ranking)
final={'stage':'MedAIN_V16_Stage8R_C','status':'PASS','comparator_external_models_expected':90,'comparator_external_models_passed':90,'canonical_medain_models_reused':15,'external_images_per_model':769,'annotation_comparisons_per_model':1257,'external_training':False,'external_tuning':False,'checkpoint_reselection':False,'primary_endpoint':'equal-weight macro-average of OD/OC mean Dice across REFUGE, Drishti-GS, ORIGA, and PAPILA','bootstrap_resamples':100000,'primary_multiplicity':'Holm across six MedAIN-vs-comparator tests','errors':[],'primary_results_file':'results/stage8rc_primary_medain_vs_comparators.csv','ranking_file':'results/stage8rc_primary_ranking.csv'}
(aud/'stage8rc_final_audit.json').write_text(json.dumps(final,indent=2)+'\n'); print(json.dumps(final,indent=2))
PY

rm -f "$AUD/RESULT_SHA256SUMS.txt"
find "$RES" "$AUD" "$ROOT/qualitative" -type f -print0 | sort -z | xargs -0 sha256sum > "$AUD/RESULT_SHA256SUMS.txt"
ARCH="$H/MedAIN_V16_STAGE8RC_ONE_SHOT_EXTERNAL_CORE.tar.gz"
rm -f "$ARCH"
tar -czf "$ARCH" -C "$ROOT" src results audit qualitative logs
sha256sum "$ARCH" | tee "$AUD/STAGE8RC_ARCHIVE.sha256"
echo "$(date) STAGE8R-C COMPLETE" | tee -a "$LOG/master.log"
echo "ARCHIVE: $ARCH" | tee -a "$LOG/master.log"
