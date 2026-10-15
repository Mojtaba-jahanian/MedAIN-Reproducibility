#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate medain-v10
H="$HOME"; ROOT="$H/MedAIN_V15_STAGE8"; SRC="$ROOT/src"; RES="$ROOT/results/stage8rb"; LOG="$ROOT/logs/stage8rb"; AUD="$ROOT/audit"
RUNNER="$SRC/stage8rb_comparator_runner.py"
mkdir -p "$LOG/jobs" "$LOG/slots"

python - <<'PY'
from pathlib import Path
import csv
H=Path.home(); root=H/'MedAIN_V15_STAGE8'; rows=list(csv.DictReader((root/'audit/STAGE8RB_JOBS.csv').open()))
for slot in range(6):
    p=root/f'audit/stage8rb_slot_{slot}.tsv'
    with p.open('w') as f:
        for i,r in enumerate(rows):
            if i%6==slot: f.write(f"{r['method']}\t{r['target_domain']}\t{r['seed']}\n")
PY

worker () {
  local slot="$1" cpulo=$((slot*4)) cpuphi=$((slot*4+3))
  while IFS=$'\t' read -r method domain seed; do
    [ -n "$method" ] || continue
    run="${method}_${domain}_seed${seed}"
    summary="$RES/$run/summary.json"
    if [ -f "$summary" ] && python - "$summary" "$RUNNER" <<'PY'
import json,sys
try:
 d=json.load(open(sys.argv[1])); import hashlib,pathlib; rh=hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(); ok=d.get('status')=='PASS' and d.get('stage')=='MedAIN_V15_Stage8R_B' and d.get('external_inference') is False and d.get('target_test_evaluated') is False and d.get('identity_key')=='sample_uid' and d.get('identity_rows')==744 and d.get('runner_sha256')==rh
except Exception: ok=False
raise SystemExit(0 if ok else 1)
PY
    then
      echo "$(date) SKIP-PASS slot=$slot $run" | tee -a "$LOG/master.log"
      continue
    fi
    for attempt in 1 2; do
      echo "$(date) START slot=$slot cpus=$cpulo-$cpuphi $run attempt=$attempt" | tee -a "$LOG/master.log"
      if taskset -c "$cpulo-$cpuphi" python -u "$RUNNER" --method "$method" --target-domain "$domain" --seed "$seed" --threads 4 --batch-size 8 --epochs 60 --patience 10 \
           > "$LOG/jobs/${run}.log" 2>&1; then
        if [ -f "$summary" ]; then
          echo "$(date) PASS $run" | tee -a "$LOG/master.log"; break
        fi
      fi
      echo "$(date) RETRY/FAIL $run attempt=$attempt" | tee -a "$LOG/master.log"
      if [ "$attempt" -eq 2 ]; then echo "$run" >> "$LOG/FAILED_RUNS.txt"; return 1; fi
      sleep 3
    done
  done < "$AUD/stage8rb_slot_${slot}.tsv"
}

: > "$LOG/master.log"
rm -f "$LOG/FAILED_RUNS.txt"
pids=()
for slot in 0 1 2 3 4 5; do worker "$slot" > "$LOG/slots/slot${slot}.log" 2>&1 & pids+=("$!"); done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
if [ "$status" -ne 0 ]; then echo "$(date) STAGE8RB WORKER FAILURE" | tee -a "$LOG/master.log"; exit 90; fi

echo "$(date) ALL WORKERS DONE; FINAL AUDIT" | tee -a "$LOG/master.log"
python - <<'PY'
from pathlib import Path
import csv,json,hashlib,statistics
H=Path.home(); root=H/'MedAIN_V15_STAGE8'; res=root/'results/stage8rb'; ck=root/'checkpoints/stage8rb'; aud=root/'audit'
jobs=list(csv.DictReader((aud/'STAGE8RB_JOBS.csv').open()))
summ=[]; errors=[]
for j in jobs:
 run=f"{j['method']}_{j['target_domain']}_seed{j['seed']}"; p=res/run/'summary.json'; cp=ck/run/'best_model.pt'
 if not p.exists(): errors.append(f'missing summary {run}'); continue
 d=json.loads(p.read_text())
 runner=root/'src/stage8rb_comparator_runner.py'
 rh=hashlib.sha256(runner.read_bytes()).hexdigest()
 if (d.get('status')!='PASS' or d.get('stage')!='MedAIN_V15_Stage8R_B' or
     d.get('external_inference') is not False or d.get('target_test_evaluated') is not False or
     d.get('identity_key')!='sample_uid' or d.get('identity_rows')!=744 or d.get('runner_sha256')!=rh):
  errors.append(f'bad provenance/summary {run}')
 if not cp.exists(): errors.append(f'missing checkpoint {run}')
 else:
  h=hashlib.sha256(cp.read_bytes()).hexdigest()
  if h!=d.get('best_checkpoint_sha256'): errors.append(f'checkpoint hash mismatch {run}')
 summ.append(d)
# Complete matrix constraints
if len(summ)!=90: errors.append(f'summaries={len(summ)} expected=90')
# Fishr diagnostic: record, do not alter protocol.
fish=[d for d in summ if d['method']=='Fishr']
fish_active=sum(1 for d in fish if d.get('fishr_active_updates',0)>0)
fish_valid=sum(1 for d in fish if d.get('fishr_selection_unlocked') is True and d.get('fishr_post_anneal_best_checkpoint') is True and d.get('fishr_active_updates',0)>0)
if len(fish)!=15: errors.append(f'Fishr summaries={len(fish)} expected=15')
if fish_valid!=15: errors.append(f'Fishr post-anneal valid runs={fish_valid}/15; refusing comparator PASS')
# Source-validation aggregate by method
rows=[]
for m in sorted(set(d['method'] for d in summ)):
 vals=[d['best_val_mean_fg_dice'] for d in summ if d['method']==m]
 rows.append({'method':m,'n':len(vals),'mean_best_source_val_dice':sum(vals)/len(vals),'std_best_source_val_dice':statistics.stdev(vals) if len(vals)>1 else 0.0})
with (res/'stage8rb_source_validation_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# checkpoint hash manifest
with (aud/'STAGE8RB_CHECKPOINT_SHA256.csv').open('w',newline='') as f:
 w=csv.writer(f); w.writerow(['run_name','sha256','bytes'])
 for d in sorted(summ,key=lambda x:x['run_name']):
  p=ck/d['run_name']/'best_model.pt'; w.writerow([d['run_name'],hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size])
final={'stage':'MedAIN_V15_Stage8R_B','status':'PASS' if not errors else 'FAIL','formal_runs_expected':90,'formal_runs_passed':len(summ),'external_inference':False,'held_out_riga_test_evaluated':False,'identity_key':'sample_uid','identity_rows':744,'legacy_sample_id_lookup':False,
       'errors':errors,'fishr_runs':len(fish),'fishr_runs_with_penalty_active_updates_gt0':fish_active,'fishr_post_anneal_valid_runs':fish_valid,
       'note':'Fishr lambda/anneal/EMA remain frozen. Pre-anneal epochs are warm-up only; formal Fishr checkpoints must be selected after penalty activation.'}
(aud/'stage8rb_final_audit.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps(final,indent=2))
if errors: raise SystemExit(1)
PY

# Core archive excludes large checkpoints but preserves all summaries, histories,
# source, audit, logs, and the checkpoint SHA manifest.
ARCH="$H/MedAIN_V15_STAGE8RB_TRAINING_CORE.tar.gz"
tar -czf "$ARCH" -C "$ROOT" audit src results/stage8rb logs/stage8rb
sha256sum "$ARCH" | tee "$AUD/STAGE8RB_ARCHIVE.sha256"
echo "$(date) STAGE8RB COMPLETE" | tee -a "$LOG/master.log"
echo "ARCHIVE: $ARCH" | tee -a "$LOG/master.log"
sha256sum "$ARCH" | tee -a "$LOG/master.log"
