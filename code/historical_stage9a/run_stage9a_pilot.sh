#!/usr/bin/env bash
set -euo pipefail

PKG="$HOME/MedAIN_STAGE9A_PROSPECTIVE_PILOT_V1"
ROOT="$HOME/MedAIN_V19_STAGE9"
VENV="$HOME/medain-stage9-torch26"
LOG="$ROOT/logs/stage9a_pilot"
DOMAINS=(BinRushed Magrabia MESSIDOR_Base1 MESSIDOR_Base2 MESSIDOR_Base3)

mkdir -p "$ROOT/anchors/results" "$ROOT/anchors/checkpoints" "$ROOT/stage9a_pilot" "$LOG"

echo "============================================================"
echo " MedAIN Stage9A — PROSPECTIVE PILOT"
echo " 5 ERM source-validation anchors, seed 1103"
echo " Then TP vs AO vs RAND frozen-basis novelty dissection"
echo " NO TARGET TEST / NO EXTERNAL DATA"
echo "============================================================"

test -x "$VENV/bin/python" || { echo "Missing venv: $VENV"; exit 10; }
test -f "$HOME/MedAIN_V13_IDENTITY_REPAIR/audit.json" || { echo "Missing identity audit"; exit 11; }
test -f "$HOME/MedAIN_V13_IDENTITY_REPAIR/STAGE9_LODO_SCHEMA_REPAIR_AUDIT.json" || { echo "Missing schema audit"; exit 12; }

"$VENV/bin/python" - <<'PY'
import json
from pathlib import Path
H=Path.home()
a=json.loads((H/"MedAIN_V13_IDENTITY_REPAIR/audit.json").read_text())
s=json.loads((H/"MedAIN_V13_IDENTITY_REPAIR/STAGE9_LODO_SCHEMA_REPAIR_AUDIT.json").read_text())
assert a["status"]=="PASS"
assert a["rows"]==744 and a["unique_sample_uid"]==744
assert a["aggregate_preprocessing_match"] is True
assert a["representative_validation_pass"] is True
assert s["status"]=="PASS"
assert s["split_assignments_changed"] is False
assert s["sample_uid_changed"] is False
print("DATA + IDENTITY + SPLIT GATE: PASS")
PY

echo
echo "===== COMPILE ====="
"$VENV/bin/python" -m py_compile \
  "$PKG/stage9_anchor_train.py" \
  "$PKG/stage9a_mechanistic_eval.py" \
  "$PKG/stage9a_aggregate.py"
echo "COMPILE: PASS"

NCPU="$(nproc)"
THREADS=4
if [ "$NCPU" -lt 4 ]; then THREADS="$NCPU"; fi
if [ "$NCPU" -ge 8 ]; then MAXJOBS=2; else MAXJOBS=1; fi
echo "nproc=$NCPU threads_per_anchor=$THREADS max_parallel_anchors=$MAXJOBS"

run_anchor () {
  local D="$1"
  local RN="stage9a_anchor_erm_${D}_seed1103_r2_a0p5"
  local PASS="$ROOT/anchors/results/$RN/summary.json"
  local CKPT="$ROOT/anchors/checkpoints/$RN/best_model.pt"
  local L="$LOG/anchor_${D}.log"

  if [ -f "$PASS" ] && [ -f "$CKPT" ]; then
    echo "ANCHOR $D: already complete; skipping"
    return 0
  fi

  echo "ANCHOR $D: START"
  OMP_NUM_THREADS="$THREADS" \
  MKL_NUM_THREADS="$THREADS" \
  OPENBLAS_NUM_THREADS="$THREADS" \
  NUMEXPR_NUM_THREADS="$THREADS" \
  "$VENV/bin/python" -u "$PKG/stage9_anchor_train.py" \
    --method erm \
    --target-domain "$D" \
    --seed 1103 \
    --epochs 60 \
    --patience 10 \
    --batch-size 8 \
    --threads "$THREADS" \
    --rank 2 \
    --alpha 0.5 \
    --beta 0.001 \
    --basis-update-every 5 \
    --basis-samples 24 \
    --skip-test \
    > "$L" 2>&1
  echo "ANCHOR $D: PASS"
}

echo
echo "===== ANCHOR TRAINING ====="
active=0
for D in "${DOMAINS[@]}"; do
  run_anchor "$D" &
  active=$((active+1))
  if [ "$active" -ge "$MAXJOBS" ]; then
    wait -n
    active=$((active-1))
  fi
done
wait
echo "ALL 5 ANCHORS: COMPLETE"

echo
echo "===== MECHANISTIC EVALUATION ====="
for D in "${DOMAINS[@]}"; do
  if [ -f "$ROOT/stage9a_pilot/$D/PASS" ]; then
    echo "MECHANISM $D: already complete; skipping"
  else
    echo "MECHANISM $D: START"
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    "$VENV/bin/python" -u "$PKG/stage9a_mechanistic_eval.py" \
      --target-domain "$D" \
      > "$LOG/mechanism_${D}.log" 2>&1
    echo "MECHANISM $D: PASS"
  fi
done

echo
echo "===== PILOT AGGREGATE ====="
"$VENV/bin/python" "$PKG/stage9a_aggregate.py" \
  | tee "$LOG/STAGE9A_PILOT_AGGREGATE.log"

echo
echo "===== HASH ALL SCIENTIFIC OUTPUTS ====="
find "$ROOT/stage9a_pilot" "$ROOT/anchors/results" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$ROOT/STAGE9A_PILOT_OUTPUT_SHA256.txt"

tar -czf "$HOME/MedAIN_STAGE9A_PILOT_AUDIT_RESULTS.tar.gz" \
  -C "$ROOT" \
  stage9a_pilot \
  anchors/results \
  STAGE9A_PILOT_OUTPUT_SHA256.txt

echo
echo "===== RESULT PACKAGE ====="
ls -lh "$HOME/MedAIN_STAGE9A_PILOT_AUDIT_RESULTS.tar.gz"
sha256sum "$HOME/MedAIN_STAGE9A_PILOT_AUDIT_RESULTS.tar.gz"

echo
echo "============================================================"
echo " STAGE9A PROSPECTIVE PILOT COMPLETE"
echo " NO TARGET TEST EVALUATED"
echo " NO EXTERNAL DATA USED"
echo "============================================================"
