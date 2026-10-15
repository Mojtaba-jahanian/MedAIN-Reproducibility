#!/usr/bin/env bash
ROOT="$HOME/MedAIN_V19_STAGE9"
LOG="$ROOT/logs/stage9a_pilot"
echo "===== STAGE9A PILOT STATUS ====="
echo "Processes:"
ps -eo pid,pcpu,pmem,etime,cmd | grep -E 'stage9_anchor_train.py|stage9a_mechanistic_eval.py|run_stage9a_pilot.sh' | grep -v grep || true
echo
echo "Completed anchor summaries:"
find "$ROOT/anchors/results" -name summary.json -type f 2>/dev/null | sort || true
echo
echo "Completed mechanism folds:"
find "$ROOT/stage9a_pilot" -name PASS -type f 2>/dev/null | sort || true
echo
echo "Latest anchor logs:"
for f in "$LOG"/anchor_*.log; do
  [ -f "$f" ] || continue
  echo "--- $f"
  tail -5 "$f"
done
echo
echo "Aggregate:"
if [ -f "$ROOT/stage9a_pilot/STAGE9A_PILOT_AGGREGATE.json" ]; then
  cat "$ROOT/stage9a_pilot/STAGE9A_PILOT_AGGREGATE.json"
else
  echo "Not available yet."
fi
