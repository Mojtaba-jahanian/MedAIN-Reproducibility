from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np

HOME=Path.home()
ROOT=HOME/"MedAIN_V19_STAGE9"
DOMAINS=["BinRushed","Magrabia","MESSIDOR_Base1","MESSIDOR_Base2","MESSIDOR_Base3"]

results=[]
for d in DOMAINS:
    p=ROOT/"stage9a_pilot"/d/"result.json"
    if not p.exists():
        raise SystemExit(f"MISSING {p}")
    r=json.loads(p.read_text())
    if r.get("status")!="PASS":
        raise SystemExit(f"NON-PASS {p}")
    results.append(r)

def cnt(stage,key,cmp):
    return sum(cmp(
        r["mechanism"]["TP"][stage][key],
        r["mechanism"]["AO"][stage][key]
    ) for r in results)

leak_better={s:cnt(s,"task_leakage_ratio",lambda a,b:a<b) for s in ("stage1","stage2")}
eff_better={s:cnt(s,"protection_efficiency",lambda a,b:a>b) for s in ("stage1","stage2")}
cap_vs_rand={
    s:sum(
        r["mechanism"]["TP"][s]["acquisition_capture_ratio"] >
        r["mechanism"]["RAND"][s]["acquisition_capture_ratio"]
        for r in results
    )
    for s in ("stage1","stage2")
}
dice_tp=np.asarray([r["validation"]["TP"]["mean_fg_dice"] for r in results],float)
dice_ao=np.asarray([r["validation"]["AO"]["mean_fg_dice"] for r in results],float)
dice_folds=int(np.sum(dice_tp>=dice_ao))
dice_mean_diff=float(np.mean(dice_tp-dice_ao))

criteria={
    "TP_leakage_lower_than_AO_4of5_both_stages":
        leak_better["stage1"]>=4 and leak_better["stage2"]>=4,
    "TP_efficiency_higher_than_AO_4of5_both_stages":
        eff_better["stage1"]>=4 and eff_better["stage2"]>=4,
    "TP_capture_above_RAND_4of5_both_stages":
        cap_vs_rand["stage1"]>=4 and cap_vs_rand["stage2"]>=4,
    "validation_task_preservation_not_contrary":
        (dice_folds>=3) or (dice_mean_diff>=0.0),
}
decision="GO_CONFIRMATORY" if all(criteria.values()) else "NO_GO_FREEZE_MEDAIN_NOVELTY"

summary={
    "stage":"MedAIN_Stage9A_Prospective_Pilot_Aggregate",
    "status":"PASS",
    "n_folds":5,
    "seed":1103,
    "leakage_better_fold_counts":leak_better,
    "efficiency_better_fold_counts":eff_better,
    "TP_capture_above_RAND_fold_counts":cap_vs_rand,
    "TP_val_dice_ge_AO_fold_count":dice_folds,
    "mean_TP_minus_AO_val_dice":dice_mean_diff,
    "criteria":criteria,
    "decision":decision,
    "pilot_is_final_inferential_evidence":False,
    "target_test_evaluated":False,
    "external_data_used":False,
}
p=ROOT/"stage9a_pilot"/"STAGE9A_PILOT_AGGREGATE.json"
p.write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
