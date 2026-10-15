#!/usr/bin/env python3
"""Independent arithmetic check of shipped archived and Stage10AB JSON data.
No patient images, checkpoints or inference are needed. Standard library only.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict
from statistics import mean

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / 'results'
METHODS = ('OFF', 'TP', 'AO', 'RAND')
FOLDS = ('BinRushed', 'Magrabia', 'MESSIDOR_Base1', 'MESSIDOR_Base2', 'MESSIDOR_Base3')
SEEDS = (2207, 3301)

def verify_stage10():
    summary = json.loads((RESULTS / 'stage10ab/stage10ab_summary.json').read_text())
    assert summary['n_reports'] == 10
    assert summary['n_image_seed_pairs'] == 842
    measured = {m: [] for m in METHODS}
    large = {m: [] for m in METHODS}
    total = 0
    for domain in FOLDS:
        for seed in SEEDS:
            f = RESULTS / 'stage10ab/runs' / f'{domain}_seed{seed}.json'
            item = json.loads(f.read_text())
            assert item['domain'] == domain and item['seed'] == seed
            assert item['status'] == 'PASS' and item['target_test_evaluated'] is False
            assert item['training_performed'] is False
            assert len(item['rows']) == item['n_source_validation']
            total += len(item['rows'])
            for row in item['rows']:
                for m in METHODS:
                    measured[m].append(int(row[m+'_remote_fp_gt32px']))
                    large[m].append(int(row[m+'_components_ge64_gt32px'] > 0))
    assert total == 842
    print('Stage 10-AB: runs=10, image-seed observations=',total)
    for m in METHODS:
        observed = mean(measured[m]); severe = mean(large[m])
        reported = summary['summary'][m]
        assert math.isclose(observed,reported['mean_remote_fp_pixels'],abs_tol=1e-12)
        assert math.isclose(severe,reported['fraction_component_ge64px'],abs_tol=1e-12)
        assert sum(measured[m]) == reported['total_remote_fp_pixels']
        print(f'  {m:4s} mean remote FP pixels={observed:.6f}, large-island fraction={severe:.6f}')
    return summary

def read_jsonl(p):
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip(): yield json.loads(line)

def verify_external():
    out = {}
    for stage, expected_count in [('stage7rc', 30), ('stage8rc', 90)]:
        path = RESULTS / 'archived' / (stage+'_run_summaries.jsonl')
        rows = list(read_jsonl(path))
        assert len(rows) == expected_count
        grouped = defaultdict(list)
        for item in rows:
            r = item['record']
            assert r['status'] == 'PASS'
            grouped[r['method']].append(float(r['overall_domain_macro']['mean_dice']))
        out[stage] = {m:mean(vals) for m,vals in sorted(grouped.items())}
        print(f'{stage}: {len(rows)} archived reports (arithmetic reconstruction only)')
        for m, val in out[stage].items():
            print(f'  {m:12s} external domain-macro Dice={val:.6f}')
    assert math.isclose(out['stage8rc']['IBN'],.8384913471002463,abs_tol=1e-11)
    assert math.isclose(out['stage8rc']['StrongAug'],.8350202275467615,abs_tol=1e-11)
    assert math.isclose(out['stage8rc']['ERM_stage8'],.8162452525432535,abs_tol=1e-11)
    assert math.isclose(out['stage7rc']['medain'],.812094,abs_tol=2e-5)
    return out

if __name__ == '__main__':
    verify_stage10()
    verify_external()
    print('RESULT = PASS (summary arithmetic only; NO inference replication)')
