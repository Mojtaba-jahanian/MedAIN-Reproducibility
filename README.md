# MedAIN — Acquisition-Sensitivity Attenuation and its Limits in Retinal Segmentation

**Research artifacts and audited results | Single author: Mojtaba Jahanian | Research snapshot: 2026-10-15**

This repository accompanies a work-in-progress research manuscript on **MedAIN**, task-protected/acquisition-only subspace interventions, domain shift and optic-disc/optic-cup segmentation. The current paper candidate is **V21.25** (target journal under consideration: *Medical Image Analysis*). **The manuscript has not been accepted or published.**

> **Scientific finding, not a performance claim.** The archived four-external-dataset benchmark does **not** establish that trained MedAIN outperforms strong existing baselines. Archived external domain-macro Dice: IBN **0.838491**, StrongAug **0.835020**, ERM **0.816245**, trained MedAIN **0.812094**. Post-training TP showed lower **source-validation** Dice than OFF in **all 10 fold×seed runs**, even though the in-sample task-leakage mechanisms favored TP versus AO. Negative results and limitations are preserved.

## What is here

| Location | Contents | Reproduction level |
|---|---|---|
| `results/stage10ab/` | Ten per-run source-validation spatial false-positive reports (842 image–seed observations), all-image CSV, frozen aggregation summary | **Recomputable exactly from released outputs** |
| `results/archived/` | 30 Stage7R-C and 90 Stage8R-C run summaries, one record per line | **Recomputable from archived summary values**; not fresh model inference |
| `code/stage10ab/` | Source-validation-only spatial error audit Python code, with configurable local project root | **Requires local Stage9B trainer, manifests and 10 trained checkpoints (not included)** |
| `code/historical_stage9a/` | Preserved historical Stage9A pilot code and prospective protocol lock | Historical code, **NOT proof** that original pilot GO decision was executed |
| `code/legacy_stage7/`, `code/legacy_stage8/` | Historical external evaluation and comparator training sources | Provided as archival scripts, without their complete execution environments and model weights |
| `code/core/`, `code/data_preparation/`, `code/smoke_tests/` | Earlier algorithm prototypes and data workflow helpers | May require original dataset layouts |
| `scripts/verify_results.py` | Python standard-library audit for all supplied aggregate results | **Runs immediately without datasets or a GPU** |
| `figures/stage10ab_spatial_error_bar.png` | Derived aggregate visualisation (no underlying fundus images) | Chart from the supplied Stage10AB output |
| `docs/` | Data access, provenance, GitHub workflow and release limitations | Read before reuse |

## Quick verification — no installation required

With Python 3.11 or later:

```bash
python scripts/verify_results.py
```

This verifies the 842 image–seed observations and compares each published aggregate number against the underlying rows, then reconstructs external domain-macro summary means from the original archived numerical records. It does **not** regenerate predictions from raw images.

## Spatial false-positive endpoint (Stage 10-AB)

On each **source-validation** optic-disc/cup mask (class 1/2), remote false-positive pixels are predicted foreground located on reference background **more than 32 pixels** away from *any* reference foreground pixel; large connected islands have at least **64 pixels**, using **8-connectivity**. Secondary sensitivity thresholds: 16 and 48 pixels. The masks are 256×256 cached representations. The measure is **pixel counts per image**, not a percentage or independent patient-level incidence.

Across the 842 **dependent** image–seed observations: OFF **0.6936**, TP **6.9667**, AO **9.1378**, RAND **0.7743** mean remote false-positive pixels. Large-island frequencies are respectively 2/842, 8/842, 12/842 and 2/842. These statistics describe *exploratory source validation*; they are not target-test or external generalization outcomes.

## Data and full reruns

No original fundus scans, masks, patient data, checkpoint weights, or hospital/source-site records are redistributed here. Use the original dataset hosts and honor their respective licences and permitted uses. See [DATA_ACCESS.md](docs/DATA_ACCESS.md) and [REPRODUCIBILITY_LIMITATIONS.md](docs/REPRODUCIBILITY_LIMITATIONS.md).

**Critical limitation:** The exact Stage9B production trainer `src/stage9b_anchor_train.py`, audited identity-safe cached manifests and ten ERM checkpoint files are **not currently available in the deliverable source archives**; therefore an independent one-command pixel-level regeneration cannot be promised by this repository. An external researcher can immediately reproduce the disclosed *summary arithmetic* and inspect the archived code, but cannot replay the full original pipeline from this snapshot alone.

## Authorship / license / status

Research author: **Mojtaba Jahanian**. This code release has **no explicit redistribution license yet**; choose and add a code license only after reviewing third-party dependencies and rights. Dataset licenses are separate. **No DOI, journal acceptance or public archive deposit is claimed**. No manuscript draft or journal correspondence is included in this code-focused package.

For paper claims, always distinguish archived Stage7/8 external inference outputs, Stage9A historical prospective plan (pilot GO evidence not recovered) and later exploratory Stage10 source-validation reconstructions.
