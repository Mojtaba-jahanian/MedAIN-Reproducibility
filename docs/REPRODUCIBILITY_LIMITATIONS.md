# Scientific and technical scope of this public snapshot

**Auditable now:** 10/10 Stage10AB report integrity and arithmetic across 842 dependent image–seed observations; 90 archived Stage8RC and 30 Stage7RC external run summaries; examples of prototype/historical model and comparator source.

**Not independently rerunnable from the package alone:** Original Stage9B production `stage9b_anchor_train.py`, identity-safe manifests and cache, 10 ERM checkpoint weights, external evaluation checkpoint files, complete historical environment and all source images. Consequently this is **not** a full independent reproduction of the published/unpublished experimental pipeline.

**Not verified:** Historical Stage9A 5-fold pilot with seed1103 followed by a documented prospective GO decision. Subsequent Stage10 source-validation evaluations do **not** retroactively prove the prospective GO. Several historic manifest byte-level SHA digests differ from identity-safe reconstructions; scientific semantics and numerical provenance need to be described precisely.

**Dependence:** The 842 records are image×seed assessments using five overlapping source-validation folds and two model seeds. They are not 842 independent patients, and no inferential claim is derived by treating them as independent.

**Negative results are essential:** post-training TP Dice is lower than OFF/ERM in all ten source fold–seed comparisons; archived external trained MedAIN mean Dice is lower than IBN, StrongAug and ERM. The in-sample mechanism metrics cannot replace generalization/accuracy evidence.

**Privacy/security:** Stage7RC original archive metadata contained absolute user-specific checkpoint paths. Only the path string was replaced by `checkpoints/<filename>` in the `stage7rc_run_summaries.jsonl` records. `source_file_sha256` preserves the hash of the original source JSON file; it cannot be recomputed from the redacted record. No secret or source data access is provided.

**Stage10AB source adaptation:** The original runner source SHA256 is `08985b61df5b550fb1666b4a1fc0a87ffcb44b124101eb1c65a852ee6bbece6f`. The public copy only changes the hardcoded execution root to `MEDAIN_ROOT` (or `~/MedAIN_Stage9B_Reproduction`) for portability. No change to the metric algorithm is intended; this adapted source has not been rerun for a fresh inference experiment.

**Use:** This is an **exploratory** evidence/code archive and should not be used to claim confirmatory evaluation, clinical readiness, or medical performance improvement.
