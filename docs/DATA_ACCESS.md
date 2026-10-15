# Data access and responsible reuse

- Main development dataset: **RIGA+** retinal fundus/optic-disc/cup annotations. The preserved research protocol references https://zenodo.org/records/6325549 ; confirm record, terms and current access before downloading.
- Archived external evaluation datasets named in run summaries: **REFUGE**, **Drishti-GS**, **ORIGA**, **PAPILA**. They may have distinct usage, registration, clinical-data and redistribution conditions. Obtain from their authorized hosts; this release includes no original pixels or annotations.
- The dataset paths / preprocessing caches used by the Stage9B trainer are **not in this public package**. The historical data manifest generator is a legacy helper, not a proven full reconstruction of the current cache.
- `sample_uid` values in derived source-validation result files are dataset record labels. Check dataset policies before posting derived per-image metrics to a fully public repository.
- No raw source image, mask, patient identifier, dataset copy, trained checkpoint or unauthorized sample panel is included.
