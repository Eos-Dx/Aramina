# Attenuation Experiment v0.1

## Scope

Research-only feature and evaluation code for a validated attenuation or
optical-density measurement at P1, P2, and P3 per breast. It does not change
the Aramina product model, feature schema, threshold, preprocessing, intended
use, output, or clinical contract. It is not for autonomous diagnosis.

## Raw Archive Audit

Audited source: `data/product-aramis-data/combined_archive.h5`.

| Raw field or set | Observed archive state | Treatment |
|---|---|---|
| `Sample Transmission` sets | 249 total; 231 at P1-P3 and immediately followed by same-position `Sample Main` | Pairing metadata only |
| `transmission_pct` | Present on 32 sets, always `10.0`; missing on 217 | Never treated as attenuation |
| `correction_factor` | Present on the same 32 sets, always `1.5964`; missing on 217 | Never treated as attenuation |
| Explicit attenuation value/formula/reference/provenance/units | Absent | No valid attenuation input |

The transmission set contains a raw frame and acquisition fields, but the
archive does not identify an open/reference acquisition, a reference set ID, a
validated formula, or a provenance status for an attenuation coefficient.
`Sample Transmission` metadata therefore cannot support a calculation such as
an optical-density transform. The experiment reports it as unavailable.

Raw P1-P3 pairing coverage is not valid attenuation coverage. Before the
provenance gate, 76 breast sessions have raw P1-P3 pairs. The exact canonical
`LABEL_MAP` admits 51 sessions from 36 patients; 32 sessions from 31 patients
are biopsy-labelled, and 15 patients have canonical-labelled raw pairs on both
sides. Eight breast sessions carry both numeric placeholders at P1-P3.
Validated attenuation coverage is zero because all required provenance fields
are absent.

Archive coverage uses the product's exact `LABEL_MAP`: `BENIGN=0` and
`CANCER=1`. `NORMAL`, `ATYPICAL`, `PRE_CANCEROUS`, mixed-case variants, and
other statuses remain unlabeled in the audit and cannot enter the experiment.

Run `audit_archive_transmission_metadata()` against the exact archive before a
future research run. It writes the set inventory and session coverage in memory
and fails closed: `transmission_pct`, `correction_factor`, and
`breast_density` cannot become attenuation inputs.

For reproducible artifacts, run:

```bash
PYTHONPATH=src python scripts/audit_attenuation_archive.py \
  /path/to/combined_archive.h5 \
  /path/to/attenuation-archive-audit
```

The runner writes `transmission_inventory.csv`, `transmission_coverage.csv`,
and `attenuation_archive_status.json`. The status artifact records source path,
availability, raw P1-P3 coverage, canonical-labelled coverage, and validated
coverage.

## Input Contract

`extract_three_point_attenuation_features()` requires one row per standardized
point with:

```text
patientId, side, position, attenuation_value,
attenuation_provenance_status=measured_validated,
attenuation_formula_id, attenuation_reference_id, attenuation_units
```

`breast_density` is categorical mammographic metadata. It is neither a numeric
attenuation value nor a fallback. Missing point data, repeated P1/P2/P3 rows,
non-numeric values, or incomplete provenance mark that breast unavailable; the
extractor never selects a row or imputes a value.

The physical attenuation formula is not assigned by this code. The approved
formula must be named in `attenuation_formula_id` and linked to its source in
`attenuation_reference_id` before a numeric value is accepted.

## Exploratory Relative Optical Density

The current archive also cannot support an `exploratory_relative_optical_density`
feature. A future exploratory calculation could define, at each matched point,
`log(I_contralateral / I_target)`, but only after a controlled direct-beam ROI
or integration method, detector correction, and same-session/stability protocol
are defined and recorded. Existing `keele_helpers/baseline_gfrm.py` reads
external `beam_roi.txt` and `transmission.txt` sidecars; neither input nor a ROI
identifier is stored in the transmission sets. The archive has 249 raw
512x768 transmission frames, but no beam-ROI, direct-reference, or optical
density formula metadata. Left and right breasts are stored in separate H5
sessions, with zero bilateral pairs in one H5 session. Relative optical-density
coverage is therefore zero and no value is derived from the frames.

## Features

For validated point values \(a_{P1}, a_{P2}, a_{P3}\):

```text
attenuation_mean = mean(a_P1, a_P2, a_P3)
attenuation_std = population_std(a_P1, a_P2, a_P3)
attenuation_range = max(a) - min(a)
delta_Pk = a_target,Pk - a_contralateral,Pk
mean_abs_delta = mean(abs(delta_P1), abs(delta_P2), abs(delta_P3))
rms_delta = sqrt(mean(delta_P1^2, delta_P2^2, delta_P3^2))
```

Symmetry fields require complete validated P1-P3 values for both breasts. They
remain unavailable otherwise. No categorical breast-density feature is used.

## Paired Evaluation

`evaluate_paired_attenuation_contribution()` accepts caller-supplied,
pre-existing baseline predictors and compares a research LogisticRegression
baseline with the same model plus all attenuation/symmetry fields. Both use the
same complete cases, patient-level repeated stratified folds, per-fold training
thresholds, and held-out cases. It reports ROC AUC, sensitivity, specificity,
balanced accuracy, PPV, NPV, threshold, and confusion counts, plus per-fold
augmented-minus-baseline deltas.

This is a research experiment only. It must remain unavailable when the input
is single class, has too few patients per class, has duplicate target cases,
lacks complete validated bilateral data, or lacks traceable measurement
provenance.
